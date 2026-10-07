"""Batched (GPU-capable) trainer for the AdaFedGCDR base representation.

Reimplements the federated knowledge-transfer stage of ``model/fedgcdr`` with the
clients of one server round processed as a padded batch, so the tiny per-client
GAT/MLP work runs as a handful of dense tensor ops instead of a Python loop.

Deliberate deviations from ``model/fedgcdr`` (documented, not hidden):
  * Local item parameters are restricted to the rows a client actually touches
    (the sampled candidate rows).  The original clones and optimises the whole
    ``V`` matrix per client; only sampled rows get non-zero gradients and the
    returned delta is only read at sampled rows, so the update is equivalent --
    it just stops paying O(num_items) per client.
  * With ``user_batch > 1`` the chunk trains one set of shared leaves (GAT
    attention, target-domain MLP) under summed gradients and a single Adam
    state, and ``V``/attention/MLP are aggregated with a length-weighted
    average instead of the original client-by-client EMA.  At ``user_batch=1``
    the aggregation semantics coincide with the original.

The RNG stream (numpy shuffle -> per-client DP ``torch.normal`` draws ->
``torch.randint`` negative sampling, every draw on CPU) matches the original
exactly, but the re-expressed tensor ops (bmm/einsum vs matmul/Linear) change
float summation order, so results are NOT bit-identical to ``train_base``:
ULP-level differences amplify over training rounds.  Reproduce the original's
numbers with ``train_base``; treat the two implementations as separate
baselines and never mix their checkpoints.

``config.fast_sampling`` (CLI ``--fast-sampling``) only batches the per-row
``unique``; negative sampling and DP noise stay per-client and the RNG stream
is unchanged, so it is a pure speed switch: ``train_base_batched`` produces
bit-identical artifacts with and without the flag (asserted in
``tests/test_ada_batched.py``).
"""

import copy
import math
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from ..fedgcdr.model import MLP
from ..fedgcdr.party import Server
from .base import BaseArtifacts, BaseConfig
from .data import SplitManifest, manifest_sha256, user_views
from .device import resolve_device
from .unique_batch import batched_unique, group_values

ALPHA = 0.01
BETA = 0.01


def _dummy_args(num_domain, num_users, config):
    # ``device`` is pinned to CPU on purpose: these objects exist only to replay
    # the original initialisation RNG (and thus the initial U/V/GAT values) in a
    # device-independent way. Passing the real device here would make the seeds
    # device-dependent and break reproducibility.
    return SimpleNamespace(
        num_domain=num_domain,
        num_users=num_users,
        embedding_size=config.embedding_size,
        device="cpu",
        user_batch=config.user_batch,
        lr_gat=config.lr_gat,
        lr_mf=0.005,
        local_epoch=config.local_epoch,
        eps=config.eps,
        delta=config.delta,
        dp=config.dp,
    )


def _gat_layer(x, valid, a):
    """Star-graph GATLayer.

    ``GATLayer`` uses an adjacency of ``eye(N)`` with row/column 0 set to 1, i.e.
    the centre node (index 0, the user) attends to every valid node while each
    other node attends exactly to ``{0, itself}``.  So the N x N attention matrix
    is never needed -- this computes the same thing in O(N*D), which keeps long
    histories off the memory limit.
    """
    f = a.shape[1] // 2
    h1 = torch.bmm(x, a[:, f:, :]).squeeze(-1)  # (B,N)
    h2 = torch.bmm(x, a[:, :f, :]).squeeze(-1)  # (B,N)
    neg = -9e15
    s0 = (h1[:, 0:1] + h2).masked_fill(~valid, neg)
    w0 = torch.nan_to_num(torch.softmax(s0, dim=-1), nan=0.0)
    out0 = torch.einsum("bn,bnd->bd", w0, x)
    if x.shape[1] == 1:
        return out0.unsqueeze(1)
    pair = torch.stack([h1[:, 1:] + h2[:, 0:1], h1[:, 1:] + h2[:, 1:]], dim=-1)
    w = torch.softmax(pair, dim=-1)
    out_i = w[..., 0:1] * x[:, 0:1, :] + w[..., 1:2] * x[:, 1:, :]
    return torch.cat([out0.unsqueeze(1), out_i], dim=1)


def _ls_lm(f_t, tv):
    with torch.no_grad():
        sim = (F.cosine_similarity(tv, f_t[:, None, :], dim=-1) + 1) / 2
        total = sim.sum(dim=1, keepdim=True)
    f_s = (sim[:, :, None] * tv).sum(dim=1) / total
    ls = (ALPHA / 2) * ((f_t - f_s) ** 2).sum(dim=1)
    lm = (BETA / 2) * ((tv - f_t[:, None, :]) ** 2).mean(dim=-1).sum(dim=1)
    return ls, lm


def _mlp_forward(tv, p):
    w1, b1, w2, b2, w3, b3 = p
    x = torch.tanh(torch.einsum("bki,bkoi->bko", tv, w1) + b1)
    x = torch.tanh(torch.einsum("bki,bkoi->bko", x, w2) + b2)
    x = torch.tanh(torch.einsum("bki,bkoi->bko", x, w3) + b3)
    return x


def _adam_step(params, grads, states, lr, step):
    b1, b2, eps = 0.9, 0.999, 1e-8
    for p, g, st in zip(params, grads, states):
        m, v = st
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        mhat = m / (1 - b1**step)
        vhat = v / (1 - b2**step)
        p.data.addcdiv_(mhat, vhat.sqrt().add_(eps), value=-lr)


class _Domain:
    def __init__(self, name, idx, users, local_of, num_items, device):
        self.name = name
        self.idx = idx
        self.users = users
        self.local_of = local_of
        self.num_items = num_items
        self.device = device
        self.U = None
        self.V = None
        self.a_in = None
        self.a_out = None
        self.attn = None
        self.mlp = None
        self.items_flat = None
        self.items_off = None
        self.pos_of = None
        self.local_of_t = None

    def build_csr(self, train_data, num_users):
        """一次性把本域的交互展平成 CSR，避免每 chunk 回头查 ``list[list[int]]``。

        只保留长度索引（``items_off``）：交互本体仍按原顺序从 ``train_data`` 取，
        以维持负采样的 RNG 消耗顺序。``pos_of`` / ``local_of_t`` 按“全局 client
        数”开长度，域内用户只占其中一段；全量下这些查找每轮会做
        ``用户数 × 轮数`` 次。
        """
        flat, offsets = [], [0]
        for cid in self.users:
            flat.append(len(train_data[cid][self.idx]))
            offsets.append(offsets[-1] + flat[-1])
        self.items_off = torch.tensor(offsets, dtype=torch.int64)
        self.pos_of = torch.full((num_users,), -1, dtype=torch.int64)
        self.local_of_t = torch.zeros(num_users, dtype=torch.int64)
        for mid, cid in enumerate(self.users):
            self.pos_of[cid] = mid
            self.local_of_t[cid] = self.local_of[cid]

    def item_lengths(self, cids):
        """本域各 client 的交互条数，一次 gather 返回。"""
        positions = self.pos_of[cids]
        return self.items_off[positions + 1] - self.items_off[positions]


def _batched_kt(dom, tf_flag, train_data, knowledge, knowledge_mask, sensitivity,
                dp_std, config, num_domain, use_dp, fast_unique=False):
    d = config.embedding_size
    bsz = config.user_batch
    device = dom.device
    users = dom.users
    n = len(users)
    ids = list(users)
    np.random.shuffle(ids)
    batch_num = math.ceil(n / bsz) if n else 0

    for bt in range(batch_num):
        chunk = ids[bt * bsz : (bt + 1) * bsz]
        B = len(chunk)
        cids = torch.tensor(chunk, dtype=torch.int64)
        # ---- per-client RNG-faithful preparation ----
        # ``total_item`` / ``ratings`` stay per-client because the negative draws
        # are consumed in client order and must match the original RNG stream.
        # Lengths come from the CSR instead of ``len(train_data[cid][idx])``, which
        # removes a per-client list lookup from every round.
        lengths_i = dom.item_lengths(cids)
        infos = []
        for cid, L in zip(chunk, lengths_i.tolist()):
            items = train_data[cid][dom.idx]
            tv = None
            if tf_flag:
                tv = torch.zeros(num_domain, d)
                for j in range(num_domain):
                    if j == dom.idx:
                        continue
                    if bool(knowledge_mask[cid, j]):
                        temp = knowledge[cid, j].clone()
                        nrm = temp.norm()
                        if nrm > sensitivity:
                            temp = temp / nrm * sensitivity
                    else:
                        temp = torch.zeros(d)
                    noise = torch.normal(0.0, dp_std, size=(1, d)).squeeze()
                    tv[j] = temp + (noise if use_dp else torch.zeros(d))
            neg = torch.randint(0, dom.num_items, (4 * L, 1), dtype=torch.int64).squeeze()
            items_t = torch.tensor(items, dtype=torch.int64)
            total_item = torch.cat([items_t, neg])
            ratings = torch.cat([torch.ones(L), torch.zeros(4 * L)])
            infos.append((cid, items_t, total_item, ratings, tv, L))

        lmax = max(i[5] for i in infos)
        smax = 5 * lmax
        uniqs = None if fast_unique else [torch.unique(i[2], return_inverse=True) for i in infos]
        rows_init = None
        inv_pad = torch.zeros(B, smax, dtype=torch.int64, device=device)
        invitems_pad = torch.zeros(B, lmax, dtype=torch.int64, device=device)
        lengths = torch.tensor([i[5] for i in infos], dtype=torch.float32, device=device)

        # ---- batched masks: derived purely from lengths, no per-client copy ----
        ar5 = torch.arange(smax, device=device)
        lens_i = lengths.long()
        valid_score = ar5[None, :] < (5 * lens_i)[:, None]
        # first L slots are positives (rating 1), the following 4L are negatives
        ratings_pad = (ar5[None, :] < lens_i[:, None]).float()

        map_ids = dom.local_of_t[cids].to(device)
        user_emb_init = dom.U[map_ids]

        if fast_unique:
            # padding positions get an out-of-range sentinel so they form one extra
            # group; ``V`` is extended with a zero row to receive that group.
            total_pad = torch.full((B, smax), dom.num_items, dtype=torch.int64)
            for b, info in enumerate(infos):
                total_pad[b, : info[2].shape[0]] = info[2]
            inv_b, counts_b = batched_unique(total_pad, dom.num_items)
            inv_b = inv_b.to(device)
            counts_b = counts_b.to(device)
            total_pad_dev = total_pad.to(device)
            rmax = int(counts_b.max())
            real_mask = torch.arange(rmax, device=device)[None, :] < counts_b[:, None]
            # group id -> row of V: column g holds the g-th ascending unique item;
            # the sentinel group collects ``num_items`` and reads the appended
            # zero row of ``v_ext``.
            group_of = group_values(total_pad_dev, inv_b, rmax, dom.num_items)
            v_ext = torch.cat([dom.V, torch.zeros(1, d, device=device)], dim=0)
            rows_init = v_ext[group_of] * real_mask[..., None]
            inv_pad = inv_b
            invitems_pad = inv_b[:, :lmax]
        else:
            rmax = max(u[0].shape[0] for u in uniqs)
            rows_init = torch.zeros(B, rmax, d, device=device)
            for b, (cid, items_t, total_item, ratings, tv, L) in enumerate(infos):
                uniq, inv = uniqs[b]
                uniq_dev = uniq.to(device)
                rows_init[b, : uniq.shape[0]] = dom.V[uniq_dev]
                inv_pad[b, : 5 * L] = inv.to(device)
                invitems_pad[b, :L] = inv[:L].to(device)
        tv_batch = None
        if tf_flag:
            tv_batch = torch.stack([i[4] for i in infos]).to(device)

        node_valid = torch.zeros(B, 1 + lmax, dtype=torch.bool, device=device)
        node_valid[:, 0] = True
        node_valid[:, 1:] = torch.arange(lmax, device=device)[None, :] < lengths[:, None]

        # ---- trainable leaves ----
        user_emb = user_emb_init.clone().requires_grad_(True)
        rows = rows_init.clone().requires_grad_(True)
        a_in = dom.a_in.clone().unsqueeze(0).repeat(B, 1, 1).requires_grad_(True)
        a_out = dom.a_out.clone().unsqueeze(0).repeat(B, 1, 1).requires_grad_(True)
        mlp_batch = []
        if tf_flag:
            for ln, pn in (("L1", "weight"), ("L1", "bias"), ("L2", "weight"),
                           ("L2", "bias"), ("L3", "weight"), ("L3", "bias")):
                stacked = torch.stack([getattr(getattr(m, ln), pn).detach() for m in dom.mlp])
                mlp_batch.append(stacked.unsqueeze(0).repeat(B, *([1] * stacked.dim())).clone().requires_grad_(True))

        params = [user_emb, rows, a_in, a_out] + mlp_batch
        states = [(torch.zeros_like(p), torch.zeros_like(p)) for p in params]

        def build_x0():
            node_items = rows.gather(1, invitems_pad[..., None].expand(-1, -1, d))
            return torch.cat([user_emb.unsqueeze(1), node_items], dim=1)

        for step in range(1, config.local_epoch + 1):
            x0 = build_x0()
            h1 = _gat_layer(x0, node_valid, a_in)
            if tf_flag:
                temp_vec = _mlp_forward(tv_batch, mlp_batch)
                ls, lm = _ls_lm(h1[:, 0, :], temp_vec)
                x2 = torch.cat([h1, temp_vec], dim=1)
                v2 = torch.cat([node_valid, torch.ones(B, num_domain, dtype=torch.bool, device=device)], dim=1)
            else:
                ls = lm = torch.zeros(B, device=device)
                x2 = h1
                v2 = node_valid
            out = _gat_layer(x2, v2, a_out)
            scoring = rows.gather(1, inv_pad[..., None].expand(-1, -1, d))
            logits = (out[:, 0, :][:, None, :] * scoring).sum(-1)
            bce = F.binary_cross_entropy(torch.sigmoid(logits), ratings_pad, reduction="none")
            bce = (bce * valid_score).sum(dim=1) / (5 * lengths)
            loss = (bce + ls + lm).sum()
            grads = torch.autograd.grad(loss, params)
            _adam_step(params, grads, states, config.lr_gat, step)

        # ---- final no_grad pass (raw transfer_vec, as in the original) ----
        with torch.no_grad():
            x0 = build_x0()
            h1 = _gat_layer(x0, node_valid, a_in)
            if tf_flag:
                x2 = torch.cat([h1, tv_batch], dim=1)
                v2 = torch.cat([node_valid, torch.ones(B, num_domain, dtype=torch.bool, device=device)], dim=1)
            else:
                x2 = h1
                v2 = node_valid
            out = _gat_layer(x2, v2, a_out)
        intermediate = h1[:, 0, :]
        # ``u_att`` mirrors the original's per-client attention output, which
        # ``train_base`` stores into ``user_embedding_with_attention`` but the
        # adafedgcdr export never reads.  Kept for parity; drop together with
        # that field if it is ever removed upstream.
        u_att = out[:, 0, :]

        # ---- aggregation (weighted average over the chunk) ----
        p = lengths / lengths.sum()
        dom.a_in = torch.einsum("b,bij->ij", p, a_in.detach())
        dom.a_out = torch.einsum("b,bij->ij", p, a_out.detach())
        dom.U[map_ids] = user_emb.detach()
        for b, cid in enumerate(chunk):
            knowledge[cid, dom.idx] = intermediate[b]
            knowledge_mask[cid, dom.idx] = True

        # V update: sum over clients of (snapshot - local) / clients-using-row
        if fast_unique:
            delta_real = rows_init - rows.detach() * real_mask[..., None]
            flat_group = group_of.reshape(-1)
            flat_real = real_mask.reshape(-1)
            # 哨兵组的 delta 恒为零（初始化即零行、无梯度），把它排除后索引
            # 天然不越过 dom.V。除法与累加顺序刻意与非 fast 分支一致：
            # 逐次出现除以触达计数后按 chunk 内 client 顺序 index_add，
            # 保证两条路径逐位一致（tests/test_ada_batched.py 有对拍）。
            keep = flat_real & (flat_group < dom.num_items)
            counts = torch.zeros(dom.num_items, device=device)
            counts.index_add_(0, flat_group[keep], flat_real.to(counts.dtype)[keep])
            counts.clamp_min_(1)
            divisor = counts[group_of.clamp_max(dom.num_items - 1)][..., None]
            scaled = (-delta_real / divisor).reshape(-1, d)[keep]
            dom.V.index_add_(0, flat_group[keep], scaled)
        else:
            all_idx, all_delta = [], []
            for b, (cid, items_t, total_item, ratings, tv, L) in enumerate(infos):
                uniq, inv = uniqs[b]
                uniq_dev = uniq.to(device)
                delta = rows_init[b, : uniq.shape[0]] - rows.detach()[b, : uniq.shape[0]]
                all_idx.append(uniq_dev)
                all_delta.append(delta)
            idx_cat = torch.cat(all_idx)
            delta_cat = torch.cat(all_delta)
            counts = torch.zeros(dom.num_items, device=device)
            counts.index_add_(0, idx_cat, torch.ones_like(idx_cat, dtype=torch.float32))
            counts.clamp_min_(1)
            dom.V.index_add_(0, idx_cat, -delta_cat / counts[idx_cat, None])

        if tf_flag:
            for idx, par_t in enumerate(mlp_batch):
                avg = torch.einsum("b,b...->...", p, par_t.detach())
                for mid, m in enumerate(dom.mlp):
                    list(m.parameters())[idx].data.copy_(avg[mid])


def train_base_batched(events, manifest: SplitManifest, config: BaseConfig) -> BaseArtifacts:
    if config.target_domain != manifest.target_domain:
        raise ValueError(f"配置目标域 {config.target_domain} 与拆分目标域 {manifest.target_domain} 不一致")
    d = config.embedding_size
    device = resolve_device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    target = config.target_domain
    views = user_views(events, target)
    user_ids = sorted(views)
    user_index = {u: i for i, u in enumerate(user_ids)}
    base_users = set(manifest.users["base"])
    source_names = sorted({e.domain for e in events if e.domain != target})
    domain_names = source_names + [target]
    num_domain = len(domain_names)

    source_item_ids = {n: set() for n in source_names}
    for uid in user_ids:
        for e in views[uid]["usable_sources"]:
            source_item_ids[e.domain].add(e.item_id)
    source_tables = {n: {it: i for i, it in enumerate(sorted(s))} for n, s in source_item_ids.items()}
    target_item_ids = sorted(
        {e.item_id for uid in manifest.users["base"] if uid in views for e in views[uid]["target_events"]}
    )
    target_items = {it: i for i, it in enumerate(target_item_ids)}
    num_items = [len(source_tables[n]) for n in source_names] + [len(target_items)]

    train_data = []
    domain_users = {n: [] for n in domain_names}
    usable_counts = {n: [0] * len(user_ids) for n in source_names}
    for uid in user_ids:
        ui = user_index[uid]
        per = []
        for n in source_names:
            events_n = [e for e in views[uid]["usable_sources"] if e.domain == n]
            usable_counts[n][ui] = len(events_n)
            idx = [source_tables[n][e.item_id] for e in events_n]
            per.append(idx)
            if idx:
                domain_users[n].append(ui)
        if uid in base_users and views[uid]["target_events"]:
            per.append([target_items[e.item_id] for e in views[uid]["target_events"]])
            domain_users[target].append(ui)
        else:
            per.append([])
        train_data.append(per)
    user_dic = {i: {} for i in range(len(user_ids))}
    for n in domain_names:
        for map_id, cid in enumerate(domain_users[n]):
            user_dic[cid][n] = map_id

    # build original objects on CPU only to reproduce init RNG + initial values
    #
    # ``Client.__init__`` consumes no RNG (party.py only reads ``len(clients)`` in
    # ``Server.__init__``), and ``train_base_batched`` never calls ``kt_stage`` /
    # ``Client.train`` -- it only reads the initial U / V / GAT / attention values.
    # So the per-user objects are placeholders: skipping them drops 55k object
    # constructions and 55k list->tensor conversions with no RNG change.
    args = _dummy_args(num_domain, len(user_ids), config)
    clients = [None] * len(user_ids)
    orig = {}
    for j, n in enumerate(source_names):
        orig[n] = Server(j, n, num_items[j], clients, domain_users[n], [], user_dic, args)
    orig[target] = Server(len(source_names), target, num_items[-1], clients, domain_users[target], [], user_dic, args)
    mlp_modules = [MLP(d).to("cpu") for _ in range(num_domain)]
    orig[target].mlp = mlp_modules

    domains = {}
    for j, n in enumerate(domain_names):
        srv = orig[n]
        dom = _Domain(n, j, domain_users[n], {}, num_items[j], device)
        dom.local_of = {cid: mid for mid, cid in enumerate(domain_users[n])}
        dom.U = srv.U.detach().clone().to(device)
        dom.V = srv.V.detach().clone().to(device)
        dom.a_in = srv.item_gat.in2hidden.A.detach().clone().to(device)
        dom.a_out = srv.item_gat.hidden2out.A.detach().clone().to(device)
        dom.attn = srv.domain_attention.detach().clone().to(device)
        dom.build_csr(train_data, len(user_ids))
        domains[n] = dom
    domains[target].mlp = [copy.deepcopy(m).to(device) for m in mlp_modules]

    knowledge = torch.zeros(len(user_ids), num_domain, d)
    knowledge_mask = torch.zeros(len(user_ids), num_domain, dtype=torch.bool)
    sensitivity = math.sqrt(1.0)
    dp_std = math.sqrt(2 * math.log(1.25 / config.delta)) / config.eps
    fast_unique = config.fast_sampling

    for name in source_names:
        for _ in range(config.source_rounds):
            _batched_kt(domains[name], False, train_data, knowledge, knowledge_mask,
                        sensitivity, dp_std, config, num_domain, config.dp, fast_unique)
    for _ in range(config.target_rounds):
        _batched_kt(domains[target], True, train_data, knowledge, knowledge_mask,
                    sensitivity, dp_std, config, num_domain, config.dp, fast_unique)

    std = math.sqrt(2 * math.log(1.25 / config.delta)) / config.eps
    source_vectors, source_counts, source_mask = {}, {}, {}
    with torch.no_grad():
        for i, uid in enumerate(user_ids):
            rows, counts, masks = [], [], []
            for j, n in enumerate(source_names):
                count = float(usable_counts[n][i])
                present = bool(knowledge_mask[i, j])
                if count > 0 and present:
                    vec = knowledge[i, j].detach().to(device)
                    nrm = vec.norm()
                    if nrm > sensitivity:
                        vec = vec / nrm * sensitivity
                    if config.dp:
                        vec = vec + torch.normal(0.0, std, size=(1, d)).to(device).squeeze()
                    vec = domains[target].mlp[j](vec).detach().cpu()
                    masks.append(True)
                else:
                    vec = torch.zeros(d)
                    masks.append(False)
                rows.append(vec)
                counts.append(count)
            source_vectors[uid] = torch.stack(rows) if rows else torch.zeros(0, d)
            source_counts[uid] = torch.tensor(counts, dtype=torch.float32)
            source_mask[uid] = torch.tensor(masks, dtype=torch.bool)

    return BaseArtifacts(
        domain_names=list(source_names),
        source_vectors=source_vectors,
        source_counts=source_counts,
        source_mask=source_mask,
        target_items=target_items,
        target_item_vectors=domains[target].V.detach().cpu().clone(),
        target_train_users=[user_ids[i] for i in domain_users[target]],
        split_sha256=manifest_sha256(manifest),
    )
