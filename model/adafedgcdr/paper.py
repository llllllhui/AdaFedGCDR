"""Controlled paper experiments using the unmodified FedGCDR Client/Server.

Sparse protocol: A/B are training users, C/D expose only their observed prefix.
Cold protocol: only A target interactions enter the backbone; B trains heads.
Never import main.py, which trains and selects checkpoints at import time.
"""
import copy
import math
import random
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from ..fedgcdr.model import MLP
from ..fedgcdr.party import Client, Server
from .base import BaseArtifacts
from .data import Case, SplitManifest, manifest_sha256, user_views
from .experiment import (
    FusionConfig, _case_records, _collate, _prepare_rows, fuse_user_vec,
    stage_balanced_loss,
)
from .fusion import AdaptiveFusion
from .rank import batched_rank_metrics, metrics_to_python


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def quarantine_development_users(manifest, development_users):
    """Previously inspected test users become validation users, never test again."""
    moved=set(manifest.users['test']) & set(development_users)
    users=dict(manifest.users)
    users['test']=tuple(sorted(set(users['test'])-moved))
    users['validation']=tuple(sorted(set(users['validation'])|moved))
    return SplitManifest(manifest.target_domain,manifest.seed,users),tuple(sorted(moved))


def prefix_cases(events, manifest, group, k_values, fixed_label=False):
    allowed = set(manifest.users[group])
    cases = []
    for user, view in user_views(events, manifest.target_domain).items():
        history = view['target_events']
        if user not in allowed or len(history) < 11 or not view['usable_sources']:
            continue
        for k in k_values:
            cases.append(Case(user, k, tuple(e.item_id for e in history[:k]), history[10 if fixed_label else k].item_id))
    return cases


def target_inputs(views, manifest, protocol, k=None, item_core=1):
    """Catalog uses training groups only; C/D labels never select items."""
    groups = ('base', 'fusion') if protocol == 'sparse' else ('base',)
    train_users = {u for g in groups for u in manifest.users[g]}
    counts = Counter(e.item_id for u in train_users for e in views[u]['target_events'])
    catalog = {i for i, count in counts.items() if count >= item_core}
    visible = {}
    for user, view in views.items():
        if user in train_users:
            events = view['target_events']
        elif protocol == 'sparse':
            if k is None or k <= 0:
                raise ValueError('Sparse protocol requires k > 0')
            events = view['target_events'][:k]
        else:
            events = []
        visible[user] = tuple(e.item_id for e in events if e.item_id in catalog)
    return tuple(sorted(catalog)), visible, train_users


@dataclass
class SourceState:
    users: tuple
    names: tuple
    item_tables: dict
    knowledge: dict
    vectors: dict


def make_args(seed, device, user_count, batch=16):
    return SimpleNamespace(num_domain=4, num_users=user_count, embedding_size=16,
        device=device, user_batch=batch, lr_gat=0.001, lr_mf=0.005,
        local_epoch=3, eps=8., delta=1e-5, dp=True)


def train_sources(events, manifest, seed, device, rounds, progress=print):
    seed_all(seed)
    views = user_views(events, manifest.target_domain)
    users = tuple(sorted(views))
    names = tuple(sorted({e.domain for e in events if e.domain != manifest.target_domain}))
    if len(names) != 3:
        raise ValueError('This experiment requires three source domains')
    tables = {name: {item: i for i, item in enumerate(sorted({e.item_id for v in views.values() for e in v['usable_sources'] if e.domain == name}))} for name in names}
    sizes = [len(tables[n]) for n in names] + [1]
    args = make_args(seed, device, len(users))
    data = [[[tables[n][e.item_id] for e in views[u]['usable_sources'] if e.domain == n] for n in names] + [[]] for u in users]
    clients = [Client(i, data[i], sizes, 0, list(names)+[manifest.target_domain], args) for i in range(len(users))]
    mapping = {i: {} for i in range(len(users))}
    servers = []
    for j, name in enumerate(names):
        ids = [i for i in range(len(users)) if data[i][j]]
        for mapped, i in enumerate(ids):
            mapping[i][name] = mapped
        servers.append(Server(j, name, sizes[j], clients, ids, [], mapping, args))
    for server in servers:
        for epoch in range(rounds):
            progress(f'source={server.domain_name} round={epoch+1}/{rounds}', flush=True)
            server.kt_stage(False)
    knowledge = {u: [c.knowledge[j][0].detach().cpu().clone() if c.knowledge[j] else None for j in range(3)] for u, c in zip(users, clients)}
    return SourceState(users, names, tables, knowledge, {s.domain_name: s.V.detach().cpu().clone() for s in servers})


def train_backbone(events, manifest, source, seed, device, protocol, k,
                   warm_rounds, transfer_rounds, fine_rounds, patience,
                   item_core=1, validate=None, progress=print):
    seed_all(seed + 10000)
    views = user_views(events, manifest.target_domain)
    item_ids, visible, train_users = target_inputs(views, manifest, protocol, k, item_core)
    table = {item: i for i, item in enumerate(item_ids)}
    names = list(source.names) + [manifest.target_domain]
    sizes = [len(source.item_tables[n]) for n in source.names] + [len(table)]
    args = make_args(seed, device, len(source.users))
    data = [[[source.item_tables[n][e.item_id] for e in views[u]['usable_sources'] if e.domain == n] for n in source.names]+[[table[i] for i in visible[u]]] for u in source.users]
    clients = [Client(i, data[i], sizes, 0, names, args) for i in range(len(source.users))]
    mapping = {i: {} for i in range(len(source.users))}
    ids = [i for i, u in enumerate(source.users) if visible[u]]
    for mapped, i in enumerate(ids):
        mapping[i][manifest.target_domain] = mapped
    for u, client in zip(source.users, clients):
        client.user_id = u
        client.knowledge[:3] = [[z.to(device)] if z is not None else [] for z in source.knowledge[u]]
    server = Server(3, manifest.target_domain, len(table), clients, ids, [], mapping, args)
    server.mlp = [MLP(16).to(device) for _ in range(4)]
    trace = []
    for phase, rounds, transfer in [('target-warm', warm_rounds, False), ('target-transfer', transfer_rounds, True)]:
        for epoch in range(rounds):
            progress(f'{protocol} k={k} {phase} round={epoch+1}/{rounds}', flush=True)
            server.kt_stage(transfer)
    # The original entry point hands the final GAT user representations to MF.
    server.U = server.user_embedding_with_attention.detach().clone()
    best = None
    best_score = -float('inf')
    stale = 0
    for epoch in range(fine_rounds):
        progress(f'{protocol} k={k} fine-tune round={epoch+1}/{fine_rounds}', flush=True)
        server.mf_train()
        score = validate(server, table, source.users, mapping) if validate else None
        trace.append({'epoch': epoch, 'validation_ndcg@10': score})
        if score is None or score > best_score:
            best_score = score if score is not None else best_score
            best = (server.U.detach().clone(), server.V.detach().clone(), epoch)
            stale = 0
        else:
            stale += 1
        if validate and stale >= patience:
            break
    if best:
        server.U, server.V, best_epoch = best
    else:
        best_epoch = None
    # Cache one DP export for every method using this backbone.
    seed_all(seed + 20000)
    std = math.sqrt(2*math.log(1.25/args.delta))/args.eps
    vectors, counts, masks = {}, {}, {}
    with torch.no_grad():
        for user in source.users:
            rows, count_row, mask_row = [], [], []
            for j, name in enumerate(source.names):
                z = source.knowledge[user][j]
                count = sum(e.domain == name for e in views[user]['usable_sources'])
                if z is None:
                    rows.append(torch.zeros(16))
                    mask_row.append(False)
                else:
                    noisy = Client.l2_clip(z.to(device), clients[0].sensitivity)
                    noisy = noisy + torch.normal(0.,std,size=(16,)).to(device)
                    rows.append(server.mlp[j](noisy).detach().cpu())
                    mask_row.append(True)
                count_row.append(float(count))
            vectors[user] = torch.stack(rows)
            counts[user] = torch.tensor(count_row)
            masks[user] = torch.tensor(mask_row)
    base = BaseArtifacts(list(source.names), vectors, counts, masks, table,
        server.V.detach().cpu().clone(), sorted(train_users), manifest_sha256(manifest))
    embeddings = {source.users[i]: server.U[mapping[i][manifest.target_domain]].detach().cpu().clone() for i in ids}
    return base, embeddings, {'fine_trace': trace, 'selected_fine_epoch': best_epoch,
        'catalog_size': len(table), 'target_clients': len(ids), 'protocol': protocol, 'k': k}


def original_records(rows, embeddings, item_table, item_vectors, device):
    records = []
    for start in range(0, len(rows), 64):
        subset = rows[start:start+64]
        batch = _collate(subset, device)
        if any(row['user_id'] not in embeddings for row in subset):
            raise ValueError('Original FedGCDR requires a trained target user embedding')
        user_vec = torch.stack([embeddings[r['user_id']] for r in subset]).to(device)
        scores = torch.einsum('bd,bcd->bc', user_vec, batch['candidate_item_vectors'])
        labels = torch.full((len(subset),),scores.shape[1]-1,device=device,dtype=torch.long)
        result = batched_rank_metrics(scores, labels)
        records.extend(metrics_to_python(result, subset, [r['candidate_items'] for r in subset]))
    return records


def popularity_records(rows, device):
    records = []
    for start in range(0,len(rows),64):
        subset=rows[start:start+64]
        batch=_collate(subset,device)
        scores=batch['candidate_pop']
        labels=torch.full((len(subset),),scores.shape[1]-1,device=device,dtype=torch.long)
        records.extend(metrics_to_python(batched_rank_metrics(scores,labels),subset,[r['candidate_items'] for r in subset]))
    return records


def prepare_rows(cases, base, catalog, sparse=False):
    rows, skipped = _prepare_rows(cases, base, catalog, FusionConfig(seed=2026,candidate_count=100))
    if sparse:
        no_history = [r for r in rows if not r['history_mask'].any()]
        rows = [r for r in rows if r['history_mask'].any()]
    else:
        no_history = []
    return rows, {'total':len(cases),'unknown_label':len(skipped),'no_known_prefix':len(no_history),'evaluated':len(rows)}


def fit_head(method, train_rows, val_rows, seed, device, max_epochs, patience):
    if not train_rows or not val_rows:
        raise ValueError(f'No training or validation rows for {method}')
    seed_all(seed + 30000)
    model = AdaptiveFusion(train_rows[0]['source_vec'].shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    best, score_best, stale, trace = None, -float('inf'), 0, []
    config = FusionConfig(seed=2026,candidate_count=100,device=device)
    for epoch in range(max_epochs):
        model.train()
        order = list(range(len(train_rows)))
        random.Random(seed+epoch).shuffle(order)
        for start in range(0,len(order),64):
            batch = _collate([train_rows[i] for i in order[start:start+64]],device)
            user, _ = fuse_user_vec(method,model,batch)
            scores = torch.einsum('bd,bcd->bc',user,batch['candidate_item_vectors'])
            labels = torch.full((scores.shape[0],),scores.shape[1]-1,device=device,dtype=torch.long)
            loss = stage_balanced_loss(F.cross_entropy(scores,labels,reduction='none'),batch['k'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        records = _case_records(model,val_rows,config,method)
        # Equal stage weight, without consulting test labels or the train loss.
        score = np.mean([np.mean([r['ndcg@10'] for r in records if r['k']==k]) for k in sorted({r['k'] for r in records})])
        trace.append({'epoch':epoch,'validation_ndcg@10':float(score)})
        if score > score_best:
            best = {n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
            score_best, stale = score, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best)
    return model, {'trace':trace,'best_validation_ndcg@10':float(score_best),
        'best_epoch':next(t['epoch'] for t in trace if t['validation_ndcg@10']==score_best),
        'method':method}
