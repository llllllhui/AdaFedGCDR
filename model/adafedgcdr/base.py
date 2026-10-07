import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..fedgcdr.model import MLP
from ..fedgcdr.party import Client, Server
from .data import SplitManifest, manifest_sha256, user_views
from .device import resolve_device


@dataclass(frozen=True)
class BaseConfig:
    target_domain: str
    embedding_size: int = 16
    source_rounds: int = 150
    target_rounds: int = 150
    local_epoch: int = 3
    user_batch: int = 16
    lr_gat: float = 0.001
    eps: float = 8.0
    delta: float = 1e-5
    dp: bool = True
    device: str = "cpu"
    fast_sampling: bool = False
    seed: int = 2026


@dataclass(frozen=True)
class BaseArtifacts:
    domain_names: list[str]
    source_vectors: dict[str, torch.Tensor]
    source_counts: dict[str, torch.Tensor]
    source_mask: dict[str, torch.Tensor]
    target_items: dict[str, int]
    target_item_vectors: torch.Tensor
    target_train_users: list[str]
    split_sha256: str


def train_base(events, manifest: SplitManifest, config: BaseConfig) -> BaseArtifacts:
    if config.target_domain != manifest.target_domain:
        raise ValueError(f"配置目标域 {config.target_domain} 与拆分目标域 {manifest.target_domain} 不一致")
    target = config.target_domain
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    views = user_views(events, target)
    if not views:
        raise ValueError("事件为空，无法训练基础表示")
    user_ids = sorted(views)
    user_index = {user_id: i for i, user_id in enumerate(user_ids)}
    base_users = set(manifest.users["base"])

    source_names = sorted({event.domain for event in events if event.domain != target})
    domain_names = source_names + [target]
    num_domain = len(domain_names)

    source_item_ids = {name: set() for name in source_names}
    for user_id in user_ids:
        for event in views[user_id]["usable_sources"]:
            source_item_ids[event.domain].add(event.item_id)
    source_tables = {name: {item: i for i, item in enumerate(sorted(items))} for name, items in source_item_ids.items()}
    target_item_ids = sorted(
        {event.item_id for user_id in manifest.users["base"] if user_id in views for event in views[user_id]["target_events"]}
    )
    target_items = {item: i for i, item in enumerate(target_item_ids)}
    num_items = [len(source_tables[name]) for name in source_names] + [len(target_items)]

    train_data = []
    domain_users = {name: [] for name in domain_names}
    for user_id in user_ids:
        per_domain = []
        for name in source_names:
            indices = [source_tables[name][e.item_id] for e in views[user_id]["usable_sources"] if e.domain == name]
            per_domain.append(indices)
            if indices:
                domain_users[name].append(user_index[user_id])
        if user_id in base_users and views[user_id]["target_events"]:
            per_domain.append([target_items[e.item_id] for e in views[user_id]["target_events"]])
            domain_users[target].append(user_index[user_id])
        else:
            per_domain.append([])
        train_data.append(per_domain)

    user_dic = {i: {} for i in range(len(user_ids))}
    for name in domain_names:
        for map_id, client_id in enumerate(domain_users[name]):
            user_dic[client_id][name] = map_id

    args = SimpleNamespace(
        num_domain=num_domain,
        num_users=len(user_ids),
        embedding_size=config.embedding_size,
        device=str(device),
        user_batch=config.user_batch,
        lr_gat=config.lr_gat,
        lr_mf=0.005,
        local_epoch=config.local_epoch,
        eps=config.eps,
        delta=config.delta,
        dp=config.dp,
    )

    clients = [Client(i, train_data[i], num_items, 0, domain_names, args) for i in range(len(user_ids))]
    for i, user_id in enumerate(user_ids):
        clients[i].user_id = user_id

    source_servers = {}
    for j, name in enumerate(source_names):
        source_servers[name] = Server(j, name, num_items[j], clients, domain_users[name], [], user_dic, args)
    target_server = Server(len(source_names), target, num_items[-1], clients, domain_users[target], [], user_dic, args)
    mlp_modules = [MLP(config.embedding_size).to(device) for _ in range(num_domain)]
    target_server.mlp = mlp_modules

    for name in source_names:
        for _ in range(config.source_rounds):
            source_servers[name].kt_stage(False)
    for _ in range(config.target_rounds):
        target_server.kt_stage(True)

    std = math.sqrt(2 * math.log(1.25 / config.delta)) / config.eps
    source_vectors, source_counts, source_mask = {}, {}, {}
    with torch.no_grad():
        for i, user_id in enumerate(user_ids):
            client = clients[i]
            rows, counts, masks = [], [], []
            for j, name in enumerate(source_names):
                count = float(sum(1 for e in views[user_id]["usable_sources"] if e.domain == name))
                knowledge = client.knowledge[j]
                if count > 0 and len(knowledge) > 0:
                    vec = Client.l2_clip(knowledge[0].detach().clone().to(device), client.sensitivity)
                    if config.dp:
                        vec = vec + torch.normal(mean=0.0, std=std, size=(1, config.embedding_size)).to(device).squeeze()
                    vec = mlp_modules[j](vec).detach().cpu()
                    masks.append(True)
                else:
                    vec = torch.zeros(config.embedding_size)
                    masks.append(False)
                rows.append(vec)
                counts.append(count)
            source_vectors[user_id] = torch.stack(rows) if rows else torch.zeros(0, config.embedding_size)
            source_counts[user_id] = torch.tensor(counts, dtype=torch.float32)
            source_mask[user_id] = torch.tensor(masks, dtype=torch.bool)

    artifacts = BaseArtifacts(
        domain_names=list(source_names),
        source_vectors=source_vectors,
        source_counts=source_counts,
        source_mask=source_mask,
        target_items=target_items,
        target_item_vectors=target_server.V.detach().cpu().clone(),
        target_train_users=[user_ids[i] for i in domain_users[target]],
        split_sha256=manifest_sha256(manifest),
    )
    return artifacts


def save_base(path, artifacts: BaseArtifacts, config: BaseConfig, events_sha256: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "events_sha256": events_sha256,
            "domain_names": artifacts.domain_names,
            "source_vectors": artifacts.source_vectors,
            "source_counts": artifacts.source_counts,
            "source_mask": artifacts.source_mask,
            "target_items": artifacts.target_items,
            "target_item_vectors": artifacts.target_item_vectors,
            "target_train_users": artifacts.target_train_users,
            "split_sha256": artifacts.split_sha256,
        },
        path,
    )


def load_base(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    artifacts = BaseArtifacts(
        domain_names=list(payload["domain_names"]),
        source_vectors=dict(payload["source_vectors"]),
        source_counts=dict(payload["source_counts"]),
        source_mask=dict(payload["source_mask"]),
        target_items=dict(payload["target_items"]),
        target_item_vectors=payload["target_item_vectors"],
        target_train_users=list(payload["target_train_users"]),
        split_sha256=payload["split_sha256"],
    )
    return artifacts, BaseConfig(**payload["config"]), payload["events_sha256"]
