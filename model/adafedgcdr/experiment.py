import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseArtifacts
from .data import Case, SplitManifest
from .fusion import AdaptiveFusion

K_VALUES = (0, 1, 3, 5, 10)


@dataclass(frozen=True)
class FusionConfig:
    seed: int = 2026
    lr: float = 1e-3
    max_epochs: int = 50
    patience: int = 5
    candidate_count: int = 100
    batch_size: int = 64
    device: str = "cpu"


@dataclass(frozen=True)
class Catalog:
    items: tuple[str, ...]
    user_items: dict[str, frozenset]
    popularity: dict[str, int]


@dataclass(frozen=True)
class FusionCheckpoint:
    fusion_state: dict[str, torch.Tensor]
    config: FusionConfig
    split_sha256: str
    best_epoch: int
    val_metrics: dict


@dataclass(frozen=True)
class EvalCheckpoint:
    base: BaseArtifacts
    fusion_state: dict[str, torch.Tensor]
    config: FusionConfig


def build_catalog(events, manifest: SplitManifest, artifacts: BaseArtifacts) -> Catalog:
    items = tuple(sorted(artifacts.target_items))
    base_users = set(manifest.users["base"])
    popularity = dict.fromkeys(items, 0)
    user_items: dict[str, set] = {}
    for event in events:
        if event.domain != manifest.target_domain:
            continue
        user_items.setdefault(event.user_id, set()).add(event.item_id)
        if event.user_id in base_users and event.item_id in popularity:
            popularity[event.item_id] += 1
    return Catalog(
        items=items,
        user_items={user_id: frozenset(seen) for user_id, seen in user_items.items()},
        popularity=popularity,
    )


def sample_candidates(case: Case, catalog: Catalog, candidate_count: int, seed: int) -> list[str]:
    if candidate_count < 2:
        raise ValueError("candidate_count 至少为 2（1 个正物品 + 1 个负物品）")
    if case.label_item_id not in catalog.popularity:
        raise ValueError(f"标签物品 {case.label_item_id} 不在目标域目录中")
    interacted = catalog.user_items.get(case.user_id, frozenset())
    available = [item for item in catalog.items if item not in interacted]
    needed = candidate_count - 1
    if len(available) < needed:
        raise ValueError(
            f"用户 {case.user_id} 可用负物品不足：需要 {needed}，实际 {len(available)}（请调低 candidate_count）"
        )
    rng = random.Random(f"{seed}:{case.user_id}:{case.k}")
    negatives = rng.sample(available, needed)
    return negatives + [case.label_item_id]


def stage_balanced_loss(losses: torch.Tensor, stages: torch.Tensor) -> torch.Tensor:
    means = [losses[stages == stage].mean() for stage in torch.unique(stages)]
    return torch.stack(means).mean()


def rank_metrics(scores: torch.Tensor, label_index: int) -> dict[str, float]:
    values = scores.detach().cpu().tolist()
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    rank = order.index(label_index)
    metrics = {}
    for k in (5, 10):
        metrics[f"hr@{k}"] = 1.0 if rank < k else 0.0
        metrics[f"ndcg@{k}"] = 1.0 / math.log2(rank + 2) if rank < k else 0.0
    return metrics


def _prepare_rows(cases, artifacts: BaseArtifacts, catalog: Catalog, config: FusionConfig):
    rows = []
    skipped = 0
    dimension = artifacts.target_item_vectors.shape[1]
    num_sources = len(artifacts.domain_names)
    max_k = max((len(case.history_item_ids) for case in cases), default=1)
    for case in cases:
        if case.label_item_id not in catalog.popularity:
            skipped += 1
            continue
        history_vec = torch.zeros(max_k, dimension)
        history_mask = torch.zeros(max_k, dtype=torch.bool)
        for position, item_id in enumerate(case.history_item_ids):
            index = artifacts.target_items.get(item_id)
            if index is not None:
                history_vec[position] = artifacts.target_item_vectors[index]
                history_mask[position] = True
        candidates = sample_candidates(case, catalog, config.candidate_count, config.seed)
        candidate_item_vectors = torch.stack(
            [artifacts.target_item_vectors[artifacts.target_items[item]] for item in candidates]
        )
        candidate_pop = torch.tensor([float(catalog.popularity[item]) for item in candidates])
        rows.append(
            {
                "user_id": case.user_id,
                "k": case.k,
                "source_vec": artifacts.source_vectors[case.user_id].reshape(num_sources, dimension),
                "source_mask": artifacts.source_mask[case.user_id],
                "source_count": artifacts.source_counts[case.user_id],
                "history_vec": history_vec,
                "history_mask": history_mask,
                "candidate_item_vectors": candidate_item_vectors,
                "candidate_pop": candidate_pop,
                "label_index": len(candidates) - 1,
            }
        )
    return rows, skipped


def _collate(rows, device):
    batch = {
        "user_id": [row["user_id"] for row in rows],
        "k": torch.tensor([row["k"] for row in rows], dtype=torch.long),
        "label_index": torch.tensor([row["label_index"] for row in rows], dtype=torch.long),
    }
    for key in ("source_vec", "source_mask", "source_count", "history_vec", "history_mask", "candidate_item_vectors", "candidate_pop"):
        batch[key] = torch.stack([row[key] for row in rows]).to(device)
    return batch


def _score(model: AdaptiveFusion, batch, fallback: bool):
    user_vec, target_weight, source_weight, personalized_mask = model(
        batch["source_vec"], batch["source_mask"], batch["source_count"], batch["history_vec"], batch["history_mask"]
    )
    scores = torch.einsum("bd,bcd->bc", user_vec, batch["candidate_item_vectors"])
    if fallback:
        scores = torch.where(personalized_mask[:, None], scores, batch["candidate_pop"])
    return scores, target_weight, source_weight, personalized_mask


def _case_records(model, rows, config: FusionConfig, fallback: bool = True) -> list[dict]:
    records = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), config.batch_size):
            batch = _collate(rows[start : start + config.batch_size], torch.device(config.device))
            scores, _, _, _ = _score(model, batch, fallback)
            for position in range(len(batch["user_id"])):
                metrics = rank_metrics(scores[position], int(batch["label_index"][position]))
                records.append({"user_id": batch["user_id"][position], "k": int(batch["k"][position]), **metrics})
    return records


def _aggregate(records: list[dict]) -> dict:
    def summarize(subset):
        count = len(subset)
        summary = {"count": count}
        for key in ("hr@5", "hr@10", "ndcg@5", "ndcg@10"):
            summary[key] = sum(record[key] for record in subset) / count if count else None
        return summary

    stages = {}
    for k in K_VALUES:
        stages[str(k)] = summarize([record for record in records if record["k"] == k])
    return {"overall": summarize(records), "stages": stages}


def train_fusion(artifacts: BaseArtifacts, cases: dict, config: FusionConfig, catalog: Catalog) -> FusionCheckpoint:
    if set(cases) - {"fusion", "validation"}:
        raise ValueError(f"训练函数只接受 fusion/validation 组用例，实际包含 {sorted(cases)}")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    model = AdaptiveFusion(num_sources=len(artifacts.domain_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_rows, train_skipped = _prepare_rows(cases.get("fusion", []), artifacts, catalog, config)
    val_rows, val_skipped = _prepare_rows(cases.get("validation", []), artifacts, catalog, config)
    if not train_rows:
        raise ValueError("没有可训练的融合样本（标签均不在目标域目录中）")
    print(
        f"[train-fusion] seed={config.seed} train={len(train_rows)} 跳过={train_skipped} "
        f"validation={len(val_rows)} 跳过={val_skipped}"
    )

    best_score = -1.0
    best_state = None
    best_epoch = -1
    best_metrics: dict = {}
    stale = 0
    for epoch in range(config.max_epochs):
        model.train()
        order = list(range(len(train_rows)))
        random.Random(config.seed + epoch).shuffle(order)
        for start in range(0, len(order), config.batch_size):
            batch = _collate([train_rows[index] for index in order[start : start + config.batch_size]], device)
            scores, _, _, _ = _score(model, batch, fallback=False)
            labels = torch.full((scores.shape[0],), scores.shape[1] - 1, dtype=torch.long, device=device)
            losses = F.cross_entropy(scores, labels, reduction="none")
            loss = stage_balanced_loss(losses, batch["k"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        records = _case_records(model, val_rows, config) if val_rows else []
        summary = _aggregate(records)
        score = summary["overall"]["ndcg@10"]
        score_value = score if score is not None else -loss.item()
        stage_line = " ".join(
            f"k={k}: ndcg@10={summary['stages'][str(k)]['ndcg@10']} hr@10={summary['stages'][str(k)]['hr@10']}"
            for k in K_VALUES
        )
        print(f"[epoch {epoch}] seed={config.seed} val={len(val_rows)} ndcg@10={score} {stage_line}")

        if best_state is None or score_value > best_score:
            best_score = score_value
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = summary
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    print(f"[train-fusion] early_stop_epoch={best_epoch} seed={config.seed} best_val_ndcg@10={best_score}")
    model.load_state_dict(best_state)
    return FusionCheckpoint(
        fusion_state=best_state,
        config=config,
        split_sha256=artifacts.split_sha256,
        best_epoch=best_epoch,
        val_metrics=best_metrics,
    )


def evaluate(checkpoint: EvalCheckpoint, cases: list, catalog: Catalog) -> dict:
    model = AdaptiveFusion(num_sources=len(checkpoint.base.domain_names)).to(torch.device(checkpoint.config.device))
    model.load_state_dict(checkpoint.fusion_state)
    rows, skipped = _prepare_rows(cases, checkpoint.base, catalog, checkpoint.config)
    records = _case_records(model, rows, checkpoint.config, fallback=True)
    summary = _aggregate(records)
    total = len(cases)
    metrics = {
        "k_values": list(K_VALUES),
        "seed": checkpoint.config.seed,
        "candidate_count": checkpoint.config.candidate_count,
        "cases": {
            "total": total,
            "evaluated": len(rows),
            "unrecommendable": {"count": skipped, "ratio": (skipped / total) if total else 0.0},
        },
        "methods": {"full": dict(summary["stages"])},
    }
    return metrics


def save_fusion(path, checkpoint: FusionCheckpoint, events_sha256: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fusion_state": checkpoint.fusion_state,
            "config": asdict(checkpoint.config),
            "split_sha256": checkpoint.split_sha256,
            "best_epoch": checkpoint.best_epoch,
            "val_metrics": checkpoint.val_metrics,
            "events_sha256": events_sha256,
        },
        path,
    )


def load_fusion(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint = FusionCheckpoint(
        fusion_state=dict(payload["fusion_state"]),
        config=FusionConfig(**payload["config"]),
        split_sha256=payload["split_sha256"],
        best_epoch=payload["best_epoch"],
        val_metrics=dict(payload["val_metrics"]),
    )
    return checkpoint, payload["events_sha256"]
