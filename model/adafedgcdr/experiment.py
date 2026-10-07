import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseArtifacts
from .data import Case, SplitManifest
from .device import resolve_device
from .fusion import AdaptiveFusion
from .rank import batched_rank_metrics, metrics_to_python

K_VALUES = (0, 1, 3, 5, 10)
METHODS = ("source-only", "target-only", "fixed-0.5", "dynamic-only", "reliability-only", "full")
COMPARABLE_METHODS = ("source-only", "fixed-0.5", "dynamic-only", "reliability-only", "full")
METRIC_KEYS = ("hr@5", "hr@10", "ndcg@5", "ndcg@10")


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


def _rank_order(values) -> list[int]:
    return sorted(range(len(values)), key=lambda i: (-values[i], i))


def _metrics_from_rank(rank: int) -> dict[str, float]:
    metrics = {}
    for k in (5, 10):
        metrics[f"hr@{k}"] = 1.0 if rank < k else 0.0
        metrics[f"ndcg@{k}"] = 1.0 / math.log2(rank + 2) if rank < k else 0.0
    return metrics


def rank_metrics(scores: torch.Tensor, label_index: int) -> dict[str, float]:
    values = scores.detach().cpu().tolist()
    order = _rank_order(values)
    return _metrics_from_rank(order.index(label_index))


def bootstrap_ci(values, seed: int, draws: int = 1000):
    if not values:
        return (None, None)
    rng = random.Random(seed)
    count = len(values)
    means = []
    for _ in range(draws):
        total = 0.0
        for _ in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    low = means[int(round(0.025 * (draws - 1)))]
    high = means[int(round(0.975 * (draws - 1)))]
    return (low, high)


def paired_bootstrap_ci(first, second, seed: int, draws: int = 1000):
    if len(first) != len(second):
        raise ValueError("配对 bootstrap 需要两组等长的用户统计量")
    return bootstrap_ci([a - b for a, b in zip(first, second)], seed, draws)


def fuse_user_vec(method: str, fusion: AdaptiveFusion, batch):
    if method not in METHODS:
        raise ValueError(f"未知方法 {method}，应为 {METHODS}")
    source_vec = batch["source_vec"]
    source_mask = batch["source_mask"]
    source_count = batch["source_count"]
    history_vec = batch["history_vec"]
    history_mask = batch["history_mask"]
    count = history_mask.sum(dim=1)
    local = (history_vec * history_mask[..., None]).sum(dim=1) / count.clamp_min(1)[:, None]
    agreement = F.cosine_similarity(source_vec, local[:, None, :], dim=-1)
    agreement = torch.where(count[:, None] > 0, agreement, torch.zeros_like(agreement))
    if method in ("reliability-only", "full"):
        logits = fusion.domain_bias + fusion.count_scale * torch.log1p(source_count) + fusion.agreement_scale * agreement
        source_weight = torch.softmax(logits.masked_fill(~source_mask, -1e9), dim=1) * source_mask
        source_weight = source_weight / source_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        source_weight = source_mask.float() / source_mask.sum(dim=1, keepdim=True).clamp_min(1)
    source = (source_weight[..., None] * source_vec).sum(dim=1)
    has_source = source_mask.any(dim=1)
    if method == "target-only":
        weight = torch.ones(count.shape[0], device=source_vec.device)
    elif method == "source-only":
        weight = torch.zeros(count.shape[0], device=source_vec.device)
    elif method in ("fixed-0.5", "reliability-only"):
        weight = torch.full((count.shape[0],), 0.5, device=source_vec.device)
    else:
        weight = count / (count + F.softplus(fusion.tau_raw))
    weight = torch.where(count > 0, weight, torch.zeros_like(weight))
    if method != "source-only":
        weight = torch.where(has_source, weight, (count > 0).float())
    user_vec = (1 - weight[:, None]) * source + weight[:, None] * local
    personalized_mask = has_source | (count > 0)
    return user_vec, personalized_mask


def _prepare_rows(cases, artifacts: BaseArtifacts, catalog: Catalog, config: FusionConfig):
    rows = []
    skipped = []
    dimension = artifacts.target_item_vectors.shape[1]
    num_sources = len(artifacts.domain_names)
    max_k = max((len(case.history_item_ids) for case in cases), default=1)
    for case in cases:
        if case.label_item_id not in catalog.popularity:
            skipped.append(case)
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
                "candidate_items": candidates,
                "candidate_item_vectors": candidate_item_vectors,
                "candidate_pop": candidate_pop,
                "label_index": len(candidates) - 1,
            }
        )
    return rows, skipped


def _collate(rows, device):
    batch = {
        "user_id": [row["user_id"] for row in rows],
        "k": torch.tensor([row["k"] for row in rows], dtype=torch.long, device=device),
        "label_index": torch.tensor([row["label_index"] for row in rows], dtype=torch.long, device=device),
        "candidate_items": [row["candidate_items"] for row in rows],
    }
    for key in ("source_vec", "source_mask", "source_count", "history_vec", "history_mask", "candidate_item_vectors", "candidate_pop"):
        batch[key] = torch.stack([row[key] for row in rows]).to(device)
    return batch


def _score(method: str, model: AdaptiveFusion, batch, fallback: bool):
    user_vec, personalized_mask = fuse_user_vec(method, model, batch)
    scores = torch.einsum("bd,bcd->bc", user_vec, batch["candidate_item_vectors"])
    if fallback:
        scores = torch.where(personalized_mask[:, None], scores, batch["candidate_pop"])
    return scores, personalized_mask


def _case_records(model, rows, config: FusionConfig, method: str = "full", fallback: bool = True) -> list[dict]:
    records = []
    model.eval()
    device = resolve_device(config.device)
    with torch.inference_mode():
        for start in range(0, len(rows), config.batch_size):
            batch = _collate(rows[start : start + config.batch_size], device)
            if method == "target-only":
                keep = [i for i, k in enumerate(batch["k"].tolist()) if k > 0]
                if not keep:
                    continue
                batch = {
                    key: ([value[i] for i in keep] if key in ("candidate_items", "user_id") else value[keep])
                    for key, value in batch.items()
                }
            scores, _ = _score(method, model, batch, fallback)
            label_index = batch["label_index"]
            metrics = batched_rank_metrics(scores, label_index)
            rows_batch = rows[start : start + config.batch_size]
            if method == "target-only":
                rows_batch = [rows_batch[i] for i in keep]
            records.extend(metrics_to_python(metrics, rows_batch, batch["candidate_items"]))
    return records


def _mean(values):
    return sum(values) / len(values) if values else None


def _user_means(records, k_values) -> dict[str, float]:
    per_user: dict[str, list] = {}
    for record in records:
        if record["k"] in k_values:
            per_user.setdefault(record["user_id"], []).append(record["ndcg@10"])
    return {user_id: _mean(values) for user_id, values in per_user.items()}


def _method_summary(method: str, records: list[dict], seed: int, catalog_size: int) -> dict:
    applicable_k = K_VALUES if method != "target-only" else tuple(k for k in K_VALUES if k > 0)
    summary = {}
    for k in K_VALUES:
        if k not in applicable_k:
            summary[str(k)] = {
                "applicable": False,
                "count": 0,
                **{key: None for key in METRIC_KEYS},
                **{f"{key}_ci": [None, None] for key in METRIC_KEYS},
                "coverage@10": None,
            }
            continue
        subset = [record for record in records if record["k"] == k]
        entry = {"applicable": True, "count": len(subset)}
        for key in METRIC_KEYS:
            values = [record[key] for record in subset]
            entry[key] = _mean(values)
            entry[f"{key}_ci"] = list(bootstrap_ci(values, seed)) if values else [None, None]
        seen = set()
        for record in subset:
            seen.update(record["top10_items"])
        entry["coverage@10"] = len(seen) / catalog_size if catalog_size and subset else None
        summary[str(k)] = entry
    user_values = list(_user_means(records, applicable_k).values())
    summary["user_mean_ndcg@10"] = {
        "value": _mean(user_values),
        "ci": list(bootstrap_ci(user_values, seed)),
        "k_values": list(applicable_k),
    }
    return summary


def _stage_info(rows: list[dict], skipped_cases: list) -> dict:
    stages = {}
    for k in K_VALUES:
        subset = [row for row in rows if row["k"] == k]
        skipped_k = [case for case in skipped_cases if case.k == k]
        attempted = len(subset) + len(skipped_k)
        missing = _mean([1.0 - float(row["source_mask"].float().mean()) for row in subset])
        stages[str(k)] = {
            "attempted": attempted,
            "count": len(subset),
            "unrecommendable_ratio": (len(skipped_k) / attempted) if attempted else 0.0,
            "missing_source_ratio": missing,
        }
    return stages


def _negative_transfer(method_records: dict) -> dict:
    full = {(record["user_id"], record["k"]): record for record in method_records["full"]}
    target_only = {(record["user_id"], record["k"]): record for record in method_records["target-only"]}
    per_stage = {"0": None}
    for k in K_VALUES:
        if k == 0:
            continue
        users = sorted(user_id for user_id, stage in target_only if stage == k)
        harmed = [user_id for user_id in users if full[(user_id, k)]["ndcg@10"] < target_only[(user_id, k)]["ndcg@10"]]
        per_stage[str(k)] = (len(harmed) / len(users)) if users else None
    users = sorted({user_id for user_id, _ in target_only})
    app_k = tuple(k for k in K_VALUES if k > 0)
    full_means = _user_means(method_records["full"], app_k)
    target_means = _user_means(method_records["target-only"], app_k)
    harmed = [user_id for user_id in users if full_means[user_id] < target_means[user_id]]
    return {
        "definition": "同一 k 下 full 的 NDCG@10 低于 target-only 的用户占比；整体口径为 k∈{1,3,5,10} 用户等权均值对比",
        "per_stage": per_stage,
        "user_ratio": (len(harmed) / len(users)) if users else None,
    }


def _comparison(methods_metrics: dict, method_records: dict, seed: int) -> dict:
    baseline_values = {
        method: methods_metrics[method]["user_mean_ndcg@10"]["value"]
        for method in COMPARABLE_METHODS
        if method != "full" and methods_metrics[method]["user_mean_ndcg@10"]["value"] is not None
    }
    strongest = max(baseline_values, key=lambda method: (baseline_values[method], method))
    full_means = _user_means(method_records["full"], K_VALUES)
    base_means = _user_means(method_records[strongest], K_VALUES)
    common = sorted(set(full_means) & set(base_means))
    paired = paired_bootstrap_ci(
        [full_means[user_id] for user_id in common], [base_means[user_id] for user_id in common], seed
    )
    full_k0 = {(record["user_id"]): record["ndcg@10"] for record in method_records["full"] if record["k"] == 0}
    source_k0 = {(record["user_id"]): record["ndcg@10"] for record in method_records["source-only"] if record["k"] == 0}
    k0_common = sorted(set(full_k0) & set(source_k0))
    k0_paired = paired_bootstrap_ci(
        [full_k0[user_id] for user_id in k0_common], [source_k0[user_id] for user_id in k0_common], seed
    )
    full_value = methods_metrics["full"]["user_mean_ndcg@10"]["value"]
    k0_full_mean = _mean([full_k0[user_id] for user_id in k0_common])
    k0_source_mean = _mean([source_k0[user_id] for user_id in k0_common])
    effective = (
        full_value is not None
        and baseline_values
        and full_value > baseline_values[strongest]
        and paired[0] is not None
        and paired[0] > 0
        and k0_full_mean is not None
        and k0_full_mean >= k0_source_mean
    )
    return {
        "criteria": (
            "预先固定判据：五个 k 的用户等权平均 NDCG@10 高于最强可比基线，配对用户 bootstrap 差值 95% 区间下界大于 0，"
            "且 k=0 的 NDCG@10 不低于源域单独推荐；target-only 缺 k=0，不参与五 k 平均比较"
        ),
        "strongest_baseline": strongest,
        "full_user_mean_ndcg@10": full_value,
        "strongest_baseline_user_mean_ndcg@10": baseline_values.get(strongest),
        "paired_ndcg@10_diff": {"value": _mean([full_means[u] - base_means[u] for u in common]), "ci": list(paired)},
        "k0_vs_source_only": {
            "full": k0_full_mean,
            "source_only": k0_source_mean,
            "paired_diff_ci": list(k0_paired),
        },
        "effective": bool(effective),
    }


def train_fusion(artifacts: BaseArtifacts, cases: dict, config: FusionConfig, catalog: Catalog) -> FusionCheckpoint:
    if set(cases) - {"fusion", "validation"}:
        raise ValueError(f"训练函数只接受 fusion/validation 组用例，实际包含 {sorted(cases)}")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    model = AdaptiveFusion(num_sources=len(artifacts.domain_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_rows, train_skipped = _prepare_rows(cases.get("fusion", []), artifacts, catalog, config)
    val_rows, val_skipped = _prepare_rows(cases.get("validation", []), artifacts, catalog, config)
    if not train_rows:
        raise ValueError("没有可训练的融合样本（标签均不在目标域目录中）")
    if not val_rows:
        raise ValueError("没有可早停的验证样本（validation 组为空或标签均不在目标域目录中）")
    print(
        f"[train-fusion] seed={config.seed} train={len(train_rows)} 跳过={len(train_skipped)} "
        f"validation={len(val_rows)} 跳过={len(val_skipped)}"
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
            scores, _ = _score("full", model, batch, fallback=False)
            labels = torch.full((scores.shape[0],), scores.shape[1] - 1, dtype=torch.long, device=device)
            losses = F.cross_entropy(scores, labels, reduction="none")
            loss = stage_balanced_loss(losses, batch["k"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        records = _case_records(model, val_rows, config, "full") if val_rows else []
        summary = _method_summary("full", records, config.seed, len(catalog.items))
        score = summary[str(10)]["ndcg@10"]
        score_value = -loss.item() if score is None else score
        stage_line = " ".join(
            f"k={k}: ndcg@10={summary[str(k)]['ndcg@10']} hr@10={summary[str(k)]['hr@10']}" for k in K_VALUES
        )
        print(f"[epoch {epoch}] seed={config.seed} val={len(val_rows)} ndcg@10={score} {stage_line}")

        if best_state is None or score_value > best_score:
            best_score = score_value
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = {"overall": summary["user_mean_ndcg@10"], "stages": {str(k): summary[str(k)] for k in K_VALUES}}
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
    model = AdaptiveFusion(num_sources=len(checkpoint.base.domain_names)).to(resolve_device(checkpoint.config.device))
    model.load_state_dict(checkpoint.fusion_state)
    rows, skipped_cases = _prepare_rows(cases, checkpoint.base, catalog, checkpoint.config)
    if not rows:
        raise ValueError("测试组没有可评估用例（test 组为空或标签均不在目标域目录中）")
    seed = checkpoint.config.seed
    method_records = {method: _case_records(model, rows, checkpoint.config, method) for method in METHODS}
    methods_metrics = {
        method: _method_summary(method, method_records[method], seed, len(catalog.items)) for method in METHODS
    }
    total = len(cases)
    return {
        "k_values": list(K_VALUES),
        "seed": seed,
        "candidate_count": checkpoint.config.candidate_count,
        "cases": {
            "total": total,
            "evaluated": len(rows),
            "unrecommendable": {"count": len(skipped_cases), "ratio": (len(skipped_cases) / total) if total else 0.0},
        },
        "stages": _stage_info(rows, skipped_cases),
        "methods": methods_metrics,
        "comparison": _comparison(methods_metrics, method_records, seed),
        "negative_transfer": _negative_transfer(method_records),
    }


def _fmt(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(metrics: dict) -> str:
    lines = ["# AdaFedGCDR 冷启动对照评估报告", ""]
    lines.append("> 本报告由 `python -m model.adafedgcdr evaluate` 自动生成，只描述离线评估结果，不代表线上效果。")
    lines += [
        "",
        "## 配置与数据概况",
        "",
        f"- 种子：{metrics['seed']}；候选物品数（含 1 个正物品）：{metrics['candidate_count']}；阶段 k：{metrics['k_values']}",
        f"- 用例总数 {metrics['cases']['total']}，可评估 {metrics['cases']['evaluated']}，"
        f"不可推荐 {metrics['cases']['unrecommendable']['count']}"
        f"（{_fmt(metrics['cases']['unrecommendable']['ratio'])}）",
        "",
        "## 各方法分阶段指标（用户级 bootstrap 95% 区间）",
        "",
        "| 方法 | k | 样本数 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | NDCG@10 95% CI | 覆盖率@10 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for method in METHODS:
        for k in metrics["k_values"]:
            entry = metrics["methods"][method][str(k)]
            ci = entry["ndcg@10_ci"]
            ci_text = "不适用" if not entry["applicable"] else f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"
            lines.append(
                f"| {method} | {k} | {entry['count']} | {_fmt(entry['hr@5'])} | {_fmt(entry['hr@10'])} "
                f"| {_fmt(entry['ndcg@5'])} | {_fmt(entry['ndcg@10'])} | {ci_text} | {_fmt(entry['coverage@10'])} |"
            )
    lines += ["", "## 五阶段用户等权平均 NDCG@10", "", "| 方法 | 参与 k | 均值 | 95% CI |", "|---|---|---|---|"]
    for method in METHODS:
        summary = metrics["methods"][method]["user_mean_ndcg@10"]
        lines.append(
            f"| {method} | {summary['k_values']} | {_fmt(summary['value'])} "
            f"| [{_fmt(summary['ci'][0])}, {_fmt(summary['ci'][1])}] |"
        )
    comparison = metrics["comparison"]
    paired = comparison["paired_ndcg@10_diff"]
    k0 = comparison["k0_vs_source_only"]
    lines += [
        "",
        "## 判据与结论",
        "",
        f"- 判据（预先固定）：{comparison['criteria']}",
        f"- 最强可比基线：{comparison['strongest_baseline']}（均值 {_fmt(comparison['strongest_baseline_user_mean_ndcg@10'])}），"
        f"full 均值 {_fmt(comparison['full_user_mean_ndcg@10'])}",
        f"- 配对用户 bootstrap 差值：{_fmt(paired['value'])}，95% CI [{_fmt(paired['ci'][0])}, {_fmt(paired['ci'][1])}]",
        f"- k=0 对比源域单独推荐：full {_fmt(k0['full'])} vs source-only {_fmt(k0['source_only'])}，"
        f"配对差值 95% CI [{_fmt(k0['paired_diff_ci'][0])}, {_fmt(k0['paired_diff_ci'][1])}]",
        f"- 结论：{'达到预先固定的有效判据' if comparison['effective'] else '未达到预先固定的有效判据'}（如实报告，不因结果调整阈值）",
        "",
        "## 每阶段数据质量与负迁移",
        "",
        "| k | 尝试样本 | 可评估样本 | 不可推荐比例 | 缺失源域比例 | 负迁移用户比例 |",
        "|---|---|---|---|---|---|",
    ]
    negative = metrics["negative_transfer"]
    for k in metrics["k_values"]:
        stage = metrics["stages"][str(k)]
        lines.append(
            f"| {k} | {stage['attempted']} | {stage['count']} | {_fmt(stage['unrecommendable_ratio'])} "
            f"| {_fmt(stage['missing_source_ratio'])} | {_fmt(negative['per_stage'][str(k)])} |"
        )
    lines += [
        "",
        f"- 负迁移口径：{negative['definition']}；整体负迁移用户比例：{_fmt(negative['user_ratio'])}",
        "",
        "## 已知限制与结论边界",
        "",
        "- Amazon 数据的 user-core/item-core 过滤造成样本选择偏差：留下的用户与物品比真实分布更活跃、更热门。",
        "- 真实新用户的目标域历史可能短于过滤阈值，本评估覆盖的用户不一定代表最稀疏的真实冷启动用户。",
        "- 本评估为离线实验，不能称为线上效果；未新增任何隐私保证。",
        "- `base.pt` 含用户表示，只应保存在本地忽略目录；导出到生产服务需要单独的隐私设计与预算核算。",
        "- 旧 FedGCDR 的现有评估依赖已训练的目标域用户 ID embedding，不与零交互结果同表比较。",
    ]
    return "\n".join(lines) + "\n"


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
