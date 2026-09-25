import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.adafedgcdr.base import BaseArtifacts
from model.adafedgcdr.data import Case
from model.adafedgcdr.experiment import (
    Catalog,
    EvalCheckpoint,
    FusionConfig,
    _collate,
    _prepare_rows,
    bootstrap_ci,
    evaluate,
    fuse_user_vec,
    paired_bootstrap_ci,
    rank_metrics,
    sample_candidates,
    stage_balanced_loss,
    train_fusion,
)
from model.adafedgcdr.fusion import AdaptiveFusion

D = 4
SEED = 2026
K_VALUES = (0, 1, 3, 5, 10)


def make_catalog(num_items=30, users=("u0", "u1", "u2", "u3")):
    items = tuple(f"t{i:02d}" for i in range(num_items))
    user_items = {}
    for u_index, user_id in enumerate(users):
        user_items[user_id] = frozenset(items[(i + u_index * 13) % num_items] for i in range(12))
    popularity = {item: 1 for item in items}
    return Catalog(items=items, user_items=user_items, popularity=popularity)


def make_user_cases(user_id, item_seq):
    return [Case(user_id, k, tuple(item_seq[:k]), item_seq[k]) for k in K_VALUES]


def make_fixture(num_items=30):
    catalog = make_catalog(num_items)
    items = list(catalog.items)
    target_items = {item: i for i, item in enumerate(items)}
    target_item_vectors = torch.stack([torch.full((D,), (i + 1) / 10.0) for i in range(num_items)])
    users = ["u0", "u1", "u2", "u3"]
    source_vectors = {}
    source_counts = {}
    source_mask = {}
    for u_index, user_id in enumerate(users):
        source_vectors[user_id] = torch.full((1, D), float(u_index + 1))
        source_counts[user_id] = torch.tensor([float(u_index + 1)])
        source_mask[user_id] = torch.tensor([True])
    artifacts = BaseArtifacts(
        domain_names=["CDs"],
        source_vectors=source_vectors,
        source_counts=source_counts,
        source_mask=source_mask,
        target_items=target_items,
        target_item_vectors=target_item_vectors,
        target_train_users=["u0", "u1"],
        split_sha256="split-digest",
    )
    sequences = {user_id: sorted(catalog.user_items[user_id]) for user_id in users}
    cases = {
        "fusion": make_user_cases("u0", sequences["u0"]) + make_user_cases("u1", sequences["u1"]),
        "validation": make_user_cases("u2", sequences["u2"]),
    }
    return artifacts, catalog, cases


def make_eval_cases(artifact_user="u3", num_items=30):
    catalog = make_catalog(num_items)
    seq = sorted(catalog.user_items[artifact_user])
    return make_user_cases(artifact_user, seq)


def small_config(**overrides):
    values = dict(seed=SEED, candidate_count=8, max_epochs=2, patience=2, batch_size=4, device="cpu")
    values.update(overrides)
    return FusionConfig(**values)


def test_negative_sampling_contract():
    catalog = make_catalog(150)
    case = Case(user_id="u0", k=3, history_item_ids=("t00", "t01", "t02"), label_item_id="t03")
    candidates = sample_candidates(case, catalog, candidate_count=100, seed=SEED)
    assert len(candidates) == 100
    negatives = candidates[:-1]
    assert candidates[99] == case.label_item_id
    assert len(set(negatives)) == 99
    assert all(item in catalog.items for item in negatives)
    assert not (set(negatives) & set(catalog.user_items["u0"]))
    assert sample_candidates(case, catalog, candidate_count=100, seed=SEED) == candidates


def test_negative_sampling_varies_with_seed_and_stage():
    catalog = make_catalog(150)
    case_k3 = Case("u0", 3, ("t00",), "t03")
    case_k5 = Case("u0", 5, ("t00",), "t03")
    assert sample_candidates(case_k3, catalog, 10, SEED) != sample_candidates(case_k5, catalog, 10, SEED)
    assert sample_candidates(case_k3, catalog, 10, SEED) != sample_candidates(case_k3, catalog, 10, SEED + 1)


def test_negative_sampling_requires_enough_available_items():
    catalog = make_catalog(5)
    case = Case("u0", 3, ("t00", "t01", "t02"), "t03")
    with pytest.raises(ValueError):
        sample_candidates(case, catalog, candidate_count=10, seed=SEED)


def test_stage_balanced_loss_gives_equal_stage_weights():
    losses = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0])
    stages = torch.tensor([0, 0, 0, 0, 1])
    assert stage_balanced_loss(losses, stages).item() == pytest.approx(2.0)


def test_stage_balanced_loss_over_five_stages():
    losses = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])
    stages = torch.tensor([0, 0, 1, 1, 3, 3, 5, 5, 10, 10])
    assert stage_balanced_loss(losses, stages).item() == pytest.approx(9.0)


def test_train_fusion_rejects_test_group():
    artifacts, catalog, cases = make_fixture()
    with pytest.raises(ValueError):
        train_fusion(artifacts, {"fusion": cases["fusion"], "test": cases["validation"]}, small_config(), catalog)


def test_train_fusion_requires_validation_rows():
    artifacts, catalog, cases = make_fixture()
    with pytest.raises(ValueError, match="验证样本"):
        train_fusion(artifacts, {"fusion": cases["fusion"], "validation": []}, small_config(), catalog)


def test_evaluate_requires_evaluable_rows():
    artifacts, catalog, cases = make_fixture()
    config = small_config()
    checkpoint = train_fusion(artifacts, cases, config, catalog)
    eval_checkpoint = EvalCheckpoint(base=artifacts, fusion_state=checkpoint.fusion_state, config=config)
    with pytest.raises(ValueError, match="可评估用例"):
        evaluate(eval_checkpoint, [], catalog)


def test_train_fusion_and_evaluate_are_deterministic():
    artifacts, catalog, cases = make_fixture()
    config = small_config()
    first = train_fusion(artifacts, cases, config, catalog)
    second = train_fusion(artifacts, cases, config, catalog)
    assert first.best_epoch == second.best_epoch
    assert first.split_sha256 == artifacts.split_sha256
    for name in first.fusion_state:
        assert torch.equal(first.fusion_state[name], second.fusion_state[name])
    eval_cases = make_eval_cases()
    checkpoint = EvalCheckpoint(base=artifacts, fusion_state=first.fusion_state, config=config)
    metrics_first = evaluate(checkpoint, eval_cases, catalog)
    metrics_second = evaluate(checkpoint, eval_cases, catalog)
    assert metrics_first == metrics_second
    assert {"0", "1", "3", "5", "10"} <= set(metrics_first["methods"]["full"])


def test_evaluate_separates_unrecommendable_labels():
    artifacts, catalog, cases = make_fixture()
    config = small_config()
    checkpoint = train_fusion(artifacts, cases, config, catalog)
    eval_cases = make_eval_cases()
    broken = Case("u3", 5, tuple(sorted(catalog.user_items["u3"])[:5]), "ghost-item")
    metrics = evaluate(EvalCheckpoint(base=artifacts, fusion_state=checkpoint.fusion_state, config=config), eval_cases + [broken], catalog)
    assert metrics["cases"]["total"] == len(eval_cases) + 1
    assert metrics["cases"]["evaluated"] == len(eval_cases)
    assert metrics["cases"]["unrecommendable"]["count"] == 1


def test_rank_metrics_positive_first():
    scores = torch.zeros(100)
    scores[99] = 1.0
    metrics = rank_metrics(scores, 99)
    assert metrics["hr@5"] == metrics["hr@10"] == metrics["ndcg@5"] == metrics["ndcg@10"] == 1.0


def test_rank_metrics_positive_sixth():
    scores = torch.zeros(100)
    scores[:5] = 2.0
    scores[99] = 1.0
    metrics = rank_metrics(scores, 99)
    assert metrics["hr@5"] == metrics["ndcg@5"] == 0.0
    assert metrics["hr@10"] == 1.0
    assert metrics["ndcg@10"] == pytest.approx(1.0 / math.log2(7))


def test_bootstrap_ci_is_reproducible():
    values = [0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 1.0]
    first = bootstrap_ci(values, seed=2026, draws=1000)
    second = bootstrap_ci(values, seed=2026, draws=1000)
    assert first == second
    assert first[0] <= first[1]
    assert bootstrap_ci(values, seed=2027, draws=1000) != first


def test_paired_bootstrap_ci_is_reproducible():
    full = [0.5, 0.6, 0.7, 0.4, 0.9]
    baseline = [0.4, 0.6, 0.5, 0.5, 0.8]
    first = paired_bootstrap_ci(full, baseline, seed=2026, draws=1000)
    assert first == paired_bootstrap_ci(full, baseline, seed=2026, draws=1000)
    assert first[0] <= first[1]


def test_full_method_matches_fusion_module():
    artifacts, catalog, cases = make_fixture()
    model = AdaptiveFusion(num_sources=1)
    rows, _ = _prepare_rows(cases["validation"][:1], artifacts, catalog, small_config())
    batch = _collate(rows, torch.device("cpu"))
    user_vec, personalized_mask = fuse_user_vec("full", model, batch)
    ref_user_vec, _, _, ref_mask = model(
        batch["source_vec"], batch["source_mask"], batch["source_count"], batch["history_vec"], batch["history_mask"]
    )
    assert torch.allclose(user_vec, ref_user_vec)
    assert torch.equal(personalized_mask, ref_mask)


def test_reliability_only_uses_gated_sources_with_fixed_fusion():
    artifacts, catalog, cases = make_fixture()
    model = AdaptiveFusion(num_sources=1)
    rows, _ = _prepare_rows(cases["validation"], artifacts, catalog, small_config())
    batch = _collate(rows, torch.device("cpu"))
    _, _, source_weight, _ = model(
        batch["source_vec"], batch["source_mask"], batch["source_count"], batch["history_vec"], batch["history_mask"]
    )
    gated_source = (source_weight[..., None] * batch["source_vec"]).sum(dim=1)
    count = batch["history_mask"].sum(dim=1)
    local = (batch["history_vec"] * batch["history_mask"][..., None]).sum(dim=1) / count.clamp_min(1)[:, None]
    expected = torch.where((count > 0)[:, None], 0.5 * gated_source + 0.5 * local, gated_source)

    reliability_vec, _ = fuse_user_vec("reliability-only", model, batch)
    assert torch.allclose(reliability_vec, expected)

    full_vec, _ = fuse_user_vec("full", model, batch)
    dynamic = count / (count + torch.nn.functional.softplus(model.tau_raw))
    dynamic_differs = (count > 0) & (dynamic != 0.5)
    assert dynamic_differs.any()
    assert torch.allclose(reliability_vec[~dynamic_differs], full_vec[~dynamic_differs])
    assert not torch.allclose(reliability_vec[dynamic_differs], full_vec[dynamic_differs])

    with torch.no_grad():
        model.tau_raw.fill_(3.0)
    reliability_after, _ = fuse_user_vec("reliability-only", model, batch)
    full_after, _ = fuse_user_vec("full", model, batch)
    assert torch.allclose(reliability_after, reliability_vec)
    assert not torch.allclose(full_after, full_vec)


def test_evaluate_reports_all_methods_with_statistics():
    artifacts, catalog, cases = make_fixture()
    config = small_config()
    checkpoint = train_fusion(artifacts, cases, config, catalog)
    metrics = evaluate(EvalCheckpoint(base=artifacts, fusion_state=checkpoint.fusion_state, config=config), make_eval_cases(), catalog)
    assert set(metrics["methods"]) == {"source-only", "target-only", "fixed-0.5", "dynamic-only", "reliability-only", "full"}
    for k in ("0", "1", "3", "5", "10"):
        assert k in metrics["methods"]["full"]
        assert 0.0 <= metrics["stages"][k]["missing_source_ratio"] <= 1.0
    assert metrics["methods"]["target-only"]["0"]["applicable"] is False
    assert metrics["methods"]["target-only"]["0"]["ndcg@10"] is None
    for method in ("source-only", "fixed-0.5", "dynamic-only", "reliability-only", "full"):
        summary = metrics["methods"][method]["user_mean_ndcg@10"]
        assert summary["ci"][0] <= summary["ci"][1]
    comparison = metrics["comparison"]
    assert comparison["strongest_baseline"] in ("source-only", "fixed-0.5", "dynamic-only", "reliability-only")
    assert comparison["paired_ndcg@10_diff"]["ci"][0] <= comparison["paired_ndcg@10_diff"]["ci"][1]
    assert comparison["effective"] in (True, False)
    assert 0.0 <= metrics["negative_transfer"]["user_ratio"] <= 1.0
