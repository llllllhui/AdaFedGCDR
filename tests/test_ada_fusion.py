import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.adafedgcdr.fusion import AdaptiveFusion

D = 4
S = 2


def make_inputs(k_values, present=(True, True), dim=D, source_values=None, source_counts=None):
    batch = len(k_values)
    num_sources = len(present)
    width = max(max(k_values), 1)
    source_vec = torch.zeros(batch, num_sources, dim)
    source_mask = torch.tensor(present, dtype=torch.bool).unsqueeze(0).repeat(batch, 1)
    source_count = torch.zeros(batch, num_sources)
    history_vec = torch.zeros(batch, width, dim)
    history_mask = torch.zeros(batch, width, dtype=torch.bool)
    for b, k in enumerate(k_values):
        for s in range(num_sources):
            if present[s]:
                value = torch.full((dim,), float(s + 1)) if source_values is None else source_values[s]
                source_vec[b, s] = value
                source_count[b, s] = float(10 * (s + 1)) if source_counts is None else source_counts[s]
        for j in range(k):
            history_vec[b, j] = torch.full((dim,), 1.0)
            history_mask[b, j] = True
    return source_vec, source_mask, source_count, history_vec, history_mask


def test_target_weight_rules():
    model = AdaptiveFusion(num_sources=S)
    k_values = [0, 1, 10]
    source_vec, source_mask, source_count, history_vec, history_mask = make_inputs(k_values)
    user_vec, target_weight, source_weight, personalized_mask = model(
        source_vec, source_mask, source_count, history_vec, history_mask
    )
    k = torch.tensor(k_values)
    assert torch.allclose(target_weight[k == 0], torch.zeros_like(target_weight[k == 0]))
    assert torch.all(target_weight[k == 10] > target_weight[k == 1])
    assert torch.count_nonzero(source_weight[~source_mask]) == 0
    assert torch.isfinite(user_vec).all()
    assert personalized_mask.dtype == torch.bool


def test_missing_source_is_zeroed_and_renormalized():
    model = AdaptiveFusion(num_sources=S)
    inputs = make_inputs([0, 3], present=(True, False))
    user_vec, target_weight, source_weight, _ = model(*inputs)
    assert torch.count_nonzero(source_weight[:, 1]) == 0
    assert torch.allclose(source_weight[:, 0], torch.ones(2))
    assert torch.isfinite(user_vec).all()


def test_all_sources_missing():
    model = AdaptiveFusion(num_sources=S)
    inputs = make_inputs([0, 3], present=(False, False))
    user_vec, target_weight, source_weight, personalized_mask = model(*inputs)
    k = torch.tensor([0, 3])
    assert torch.allclose(target_weight[k == 0], torch.zeros_like(target_weight[k == 0]))
    assert torch.allclose(target_weight[k == 3], torch.ones_like(target_weight[k == 3]))
    assert torch.count_nonzero(source_weight) == 0
    assert torch.isfinite(user_vec).all()
    assert personalized_mask.tolist() == [False, True]
    assert torch.allclose(user_vec[0], torch.zeros(D))


def test_low_reliability_source_is_downweighted():
    model = AdaptiveFusion(num_sources=S)
    with torch.no_grad():
        model.domain_bias.zero_()
        model.count_scale.fill_(1.0)
        model.agreement_scale.fill_(1.0)
        model.tau_raw.zero_()
    ones = torch.full((D,), 1.0)
    anti = torch.full((D,), -1.0)
    source_values = [ones, anti]
    source_counts = [10000.0, 0.0]
    source_vec, source_mask, source_count, history_vec, history_mask = make_inputs(
        [5], source_values=source_values, source_counts=source_counts
    )
    _, _, source_weight, _ = model(source_vec, source_mask, source_count, history_vec, history_mask)
    assert source_weight[0, 1] < source_weight[0, 0]
    assert source_weight[0, 1] < 1e-3


def test_agreement_term_is_disabled_at_k0():
    model = AdaptiveFusion(num_sources=S)
    base = torch.full((D,), 1.0)
    flipped = torch.stack([torch.full((D,), -2.0), torch.full((D,), 7.0)])
    source_vec, source_mask, source_count, history_vec, history_mask = make_inputs([0, 0])
    source_vec[1] = flipped
    _, _, source_weight, _ = model(source_vec, source_mask, source_count, history_vec, history_mask)
    assert torch.allclose(source_weight[0], source_weight[1])
    assert torch.allclose(source_vec[0, 0], base)


def test_backward_produces_finite_gradients():
    model = AdaptiveFusion(num_sources=S)
    inputs = make_inputs([0, 1, 10])
    user_vec, target_weight, source_weight, _ = model(*inputs)
    loss = user_vec.sum() + target_weight.sum() + source_weight.sum()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_forward_is_deterministic():
    model = AdaptiveFusion(num_sources=S)
    inputs = make_inputs([0, 1, 10])
    first = model(*inputs)
    second = model(*inputs)
    for a, b in zip(first, second):
        assert torch.equal(a, b)
