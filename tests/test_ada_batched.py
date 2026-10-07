"""批处理训练器的对拍回归。

两条硬约束：

1. ``--fast-sampling`` 只批量逐行 ``unique``，RNG 流不变，因此
   ``train_base_batched`` 带与不带该开关必须**逐位一致**（fast 路径的
   组→值映射与 V 更新历史上错行，本测试即其回归）。
2. 批处理与逐 client 的 ``train_base`` 算法等价、RNG 流相同，但算子形状
   不同导致浮点求和顺序不同，数值只要求 ULP 级接近（1+1 轮实测 ~1e-6，
   阈值留两个数量级余量）；语义级偏差（如错行索引）会立即放大到 1e-2 以上。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.adafedgcdr.base import BaseConfig, train_base
from model.adafedgcdr.batched import train_base_batched
from tests.test_ada_base import TARGET, build_synthetic_events
from model.adafedgcdr.data import build_split

SEED = 2026


def _config(rounds=1, user_batch=1, fast=False):
    return BaseConfig(
        target_domain=TARGET,
        source_rounds=rounds,
        target_rounds=rounds,
        user_batch=user_batch,
        device="cpu",
        fast_sampling=fast,
        seed=SEED,
    )


def _assert_artifacts_equal(first, second):
    assert torch.equal(first.target_item_vectors, second.target_item_vectors)
    assert set(first.source_vectors) == set(second.source_vectors)
    for user_id in first.source_vectors:
        assert torch.equal(first.source_vectors[user_id], second.source_vectors[user_id])
        assert torch.equal(first.source_mask[user_id], second.source_mask[user_id])
        assert torch.equal(first.source_counts[user_id], second.source_counts[user_id])


def test_fast_sampling_is_bit_identical():
    events = build_synthetic_events()
    split = build_split(events, TARGET, SEED)
    # user_batch=7 与 100 用户不成整除，chunk 必然带 padding，覆盖哨兵组路径
    slow = train_base_batched(events, split, _config(user_batch=7))
    fast = train_base_batched(events, split, _config(user_batch=7, fast=True))
    _assert_artifacts_equal(slow, fast)


def test_batched_tracks_original_at_user_batch_one():
    events = build_synthetic_events()
    split = build_split(events, TARGET, SEED)
    original = train_base(events, split, _config())
    batched = train_base_batched(events, split, _config())
    target_diff = (original.target_item_vectors - batched.target_item_vectors).abs().max()
    assert target_diff < 1e-4, f"target 向量偏差 {target_diff:.3e} 超出 ULP 级预期"
    for user_id in original.source_vectors:
        diff = (original.source_vectors[user_id] - batched.source_vectors[user_id]).abs().max()
        assert diff < 1e-4, f"user {user_id} 源向量偏差 {diff:.3e} 超出 ULP 级预期"
