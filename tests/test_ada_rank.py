"""评估端批量排名与旧实现 ``_rank_order`` 的等价性。

``experiment.py`` 原先对每个样本做一次 ``.cpu().tolist()`` + Python ``sorted``，
在 GPU 上是每样本一次同步。批量化后用 ``torch.argsort(stable=True)``：稳定排序
保留同分元素的原下标顺序，与旧实现 ``key=(-v, i)`` 严格等价——这一点是硬约束，
``torch.topk`` 在 CUDA 上不保证稳定顺序，不能直接替换。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.adafedgcdr.experiment import _rank_order
from model.adafedgcdr.rank import batched_rank_metrics


def reference_ranks(scores, label_index):
    """旧实现：Python sorted，返回每个样本的正物品排名。"""
    ranks = []
    for b, row in enumerate(scores):
        values = row.tolist()
        order = _rank_order(values)
        ranks.append(order.index(int(label_index[b])))
    return ranks


def test_matches_python_sorted_on_random_scores():
    torch.manual_seed(0)
    scores = torch.randn(7, 20)
    label = torch.tensor([19] * 7)
    got = batched_rank_metrics(scores, label, (5, 10))
    want = reference_ranks(scores, label)
    assert got["rank"].tolist() == want


def test_ties_break_by_original_index():
    """大量同分时必须按候选原下标升序打破平局。"""
    scores = torch.zeros(3, 10)
    # 正物品在最后一列，同分时应排在所有更小下标之后
    label = torch.tensor([9, 9, 9])
    got = batched_rank_metrics(scores, label, (5, 10))
    # 全同分时，下标 9 排最后 -> rank == 9
    assert got["rank"].tolist() == [9, 9, 9]


def test_partial_ties_mixed():
    scores = torch.tensor(
        [
            [1.0, 1.0, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.3, 0.9, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        ]
    )
    label = torch.tensor([2, 5, 0])
    got = batched_rank_metrics(scores, label, (5, 10))
    want = reference_ranks(scores, label.tolist())
    assert got["rank"].tolist() == want


def test_metrics_match_definition():
    scores = torch.tensor([[0.9, 0.1, 0.5, 0.3], [0.2, 0.8, 0.4, 0.6]])
    label = torch.tensor([2, 0])
    got = batched_rank_metrics(scores, label, (5, 10))
    import math

    for b, row in enumerate(scores.tolist()):
        order = _rank_order(row)
        rank = order.index(int(label[b]))
        for k in (5, 10):
            assert got[f"hr@{k}"][b].item() == (1.0 if rank < k else 0.0)
            expected_ndcg = 1.0 / math.log2(rank + 2) if rank < k else 0.0
            assert abs(got[f"ndcg@{k}"][b].item() - expected_ndcg) < 1e-6


def test_topk_order_prefix():
    """top-N 顺序也必须与 Python sorted 的前 N 一致。"""
    torch.manual_seed(3)
    scores = torch.randint(0, 4, (5, 30)).float()  # 大量同分
    got = batched_rank_metrics(scores, torch.tensor([29] * 5), (5, 10), top_n=10)
    for b, row in enumerate(scores.tolist()):
        want = _rank_order(row)[:10]
        assert got["topn"][b].tolist() == want
