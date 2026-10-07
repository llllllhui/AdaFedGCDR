"""评估端的批量排名与指标计算。

旧实现对每个样本做一次 ``.cpu().tolist()`` + Python ``sorted``，在 GPU 上意味着每
样本一次同步。这里把整批排名留在设备上算，只在最后回拉一次。

**平局规则是硬约束**：``_rank_order`` 用 ``key=(-value, index)``，同分按候选原
下标升序。``torch.topk`` 在 CUDA 上不保证稳定顺序，会改变 HR/NDCG，因此必须用
``torch.argsort(..., stable=True)``——稳定排序精确保留同分元素的原下标顺序。
"""

import torch

RANKS = (5, 10)


def batched_rank_metrics(scores, label_index, ranks=RANKS, top_n: int = 10):
    """返回正物品排名、HR/NDCG 与 top-N 下标。

    :param scores: ``[B, C]`` 候选得分。
    :param label_index: ``[B]`` 正物品在候选中的下标。
    :returns: ``{"rank": [B], "hr@k": [B], "ndcg@k": [B], "topn": [B, top_n]}``，
        全部留在 ``scores.device`` 上。
    """
    # stable=True 是与旧实现等价的关键：同分元素按原下标升序。
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    hit = order == label_index[:, None]
    rank = hit.float().argmax(dim=1)

    out = {"rank": rank, "topn": order[:, :top_n]}
    # NDCG discount as a lookup over candidate rank: the old implementation used
    # Python ``math.log2`` in float64, so the table is built in float64 on the CPU
    # and only ever gathered. Computing ``log2`` in float32 drifts in the 8th
    # decimal, which is enough to change the reported metrics.
    table = _discount_table(scores.shape[1], scores.device)
    discount = table[rank]
    for k in ranks:
        within = (rank < k)
        out[f"hr@{k}"] = within.to(table.dtype)
        out[f"ndcg@{k}"] = torch.where(within, discount, torch.zeros_like(discount))
    return out


_DISCOUNT_CACHE = {}


def _discount_table(num_candidates: int, device):
    """NDCG 折扣表。

    旧实现用 Python ``math.log2``（float64）逐样本算折扣，因此在 CPU 上先按
    float64 算好再降精度搬到设备：这样数值与旧实现一致，而 MPS 也不必支持
    float64（它没有该 dtype，转换会直接报错）。
    """
    key = (num_candidates, str(device))
    cached = _DISCOUNT_CACHE.get(key)
    if cached is None:
        ranks = torch.arange(num_candidates, dtype=torch.float64)
        table = (1.0 / torch.log2(ranks + 2.0))
        target = torch.float32 if device.type == "mps" else table.dtype
        cached = table.to(device=device, dtype=target)
        _DISCOUNT_CACHE[key] = cached
    return cached


def metrics_to_python(metrics, rows, top_items_per_row):
    """把设备端结果转成 ``_case_records`` 需要的 dict 列表。

    ``top_items_per_row[b]`` 是该样本的候选物品名列表，顺序与候选下标一致。
    """
    rank = metrics["rank"].cpu().tolist()
    topn = metrics["topn"].cpu().tolist()
    hr = {k: metrics[f"hr@{k}"].cpu().tolist() for k in RANKS}
    ndcg = {k: metrics[f"ndcg@{k}"].cpu().tolist() for k in RANKS}

    records = []
    for b, row in enumerate(rows):
        records.append(
            {
                "user_id": row["user_id"],
                "k": int(row["k"]),
                "top10_items": [top_items_per_row[b][i] for i in topn[b]],
                **{f"hr@{k}": hr[k][b] for k in RANKS},
                **{f"ndcg@{k}": ndcg[k][b] for k in RANKS},
            }
        )
    return records
