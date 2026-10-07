"""批量化 per-row 去重，与逐 client ``torch.unique(..., return_inverse=True)`` 等价。

``batched.py`` 每个 chunk 对 B 个 client 各调一次 ``torch.unique``，全量下是
``轮数 × 用户数`` 次调用。批量化后只剩一次排序，但必须精确保留两点语义：

1. ``torch.unique`` 返回的 unique 值是**升序**的，``inv`` 指向该升序数组的下标；
2. 同一行内相同值映射到同一个下标。

padding 位置用越界哨兵（``sentinel``）填充，它会成为一个额外的“伪 unique 组”，
其下标等于真实唯一值个数，因此 ``V`` 需要多一行零向量来承接（见
``batched.py`` 中的 ``V_ext``）。
"""

import torch


def batched_unique(padded, sentinel):
    """对 ``[B, N]`` 的 int64 张量逐行去重。

    返回 ``(inv, counts)``：``inv[b, i]`` 是 ``padded[b, i]`` 在该行升序 unique
    数组中的下标（含哨兵组），``counts[b]`` 是含哨兵组的总数。当某行没有 padding
    时 ``counts[b]`` 等于真实唯一值个数加一。

    调用方须保证 ``padded`` 中的 padding 位置都已写成 ``sentinel``。
    """
    order = torch.argsort(padded, dim=1, stable=True)
    ordered = torch.gather(padded, 1, order)
    is_new = torch.ones_like(ordered, dtype=torch.bool)
    is_new[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    group = torch.cumsum(is_new, dim=1) - 1  # 升序 unique 下标
    counts = is_new.sum(dim=1)

    inv = torch.empty_like(group)
    inv.scatter_(1, order, group)
    return inv, counts


def group_values(padded, inv, rmax, sentinel):
    """构造「组下标 → 物品值」映射，返回形状 ``[B, rmax]`` 的 int64 张量。

    对每个位置 ``i`` 把 ``padded[b, i]`` 写入第 ``inv[b, i]`` 列。同一组的所有
    位置按定义持有同一个升序 unique 值（padding 位置持有哨兵值），因此重复写
    覆盖的是相同的值，写序不影响结果；第 ``g`` 列即为该行第 ``g`` 个升序
    unique 值，哨兵组收集到 ``sentinel`` 本身。注意：不能用 ``gather(padded,
    inv[:, :rmax])`` 代替——那是「位置→值」方向的查表，组下标不是位置下标。
    未写到的列（``g >= counts[b]``）初始化为 ``sentinel``，调用方须用
    ``arange(rmax) < counts`` 的 real_mask 屏蔽。
    """
    out = torch.full((padded.shape[0], rmax), sentinel, dtype=padded.dtype, device=padded.device)
    out.scatter_(1, inv, padded)
    return out
