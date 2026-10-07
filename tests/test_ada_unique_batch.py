import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.adafedgcdr.unique_batch import batched_unique, group_values


def reference_unique(total_items):
    """逐行 ``torch.unique(return_inverse=True)``。"""
    return [torch.unique(row, return_inverse=True) for row in total_items]


def pad_rows(total_items, sentinel, smax):
    B = len(total_items)
    padded = torch.full((B, smax), sentinel, dtype=torch.int64)
    for b, row in enumerate(total_items):
        padded[b, : row.shape[0]] = row
    return padded


def test_matches_per_row_unique_without_padding():
    torch.manual_seed(0)
    rows = [torch.randint(0, 20, (5 * 3,), dtype=torch.int64) for _ in range(4)]
    sentinel = 999
    padded = pad_rows(rows, sentinel, rows[0].shape[0])
    inv, counts = batched_unique(padded, sentinel)
    ref = reference_unique(rows)
    for b, (uniq, _) in enumerate(ref):
        got = int(counts[b])
        assert got == uniq.shape[0], f"row {b}: count {got} != {uniq.shape[0]}"
        # inv[b] 的取值域应恰好是 [0, uniq.shape[0])
        assert int(inv[b].min()) == 0 and int(inv[b].max()) == uniq.shape[0] - 1
        # 映射出的值必须等于 unique 的升序数组
        mapped = uniq[inv[b]]
        assert torch.equal(mapped, rows[b]), f"row {b}: 映射后的值与原序列不一致"


def test_with_padding_sentinel_forms_extra_group():
    """padding 哨兵应成为下标最大的额外组，且不影响真实值的前缀下标。

    只有 L < lmax 的行才会出现 padding；L == lmax 的行整行都是真实值，哨兵不出现，
    组数恰为真实唯一值个数。
    """
    torch.manual_seed(1)
    lens = [3, 5, 2, 4]
    rows = [torch.randint(0, 20, (5 * L,), dtype=torch.int64) for L in lens]
    sentinel = 999
    smax = 5 * max(lens)
    padded = pad_rows(rows, sentinel, smax)
    inv, counts = batched_unique(padded, sentinel)
    for b, L in enumerate(lens):
        uniq, _ = torch.unique(rows[b], return_inverse=True)
        real = uniq.shape[0]
        # 真实值下标必须与 torch.unique 一致（哨兵是最大值，不占前缀）
        assert torch.equal(uniq[inv[b, : 5 * L]], rows[b]), f"row {b} 下标错位"
        has_padding = 5 * L < smax
        assert int(counts[b]) == real + (1 if has_padding else 0), (
            f"row {b}: 期望 {real + (1 if has_padding else 0)} 组，实得 {int(counts[b])}"
        )
        if has_padding:
            # padding 段全部映射到哨兵组
            assert bool((inv[b, 5 * L :] == real).all()), f"row {b}: padding 未指向哨兵组"
        else:
            assert int(counts[b]) == real, f"row {b}: 无 padding 行的组数应等于真实值个数"


def test_single_row_and_all_padding_edge():
    sentinel = 999
    rows = [torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)]
    padded = pad_rows(rows, sentinel, 5)
    inv, counts = batched_unique(padded, sentinel)
    assert int(counts[0]) == 5
    assert torch.equal(inv[0], torch.tensor([0, 1, 2, 3, 4]))

    # 全部为 padding 的行
    padded_all = torch.full((1, 4), sentinel, dtype=torch.int64)
    inv2, counts2 = batched_unique(padded_all, sentinel)
    assert int(counts2[0]) == 1
    assert bool((inv2 == 0).all())


def test_duplicate_heavy_rows():
    """大量重复时 unique 组数与 torch.unique 一致。"""
    rows = [torch.tensor([7, 7, 7, 7, 7], dtype=torch.int64), torch.tensor([1, 2, 1, 2, 1], dtype=torch.int64)]
    sentinel = 999
    padded = pad_rows(rows, sentinel, 5)
    inv, counts = batched_unique(padded, sentinel)
    assert int(counts[0]) == 1
    assert int(counts[1]) == 2
    for b, row in enumerate(rows):
        uniq, _ = torch.unique(row, return_inverse=True)
        assert torch.equal(uniq[inv[b]], row)


def test_group_values_matches_per_row_unique():
    """group_values 的第 g 列必须等于该行第 g 个升序 unique 值。

    这是 batched.py fast 路径 V 初始化与 V 更新的索引来源；历史上曾把
    「位置→组」的 inv 误用作「组→值」查表（gather），导致大范围打分与
    V 更新错行——本测试是对该 bug 的单元级回归。
    """
    torch.manual_seed(2)
    lens = [3, 5, 2, 4]
    rows = [torch.randint(0, 20, (5 * L,), dtype=torch.int64) for L in lens]
    sentinel = 999
    smax = 5 * max(lens)
    padded = pad_rows(rows, sentinel, smax)
    inv, counts = batched_unique(padded, sentinel)
    rmax = int(counts.max())
    gv = group_values(padded, inv, rmax, sentinel)
    for b, row in enumerate(rows):
        uniq, _ = torch.unique(row, return_inverse=True)
        real = uniq.shape[0]
        assert torch.equal(gv[b, :real], uniq), f"row {b}: 组→值映射错位"
        if 5 * lens[b] < smax:
            # 有 padding 的行多出一个哨兵组，其值恰为哨兵本身
            assert int(gv[b, real]) == sentinel, f"row {b}: 哨兵组未收集到哨兵值"
        # 未写到的列保持哨兵初值（调用方用 real_mask 屏蔽）
        tail = real + (1 if 5 * lens[b] < smax else 0)
        assert bool((gv[b, tail:] == sentinel).all()), f"row {b}: 未写列应保持哨兵初值"


def test_group_values_rejects_gather_construction():
    """直接用 gather(padded, inv[:, :rmax]) 得到的是位置值不是组值，必须不等。"""
    padded = torch.tensor([[5, 3, 5, 9]], dtype=torch.int64)
    sentinel = 99
    inv, counts = batched_unique(padded, sentinel)
    gv = group_values(padded, inv, int(counts.max()), sentinel)
    uniq, _ = torch.unique(padded[0], return_inverse=True)
    assert torch.equal(gv[0, :3], uniq)
    assert not torch.equal(gv[0, :3], torch.gather(padded, 1, inv[:, :3])[0])
