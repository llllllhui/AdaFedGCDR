import csv
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_amazon_events.py"
spec = importlib.util.spec_from_file_location("prepare_amazon_events", MODULE_PATH)
prepare_amazon_events = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_amazon_events)


def write_raw(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for row in rows:
            writer.writerow(row)


def read_out(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.reader(stream))


def test_convert_applies_cores_and_converts_units(tmp_path):
    books = tmp_path / "Books.csv"
    cds = tmp_path / "CDs_and_Vinyl.csv"
    write_raw(books, [
        ("b1", "u1", "5", 1000),
        ("b2", "u1", "4", 1001),
        ("b1", "u2", "5", 1002),
        ("b3", "u3", "3", 1003),
        ("b4", "u3", "2", 1004),
        ("b1", "u4", "5", 1005),
        ("b5", "u4", "1", 1006),
        ("b2", "u4", "1", 1007),
        ("bad", "row", "missing"),
        ("b6", "u1", "x", "notanint"),
    ])
    write_raw(cds, [
        ("c1", "u1", "5", 10),
        ("c2", "u1", "4", 20),
        ("c1", "u9", "5", 30),
        ("c9", "u9", "5", 40),
    ])
    out = tmp_path / "events.csv"
    summary = prepare_amazon_events.convert(
        [("Books", books), ("CDs", cds)],
        target="Books",
        user_core=(2, {}),
        item_core=(1, {"CDs": 2}),
        out_path=out,
    )
    rows = read_out(out)
    assert rows[0] == ["user_id", "domain", "item_id", "timestamp_ms"]
    body = rows[1:]
    assert ("u2", "Books", "b1", "1002000") not in [tuple(r) for r in body]
    assert ("u9", "CDs", "c9", "40000") not in [tuple(r) for r in body]
    assert ("u1", "Books", "b1", "1000000") in [tuple(r) for r in body]
    assert all(len(r) == 4 for r in body)
    stats = summary["domains"]["Books"]
    assert stats["raw_rows"] == 10
    assert stats["kept_events"] == 7
    assert stats["skipped_rows"] == 2
    assert stats["users_kept"] == 3
    assert summary["domains"]["CDs"]["items_kept"] == 1
    assert summary["bridge_users"] == 1
    assert summary["qualified_users"] == 0


def test_advisory_qualified_requires_source_before_first_target(tmp_path):
    books = tmp_path / "Books.csv"
    cds = tmp_path / "CDs.csv"
    rows = []
    for user, base in (("w", 100), ("v", 100), ("x", 100)):
        for index in range(11 if user != "x" else 10):
            rows.append((f"b{index}", user, "5", base + index))
    write_raw(books, rows)
    write_raw(cds, [
        ("c1", "w", "5", 50),
        ("c1", "v", "5", 200),
        ("c1", "x", "5", 50),
    ])
    summary = prepare_amazon_events.convert(
        [("Books", books), ("CDs", cds)],
        target="Books",
        user_core=(1, {}),
        item_core=(1, {}),
        out_path=tmp_path / "events.csv",
    )
    assert summary["bridge_users"] == 3
    assert summary["qualified_users"] == 1


def test_convert_preserves_duplicate_rows_for_pipeline_dedup(tmp_path):
    books = tmp_path / "Books.csv"
    write_raw(books, [("b1", "u1", "5", 1000), ("b1", "u1", "4", 900)])
    out = tmp_path / "events.csv"
    prepare_amazon_events.convert(
        [("Books", books)], target="Books", user_core=(1, {}), item_core=(1, {}), out_path=out
    )
    body = read_out(out)[1:]
    assert [tuple(r) for r in body] == [("u1", "Books", "b1", "1000000"), ("u1", "Books", "b1", "900000")]


def test_parse_core_specs():
    assert prepare_amazon_events.parse_core_specs(["5", "Books=96"], 10) == (5, {"Books": 96})
    assert prepare_amazon_events.parse_core_specs([], 10) == (10, {})
    with pytest.raises(ValueError):
        prepare_amazon_events.parse_core_specs(["abc"], 10)
    with pytest.raises(ValueError):
        prepare_amazon_events.parse_core_specs(["Books=0"], 10)


def test_convert_rejects_unknown_target_and_duplicate_names(tmp_path):
    books = tmp_path / "Books.csv"
    write_raw(books, [("b1", "u1", "5", 1000)])
    with pytest.raises(ValueError):
        prepare_amazon_events.convert(
            [("Books", books)], target="Movies", user_core=(1, {}), item_core=(1, {}), out_path=tmp_path / "o.csv"
        )
    with pytest.raises(ValueError):
        prepare_amazon_events.convert(
            [("Books", books), ("Books", books)], target="Books",
            user_core=(1, {}), item_core=(1, {}), out_path=tmp_path / "o.csv",
        )
