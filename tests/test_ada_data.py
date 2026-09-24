import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.adafedgcdr.data import (
    Event,
    build_split,
    load_events,
    make_cases,
    manifest_sha256,
)

TARGET = "Books"
SOURCE = "CDs"
SEED = 2026
K_VALUES = (0, 1, 3, 5, 10)


def build_synthetic_events():
    events = []
    for user in range(100):
        user_id = f"u{user}"
        events.append(Event(user_id, SOURCE, f"s{user % 30}", 0))
        for index in range(12):
            events.append(Event(user_id, TARGET, f"t{(user * 13 + index) % 200}", 1000000 + index))
    return events


def target_events_by_user(events):
    mapping = {}
    for event in events:
        if event.domain == TARGET:
            mapping.setdefault(event.user_id, []).append(event)
    for user_events in mapping.values():
        user_events.sort(key=lambda e: (e.timestamp_ms, e.item_id))
    return mapping


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("user_id", "domain", "item_id", "timestamp_ms"))
        writer.writerows(rows)


def test_split_groups_are_disjoint():
    events = build_synthetic_events()
    split = build_split(events, TARGET, SEED)
    groups = [set(split.users[name]) for name in ("base", "fusion", "validation", "test")]
    assert not any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4))
    assert set().union(*groups) == {event.user_id for event in events}


def test_case_prefixes_and_labels():
    events = build_synthetic_events()
    split = build_split(events, TARGET, SEED)
    cases = make_cases(events, split, K_VALUES)
    target_events = target_events_by_user(events)
    case = next(c for c in cases if c.k == 0)
    assert case.history_item_ids == ()
    assert case.label_item_id == target_events[case.user_id][0].item_id
    assert next(c for c in cases if c.user_id == case.user_id and c.k == 3).label_item_id == target_events[case.user_id][3].item_id


def test_every_qualified_user_has_all_k_cases():
    events = build_synthetic_events()
    split = build_split(events, TARGET, SEED)
    cases = make_cases(events, split, K_VALUES)
    by_user = {}
    for case in cases:
        by_user.setdefault(case.user_id, []).append(case)
    assert len(by_user) == 100
    for user_cases in by_user.values():
        ordered = sorted(user_cases, key=lambda c: c.k)
        assert [c.k for c in ordered] == list(K_VALUES)
        for case in ordered:
            assert len(case.history_item_ids) == case.k
        for smaller, larger in zip(ordered, ordered[1:]):
            assert larger.history_item_ids[: smaller.k] == smaller.history_item_ids


def test_disqualified_users_are_excluded():
    events = []
    for index in range(5):
        events.append(Event("few", TARGET, f"t{index}", 1000 + index))
    events.append(Event("few", SOURCE, "s0", 0))
    for index in range(12):
        events.append(Event("lonely", TARGET, f"l{index}", 1000 + index))
    for index in range(12):
        events.append(Event("late", TARGET, f"m{index}", 1000 + index))
    events.append(Event("late", SOURCE, "s0", 5000))
    events.append(Event("good", SOURCE, "s0", 0))
    for index in range(12):
        events.append(Event("good", TARGET, f"g{index}", 1000 + index))
    split = build_split(events, TARGET, SEED)
    cases = make_cases(events, split, K_VALUES)
    assert {case.user_id for case in cases} == {"good"}


def test_split_is_deterministic():
    events = build_synthetic_events()
    first = build_split(events, TARGET, SEED)
    second = build_split(list(reversed(events)), TARGET, SEED)
    assert first == second
    assert manifest_sha256(first) == manifest_sha256(second)


def test_load_events_dedup_keeps_earliest(tmp_path):
    path = tmp_path / "events.csv"
    write_csv(path, [("u0", TARGET, "t0", 5), ("u0", TARGET, "t0", 3), ("u0", TARGET, "t0", 4)])
    events = load_events(path)
    assert events == [Event("u0", TARGET, "t0", 3)]


def test_load_events_reports_line_number_on_bad_timestamp(tmp_path):
    path = tmp_path / "events.csv"
    write_csv(path, [("u0", TARGET, "t0", 1), ("u0", TARGET, "t1", "not-a-number")])
    with pytest.raises(ValueError) as info:
        load_events(path)
    assert ":3" in str(info.value)


def test_load_events_rejects_wrong_header(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("user,item,time\nu0,t0,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_events(path)


def test_load_events_roundtrip_is_sorted(tmp_path):
    path = tmp_path / "events.csv"
    write_csv(path, [("u1", TARGET, "t1", 2), ("u0", SOURCE, "s0", 3), ("u0", TARGET, "t0", 1)])
    events = load_events(path)
    assert events == [
        Event("u0", TARGET, "t0", 1),
        Event("u0", SOURCE, "s0", 3),
        Event("u1", TARGET, "t1", 2),
    ]
