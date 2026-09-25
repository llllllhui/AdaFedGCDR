import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.adafedgcdr.__main__ import check_prepare_audit, cmd_prepare
from model.adafedgcdr.data import load_manifest


def write_events(path):
    Path(path).write_text(
        "user_id,domain,item_id,timestamp_ms\n"
        + "".join(f"u{i},CDs,s{i % 30},0\n" for i in range(20))
        + "".join(f"u{i},Books,t{(i * 13 + j) % 200},{1000000 + j}\n" for i in range(20) for j in range(12)),
        encoding="utf-8",
    )


def run_prepare(tmp_path):
    events = tmp_path / "events.csv"
    write_events(events)
    split = tmp_path / "split.json"
    cmd_prepare(Namespace(events=str(events), target="Books", seed=2026, split=str(split)))
    return events, split


def test_check_prepare_audit_passes_after_prepare(tmp_path):
    events, split = run_prepare(tmp_path)
    check_prepare_audit(str(events), str(split), load_manifest(split))


def test_check_prepare_audit_rejects_tampered_events(tmp_path):
    events, split = run_prepare(tmp_path)
    events.write_text(events.read_text(encoding="utf-8") + "u99,CDs,s0,0\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        check_prepare_audit(str(events), str(split), load_manifest(split))


def test_check_prepare_audit_requires_audit_file(tmp_path):
    events, split = run_prepare(tmp_path)
    (tmp_path / "split.json.audit.json").unlink()
    with pytest.raises(SystemExit):
        check_prepare_audit(str(events), str(split), load_manifest(split))
