import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

GROUPS = ("base", "fusion", "validation", "test")
DEFAULT_K_VALUES = (0, 1, 3, 5, 10)
CSV_COLUMNS = ("user_id", "domain", "item_id", "timestamp_ms")


@dataclass(frozen=True)
class Event:
    user_id: str
    domain: str
    item_id: str
    timestamp_ms: int


@dataclass(frozen=True)
class SplitManifest:
    target_domain: str
    seed: int
    users: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class Case:
    user_id: str
    k: int
    history_item_ids: tuple[str, ...]
    label_item_id: str


def load_events(path) -> list[Event]:
    path = Path(path)
    kept: dict[tuple[str, str, str], Event] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(f"{path}: 表头必须为 {','.join(CSV_COLUMNS)}，实际为 {reader.fieldnames}")
        for row in reader:
            line = reader.line_num
            raw = row["timestamp_ms"]
            try:
                timestamp_ms = int(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{line}: timestamp_ms 无法转为整数: {raw!r}") from None
            event = Event(row["user_id"], row["domain"], row["item_id"], timestamp_ms)
            key = (event.user_id, event.domain, event.item_id)
            existing = kept.get(key)
            if existing is None or event.timestamp_ms < existing.timestamp_ms:
                kept[key] = event
    return sorted(kept.values(), key=lambda e: (e.user_id, e.domain, e.timestamp_ms, e.item_id))


def split_group(user_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{user_id}".encode()).digest()[:8]
    value = int.from_bytes(digest, "big") / float(1 << 64)
    if value < 0.6:
        return "base"
    if value < 0.8:
        return "fusion"
    if value < 0.9:
        return "validation"
    return "test"


def build_split(events: list[Event], target: str, seed: int) -> SplitManifest:
    buckets: dict[str, list[str]] = {group: [] for group in GROUPS}
    for user_id in sorted({event.user_id for event in events}):
        buckets[split_group(user_id, seed)].append(user_id)
    users = {group: tuple(names) for group, names in buckets.items()}
    return SplitManifest(target_domain=target, seed=seed, users=users)


def user_views(events: list[Event], target: str) -> dict[str, dict]:
    by_user: dict[str, list[Event]] = {}
    for event in events:
        by_user.setdefault(event.user_id, []).append(event)
    views = {}
    for user_id in sorted(by_user):
        user_events = by_user[user_id]
        target_events = sorted((e for e in user_events if e.domain == target), key=lambda e: (e.timestamp_ms, e.item_id))
        source_events = sorted((e for e in user_events if e.domain != target), key=lambda e: (e.timestamp_ms, e.domain, e.item_id))
        first_ts = target_events[0].timestamp_ms if target_events else None
        if first_ts is None:
            usable_sources = list(source_events)
        else:
            usable_sources = [e for e in source_events if e.timestamp_ms < first_ts]
        views[user_id] = {
            "target_events": target_events,
            "source_events": source_events,
            "usable_sources": usable_sources,
            "first_target_ts": first_ts,
        }
    return views


def make_cases(events: list[Event], manifest: SplitManifest, k_values=DEFAULT_K_VALUES) -> list[Case]:
    target = manifest.target_domain
    k_values = tuple(sorted(set(k_values)))
    cases: list[Case] = []
    for user_id, view in user_views(events, target).items():
        target_events = view["target_events"]
        if len(target_events) <= max(k_values):
            continue
        if not view["usable_sources"]:
            continue
        for k in k_values:
            cases.append(
                Case(
                    user_id=user_id,
                    k=k,
                    history_item_ids=tuple(e.item_id for e in target_events[:k]),
                    label_item_id=target_events[k].item_id,
                )
            )
    return cases


def audit_users(events: list[Event], target: str, k_values=DEFAULT_K_VALUES) -> dict[str, int]:
    counts = {"total_users": 0, "qualified": 0, "no_target": 0, "insufficient_target": 0, "no_usable_source": 0}
    for view in user_views(events, target).values():
        counts["total_users"] += 1
        target_events = view["target_events"]
        if not target_events:
            counts["no_target"] += 1
        elif len(target_events) <= max(k_values):
            counts["insufficient_target"] += 1
        elif not view["usable_sources"]:
            counts["no_usable_source"] += 1
        else:
            counts["qualified"] += 1
    return counts


def manifest_to_dict(manifest: SplitManifest) -> dict:
    return {
        "target_domain": manifest.target_domain,
        "seed": manifest.seed,
        "users": {group: list(manifest.users[group]) for group in GROUPS},
    }


def manifest_from_dict(data: dict) -> SplitManifest:
    users = data.get("users", {})
    if set(users) != set(GROUPS):
        raise ValueError(f"拆分名单必须包含 {GROUPS}，实际为 {tuple(sorted(users))}")
    return SplitManifest(
        target_domain=data["target_domain"],
        seed=int(data["seed"]),
        users={group: tuple(users[group]) for group in GROUPS},
    )


def save_manifest(manifest: SplitManifest, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_to_dict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_manifest(path) -> SplitManifest:
    return manifest_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def manifest_sha256(manifest: SplitManifest) -> str:
    payload = json.dumps(manifest_to_dict(manifest), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
