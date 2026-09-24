import argparse
import csv
import gzip
import sys
from collections import Counter
from pathlib import Path

CSV_HEADER = ("user_id", "domain", "item_id", "timestamp_ms")
MIN_TARGET_EVENTS = 11


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open(newline="", encoding="utf-8")


def parse_domain_spec(spec: str) -> tuple[str, Path]:
    name, sep, value = spec.partition("=")
    if not sep or not name.strip() or not value.strip():
        raise ValueError(f"--domain 需要 名称=路径 格式，实际为 {spec!r}")
    path = Path(value.strip())
    if not path.is_file():
        raise ValueError(f"数据文件不存在: {path}")
    return name.strip(), path


def parse_core_specs(values: list[str], fallback: int) -> tuple[int, dict[str, int]]:
    default = fallback
    overrides: dict[str, int] = {}
    for value in values:
        name, sep, number = value.partition("=")
        raw = number if sep else name
        try:
            core = int(raw)
        except ValueError:
            raise ValueError(f"核心阈值必须是整数: {value!r}") from None
        if core < 1:
            raise ValueError(f"核心阈值必须 >= 1: {value!r}")
        if sep:
            overrides[name] = core
        else:
            default = core
    return default, overrides


def count_domain(path: Path) -> tuple[Counter, Counter, int, int]:
    user_counts: Counter = Counter()
    item_counts: Counter = Counter()
    raw_rows = 0
    skipped = 0
    with open_text(path) as stream:
        for row in csv.reader(stream):
            raw_rows += 1
            if len(row) != 4:
                skipped += 1
                continue
            item_counts[row[0]] += 1
            user_counts[row[1]] += 1
    return user_counts, item_counts, raw_rows, skipped


def stream_domain(path: Path, allowed_users: set, allowed_items: set, on_event) -> tuple[int, int]:
    kept = 0
    skipped = 0
    with open_text(path) as stream:
        for row in csv.reader(stream):
            if len(row) != 4:
                skipped += 1
                continue
            item, user, _rating, timestamp = row
            if user not in allowed_users or item not in allowed_items:
                continue
            try:
                timestamp_ms = int(timestamp) * 1000
            except ValueError:
                skipped += 1
                continue
            on_event(user, item, timestamp_ms)
            kept += 1
    return kept, skipped


def convert(domains: list[tuple[str, Path]], target: str, user_core, item_core, out_path) -> dict:
    user_core_default, user_core_overrides = user_core
    item_core_default, item_core_overrides = item_core
    names = [name for name, _ in domains]
    if len(set(names)) != len(names):
        raise ValueError(f"域名重复: {names}")
    if target not in names:
        raise ValueError(f"目标域 {target!r} 不在 --domain 列表 {names} 中")

    user_stats: dict[str, dict[str, tuple[int, int]]] = {}
    domain_stats: dict[str, dict] = {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_HEADER)
        for name, path in domains:
            u_core = user_core_overrides.get(name, user_core_default)
            i_core = item_core_overrides.get(name, item_core_default)
            user_counts, item_counts, raw_rows, _ = count_domain(path)
            allowed_users = {user for user, count in user_counts.items() if count >= u_core}
            allowed_items = {item for item, count in item_counts.items() if count >= i_core}

            def on_event(user, item, timestamp_ms, domain=name):
                writer.writerow((user, domain, item, timestamp_ms))
                per_user = user_stats.setdefault(user, {})
                count, min_ts = per_user.get(domain, (0, None))
                per_user[domain] = (count + 1, timestamp_ms if min_ts is None else min(min_ts, timestamp_ms))

            kept, skipped = stream_domain(path, allowed_users, allowed_items, on_event)
            domain_stats[name] = {
                "path": str(path),
                "raw_rows": raw_rows,
                "kept_events": kept,
                "skipped_rows": skipped,
                "users_kept": len(allowed_users),
                "items_kept": len(allowed_items),
                "user_core": u_core,
                "item_core": i_core,
            }
            print(
                f"[{name}] 原始 {raw_rows} 行，保留 {kept} 条事件"
                f"（用户 {len(allowed_users)}，物品 {len(allowed_items)}，跳过 {skipped}）",
                file=sys.stderr,
            )

    bridge_users = sum(1 for per_user in user_stats.values() if len(per_user) >= 2)
    qualified = 0
    for per_user in user_stats.values():
        if target not in per_user:
            continue
        target_count, first_target_ts = per_user[target]
        if target_count < MIN_TARGET_EVENTS:
            continue
        if any(name != target and min_ts < first_target_ts for name, (_, min_ts) in per_user.items()):
            qualified += 1
    return {
        "total_users": len(user_stats),
        "bridge_users": bridge_users,
        "qualified_users": qualified,
        "domains": domain_stats,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="将 UCSD Amazon ratings-only CSV（无表头 item,user,rating,timestamp，Unix 秒）"
        "转换为 model/adafedgcdr 数据契约 CSV（user_id,domain,item_id,timestamp_ms）。"
        "每行评分视为一次正向交互；本脚本不去重，重复 (user,domain,item) 由管线 load_events 保留最早时间戳。"
    )
    parser.add_argument("--domain", action="append", required=True, metavar="名称=CSV",
                        help="领域短名与 ratings-only CSV 路径，可重复，如 Books=data/raw/Books.csv（支持 .gz）")
    parser.add_argument("--target", required=True, help="目标域名称，必须是 --domain 之一")
    parser.add_argument("--user-core", action="append", default=[], metavar="N|域名=N",
                        help="用户核心数阈值，默认 10；可用 域名=N 单独覆盖，如 --user-core 10 --user-core Books=96")
    parser.add_argument("--item-core", action="append", default=[], metavar="N|域名=N",
                        help="物品核心数阈值，默认 10；可用 域名=N 单独覆盖")
    parser.add_argument("--out", required=True, help="输出 CSV 路径，如 data/private/events.csv")
    args = parser.parse_args(argv)

    try:
        domains = [parse_domain_spec(spec) for spec in args.domain]
        user_core = parse_core_specs(args.user_core, 10)
        item_core = parse_core_specs(args.item_core, 10)
        summary = convert(domains, args.target, user_core, item_core, args.out)
    except ValueError as exc:
        parser.exit(2, f"错误: {exc}\n")

    print(f"已写出 {args.out}", file=sys.stderr)
    print(
        f"总用户 {summary['total_users']}，桥接用户 {summary['bridge_users']}，"
        f"预估合格用户 {summary['qualified_users']}（未去重口径，以 prepare 审计为准）",
        file=sys.stderr,
    )
    print("下一步: .venv/bin/python -m model.adafedgcdr prepare --events "
          f"{args.out} --target {args.target} --split artifacts/split.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
