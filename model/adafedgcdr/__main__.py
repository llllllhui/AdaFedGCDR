import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .base import BaseConfig, load_base, save_base, train_base
from .data import (
    DEFAULT_K_VALUES,
    audit_users,
    build_split,
    file_sha256,
    load_events,
    load_manifest,
    make_cases,
    manifest_sha256,
    save_manifest,
    select_group_cases,
)
from .experiment import (
    EvalCheckpoint,
    FusionConfig,
    build_catalog,
    evaluate,
    load_fusion,
    render_report,
    save_fusion,
    train_fusion,
)


def fail(message: str):
    raise SystemExit(f"错误：{message}")


def check_hashes(events_path, manifest, split_digest, events_digest, checkpoint_name, stored_split, stored_events):
    if stored_split != split_digest:
        fail(f"{checkpoint_name} 的拆分 SHA-256 与 {events_path} 对应拆分不符（{stored_split} != {split_digest}）")
    if stored_events != events_digest:
        fail(f"{checkpoint_name} 的数据 SHA-256 与 {events_path} 不符（{stored_events} != {events_digest}）")


def check_prepare_audit(events_path, split_path, manifest):
    audit_path = Path(str(split_path) + ".audit.json")
    if not audit_path.is_file():
        fail(f"缺少审计文件 {audit_path}，请先对同一数据运行 prepare")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    events_digest = file_sha256(events_path)
    if audit.get("events_sha256") != events_digest:
        fail(f"{events_path} 与 prepare 时的数据不符（数据在 prepare 之后被修改过）")
    if audit.get("split_sha256") != manifest_sha256(manifest):
        fail(f"{split_path} 与 prepare 审计中的拆分 SHA-256 不符")


def cmd_prepare(args):
    events = load_events(args.events)
    manifest = build_split(events, args.target, args.seed)
    save_manifest(manifest, args.split)
    audit = audit_users(events, args.target)
    audit.update(
        {
            "target_domain": args.target,
            "events_sha256": file_sha256(args.events),
            "split_sha256": manifest_sha256(manifest),
            "group_sizes": {group: len(manifest.users[group]) for group in manifest.users},
        }
    )
    audit_path = Path(str(args.split) + ".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def cmd_train_base(args):
    events = load_events(args.events)
    manifest = load_manifest(args.split)
    check_prepare_audit(args.events, args.split, manifest)
    config = BaseConfig(
        target_domain=manifest.target_domain,
        embedding_size=args.embedding_size,
        source_rounds=args.source_rounds,
        target_rounds=args.target_rounds,
        local_epoch=args.local_epoch,
        user_batch=args.user_batch,
        dp=args.dp,
        device=args.device,
        seed=manifest.seed,
    )
    artifacts = train_base(events, manifest, config)
    save_base(args.out, artifacts, config, file_sha256(args.events))
    print(
        json.dumps(
            {
                "out": args.out,
                "domain_names": artifacts.domain_names,
                "users": len(artifacts.source_vectors),
                "target_items": len(artifacts.target_items),
                "target_train_users": len(artifacts.target_train_users),
                "split_sha256": artifacts.split_sha256,
                "config": asdict(config),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_train_fusion(args):
    events = load_events(args.events)
    manifest = load_manifest(args.split)
    split_digest = manifest_sha256(manifest)
    events_digest = file_sha256(args.events)
    artifacts, _, base_events_digest = load_base(args.base)
    check_hashes(args.events, manifest, split_digest, events_digest, args.base, artifacts.split_sha256, base_events_digest)
    catalog = build_catalog(events, manifest, artifacts)
    cases = make_cases(events, manifest)
    config = FusionConfig(
        seed=manifest.seed,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        candidate_count=args.candidate_count,
        batch_size=args.batch_size,
        device=args.device,
    )
    groups = {
        "fusion": select_group_cases(cases, manifest, "fusion"),
        "validation": select_group_cases(cases, manifest, "validation"),
    }
    checkpoint = train_fusion(artifacts, groups, config, catalog)
    save_fusion(args.out, checkpoint, events_digest)
    print(
        json.dumps(
            {
                "out": args.out,
                "best_epoch": checkpoint.best_epoch,
                "split_sha256": checkpoint.split_sha256,
                "config": asdict(config),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_evaluate(args):
    events = load_events(args.events)
    manifest = load_manifest(args.split)
    split_digest = manifest_sha256(manifest)
    events_digest = file_sha256(args.events)
    artifacts, _, base_events_digest = load_base(args.base)
    check_hashes(args.events, manifest, split_digest, events_digest, args.base, artifacts.split_sha256, base_events_digest)
    fusion, fusion_events_digest = load_fusion(args.fusion)
    check_hashes(args.events, manifest, split_digest, events_digest, args.fusion, fusion.split_sha256, fusion_events_digest)
    catalog = build_catalog(events, manifest, artifacts)
    cases = select_group_cases(make_cases(events, manifest), manifest, "test")
    config = FusionConfig(
        seed=manifest.seed,
        candidate_count=args.candidate_count,
        device=args.device,
    )
    metrics = evaluate(EvalCheckpoint(base=artifacts, fusion_state=fusion.fusion_state, config=config), cases, catalog)
    text = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    report_path = Path(args.report) if args.report else Path(args.out).parent / "report.md"
    report_path.write_text(render_report(metrics), encoding="utf-8")
    print(text)
    print(f"报告已写入 {report_path}")


def build_parser():
    parser = argparse.ArgumentParser(prog="model.adafedgcdr", description="AdaFedGCDR 自适应冷启动验证版")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="读取 CSV 并写拆分及数据审计")
    prepare.add_argument("--events", required=True)
    prepare.add_argument("--target", required=True)
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.add_argument("--split", required=True)
    prepare.set_defaults(func=cmd_prepare)

    train_base_parser = sub.add_parser("train-base", help="训练并导出隔离的跨域基础表示")
    train_base_parser.add_argument("--events", required=True)
    train_base_parser.add_argument("--split", required=True)
    train_base_parser.add_argument("--source-rounds", type=int, default=150)
    train_base_parser.add_argument("--target-rounds", type=int, default=150)
    train_base_parser.add_argument("--embedding-size", type=int, default=16)
    train_base_parser.add_argument("--local-epoch", type=int, default=3)
    train_base_parser.add_argument("--user-batch", type=int, default=16)
    train_base_parser.add_argument("--device", default="cpu")
    train_base_parser.add_argument("--dp", action=argparse.BooleanOptionalAction, default=True)
    train_base_parser.add_argument("--out", required=True)
    train_base_parser.set_defaults(func=cmd_train_base)

    train_fusion_parser = sub.add_parser("train-fusion", help="按冷启动阶段训练融合器")
    train_fusion_parser.add_argument("--events", required=True)
    train_fusion_parser.add_argument("--split", required=True)
    train_fusion_parser.add_argument("--base", required=True)
    train_fusion_parser.add_argument("--candidate-count", type=int, default=100)
    train_fusion_parser.add_argument("--lr", type=float, default=1e-3)
    train_fusion_parser.add_argument("--max-epochs", type=int, default=50)
    train_fusion_parser.add_argument("--patience", type=int, default=5)
    train_fusion_parser.add_argument("--batch-size", type=int, default=64)
    train_fusion_parser.add_argument("--device", default="cpu")
    train_fusion_parser.add_argument("--out", required=True)
    train_fusion_parser.set_defaults(func=cmd_train_fusion)

    evaluate_parser = sub.add_parser("evaluate", help="在测试组上评估并写出指标")
    evaluate_parser.add_argument("--events", required=True)
    evaluate_parser.add_argument("--split", required=True)
    evaluate_parser.add_argument("--base", required=True)
    evaluate_parser.add_argument("--fusion", required=True)
    evaluate_parser.add_argument("--candidate-count", type=int, default=100)
    evaluate_parser.add_argument("--device", default="cpu")
    evaluate_parser.add_argument("--report", default=None)
    evaluate_parser.add_argument("--out", required=True)
    evaluate_parser.set_defaults(func=cmd_evaluate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
