# -*- coding: utf-8 -*-
"""Bad-case 采集：把失败自动汇进标注池

用法：
    uv run python scripts/eval/collect_bad_cases.py                       # 扫流水
    uv run python scripts/eval/collect_bad_cases.py --report eval/report-*.json
    uv run python scripts/eval/collect_bad_cases.py --list                # 看池子
    uv run python scripts/eval/collect_bad_cases.py --triage compare-two --status wontfix --note "轨迹漏发已修"
    uv run python scripts/eval/collect_bad_cases.py --promote compare-two # 出用例骨架

闭环的位置：

    线上/评测跑出失败 → 本脚本入池（eval/bad_cases.jsonl）
        → 人工定级（status: new → promoted / wontfix）
        → --promote 出 cases.yaml 骨架 → 进回归集 → 下一轮评测

**为什么留人工定级这一步**：自动进回归集是个陷阱。一条 bad case 可能是
模型的运行间抖动，也可能是判据本身写歪了；不加分诊就扩测试集，等于把噪声
固化成基准，之后每一轮都在为噪声调参。所以脚本只做"发现与去重"，
"这条值不值得进回归集"由人回答。

入池不覆盖人工状态（见 bad_case_pool 模块说明），因此本脚本可以反复跑。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.eval.bad_case_pool import (  # noqa: E402
    STATUS_NEW,
    VALID_STATUSES,
    from_provenance,
    from_report,
    load_pool,
    merge,
    triage,
)
from scripts.eval.trace_audit import audit_directory  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_POOL = PROJECT_ROOT / "eval" / "bad_cases.jsonl"
DEFAULT_CONVERSATIONS = PROJECT_ROOT / "data" / "conversations"

_PROMOTE_TEMPLATE = """  - id: {case_id}-regression
    description: 由 bad case 升级（{source}）
    queries:
      - "{query}"
    rubric:
      p0:
        # 原始成因：{reason}
        - "TODO：把上面的成因写成可判定的 P0 判据"
      p1:
        - "TODO"
      p2:
        - "TODO"
"""


def _print_pool(pool: dict) -> None:
    if not pool:
        print("池子是空的。")
        return
    by_status: dict[str, int] = {}
    for case in pool.values():
        by_status[case.status] = by_status.get(case.status, 0) + 1
    print(f"共 {len(pool)} 条：" + "，".join(f"{k} {v}" for k, v in sorted(by_status.items())))
    print()
    for case in sorted(pool.values(), key=lambda c: (c.status, c.case_id)):
        print(f"[{case.status:8}] {case.case_id:22} ({case.source})  {case.reason[:90]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=str(DEFAULT_POOL))
    parser.add_argument("--conversations", default=str(DEFAULT_CONVERSATIONS))
    parser.add_argument("--report", default=None, help="scripts/eval_regression.py 产出的 .json")
    parser.add_argument("--no-provenance", action="store_true", help="跳过流水扫描")
    parser.add_argument("--list", action="store_true", help="只列出池子内容")
    parser.add_argument("--promote", default=None, help="为指定 case_id 输出 cases.yaml 骨架")
    parser.add_argument("--triage", default=None, help="按 fingerprint 或 case_id 定级")
    parser.add_argument("--status", default=None, choices=VALID_STATUSES, help="配合 --triage")
    parser.add_argument("--note", default="", help="配合 --triage：为什么定这个级")
    parser.add_argument(
        "--force",
        action="store_true",
        help="配合 --triage：按 case_id 定级时连**已定级**的条目一起覆盖"
             "（默认跳过，避免静默冲掉人工判断）",
    )
    args = parser.parse_args()

    pool_path = Path(args.pool)

    if args.list:
        _print_pool(load_pool(pool_path))
        return

    if args.triage:
        if not args.status:
            raise SystemExit("--triage 必须配 --status")
        changed, protected = triage(
            pool_path, args.triage, args.status, args.note, force=args.force,
        )
        print(f"已把 {changed} 条标为 {args.status}" if changed else f"没有匹配 {args.triage} 的条目")
        for case in protected:
            print(
                f"  跳过（已定级为 {case.status}）：{case.fingerprint[:10]} {case.reason[:60]}",
            )
        if protected:
            print(
                "  要改这些请按指纹点名（上面那串前缀就能用），"
                "确实要整批覆盖再加 --force。",
            )
        return

    if args.promote:
        pool = load_pool(pool_path)
        matched = [case for case in pool.values() if case.case_id == args.promote]
        if not matched:
            raise SystemExit(f"池子里没有 case_id={args.promote}")
        for case in matched:
            print(_PROMOTE_TEMPLATE.format(
                case_id=case.case_id,
                source=case.source,
                query=case.buyer_query or "TODO：填入能复现该失败的买家问句",
                reason=case.reason.replace("\n", " "),
            ))
        return

    harvested = []
    if not args.no_provenance:
        harvested += from_provenance(audit_directory(Path(args.conversations)))
    if args.report:
        harvested += from_report(json.loads(Path(args.report).read_text(encoding="utf-8")))

    added, updated = merge(pool_path, harvested)
    print(f"采集 {len(harvested)} 条：新增 {added}，已存在 {updated}｜池子：{pool_path}")
    new_cases = [case for case in load_pool(pool_path).values() if case.status == STATUS_NEW]
    if new_cases:
        print(f"\n待分诊 {len(new_cases)} 条：")
        for case in sorted(new_cases, key=lambda c: c.case_id):
            print(f"  {case.case_id:22} ({case.source})  {case.reason[:90]}")


if __name__ == "__main__":
    main()
