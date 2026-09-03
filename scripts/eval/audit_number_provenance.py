# -*- coding: utf-8 -*-
"""金额出处扫描：把"回复里的数字有没有工具出处"变成一个可回归的数

用法：
    uv run python scripts/eval/audit_number_provenance.py
    uv run python scripts/eval/audit_number_provenance.py --max-ratio 0.02   # 当门禁用
    uv run python scripts/eval/audit_number_provenance.py --json

指标：**无出处金额率 = 无出处金额数 / 回复中金额总数**。

这个数是**下界**，不是全貌：只统计带货币标记的数字（¥ $ 元 USD ...），
表格里裸写的 "| 65 |" 不算。放宽到所有数字会把"续航 40 小时""库存 150 件"
一并扫进来，判据立刻不可用。宁可漏报，不要一个没人敢信的数。

分类只是诊断线索：
    suspected_sum          约等于若干工具金额之和 —— 典型是把单品到手价相加
    suspected_difference   约等于两数之差       —— 典型是"预算还剩多少"
    unsourced              找不到成因

注意分类会有巧合命中（三个金额凑出的和恰好等于另一个金额），所以它不参与
"通过与否"的判定，只写在报告里供人复核。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.eval.trace_audit import SessionAudit, audit_directory  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "conversations"


def summarize(audits: list[SessionAudit]) -> dict:
    total = sum(item.total_amounts for item in audits)
    unsourced = sum(len(item.unsourced) for item in audits)
    kinds = Counter(finding.kind for item in audits for finding in item.unsourced)
    return {
        "sessions": len(audits),
        "sessions_with_findings": sum(1 for item in audits if not item.clean),
        "sessions_flagged_at_runtime": sum(1 for item in audits if item.runtime_flagged),
        "total_amounts": total,
        "unsourced_amounts": unsourced,
        "unsourced_ratio": round(unsourced / total, 4) if total else 0.0,
        "kinds": dict(kinds),
    }


def render(audits: list[SessionAudit], summary: dict) -> str:
    lines = ["# 金额出处扫描", ""]
    lines.append(
        f"会话 {summary['sessions']} 个（{summary['sessions_with_findings']} 个有发现，"
        f"{summary['sessions_flagged_at_runtime']} 个在运行时就已告警）｜"
        f"金额 {summary['total_amounts']} 处｜无出处 {summary['unsourced_amounts']} 处"
        f"（{summary['unsourced_ratio']:.1%}）",
    )
    if summary["kinds"]:
        lines.append("成因分布：" + "，".join(f"{k} {v}" for k, v in sorted(summary["kinds"].items())))
    lines.append("")
    for item in audits:
        if item.clean:
            continue
        mark = "运行时已告警" if item.runtime_flagged else "本次补判发现"
        lines.append(f"## {item.session_id}（{len(item.unsourced)}/{item.total_amounts}，{mark}）")
        seen: set[float] = set()
        for finding in item.unsourced:
            if finding.value in seen:
                continue
            seen.add(finding.value)
            explain = f"  ← {finding.explain}" if finding.explain else ""
            lines.append(f"  - {finding.raw}  [{finding.kind}]{explain}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="conversations 目录")
    parser.add_argument("--json", action="store_true", help="只输出汇总 JSON")
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=None,
        help="无出处金额率上限，超过则退出码 1（当回归门禁用）",
    )
    args = parser.parse_args()

    audits = audit_directory(Path(args.dir))
    summary = summarize(audits)
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render(audits, summary))

    if args.max_ratio is not None and summary["unsourced_ratio"] > args.max_ratio:
        print(
            f"\n[门禁不通过] 无出处金额率 {summary['unsourced_ratio']:.1%} > 上限 {args.max_ratio:.1%}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
