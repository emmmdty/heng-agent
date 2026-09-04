# -*- coding: utf-8 -*-
"""算式自洽扫描：把"回复里写出来的算式算不算得通"变成一个可回归的门禁

用法：
    uv run python scripts/eval/audit_arithmetic.py
    uv run python scripts/eval/audit_arithmetic.py --report latest --gate   # 当门禁用
    uv run python scripts/eval/audit_arithmetic.py --json

十九期把 `check_arithmetic` 接进了编排器轮末，命中即发 `arith.inconsistent`。
**但没有任何人读这个事件**——判据写进流水就结束了，不进门禁、不进报告。
护栏发出的告警没人读，与没有护栏的外观完全一样。本脚本补的是"读"这一半。

与金额出处扫描（`audit_number_provenance.py`）是并列的两条，抓互补的两类错：

    金额出处   数字**从哪来**——没有工具出处即可疑
    算式自洽   数字**怎么来**——写出来的过程算不通

十九期那次错的三个数（886.34 / 7.5% / 6.48）**都有工具出处**，
出处扫描对它完全无感；反过来凭知识说出的 `$800` 算式扫描也看不见。

**门禁口径与出处扫描不同：不设阈值、不设样本量下限。**
无出处金额率是比率指标（对已有出处数字的修辞取整本来就占几个点），
小样本上不判定是对的；而 `886.34 × 7.5% = 6.48` 是一次具体的事实错误，
"发生了没有"不是"高了低了"（踩坑 45）。命中一处即红。

判据刻意窄：只认显式写出来的百分比乘法，346 条历史回复里只抽出 4 个算式。
所以"本轮 0 处问题"要分清是"判过了、全对"还是"压根没东西可判"——
读数里两者分开写，不能让后者冒充前者（同踩坑 33）。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.application.harness.arithmetic_check import Problem, check_arithmetic  # noqa: E402
from scripts.eval.audit_number_provenance import (  # noqa: E402
    conversations_dir_from_report,
    latest_report,
    select_audits,
    sessions_from_report,
)
from scripts.eval.trace_audit import SessionTrace, load_session  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "conversations"


@dataclass
class SessionArithmetic:
    session_id: str
    equations: int
    problems: list[Problem] = field(default_factory=list)
    # 运行时是否当场告警过。补判发现而运行时没告警，说明这条流水跑在判据落地之前
    # ——它是"改动之前"的对照数据，扔了就再也拿不到基线（同 trace_audit 的理由）。
    runtime_flagged: bool = False

    @property
    def clean(self) -> bool:
        return not self.problems


def audit_session_arithmetic(trace: SessionTrace) -> SessionArithmetic:
    """逐条回复验算式自洽。

    按回复而不是按会话聚合：算式是写在某一条回复里的，
    没有"引用上一轮"这回事（这一点与金额出处正相反）。
    """
    equations = 0
    problems: list[Problem] = []
    for reply in trace.agent_replies:
        report = check_arithmetic(reply)
        equations += report.equations
        problems.extend(report.problems)
    return SessionArithmetic(
        session_id=trace.session_id,
        equations=equations,
        problems=problems,
        runtime_flagged=bool(trace.arith_warnings),
    )


def audit_directory_arithmetic(conversations_dir: Path) -> list[SessionArithmetic]:
    return [
        audit_session_arithmetic(load_session(path))
        for path in sorted(Path(conversations_dir).glob("*.jsonl"))
    ]


def summarize(audits: list[SessionArithmetic]) -> dict:
    return {
        "sessions": len(audits),
        "sessions_with_findings": sum(1 for item in audits if not item.clean),
        "sessions_flagged_at_runtime": sum(1 for item in audits if item.runtime_flagged),
        "equations": sum(item.equations for item in audits),
        "problems": sum(len(item.problems) for item in audits),
    }


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    reason: str


def gate_verdict(summary: dict) -> GateVerdict:
    """命中一处即不通过。

    这里**故意没有 --max-ratio / --min-sessions**：那两个旋钮属于比率指标。
    算式不自洽是可以指着原文说"这一行算错了"的事实错误，
    没有"这轮抖了一下"的解释空间，也就没有摊薄它的口径。
    """
    problems = summary["problems"]
    equations = summary["equations"]
    if problems:
        return GateVerdict(False, f"{problems} 处算式等号两边对不上（本轮共抽出 {equations} 个算式）")
    if not equations:
        return GateVerdict(True, (
            f"本轮未抽出任何显式算式（{summary['sessions']} 个会话），判据无从判定。"
            f"这不是「全对」——判据只认显式写出的百分比乘法，历史上 346 条回复里也只有 4 个"
        ))
    return GateVerdict(True, f"{equations} 个算式全部自洽")


def render(audits: list[SessionArithmetic], summary: dict) -> str:
    lines = ["# 算式自洽扫描", ""]
    lines.append(
        f"会话 {summary['sessions']} 个（{summary['sessions_with_findings']} 个有发现，"
        f"{summary['sessions_flagged_at_runtime']} 个在运行时就已告警）｜"
        f"算式 {summary['equations']} 个｜不自洽 {summary['problems']} 处",
    )
    lines.append("")
    for item in audits:
        if item.clean:
            continue
        mark = "运行时已告警" if item.runtime_flagged else "本次补判发现"
        lines.append(f"## {item.session_id}（{len(item.problems)}/{item.equations}，{mark}）")
        for problem in item.problems:
            lines.append(f"  - {problem.raw}  → 应为 {problem.expected:.2f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default=None,
        help="conversations 目录（不传时：有 --report 就跟着报告记下的 DATA_DIR 走，"
             "否则用仓库默认的 data/conversations）",
    )
    parser.add_argument("--json", action="store_true", help="只输出汇总 JSON")
    parser.add_argument(
        "--report",
        default=None,
        metavar="PATH|latest",
        help="只扫该报告那一轮的会话（门禁用这个；不传则扫全目录）",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="有不自洽算式则退出码 1（当回归门禁用）",
    )
    args = parser.parse_args()

    report = None
    if args.report is not None:
        report = (latest_report() if args.report == "latest"
                  else json.loads(Path(args.report).read_text(encoding="utf-8")))

    directory = Path(args.dir) if args.dir else None
    if directory is None and report is not None:
        directory = conversations_dir_from_report(report)
    if directory is None:
        directory = DEFAULT_DIR
    if args.dir is None and directory != DEFAULT_DIR:
        print(f"（按报告记下的 DATA_DIR 扫描：{directory}）\n")

    audits = audit_directory_arithmetic(directory)
    if report is not None:
        # 一份都匹配不上时 select_audits 会报错退出：0 个算式算出的"0 处问题"
        # 会被门禁当满分放行，比红灯更危险
        audits = select_audits(audits, sessions_from_report(report))
    summary = summarize(audits)
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render(audits, summary))

    if args.gate:
        verdict = gate_verdict(summary)
        if not verdict.passed:
            print(f"\n[门禁不通过] {verdict.reason}", file=sys.stderr)
            raise SystemExit(1)
        print(f"\n[门禁通过] {verdict.reason}")


if __name__ == "__main__":
    main()
