# -*- coding: utf-8 -*-
"""组合总价错加扫描（basket_misadd）：把"相加当组合总价"变成可回归的门禁

用法：
    uv run python scripts/eval/audit_basket_sum.py
    uv run python scripts/eval/audit_basket_sum.py --report latest --gate   # 当门禁用
    uv run python scripts/eval/audit_basket_sum.py --json

缺陷形状（quote_basket_tool docstring 记录的实测）：买家问"两个一起多少钱"，
模型把两个单品 landed_price 相加当组合总价——每个加数都来自工具，
judge 判 PASS，错的是"运费按一次履约计"这条模型推不出来的口径。

与金额出处扫描的关系：basket_misadd 是 `number_provenance` 分类里
**唯一参与门禁判定的 kind**。其余 kind（suspected_sum / suspected_difference /
unsourced）是诊断线索，走比率口径；本判据是"这一行把组合总价算错了"的事实错误，
**不设阈值、不设样本量下限，命中一处即红**（口径同算式自洽、收货字段）。

四个升级条件缺一不可（宁可漏报不误报）：无出处、≥2 个 landed 值相加、
金额所在行带组合语境且不带分开语境（"分开买合计"是合法用法——
分开买本来就各付各的运费，加法恰好是对的）、会话内存在 quote_basket 报价
且组合总价与该金额不符。没有第 4 条就没有 ground truth，只作线索不定罪。

**"本轮 0 处"要分清是"判过了、全对"还是"压根没东西可判"**：
读数里两者分开写，不能让后者冒充前者（踩坑 33）。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.application.harness.number_provenance import (  # noqa: E402
    KIND_BASKET_MISADD,
    UnsourcedAmount,
    check_reply,
    collect_sources,
)
from scripts.eval.audit_number_provenance import (  # noqa: E402
    conversations_dir_from_report,
    latest_report,
    select_audits,
    sessions_from_report,
)
from scripts.eval.trace_audit import SessionTrace, load_session  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "conversations"

__all__ = [
    "SessionBasket",
    "audit_directory_basket",
    "audit_session_basket",
    "gate_verdict",
    "summarize",
]


@dataclass
class SessionBasket:
    session_id: str
    amounts: int
    violations: list[UnsourcedAmount] = field(default_factory=list)
    # 会话内是否出现过 quote_basket 报价。没有它判据只有线索可看、定不了罪，
    # 所以"本轮 0 处违规"必须按这个字段分成两说（踩坑 33）。
    has_basket_quote: bool = False
    runtime_flagged: bool = False

    @property
    def clean(self) -> bool:
        return not self.violations


def _runtime_basket_warnings(trace: SessionTrace) -> int:
    return sum(
        1
        for payload in trace.runtime_warnings
        for item in (payload.get("unsourced") or [])
        if item.get("kind") == KIND_BASKET_MISADD
    )


def audit_session_basket(trace: SessionTrace) -> SessionBasket:
    """按会话聚合出处后逐条回复判定，与运行时口径一致。"""
    sources = collect_sources(tool_results=trace.tool_results, buyer_texts=trace.buyer_texts)
    amounts = 0
    violations: list[UnsourcedAmount] = []
    for reply in trace.agent_replies:
        report = check_reply(reply, sources)
        amounts += report.total_amounts
        violations.extend(item for item in report.unsourced if item.kind == KIND_BASKET_MISADD)
    return SessionBasket(
        session_id=trace.session_id,
        amounts=amounts,
        violations=violations,
        has_basket_quote=bool(sources.basket),
        runtime_flagged=_runtime_basket_warnings(trace) > 0,
    )


def audit_directory_basket(conversations_dir: Path) -> list[SessionBasket]:
    return [
        audit_session_basket(load_session(path))
        for path in sorted(Path(conversations_dir).glob("*.jsonl"))
    ]


def summarize(audits: list[SessionBasket]) -> dict:
    return {
        "sessions": len(audits),
        "sessions_with_basket_quote": sum(1 for item in audits if item.has_basket_quote),
        "sessions_with_findings": sum(1 for item in audits if not item.clean),
        "sessions_flagged_at_runtime": sum(1 for item in audits if item.runtime_flagged),
        "amounts": sum(item.amounts for item in audits),
        "violations": sum(len(item.violations) for item in audits),
    }


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    reason: str


def gate_verdict(summary: dict) -> GateVerdict:
    """命中一处即不通过。

    这里**故意没有 --max-ratio / --min-sessions**：那两个旋钮属于比率指标。
    "组合总价把运费重复计了一次"是能指着原文说的事实错误，
    没有"这轮抖了一下"的解释空间。
    """
    violations = summary["violations"]
    quoted = summary["sessions_with_basket_quote"]
    if violations:
        return GateVerdict(
            False,
            f"{violations} 处组合总价来自单品到手价相加，而会话内 quote_basket 已报过价"
            f"（{quoted} 个会话有组合报价）",
        )
    if not quoted:
        return GateVerdict(True, (
            f"本轮 {summary['sessions']} 个会话没有任何一个调过 quote_basket_tool，"
            f"判据无从判定——这不是「全对」，别把它当绿灯读"
        ))
    return GateVerdict(
        True,
        f"{quoted} 个会话有组合报价，{summary['amounts']} 处金额全部与工具一致或仅为线索级偏离",
    )


def render(audits: list[SessionBasket], summary: dict) -> str:
    lines = ["# 组合总价错加扫描（basket_misadd）", ""]
    lines.append(
        f"会话 {summary['sessions']} 个（{summary['sessions_with_basket_quote']} 个有组合报价，"
        f"{summary['sessions_with_findings']} 个有发现，"
        f"{summary['sessions_flagged_at_runtime']} 个在运行时就已告警）｜"
        f"金额 {summary['amounts']} 处｜违规 {summary['violations']} 处",
    )
    lines.append("")
    for item in audits:
        if item.clean:
            continue
        mark = "运行时已告警" if item.runtime_flagged else "本次补判发现"
        lines.append(f"## {item.session_id}（{len(item.violations)}/{item.amounts}，{mark}）")
        for finding in item.violations:
            lines.append(f"  - {finding.raw}：{finding.explain}")
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
        help="有 basket_misadd 违规则退出码 1（当回归门禁用）",
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

    audits = audit_directory_basket(directory)
    if report is not None:
        # 一份都匹配不上时 select_audits 会报错退出：0 个金额算出的"0 处违规"
        # 会被门禁当满分放行，比红灯更危险（踩坑 33）
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
