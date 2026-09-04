# -*- coding: utf-8 -*-
"""知识库出处扫描：把"声称来自知识库而本会话并无成功返回"变成可回归的门禁

用法：
    uv run python scripts/eval/audit_knowledge_provenance.py
    uv run python scripts/eval/audit_knowledge_provenance.py --report latest --gate
    uv run python scripts/eval/audit_knowledge_provenance.py --json

来源：二十期分诊 `category-insight` 确认，judge 结构上看不到工具返回，
"知识库当时可不可用"它判不了——出处属不属实这半必须由确定性判据接管。
判据本体在 `app/application/harness/knowledge_provenance.py`，
本脚本是它的下游（二十期的教训：只做"检测"不做"暴露"的判据
在真实评测里等价于不存在）。

**门禁口径：不设阈值、不设样本量下限，命中一处即红**（同算式自洽、收货字段）
——"知识库根本没返回过，回复却说'知识库里说'"是能指着原文说的张冠李戴。

判据刻意窄（只认"知识库 / 品类洞察"字样、诚实降级不算声明），
"本轮 0 处"要分清"判过了、全对"与"压根没东西可判"（踩坑 33）：
读数按"会话内调过 category_insight_tool"分开统计。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.application.harness.knowledge_provenance import (  # noqa: E402
    KnowledgeClaim,
    check_knowledge,
    collect_knowledge_sources,
)
from scripts.eval.audit_number_provenance import (  # noqa: E402
    conversations_dir_from_report,
    latest_report,
    select_audits,
    sessions_from_report,
)
from scripts.eval.trace_audit import (  # noqa: E402
    KNOWLEDGE_WARNING_EVENT,
    SessionTrace,
    load_session,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "conversations"

__all__ = [
    "KNOWLEDGE_WARNING_EVENT",
    "SessionKnowledge",
    "audit_directory_knowledge",
    "audit_session_knowledge",
    "gate_verdict",
    "summarize",
]


@dataclass
class SessionKnowledge:
    session_id: str
    claims: int
    unsourced: list[KnowledgeClaim] = field(default_factory=list)
    # 会话内是否有过 category_insight_tool 的成功返回。没有它，
    # "0 声明"读不出"判据有没有被喂到东西"（踩坑 33）。
    kb_available: bool = False
    kb_called: bool = False
    runtime_flagged: bool = False

    @property
    def clean(self) -> bool:
        return not self.unsourced


def _kb_called(tool_results) -> bool:
    """会话内是否调过 category_insight_tool（含报错）——"0 声明"要按它分两说。"""
    return any(
        isinstance(payload, dict) and payload.get("tool") == "category_insight_tool"
        for payload in tool_results
    )


def audit_session_knowledge(trace: SessionTrace) -> SessionKnowledge:
    """按会话聚合出处状态后逐条回复判定，与运行时口径一致。"""
    sources = collect_knowledge_sources(trace.tool_results)
    claims = 0
    unsourced: list[KnowledgeClaim] = []
    for reply in trace.agent_replies:
        report = check_knowledge(reply, sources)
        claims += report.claims
        unsourced.extend(report.unsourced)
    return SessionKnowledge(
        session_id=trace.session_id,
        claims=claims,
        unsourced=unsourced,
        kb_available=sources.available,
        kb_called=_kb_called(trace.tool_results),
        runtime_flagged=bool(trace.knowledge_warnings),
    )


def audit_directory_knowledge(conversations_dir: Path) -> list[SessionKnowledge]:
    return [
        audit_session_knowledge(load_session(path))
        for path in sorted(Path(conversations_dir).glob("*.jsonl"))
    ]


def summarize(audits: list[SessionKnowledge]) -> dict:
    return {
        "sessions": len(audits),
        "sessions_kb_called": sum(1 for item in audits if item.kb_called),
        "sessions_kb_available": sum(1 for item in audits if item.kb_available),
        "sessions_with_findings": sum(1 for item in audits if not item.clean),
        "sessions_flagged_at_runtime": sum(1 for item in audits if item.runtime_flagged),
        "claims": sum(item.claims for item in audits),
        "unsourced": sum(len(item.unsourced) for item in audits),
    }


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    reason: str


def gate_verdict(summary: dict) -> GateVerdict:
    """命中一处即不通过（口径同算式自洽、收货字段）。"""
    unsourced = summary["unsourced"]
    if unsourced:
        return GateVerdict(
            False,
            f"{unsourced} 处知识库出处声明无据（本轮共 {summary['claims']} 处声明）",
        )
    if not summary["claims"]:
        return GateVerdict(True, (
            f"本轮未抽出任何知识库出处声明（{summary['sessions']} 个会话，"
            f"{summary['sessions_kb_called']} 个调过 category_insight_tool），"
            f"判据无从判定——这不是「全对」"
        ))
    return GateVerdict(True, f"{summary['claims']} 处知识库出处声明全部有据")


def render(audits: list[SessionKnowledge], summary: dict) -> str:
    lines = ["# 知识库出处扫描", ""]
    lines.append(
        f"会话 {summary['sessions']} 个（{summary['sessions_kb_called']} 个调过知识库，"
        f"{summary['sessions_kb_available']} 个有成功返回，"
        f"{summary['sessions_with_findings']} 个有发现，"
        f"{summary['sessions_flagged_at_runtime']} 个在运行时就已告警）｜"
        f"声明 {summary['claims']} 处｜无出处 {summary['unsourced']} 处",
    )
    lines.append("")
    for item in audits:
        if item.clean:
            continue
        mark = "运行时已告警" if item.runtime_flagged else "本次补判发现"
        lines.append(f"## {item.session_id}（{len(item.unsourced)}/{item.claims}，{mark}）")
        for claim in item.unsourced:
            lines.append(f"  - {claim.raw}")
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
        help="有无出处声明则退出码 1（当回归门禁用）",
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

    audits = audit_directory_knowledge(directory)
    if report is not None:
        # 一份都匹配不上时报错退出：0 处声明算出的"0 处问题"
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
