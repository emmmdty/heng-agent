# -*- coding: utf-8 -*-
"""收货字段出处扫描：把"回复里的地址/电话/邮编是不是编的"变成可回归的门禁

用法：
    uv run python scripts/eval/audit_contact_provenance.py
    uv run python scripts/eval/audit_contact_provenance.py --report latest --gate   # 当门禁用
    uv run python scripts/eval/audit_contact_provenance.py --json

二十期的教训摆在前面：`arith.inconsistent` 那条判据只做了"检测"没做"暴露"，
在 app 之外零消费方，**在真实评测里等价于不存在**。所以 `contact.unsourced`
从落地第一天就带着它的下游——就是本脚本。

与另外两条离线扫描并列，三条抓的是互不重叠的三类错：

    金额出处（audit_number_provenance）  数字**从哪来**
    算式自洽（audit_arithmetic）         数字**怎么来**
    收货字段（本脚本）                    买家的**个人信息**从哪来

二十期实测那次（`clarify-missing-address`），Agent 写的是
"您之前的记录是上海市浦东新区世纪大道100号"——**里面一个金额都没有**，
前两条扫描完全无感。

**门禁口径与算式自洽相同、与金额出处不同：不设阈值、不设样本量下限，命中一处即红。**
无出处金额率是比率指标（对已有出处数字的修辞取整本来就占几个点），
小样本上不判定是对的；而"这个地址不存在于任何地方"是能指着原文说的事实错误，
"发生了没有"不是"高了低了"（踩坑 45）。

判据刻意窄（只认完整地址、手机号、带标签的邮编；**不认收件人姓名**），
所以"本轮 0 处问题"要分清是"判过了、全对"还是"压根没东西可判"——
读数里两者分开写，不能让后者冒充前者（踩坑 33）。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.application.harness.contact_provenance import (  # noqa: E402
    ContactClaim,
    check_contact,
    collect_contact_sources,
)
from scripts.eval.audit_number_provenance import (  # noqa: E402
    conversations_dir_from_report,
    latest_report,
    select_audits,
    sessions_from_report,
)
from scripts.eval.trace_audit import (  # noqa: E402
    CONTACT_WARNING_EVENT,
    SessionTrace,
    load_session,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "conversations"

__all__ = [
    "CONTACT_WARNING_EVENT",
    "SessionContact",
    "audit_directory_contact",
    "audit_session_contact",
    "gate_verdict",
    "summarize",
]


@dataclass
class SessionContact:
    session_id: str
    claims: int
    unsourced: list[ContactClaim] = field(default_factory=list)
    # 运行时是否当场告警过。补判发现而运行时没告警，说明这份流水跑在判据落地之前
    # ——它是"改动之前"的对照数据，扔了就再也拿不到基线（同 trace_audit 的理由）。
    runtime_flagged: bool = False
    buyer_texts: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.unsourced


def audit_session_contact(trace: SessionTrace) -> SessionContact:
    """按会话聚合出处后逐条回复判定。

    出处按会话而非按轮聚合，与运行时的 `SessionContactSources` 口径一致：
    买家第 1 轮给的地址，第 3 轮复述不算编造。
    """
    sources = collect_contact_sources(
        tool_results=trace.tool_results, buyer_texts=trace.buyer_texts,
    )
    claims = 0
    unsourced: list[ContactClaim] = []
    for reply in trace.agent_replies:
        report = check_contact(reply, sources)
        claims += report.claims
        unsourced.extend(report.unsourced)
    return SessionContact(
        session_id=trace.session_id,
        claims=claims,
        unsourced=unsourced,
        runtime_flagged=bool(trace.contact_warnings),
        buyer_texts=list(trace.buyer_texts),
    )


def audit_directory_contact(conversations_dir: Path) -> list[SessionContact]:
    return [
        audit_session_contact(load_session(path))
        for path in sorted(Path(conversations_dir).glob("*.jsonl"))
    ]


def summarize(audits: list[SessionContact]) -> dict:
    return {
        "sessions": len(audits),
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
    """命中一处即不通过。

    这里**故意没有 --max-ratio / --min-sessions**：那两个旋钮属于比率指标。
    编造一个收货地址是可以指着原文说"这个地址不存在"的事实错误，
    没有"这轮抖了一下"的解释空间，也就没有摊薄它的口径。
    """
    unsourced = summary["unsourced"]
    claims = summary["claims"]
    if unsourced:
        return GateVerdict(
            False, f"{unsourced} 处收货字段没有出处（本轮共抽出 {claims} 处断言）",
        )
    if not claims:
        return GateVerdict(True, (
            f"本轮未抽出任何收货字段断言（{summary['sessions']} 个会话），判据无从判定。"
            f"这不是「全对」——判据只认完整地址、手机号与带标签的邮编，不认收件人姓名"
        ))
    return GateVerdict(True, f"{claims} 处收货字段断言全部有出处")


def render(audits: list[SessionContact], summary: dict) -> str:
    lines = ["# 收货字段出处扫描", ""]
    lines.append(
        f"会话 {summary['sessions']} 个（{summary['sessions_with_findings']} 个有发现，"
        f"{summary['sessions_flagged_at_runtime']} 个在运行时就已告警）｜"
        f"断言 {summary['claims']} 处｜无出处 {summary['unsourced']} 处",
    )
    lines.append("")
    for item in audits:
        if item.clean:
            continue
        mark = "运行时已告警" if item.runtime_flagged else "本次补判发现"
        lines.append(f"## {item.session_id}（{len(item.unsourced)}/{item.claims}，{mark}）")
        for claim in item.unsourced:
            lines.append(f"  - [{claim.kind}] {claim.raw}")
        if item.buyer_texts:
            lines.append(f"  买家原话：{' / '.join(item.buyer_texts)[:200]}")
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
        help="有无出处收货字段则退出码 1（当回归门禁用）",
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

    audits = audit_directory_contact(directory)
    if report is not None:
        # 一份都匹配不上时 select_audits 会报错退出：0 个断言算出的"0 处问题"
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
