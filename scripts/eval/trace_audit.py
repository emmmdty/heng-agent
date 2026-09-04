# -*- coding: utf-8 -*-
"""trace_audit —— 对话流水的离线审计

把 `DATA_DIR/conversations/{session_id}.jsonl` 还原成可判定的结构，
再在上面跑金额出处校验（app/application/harness/number_provenance.py）。

为什么离线侧还要跑一遍运行时已经在跑的判定：

    1. **历史流水补判**。护栏是后加的，之前跑过的评测流水里没有
       number.unsourced 事件，但它们记录的是真实回复，照样可以补判——
       这是唯一一批"改动之前"的对照数据，扔了就再也拿不到基线了。
    2. **可回归的单一数字**。运行时告警散在各个会话里，离线扫描把它聚合成
       「无出处金额率」，改一次提示词/工具就能复算一次，才能当基线用。

还原时有一个必须处理的顺序问题：JsonFileConversationStore 先写两条 turn、
再批量写 events，落盘顺序与发生顺序不同。所以这里**先整份读完再判**，
按会话聚合出处，而不是边读边判——后者会让第一轮的工具返回永远"晚于"回复，
无出处率恒为 100%，而这个错误的读数看上去和真实读数一样像模像样。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.application.harness.number_provenance import (  # noqa: E402
    UnsourcedAmount,
    check_reply,
    collect_sources,
)

# 运行时护栏发出的告警事件类型（离线侧据此区分"当时就发现了"与"这次补判才发现"）
RUNTIME_WARNING_EVENT = "number.unsourced"
# 算式自洽校验的告警（十九期）。与上面那条分开收：两者抓的是互补的两类错，
# 混在一起会让"这次补判才发现"的判断跟着串味。
ARITHMETIC_WARNING_EVENT = "arith.inconsistent"


@dataclass
class SessionTrace:
    session_id: str
    buyer_texts: list[str] = field(default_factory=list)
    agent_replies: list[str] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    runtime_warnings: list[dict] = field(default_factory=list)
    arith_warnings: list[dict] = field(default_factory=list)


@dataclass
class SessionAudit:
    session_id: str
    total_amounts: int
    unsourced: list[UnsourcedAmount]
    runtime_flagged: bool
    # 买家原话一并带出：bad case 入池要靠它复现，不带的话飞轮最后一步只能吐 TODO
    buyer_texts: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.unsourced


def load_session(path: Path) -> SessionTrace:
    """读一份会话流水。单行损坏跳过，整份缺失按空会话处理——
    审计工具本身不该因为一行脏数据就报废整轮扫描。"""
    trace = SessionTrace(session_id=Path(path).stem)
    if not Path(path).exists():
        return trace
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        kind = record.get("kind")
        if kind == "turn":
            if record.get("role") == "buyer":
                trace.buyer_texts.append(record.get("content", ""))
            elif record.get("role") == "agent":
                trace.agent_replies.append(record.get("content", ""))
        elif kind == "event":
            if record.get("type") == "tool.result":
                trace.tool_results.append(record.get("payload") or {})
            elif record.get("type") == RUNTIME_WARNING_EVENT:
                trace.runtime_warnings.append(record.get("payload") or {})
            elif record.get("type") == ARITHMETIC_WARNING_EVENT:
                trace.arith_warnings.append(record.get("payload") or {})
    return trace


def audit_session(trace: SessionTrace) -> SessionAudit:
    """按会话聚合出处后逐条回复判定。

    出处按会话而非按轮聚合，与运行时的 SessionSources 口径一致：
    模型引用上一轮检索到的价格是正常行为。
    """
    sources = collect_sources(tool_results=trace.tool_results, buyer_texts=trace.buyer_texts)
    total = 0
    unsourced: list[UnsourcedAmount] = []
    for reply in trace.agent_replies:
        report = check_reply(reply, sources)
        total += report.total_amounts
        unsourced.extend(report.unsourced)
    return SessionAudit(
        session_id=trace.session_id,
        total_amounts=total,
        unsourced=unsourced,
        runtime_flagged=bool(trace.runtime_warnings),
        buyer_texts=list(trace.buyer_texts),
    )


def audit_directory(conversations_dir: Path) -> list[SessionAudit]:
    """扫一个 conversations 目录，按会话 id 排序返回。"""
    return [
        audit_session(load_session(path))
        for path in sorted(Path(conversations_dir).glob("*.jsonl"))
    ]
