# -*- coding: utf-8 -*-
"""turn 记录的 usage 落盘 + 编排器接线（二十三期清单 2）

模型层发 llm.usage 事件只解决了"事件存在"；要让它进流水，还差两段接线：
    1. 编排器把本轮 drain 到的 llm.usage 事件**求和后写上 agent turn**——
       轮次是"每意图"的自然边界，指标脚本按 turn 聚合，不必再去流水里
       做事件-轮次的相关（落盘顺序不等于发生顺序，地雷 10）；
    2. 两个 ConversationStore 实现把字段原样落盘、原样读回。

分开测的理由与 provenance 接线相同：判定逻辑（这里是求和）与接线
（有没有真的挂上）是两种缺陷，后者外观与"故意不做"完全一样。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from agentscope.message import Msg, TextBlock

from app.application.agents.orchestrator import MainAgentOrchestrator, SubmitIntentInput
from app.domain.buyer.preference import PreferenceStore
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.json_file_stores import JsonFileConversationStore


class NullPreferenceStore(PreferenceStore):
    async def append(self, preference):
        return None

    async def list_by_buyer(self, buyer_id):
        return []

    async def delete(self, buyer_id, statement):
        return False


@dataclass
class FakeAgent:
    """回一句固定文本，回复前按剧本发事件（含模型层的 llm.usage）。"""

    bus: TradeEventBus
    session_id: str
    reply_text: str
    events: list = field(default_factory=list)  # (type, payload) 剧本
    name: str = "main"
    state: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(
            summary="", context=[], tasks_context=SimpleNamespace(tasks=[]),
        ),
    )

    async def reply_stream(self, inputs=None, yield_final_msg=True):
        for event_type, payload in self.events:
            self.bus.publish(self.session_id, event_type, payload)
        yield Msg(name="main", content=[TextBlock(type="text", text=self.reply_text)], role="assistant")


@dataclass
class FakeRegistry:
    agent: FakeAgent

    async def get_or_create(self, session_id):
        return self.agent

    async def persist(self, session_id):
        return None


def _usage(model: str, prompt: int, completion: int) -> tuple[str, dict]:
    return "llm.usage", {
        "model": model, "prompt_tokens": prompt,
        "completion_tokens": completion, "total_tokens": prompt + completion,
    }


async def _run(tmp_path, agent_events: list):
    """跑一轮并把流水读回来，返回 (turns, 原始行)。"""
    bus = TradeEventBus()
    session_id = "s-usage"
    store = JsonFileConversationStore(tmp_path)
    agent = FakeAgent(bus=bus, session_id=session_id, reply_text="找到了，¥89。", events=agent_events)
    orchestrator = MainAgentOrchestrator(
        sessions=FakeRegistry(agent), bus=bus,
        preference_store=NullPreferenceStore(), conversation_store=store,
    )
    result = await orchestrator.handle_intent(
        SubmitIntentInput(
            shopping_session_id=session_id, buyer_id="b1",
            locale="zh-CN", currency="CNY", raw_query="找个露营灯",
        ),
    )
    assert not result.final_text.startswith("[error]"), "假 Agent 没跑通，断言会假绿"
    turns = await store.list_turns(session_id)
    lines = [
        json.loads(line)
        for line in (tmp_path / "conversations" / f"{session_id}.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return turns, lines


class TestTurnUsageRecording:
    async def test_agent_turn_carries_summed_usage(self, tmp_path):
        """一轮两次模型调用（主调用 + 工具后的续调用），turn 上应是两者的和。"""
        turns, _ = await _run(tmp_path, [
            _usage("mimo-v2.5", 1200, 300),
            _usage("mimo-v2.5", 1500, 250),
            ("tool.result", {"tool": "product_search_tool", "hits": []}),
        ])
        agent_turns = [t for t in turns if t.role == "agent"]
        assert len(agent_turns) == 1
        turn = agent_turns[0]
        assert turn.prompt_tokens == 2700
        assert turn.completion_tokens == 550
        assert turn.model == "mimo-v2.5"

    async def test_buyer_turn_stays_zero(self, tmp_path):
        turns, _ = await _run(tmp_path, [_usage("mimo-v2.5", 100, 50)])
        buyer = [t for t in turns if t.role == "buyer"]
        assert len(buyer) == 1
        assert buyer[0].prompt_tokens == 0
        assert buyer[0].completion_tokens == 0
        assert buyer[0].model == ""

    async def test_no_model_call_records_zero_not_absent(self, tmp_path):
        """缓存命中轮没有模型调用：字段在、值为 0——与"旧流水没这个字段"
        是两种读数，指标脚本靠它区分"没调模型"和"没记账"。"""
        turns, lines = await _run(tmp_path, [])
        agent = [t for t in turns if t.role == "agent"][0]
        assert agent.prompt_tokens == 0 and agent.completion_tokens == 0
        agent_line = [line for line in lines if line.get("role") == "agent"][0]
        assert "prompt_tokens" in agent_line and "completion_tokens" in agent_line

    async def test_usage_events_land_in_trace_too(self, tmp_path):
        """llm.usage 事件本身也要进流水——按调用粒度归因（哪次调用烧的）靠它。"""
        _, lines = await _run(tmp_path, [_usage("mimo-v2.5", 100, 50)])
        assert [line["type"] for line in lines if line.get("kind") == "event"].count("llm.usage") == 1

    async def test_fallback_call_is_attributed(self, tmp_path):
        """回退轮的 model 应记实际服务的备用模型，不是主模型。"""
        turns, _ = await _run(tmp_path, [
            _usage("longcat-2.0", 900, 80),
            ("model.fallback", {"from": "mimo-v2.5", "to": "longcat-2.0", "reason": "503"}),
        ])
        agent = [t for t in turns if t.role == "agent"][0]
        assert agent.model == "longcat-2.0"
        assert agent.completion_tokens == 80


class TestJsonStoreRoundTrip:
    async def test_fields_survive_round_trip(self, tmp_path):
        store = JsonFileConversationStore(tmp_path)
        from app.domain.session.ports.conversation_store import ConversationTurn

        await store.append_turn(ConversationTurn(
            session_id="s-rt", buyer_id="b", role="agent", content="hi",
            model="mimo-v2.5", latency_ms=1234,
            prompt_tokens=111, completion_tokens=22,
        ))
        (turn,) = await store.list_turns("s-rt")
        assert turn.prompt_tokens == 111
        assert turn.completion_tokens == 22
        assert turn.model == "mimo-v2.5"

    async def test_old_lines_without_usage_fields_read_back_as_zero(self, tmp_path):
        """旧流水没有这两个字段，读回来必须是 0 而不是炸——审计工具
        要能扫历史目录。"""
        store = JsonFileConversationStore(tmp_path)
        from app.domain.session.ports.conversation_store import ConversationTurn

        await store.append_turn(ConversationTurn(
            session_id="s-old", buyer_id="b", role="agent", content="hi", latency_ms=5,
        ))
        (turn,) = await store.list_turns("s-old")
        assert turn.prompt_tokens == 0 and turn.completion_tokens == 0


class TestSqlStoreParity:
    """SQL 实现同样要落这两个字段——两边漂移是"换了存储实现指标就消失"的
    同型坑。用内存库跑真实 SQL。"""

    async def test_fields_survive_round_trip(self, tmp_path):
        pytest.importorskip("sqlalchemy")
        from sqlalchemy.ext.asyncio import create_async_engine

        from app.infrastructure.persistence.sql.repositories import SqlConversationStore
        from app.infrastructure.persistence.sql.tables import Base
        from app.domain.session.ports.conversation_store import ConversationTurn

        engine = create_async_engine("sqlite+aiosqlite://")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        store = SqlConversationStore(engine)
        await store.append_turn(ConversationTurn(
            session_id="s-sql", buyer_id="b", role="agent", content="hi",
            model="m", latency_ms=10, prompt_tokens=7, completion_tokens=3,
        ))
        (turn,) = await store.list_turns("s-sql")
        assert turn.prompt_tokens == 7
        assert turn.completion_tokens == 3
