# -*- coding: utf-8 -*-
"""编排器接入金额出处校验的接线测试

判定逻辑本身在 tests/test_number_provenance.py 里测；这里测的是**接线**：
判定器有没有真的挂在轮次边界上、告警有没有真的发出来。

分开测是有代价的教训：设计演进记录里，BM25 索引只在评测脚本里构造、
从没接进 composition，评测选出的最优配置根本没上线，而"忘了接线"与
"故意关掉"外观完全一样、没有任何告警。可选能力必须有一条测试钉住接线。

这里用最小假 Agent 替掉 AgentScope 的真 Agent：编排器只用到
reply_stream / state.summary / state.context / state.tasks_context，
把这四样铺出来就够跑通一轮，不必起模型。
"""
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from agentscope.message import Msg, TextBlock

from app.application.agents.orchestrator import MainAgentOrchestrator, SubmitIntentInput
from app.domain.buyer.preference import PreferenceStore
from app.infrastructure.eventbus import TradeEventBus


class NullPreferenceStore(PreferenceStore):
    async def append(self, preference):  # pragma: no cover —— 本文件用不到写路径
        return None

    async def list_by_buyer(self, buyer_id):
        return []

    async def delete(self, buyer_id, statement):  # pragma: no cover
        return False


@dataclass
class FakeAgent:
    """回一句固定文本，并在回复前按剧本发若干条 tool.result 事件。"""

    bus: TradeEventBus
    session_id: str
    reply_text: str
    tool_results: list = field(default_factory=list)
    name: str = "main"
    state: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(
            summary="", context=[], tasks_context=SimpleNamespace(tasks=[]),
        ),
    )

    async def reply_stream(self, inputs=None, yield_final_msg=True):
        for payload in self.tool_results:
            self.bus.publish(self.session_id, "tool.result", payload)
        yield Msg(name="main", content=[TextBlock(type="text", text=self.reply_text)], role="assistant")


@dataclass
class FakeRegistry:
    agent: FakeAgent

    async def get_or_create(self, session_id):
        return self.agent

    async def persist(self, session_id):
        return None


async def _run(reply_text: str, tool_results: list, raw_query: str = "两个一起多少钱") -> list:
    """跑一轮，返回本轮总线上的全部事件。"""
    bus = TradeEventBus()
    session_id = "s-prov"
    observer = bus.subscribe(session_id)
    agent = FakeAgent(bus=bus, session_id=session_id, reply_text=reply_text, tool_results=tool_results)
    orchestrator = MainAgentOrchestrator(
        sessions=FakeRegistry(agent), bus=bus, preference_store=NullPreferenceStore(),
    )
    await orchestrator.handle_intent(
        SubmitIntentInput(
            shopping_session_id=session_id, buyer_id="b1",
            locale="zh-CN", currency="CNY", raw_query=raw_query,
        ),
    )
    events = []
    while not observer.empty():
        events.append(observer.get_nowait())
    finals = [e.payload.get("text", "") for e in events if e.type == "final.result"]
    assert finals and not finals[0].startswith("[error]"), (
        f"假 Agent 没跑通，本轮被吞成异常，断言会假绿：{finals}"
    )
    return events


TWO_LANDED_PRICES = [{
    "tool": "product_search_tool",
    "hits": [
        {"landed_price": {"landed_total_major": 364.0}},
        {"landed_price": {"landed_total_major": 154.0}},
    ],
}]


class TestProvenanceWiring:
    async def test_unsourced_amount_raises_warning_event(self):
        """真实 bad case 复现：¥364 + ¥154 被相加成 ¥518，运费重复计了一次。"""
        events = await _run("两个一起买到手 ¥518。", TWO_LANDED_PRICES)

        warnings = [e for e in events if e.type == "number.unsourced"]
        assert warnings, "无出处金额必须发告警事件，否则等于没接线"
        (item,) = warnings[0].payload["unsourced"]
        assert item["value"] == 518.0
        assert item["kind"] == "suspected_sum"

    async def test_sourced_reply_raises_no_warning(self):
        events = await _run("分别是 ¥364 和 ¥154。", TWO_LANDED_PRICES)
        assert not [e for e in events if e.type == "number.unsourced"]

    async def test_buyer_quoted_number_is_not_flagged(self):
        events = await _run("你的预算 300 元够用。", [], raw_query="帮我找 300 块以内的露营灯")
        assert not [e for e in events if e.type == "number.unsourced"]

    async def test_warning_lands_in_persisted_trace(self, tmp_path):
        """告警必须进落盘轨迹——bad case 回收要靠它把失败会话捞出来。"""
        from app.infrastructure.persistence.json_file_stores import JsonFileConversationStore

        bus = TradeEventBus()
        session_id = "s-trace"
        store = JsonFileConversationStore(tmp_path)
        agent = FakeAgent(
            bus=bus, session_id=session_id,
            reply_text="两个一起买到手 ¥518。", tool_results=TWO_LANDED_PRICES,
        )
        orchestrator = MainAgentOrchestrator(
            sessions=FakeRegistry(agent), bus=bus,
            preference_store=NullPreferenceStore(), conversation_store=store,
        )
        await orchestrator.handle_intent(
            SubmitIntentInput(
                shopping_session_id=session_id, buyer_id="b1",
                locale="zh-CN", currency="CNY", raw_query="两个一起多少钱",
            ),
        )

        lines = (tmp_path / "conversations" / f"{session_id}.jsonl").read_text(encoding="utf-8").splitlines()
        kinds = [json.loads(line).get("type") for line in lines if line.strip()]
        assert "number.unsourced" in kinds

    async def test_sources_carry_over_to_later_turns(self):
        """第 2 轮复述第 1 轮检索到的价格，不该被判无出处。"""
        bus = TradeEventBus()
        session_id = "s-multi"
        observer = bus.subscribe(session_id)
        agent = FakeAgent(
            bus=bus, session_id=session_id, reply_text="找到了，¥89。",
            tool_results=[{"tool": "product_search_tool", "hits": [{"price_major": 89.0}]}],
        )
        orchestrator = MainAgentOrchestrator(
            sessions=FakeRegistry(agent), bus=bus, preference_store=NullPreferenceStore(),
        )
        intent = SubmitIntentInput(
            shopping_session_id=session_id, buyer_id="b1",
            locale="zh-CN", currency="CNY", raw_query="找个露营灯",
        )
        first = await orchestrator.handle_intent(intent)
        assert not first.final_text.startswith("[error]"), "假 Agent 没跑通，断言会假绿"

        agent.reply_text = "刚才那款 LumenGo，¥89。"
        agent.tool_results = []  # 第 2 轮没再检索
        second = await orchestrator.handle_intent(intent)
        assert not second.final_text.startswith("[error]")

        while not observer.empty():
            event = observer.get_nowait()
            assert event.type != "number.unsourced", "跨轮引用旧检索结果不是无出处"


@pytest.mark.parametrize("event_type", ["number.unsourced"])
def test_event_type_is_registered(event_type):
    from app.infrastructure.eventbus import EVENT_TYPES

    assert event_type in EVENT_TYPES


class TestPreferenceHintDoesNotHijackTheTurn:
    """偏好注入不能把买家本轮的问题挤掉。

    真实失败（干净整轮里的 memory-recall，12/13 里唯一挂的那条）：
    买家问"帮我推荐一套旅行三件套"，Agent 只回了
    "已为你记住'不要塑料材质'的长期偏好，后续推荐会自动避开塑料类产品。😊"——
    **它对着注入的偏好作答，买家的问题一个字没回。**

    根因是结构的，不是措辞的：偏好被当成**另一条 user 消息**插在买家问句之前，
    上下文里于是连着两条买家发言，模型挑了前一条当本轮的话。
    修法也应该是结构的——合并成一条消息，让"本轮买家说了什么"只有一个答案，
    而不是在提示词里求模型别理它（这条路本仓已经验证过拦不住，见踩坑 25）。
    """

    async def test_preference_and_query_arrive_as_one_message(self):
        from app.application.agents.orchestrator import MainAgentOrchestrator
        from app.domain.buyer.preference import BuyerPreference

        class OnePreferenceStore(NullPreferenceStore):
            async def list_by_buyer(self, buyer_id):
                return [BuyerPreference(buyer_id=buyer_id, kind="dislike", statement="不要塑料材质")]

        orchestrator = MainAgentOrchestrator(
            sessions=None, bus=TradeEventBus(), preference_store=OnePreferenceStore(),
        )
        intent = SubmitIntentInput(
            shopping_session_id="s1", buyer_id="b1",
            locale="zh-CN", currency="CNY", raw_query="帮我推荐一套旅行三件套。",
        )
        inputs = await orchestrator._build_inputs(intent, "s1")

        assert len(inputs) == 1, "偏好和问句必须是同一条消息，否则模型会对着偏好作答"
        text = inputs[0].get_text_content()
        assert "不要塑料材质" in text
        assert text.rstrip().endswith("帮我推荐一套旅行三件套。"), "买家问句必须在最后，是本轮真正要答的东西"

    async def test_preference_block_is_marked_as_background(self):
        from app.application.memory.preference_selector import render_preference_hint
        from app.domain.buyer.preference import BuyerPreference

        hint = render_preference_hint(
            [BuyerPreference(buyer_id="b1", kind="dislike", statement="不要塑料材质")],
        )
        assert "不是买家本轮" in hint or "不要对本段作答" in hint, (
            "块内要自证身份：它是背景资料，不是这一轮的问题"
        )


class TestContactProvenanceWiring:
    """收货字段出处校验的接线（二十期实测缺陷 `clarify-missing-address`）。

    判定逻辑本身在 tests/test_contact_provenance.py 里测；这里测的是**接线**
    ——判定器有没有真的挂在轮次边界上、告警有没有真的进事件流与落盘轨迹。
    与金额出处校验分开测的理由完全相同（见本文件开头）。
    """

    async def test_fabricated_address_raises_warning_event(self):
        """真实 bad case 复现：本轮只检索过商品，地址是编的，还说成"您之前的记录"。"""
        events = await _run(
            "收货地址：您之前的记录是上海市浦东新区世纪大道100号，这次还是这个地址吗？",
            [{"tool": "product_search_tool", "hits": [{"product_id": "P1008"}]}],
            raw_query="帮我下单 2 个 LumenGo 露营灯军绿色。",
        )

        warnings = [e for e in events if e.type == "contact.unsourced"]
        assert warnings, "编造的收货地址必须发告警事件，否则等于没接线"
        (item,) = warnings[0].payload["unsourced"]
        assert item["kind"] == "address"
        assert item["raw"] == "上海市浦东新区世纪大道100号"

    async def test_asking_for_the_address_raises_nothing(self):
        """这条用例要的正是"去问"——判据不能反过来罚正确行为。"""
        events = await _run(
            "还需要您提供收货地址、收件人和联系电话，我才能下单。",
            [{"tool": "product_search_tool", "hits": [{"product_id": "P1008"}]}],
            raw_query="帮我下单 2 个 LumenGo 露营灯军绿色。",
        )
        assert not [e for e in events if e.type == "contact.unsourced"]

    async def test_address_the_buyer_gave_raises_nothing(self):
        events = await _run(
            "好的，寄到上海市浦东新区世纪大道100号。",
            [],
            raw_query="寄到浦东世纪大道100号，帮我下单",
        )
        assert not [e for e in events if e.type == "contact.unsourced"]

    async def test_warning_lands_in_persisted_trace(self, tmp_path):
        """告警必须进落盘轨迹——离线审计与 bad case 回收都从这里读。"""
        from app.infrastructure.persistence.json_file_stores import JsonFileConversationStore

        bus = TradeEventBus()
        session_id = "s-contact-trace"
        store = JsonFileConversationStore(tmp_path)
        agent = FakeAgent(
            bus=bus, session_id=session_id,
            reply_text="收货地址：您之前的记录是上海市浦东新区世纪大道100号。",
            tool_results=[{"tool": "product_search_tool", "hits": [{"product_id": "P1008"}]}],
        )
        orchestrator = MainAgentOrchestrator(
            sessions=FakeRegistry(agent), bus=bus,
            preference_store=NullPreferenceStore(), conversation_store=store,
        )
        await orchestrator.handle_intent(
            SubmitIntentInput(
                shopping_session_id=session_id, buyer_id="b1",
                locale="zh-CN", currency="CNY", raw_query="帮我下单 2 个 LumenGo 露营灯军绿色。",
            ),
        )

        lines = (tmp_path / "conversations" / f"{session_id}.jsonl").read_text(encoding="utf-8").splitlines()
        kinds = [json.loads(line).get("type") for line in lines if line.strip()]
        assert "contact.unsourced" in kinds
