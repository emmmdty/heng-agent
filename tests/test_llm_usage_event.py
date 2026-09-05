# -*- coding: utf-8 -*-
"""模型层的 llm.usage 事件（二十三期清单 2 的地基）

任务 2 要"读 data/conversations/ 流水的 usage 与 latency_ms"，但流水里
**从来没有 usage**：模型层的 token 只进了内存预算账本（_charge_budget），
从未落进事件轨迹——观测侧缺了一环，指标脚本建成也无米下锅。

修法与 model.fallback 同一条路：模型层在每次上游调用结束后把
(input, output, total) 发成 llm.usage 事件，编排器的轮次轨迹自然会把它
写进流水。事件里的 model 取**实际服务的那个模型**——预算降级与限流回退
都会换模型，归错主等于把成本算到别人头上。

测试不打真实网关：monkeypatch 掉 OpenAIChatModel.__call__（MRO 上
`_invoke_upstream` 的 `super().__call__` 正是它）。
"""
from __future__ import annotations

import asyncio

import pytest
from agentscope.credential import OpenAICredential
from agentscope.message import TextBlock
from agentscope.model import ChatResponse, ChatUsage, OpenAIChatModel

from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.llm import ThrottledChatModel
from app.infrastructure.throttle import GatewayThrottle

pytestmark = pytest.mark.asyncio

# 打不到上游：__call__ 已被替换，凭据只用于父类建 client
_CREDENTIAL = OpenAICredential(api_key="test-key", base_url="http://127.0.0.1:9/v1")


def _response(prompt: int = 100, completion: int = 50) -> ChatResponse:
    return ChatResponse(
        content=[TextBlock(type="text", text="hi")],
        is_last=True,
        usage=ChatUsage(input_tokens=prompt, output_tokens=completion, time=0.1),
    )


def _response_without_usage() -> ChatResponse:
    return ChatResponse(content=[TextBlock(type="text", text="hi")], is_last=True)


async def _fake_call(self, messages, tools=None, tool_choice=None, **kwargs):
    return _response()


async def _fake_stream(self, messages, tools=None, tool_choice=None, **kwargs):
    async def _gen():
        yield ChatResponse(content=[TextBlock(type="text", text="a")], is_last=False)
        yield ChatResponse(content=[TextBlock(type="text", text="b")], is_last=False)
        yield _response(prompt=30, completion=7)
    return _gen()


def _model(bus=None, model_name="primary-model") -> ThrottledChatModel:
    return ThrottledChatModel(
        model=model_name,
        credential=_CREDENTIAL,
        throttle=GatewayThrottle(max_concurrency=1, min_interval_seconds=0),
        bus=bus,
    )


def _session(session_id: str):
    """模型层从 ShoppingContext 取当前会话路由事件（与 model.fallback 同一条路）。"""
    return ShoppingContext.set(ShoppingContextSnapshot(
        shopping_session_id=session_id, buyer_id="b", locale="zh-CN", currency="CNY",
    ))


def _drain(queue) -> list:
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


class TestUsageEvent:
    async def test_nonstream_publishes_usage(self, monkeypatch):
        monkeypatch.setattr(OpenAIChatModel, "__call__", _fake_call)
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        token = _session("s1")
        try:
            result = await _model(bus=bus)([])
        finally:
            ShoppingContext.reset(token)
        assert result is not None, "主链路不能被记账破坏"
        usage = [e for e in _drain(queue) if e.type == "llm.usage"]
        assert len(usage) == 1
        assert usage[0].payload == {
            "model": "primary-model", "prompt_tokens": 100,
            "completion_tokens": 50, "total_tokens": 150,
        }

    async def test_stream_publishes_usage_of_last_chunk(self, monkeypatch):
        """流式的 usage 在最后一个 chunk 上（_charge_budget 的同一条注释），
        事件必须等流耗尽再发——发早了拿到的是 None 或中间态。"""
        monkeypatch.setattr(OpenAIChatModel, "__call__", _fake_stream)
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        token = _session("s1")
        try:
            stream = await _model(bus=bus)([])
            chunks = [chunk async for chunk in stream]
        finally:
            ShoppingContext.reset(token)
        assert len(chunks) == 3
        usage = [e for e in _drain(queue) if e.type == "llm.usage"]
        assert len(usage) == 1
        assert usage[0].payload["prompt_tokens"] == 30
        assert usage[0].payload["completion_tokens"] == 7
        assert usage[0].payload["total_tokens"] == 37

    async def test_response_without_usage_publishes_nothing(self, monkeypatch):
        """usage 取不到就不计（_charge_budget 的同一取舍）——不编造零。"""

        async def _no_usage(self, messages, tools=None, tool_choice=None, **kwargs):
            return _response_without_usage()

        monkeypatch.setattr(OpenAIChatModel, "__call__", _no_usage)
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        token = _session("s1")
        try:
            await _model(bus=bus)([])
        finally:
            ShoppingContext.reset(token)
        assert [e for e in _drain(queue) if e.type == "llm.usage"] == []


class TestFallbackAttribution:
    async def test_fallback_usage_attributes_to_fallback_model(self, monkeypatch):
        """回退后真正服务的是备用模型——usage 记它，不记主模型。"""

        class FakeFallbackModel:
            model = "fallback-model"

            async def __call__(self, messages, tools=None, tool_choice=None, **kwargs):
                return _response(prompt=11, completion=22)

        async def _always_fail(self, messages, tools=None, tool_choice=None, **kwargs):
            raise RuntimeError("Throttling.Concurrency")  # 瞬时限流类，走回退

        monkeypatch.setattr(OpenAIChatModel, "__call__", _always_fail)
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        fallback = FakeFallbackModel()
        model = ThrottledChatModel(
            model="primary-model", credential=_CREDENTIAL,
            throttle=GatewayThrottle(max_concurrency=1, min_interval_seconds=0),
            fallback=fallback, max_transient_retries=0, bus=bus,
        )
        token = _session("s1")
        try:
            result = await model([])
        finally:
            ShoppingContext.reset(token)
        assert result is not None
        usage = [e for e in _drain(queue) if e.type == "llm.usage"]
        assert len(usage) == 1
        assert usage[0].payload["model"] == "fallback-model"
        assert usage[0].payload["total_tokens"] == 33


class TestSafety:
    async def test_no_bus_does_not_crash(self, monkeypatch):
        monkeypatch.setattr(OpenAIChatModel, "__call__", _fake_call)
        token = _session("s1")
        try:
            assert await _model(bus=None)([]) is not None
        finally:
            ShoppingContext.reset(token)

    async def test_publish_failure_does_not_break_main_path(self, monkeypatch):
        """记账失败绝不能影响主链路（_charge_budget 的同一条纪律）。"""
        monkeypatch.setattr(OpenAIChatModel, "__call__", _fake_call)

        class BrokenBus:
            def publish(self, *args, **kwargs):
                raise RuntimeError("bus down")

        token = _session("s1")
        try:
            result = await _model(bus=BrokenBus())([])
        finally:
            ShoppingContext.reset(token)
        assert result is not None

    async def test_abandoned_stream_closes_inner_and_releases_slot(self, monkeypatch):
        """调用方中途放弃流：内层生成器必须被同步关闭。

        内层 finally 里的 usage 发布要靠 aclose 传进去——拖到 asyncgen GC
        阶段会在错误（或已清空）的 contextvar 里求值，事件丢失或串会话。"""
        closed: list[bool] = []

        async def _abort_stream(self, messages, tools=None, tool_choice=None, **kwargs):
            async def _gen():
                try:
                    yield ChatResponse(content=[TextBlock(type="text", text="a")], is_last=False)
                    yield ChatResponse(content=[TextBlock(type="text", text="b")], is_last=False)
                finally:
                    closed.append(True)
            return _gen()

        monkeypatch.setattr(OpenAIChatModel, "__call__", _abort_stream)
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        model = _model(bus=bus)
        token = _session("s1")
        try:
            stream = await model([])
            async for _chunk in stream:  # 消费一个分片后放弃
                break
            await stream.aclose()
            # 闸门已释放：下一次调用不被卡死
            assert await asyncio.wait_for(model([]), timeout=0.5) is not None
        finally:
            ShoppingContext.reset(token)
        assert closed == [True], "内层生成器没有被关闭"
        # 没拿到 usage chunk：不发事件，更不能发一个归错会话的
        assert [e for e in _drain(queue) if e.type == "llm.usage"] == []


class TestBudgetTierAttribution:
    async def test_budget_tier_fallback_attributes_to_fallback_model(self, monkeypatch):
        """预算降级分支（_invoke_upstream 内 tier != main）的 usage
        也要归到实际服务的备用模型——两条回退路径只测过一条不算钉住。"""

        class FakeFallbackModel:
            model = "cheap-model"

            async def __call__(self, messages, tools=None, tool_choice=None, **kwargs):
                return _response(prompt=5, completion=6)

        monkeypatch.setattr("app.infrastructure.llm.current_tier", lambda: "minimal")
        monkeypatch.setattr("app.infrastructure.llm.minimal_mode_hint", lambda: None)
        monkeypatch.setattr(OpenAIChatModel, "__call__", _fake_call)
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        model = ThrottledChatModel(
            model="primary-model", credential=_CREDENTIAL,
            throttle=GatewayThrottle(max_concurrency=1, min_interval_seconds=0),
            fallback=FakeFallbackModel(), bus=bus,
        )
        token = _session("s1")
        try:
            await model([])
        finally:
            ShoppingContext.reset(token)
        usage = [e for e in _drain(queue) if e.type == "llm.usage"]
        assert len(usage) == 1, "预算降级分支的 usage 没发出来"
        assert usage[0].payload["model"] == "cheap-model"
