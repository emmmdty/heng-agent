# -*- coding: utf-8 -*-
"""四期第一部分：ThrottledChatModel 的闸门持有、退避重试与限流回退

桩替换 `_invoke_upstream`，不接触真实网关。
重点验证流式分支：名额必须持有到流耗尽——若在返回生成器时就释放，限流形同虚设。
"""
from __future__ import annotations

import asyncio

import pytest
from agentscope.credential import OpenAICredential

from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.llm import ThrottledChatModel
from app.infrastructure.settings import Settings
from app.infrastructure.throttle import GatewayThrottle

pytestmark = pytest.mark.asyncio

_CREDENTIAL = OpenAICredential(api_key="test-key", base_url="http://127.0.0.1:9/v1")


class StubUpstreamModel(ThrottledChatModel):
    """用预设行为替代真实上游：每次调用弹出一个 behavior。

    behavior 取值：
        Exception 实例      -> 抛出
        ("stream", n)       -> 返回产出 n 个分片的异步生成器
        其他                 -> 直接作为非流式返回值
    """

    def __init__(self, behaviors: list, **kwargs) -> None:
        super().__init__(**kwargs)
        self._behaviors = list(behaviors)
        self.upstream_calls = 0
        self.concurrent_peak = 0
        self._in_flight = 0

    async def _invoke_upstream(self, messages, tools, tool_choice, **kwargs):
        self.upstream_calls += 1
        behavior = self._behaviors.pop(0) if self._behaviors else "ok"
        if isinstance(behavior, BaseException):
            raise behavior
        if isinstance(behavior, tuple) and behavior[0] == "stream":
            return self._stream(behavior[1])
        return behavior

    async def _stream(self, chunks: int):
        self._in_flight += 1
        self.concurrent_peak = max(self.concurrent_peak, self._in_flight)
        try:
            for index in range(chunks):
                await asyncio.sleep(0.02)
                yield f"chunk-{index}"
        finally:
            self._in_flight -= 1


def _build(behaviors, *, throttle=None, fallback=None, retries=0, bus=None) -> StubUpstreamModel:
    return StubUpstreamModel(
        behaviors,
        credential=_CREDENTIAL,
        model="primary-model",
        throttle=throttle or GatewayThrottle(max_concurrency=1, min_interval_seconds=0),
        fallback=fallback,
        max_transient_retries=retries,
        retry_base_seconds=0.01,
        bus=bus,
    )


class FakeFallbackModel:
    """备用模型替身：只需暴露 model 属性与可等待调用。"""

    def __init__(self) -> None:
        self.model = "fallback-model"
        self.calls = 0

    async def __call__(self, messages, tools=None, tool_choice=None, **kwargs):
        self.calls += 1
        return "fallback-reply"


class TestThrottleHeldUntilStreamDrained:
    async def test_stream_holds_slot_until_drained(self):
        """并发上限 1 时，第二个流必须等第一个流读完才能开始。"""
        throttle = GatewayThrottle(max_concurrency=1, min_interval_seconds=0)
        model = _build([("stream", 3), ("stream", 3)], throttle=throttle)

        async def consume():
            stream = await model([])
            return [chunk async for chunk in stream]

        results = await asyncio.gather(consume(), consume())
        assert all(len(chunks) == 3 for chunks in results)
        assert model.concurrent_peak == 1, (
            f"闸门未持有到流耗尽，出现 {model.concurrent_peak} 个流同时在飞"
        )

    async def test_slot_released_after_stream_drained(self):
        """流读完后名额要归还，否则第二次调用会直接卡死。"""
        throttle = GatewayThrottle(max_concurrency=1, min_interval_seconds=0)
        model = _build([("stream", 2), "ok"], throttle=throttle)

        stream = await model([])
        assert [chunk async for chunk in stream] == ["chunk-0", "chunk-1"]
        assert await asyncio.wait_for(model([]), timeout=0.5) == "ok"

    async def test_slot_released_when_upstream_raises(self):
        throttle = GatewayThrottle(max_concurrency=1, min_interval_seconds=0)
        model = _build([ValueError("bad request"), "ok"], throttle=throttle)

        with pytest.raises(ValueError):
            await model([])
        assert await asyncio.wait_for(model([]), timeout=0.5) == "ok"


class TestRetryAndFallback:
    async def test_transient_error_retried_then_succeeds(self):
        model = _build([RuntimeError("Too many concurrent requests."), "ok"], retries=2)
        assert await model([]) == "ok"
        assert model.upstream_calls == 2

    async def test_business_error_not_retried(self):
        """模型不存在这类错误必须立刻抛出，不能浪费退避时间。"""
        model = _build([RuntimeError("model_not_found"), "ok"], retries=2)
        with pytest.raises(RuntimeError, match="model_not_found"):
            await model([])
        assert model.upstream_calls == 1

    async def test_falls_back_after_retries_exhausted(self):
        fallback = FakeFallbackModel()
        bus = TradeEventBus()
        queue = bus.subscribe("s-fallback")
        model = _build(
            [RuntimeError("Throttling.Concurrency")] * 3,
            fallback=fallback,
            retries=2,
            bus=bus,
        )

        # 事件需按会话路由，模型层从 ShoppingContext 取当前会话
        token = ShoppingContext.set(
            ShoppingContextSnapshot(
                shopping_session_id="s-fallback", buyer_id="b", locale="zh-CN", currency="CNY",
            ),
        )
        try:
            assert await model([]) == "fallback-reply"
        finally:
            ShoppingContext.reset(token)

        assert model.upstream_calls == 3, "主模型应先把重试次数用尽"
        assert fallback.calls == 1

        event = queue.get_nowait()
        assert event.type == "model.fallback"
        assert event.payload["from"] == "primary-model"
        assert event.payload["to"] == "fallback-model"
        assert "Throttling" in event.payload["reason"]

    async def test_raises_when_no_fallback_configured(self):
        model = _build([RuntimeError("429 rate limit")] * 2, retries=1)
        with pytest.raises(RuntimeError, match="429"):
            await model([])


class TestNonStreamingResultDetection:
    """非流式返回的识别不能依赖 hasattr。

    回归的是一个潜伏缺陷：原代码用 `hasattr(result, "__aiter__")` 判断是否流式，
    而 2.0 的 `ChatResponse` 继承自 `DictMixin` → `dict`，属性查找走字典键查找，
    缺键抛的是 **KeyError**，`hasattr` 只吞 `AttributeError`——于是这行在任何
    非流式返回上直接崩。生产默认 stream=True 才一直没暴露，但
    `create_chat_model(settings, stream=False)` 是对外暴露的合法签名，一用就炸。
    """

    async def test_dict_like_response_is_not_mistaken_for_stream(self):
        from app.infrastructure.llm import _is_async_stream

        class DictLikeResponse(dict):
            def __getattr__(self, name):  # 复刻 DictMixin 的行为
                return self[name]

        response = DictLikeResponse(content=[{"type": "text", "text": "hi"}])
        # 先证明 hasattr 确实会在这种对象上抛 KeyError（缺陷成因）
        with pytest.raises(KeyError):
            hasattr(response, "__aiter__")
        # 再证明修法不触碰属性查找协议
        assert _is_async_stream(response) is False

    async def test_async_generator_is_detected_as_stream(self):
        from app.infrastructure.llm import _is_async_stream

        async def gen():
            yield {"content": []}

        stream = gen()
        assert _is_async_stream(stream) is True
        await stream.aclose()


def _build_settings(tmp_path) -> Settings:
    return Settings(
        llm_base_url="http://127.0.0.1:9/v1", llm_api_key="k", llm_model="m",
        port=8000, log_level="info",
        embedding_base_url="", embedding_api_key="", embedding_model="", embedding_dim=8,
        qdrant_url="", qdrant_collection="c",
        reranker_base_url="", reranker_model="", tavily_api_key="",
        otlp_endpoint="", data_dir=tmp_path, category_kb_collection="kb",
        context_size=128000, tool_result_limit=20000, reply_token_budget=0,
        tool_failure_threshold=3, tool_circuit_reset_seconds=60.0,
        cors_origins=["http://localhost:5173"],
    )


class TestRetryLayersAreCollapsed:
    """重试必须只有一层，且就是本仓声明的那一层。

    回归的是一个乘积放大缺陷：openai SDK（默认 2）、AgentScope（默认 3）与本仓
    `ThrottledChatModel`（LLM_MAX_RETRIES）三层重试各自为政，一次逻辑调用在持续
    失败时会打出 3×4×3 = 36 个上游请求。对一个以「网关配额治理」为核心卖点的系统，
    这是自相矛盾的：GatewayThrottle 在前门限并发与间隔，底下两层却背着它疯狂重发，
    而且恰恰发生在网关已经限流的时刻——最该退让时反而放大了压力。
    """

    async def test_lower_layers_do_not_retry(self, tmp_path):
        from app.infrastructure.llm import create_chat_model

        settings = _build_settings(tmp_path)
        model = create_chat_model(settings, stream=False)

        assert model.max_retries == 0, "AgentScope 层不得自行重试"
        assert model.client.max_retries == 0, "openai SDK 层不得自行重试"

    async def test_total_upstream_attempts_equals_declared_budget(self, tmp_path):
        """真实上游请求数必须等于声明值，不能是各层的乘积。"""
        from app.infrastructure.llm import create_chat_model

        settings = _build_settings(tmp_path)
        model = create_chat_model(settings, stream=False)

        total = (
            (model.client.max_retries + 1)
            * (model.max_retries + 1)
            * (settings.llm_max_retries + 1)
        )
        assert total == settings.llm_max_retries + 1, (
            f"上游请求数 {total} != 声明的 {settings.llm_max_retries + 1}，重试层没收口"
        )
