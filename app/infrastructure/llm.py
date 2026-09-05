# -*- coding: utf-8 -*-
"""llm

统一创建 AgentScope 2.0 大模型对象。全项目只从这里拿 model，
主 / 子 Agent 各自持有独立实例。

2.0 的模型接入方式：OpenAICredential（携带 api_key + base_url，天然支持
OpenAI 兼容网关）→ OpenAIChatModel(credential=..., model=...)。

四期在此加两层框架不覆盖的东西：
    1. 配额闸门：闸门必须持有到「流耗尽」。流式调用返回的是异步生成器，
       若在 `async with slot()` 内直接 return，名额会在数据还没读完时释放，
       限流等于没做；
    2. 限流回退：实测网关配额池紧张时主模型单发也会 429，退避重试用尽后换用
       备用模型，并发 model.fallback 事件如实告知，不静默降级。
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from typing import Any, AsyncGenerator, Optional

from agentscope.credential import OpenAICredential
from agentscope.message import Msg
from agentscope.model import ChatResponse, OpenAIChatModel
from agentscope.tool import ToolChoice

from app.infrastructure.budget import current_tier, get_budget, minimal_mode_hint
from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.settings import Settings
from app.infrastructure.throttle import GatewayThrottle
from app.infrastructure.transient import is_transient_error

logger = logging.getLogger(__name__)


class ThrottledChatModel(OpenAIChatModel):
    """带配额闸门、退避重试与限流回退的 OpenAIChatModel。"""

    def __init__(
        self,
        *,
        throttle: GatewayThrottle,
        fallback: Optional[OpenAIChatModel] = None,
        max_transient_retries: int = 2,
        retry_base_seconds: float = 6.0,
        bus: Optional[TradeEventBus] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._throttle = throttle
        self._fallback = fallback
        self._max_transient_retries = max_transient_retries
        self._retry_base_seconds = retry_base_seconds
        self._bus = bus

    async def __call__(  # type: ignore[override]
        self,
        messages: list[Msg],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[ToolChoice] = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        # 手动进出上下文而非 async with：流式分支要把名额移交给包装生成器
        slot = self._throttle.slot()
        await slot.__aenter__()
        try:
            result = await self._call_with_fallback(messages, tools, tool_choice, **kwargs)
        except BaseException:
            await slot.__aexit__(*sys.exc_info())
            raise

        if not _is_async_stream(result):
            await slot.__aexit__(None, None, None)
            _charge_budget(result)
            return result
        return self._release_after_stream(slot, result)

    @staticmethod
    async def _release_after_stream(slot: Any, stream: Any) -> AsyncGenerator[ChatResponse, None]:
        """把闸门名额持有到流真正读完（含调用方提前中断的情况）。"""
        last: Any = None
        try:
            async for chunk in stream:
                last = chunk
                yield chunk
        finally:
            # 调用方提前放弃流（aclose / 任务取消）时必须把内层生成器也关掉：
            # 内层 finally 里的 usage 发布要在**调用方上下文**里同步执行，
            # 拖到 asyncgen GC 阶段会丢失事件，甚至串到别的会话
            # （contextvar 在 finalization 里求值）。
            try:
                await stream.aclose()
            except BaseException:  # noqa: BLE001 —— 清理不让异常外溢，包括取消窗口
                pass
            # 流式的 usage 在最后一个 chunk 上，读完再记账
            _charge_budget(last)
            await slot.__aexit__(None, None, None)

    async def _invoke_upstream(
        self,
        messages: list[Msg],
        tools: Optional[list[dict]],
        tool_choice: Optional[ToolChoice],
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """真正打上游的一跳。抽成方法便于替换与测试。

        同时是四档预算降级（16-4 章）的作用点：
        预算充足（main）走主模型；剩余不足时切到更便宜的备用模型，
        并发 model.fallback 事件如实告知——**降级不静默**。
        minimal 档额外注入简洁模式提示，压住 Think 长度。

        两个分支都在**知道实际服务模型**的位置包一层 usage 发布
        （_attach_usage）：预算降级与限流回退都会换模型，
        归错主等于把成本算到别人头上（二十三期清单 2）。
        """
        tier = current_tier()
        if tier != "main":
            hint = minimal_mode_hint()
            if hint:
                messages = [*messages, Msg(name="system", content=hint, role="system")]
            if self._fallback is not None:
                logger.info("Token 预算档位 %s，切用备用模型 %s", tier, self._fallback.model)
                self._publish_budget_tier(tier)
                result = await self._fallback(messages, tools, tool_choice, **kwargs)
                return self._attach_usage(result, self._fallback.model)
        result = await super().__call__(messages, tools, tool_choice, **kwargs)
        return self._attach_usage(result, self.model)

    def _attach_usage(
        self, result: Any, model_name: str,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """给这次调用的结果挂上 usage 发布，**不改变结果本身**。

        非流式当场发；流式包一层生成器，等流耗尽再发——usage 在最后一个
        chunk 上，发早了拿到的是 None 或中间态。没有 bus 就原样返回。
        """
        if self._bus is None:
            return result
        if _is_async_stream(result):
            return self._publish_usage_after_stream(result, model_name)
        self._publish_usage_event(result, model_name)
        return result

    def _publish_usage_after_stream(self, stream: Any, model_name: str) -> AsyncGenerator[Any, None]:
        async def _wrapped() -> AsyncGenerator[Any, None]:
            last: Any = None
            try:
                async for chunk in stream:
                    last = chunk
                    yield chunk
            finally:
                self._publish_usage_event(last, model_name)
        return _wrapped()

    def _publish_usage_event(self, response: Any, model_name: str) -> None:
        """把一次上游调用的 token 用量发成 llm.usage 事件。

        二十三期清单 2 的地基：此前 token 只进内存预算账本，流水里没有
        usage，成本指标建成也无米下锅。与 _charge_budget 同一条纪律：
        **记账失败绝不能影响主链路**，取不到 usage 就跳过，不编造零。
        """
        if self._bus is None or response is None:
            return
        try:
            counts = _token_counts(response)
            if counts is None:
                return
            prompt, completion, total = counts
            self._bus.publish(
                ShoppingContext.current_session_id(),
                "llm.usage",
                {
                    "model": model_name,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total,
                },
            )
        except Exception as err:  # noqa: BLE001
            logger.debug("llm.usage 事件发布跳过（不影响主链路）：%s", err)

    def _publish_budget_tier(self, tier: str) -> None:
        if self._bus is None or self._fallback is None:
            return
        self._bus.publish(
            ShoppingContext.current_session_id(),
            "model.fallback",
            {
                "from": self.model,
                "to": self._fallback.model,
                "reason": f"Token 预算档位 {tier}",
                "budget_tier": tier,
            },
        )

    async def aclose(self) -> None:
        """显式关闭 model 链上全部 HTTP 客户端（主 + 备用）。

        会话淘汰路径调用：openai SDK 对被回收的 client 只有 `__del__` 里
        fire-and-forget 的 aclose 兜底，执行时机不可控——在那之前每会话
        8 个 transport + 8 个 SSLContext（OpenSSL 会话缓存，单个可达数百 KB）
        一直占着内存。每次调用是幂等的（is_closed 的 client 再关无害）。
        """
        clients = [getattr(self, "client", None)]
        fallback = getattr(self, "_fallback", None)
        if fallback is not None:
            clients.append(getattr(fallback, "client", None))
        for client in clients:
            if client is None:
                continue
            # openai SDK 的关闭方法是 close()，httpx 的才是 aclose()——
            # 两层包装都可能出现在这条链上，按名字找
            closer = getattr(client, "aclose", None) or getattr(client, "close", None)
            if closer is None:
                continue
            try:
                await closer()
            except Exception as err:  # noqa: BLE001 —— 关闭失败不影响主链路
                logger.debug("模型客户端关闭跳过：%s", err)

    async def _call_with_fallback(
        self,
        messages: list[Msg],
        tools: Optional[list[dict]],
        tool_choice: Optional[ToolChoice],
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        last_error: Optional[BaseException] = None
        for attempt in range(self._max_transient_retries + 1):
            try:
                return await self._invoke_upstream(messages, tools, tool_choice, **kwargs)
            except BaseException as err:
                if not is_transient_error(err):
                    raise
                last_error = err
                if attempt < self._max_transient_retries:
                    # 指数退避：网关速率类限流对固定间隔重试不敏感
                    delay = self._retry_base_seconds * (3**attempt)
                    logger.warning(
                        "模型 %s 遇上游瞬时故障，%.0fs 后重试（第 %d/%d 次）：%s",
                        self.model, delay, attempt + 1, self._max_transient_retries, err,
                    )
                    await asyncio.sleep(delay)

        if self._fallback is None:
            raise last_error  # type: ignore[misc]

        logger.warning("模型 %s 重试用尽，回退到 %s：%s", self.model, self._fallback.model, last_error)
        self._publish_fallback(str(last_error))
        # 限流回退的这条路**不经过** _invoke_upstream（那里只包了预算降级分支），
        # usage 发布必须在这里再挂一次——漏了它，回退轮次的 token 就从流水里消失
        result = await self._fallback(messages, tools, tool_choice, **kwargs)
        return self._attach_usage(result, self._fallback.model)

    def _publish_fallback(self, reason: str) -> None:
        if self._bus is None or self._fallback is None:
            return
        session_id = ShoppingContext.current_session_id()
        self._bus.publish(
            session_id,
            "model.fallback",
            {"from": self.model, "to": self._fallback.model, "reason": reason},
        )


def _safe_field(source: Any, name: str) -> Any:
    """宽容取字段。

    坑：ChatResponse.usage 的 `__getattr__` 对缺失字段抛 **KeyError**，
    而 `getattr(obj, name, default)` 只吃 AttributeError——直接用 getattr 带默认值
    依旧会把 KeyError 抛到主链路，把整轮对话搞成 [error]。实测踩过。
    """
    if isinstance(source, dict):
        return source.get(name)
    try:
        return getattr(source, name, None)
    except Exception:  # noqa: BLE001 —— 包含 KeyError 等非标准实现
        return None


def _token_counts(response: Any) -> Optional[tuple[int, int, int]]:
    """从一次模型调用返回里取 (prompt, completion, total)。

    usage 字段各网关存在差异：input_tokens/output_tokens 与 prompt_tokens/
    completion_tokens 两套名字都见过，按两套名字兜底取。三个数一个都取不到
    （usage 缺失或全空）时返回 None——调用方据此跳过，**不编造零**。
    ChatResponse 的 `__getattr__` 对缺失字段抛 KeyError（地雷 5），
    取字段一律走 _safe_field。
    """
    usage = _safe_field(response, "usage")
    if usage is None:
        return None
    prompt = _safe_field(usage, "input_tokens")
    if prompt is None:
        prompt = _safe_field(usage, "prompt_tokens")
    completion = _safe_field(usage, "output_tokens")
    if completion is None:
        completion = _safe_field(usage, "completion_tokens")
    total = _safe_field(usage, "total_tokens")
    if prompt is None and completion is None and total is None:
        return None
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    if total is None:
        total = prompt + completion
    return prompt, completion, int(total)


def _charge_budget(response: Any) -> None:
    """把一次模型调用的 token 记进当前意图的预算账本。

    未启用预算（TOKEN_BUDGET_TOTAL=0）时直接返回，零开销。
    usage 字段各网关存在差异，取不到就不计——**记账失败绝不能影响主链路**。
    """
    try:
        budget = get_budget()
        if budget is None or response is None:
            return
        counts = _token_counts(response)
        if counts is None:
            return
        budget.charge("llm", counts[2])
    except Exception as err:  # noqa: BLE001
        logger.debug("Token 记账跳过（不影响主链路）：%s", err)


def _is_async_stream(result: Any) -> bool:
    """结果是不是异步流。

    不能用 `hasattr(result, "__aiter__")`：2.0 的 `ChatResponse` 继承自
    `DictMixin` → `dict`，属性查找走字典键查找，缺键抛的是 **KeyError**，
    而 `hasattr` 只吞 `AttributeError`——于是这行在任何**非流式**返回上直接崩。
    生产默认 stream=True 才一直没暴露，但 `create_chat_model(settings, stream=False)`
    是对外暴露的合法签名，一用就炸。
    改用 `inspect.isasyncgen` 按类型判定，不触碰对象的属性查找协议。
    """
    return inspect.isasyncgen(result) or isinstance(result, AsyncGenerator)


def create_chat_model(
    settings: Settings,
    stream: bool = True,
    throttle: Optional[GatewayThrottle] = None,
    bus: Optional[TradeEventBus] = None,
) -> OpenAIChatModel:
    credential = OpenAICredential(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    common = {
        "credential": credential,
        "stream": stream,
        # 上下文窗口：压缩触发阈值（ContextConfig.trigger_ratio）按此值比例计算
        "context_size": settings.context_size,
        # ---- 重试收口：本层是唯一的重试策略持有者 ----
        # 默认情况下有三层重试各自为政，且是**乘积**关系：
        #   openai SDK  max_retries=2  → 3 次
        #   AgentScope  max_retries=3  → 4 次
        #   本仓 ThrottledChatModel    → LLM_MAX_RETRIES+1 = 3 次
        # 一次逻辑调用在持续失败时会打出 3×4×3 = **36 个上游请求**。
        # 对一个主打「网关配额治理」的系统这是自相矛盾的：GatewayThrottle 在前门
        # 限并发与请求间隔，底下两层却在背着它疯狂重发，而且恰恰发生在网关已经
        # 限流/故障的时刻——正是最该退让的时候反而放大了压力。
        # 把下面两层都关到 0，重试与退避只由本层做，且回退与 model.fallback 事件
        # 也只在本层触发，行为可预期、可观测。
        "max_retries": 0,
        "client_kwargs": {"max_retries": 0},
    }
    fallback = (
        OpenAIChatModel(model=settings.llm_fallback_model, **common)
        if settings.llm_fallback_model and settings.llm_fallback_model != settings.llm_model
        else None
    )
    # 注意 fallback 必须是裸 OpenAIChatModel：若换成带 bus 的 ThrottledChatModel，
    # 它内部的 _attach_usage 与外层各包一次，同一调用会双发 llm.usage
    return ThrottledChatModel(
        model=settings.llm_model,
        throttle=throttle or GatewayThrottle(settings.llm_max_concurrency, settings.llm_min_interval_seconds),
        fallback=fallback,
        max_transient_retries=settings.llm_max_retries,
        bus=bus,
        **common,
    )
