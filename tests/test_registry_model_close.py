# -*- coding: utf-8 -*-
"""会话淘汰时显式关闭模型客户端（soak 内存分析的修复，二十三期清单 7）

**为什么需要这个**（诊断证据链，见 docs/交接文档 soak 段）：

每会话的 model 链 = 主模型 + 备用模型，各持一个 openai.AsyncOpenAI
（内含 httpx.AsyncClient；1 个 client 4 个 transport + 4 个 SSLContext）。
Agent 被 LRU 淘汰后，openai SDK 的 `AsyncHttpxClientWrapper.__del__`
用 `create_task(self.aclose())` fire-and-forget 兜底关闭——但这依赖
事件循环恰好有空执行那些 task；执行完之前 transport/SSLContext
（每会话 8+8 个）一直占着内存，SSLContext 尤其重（OpenSSL 会话缓存）。

修复：SessionRegistry 淘汰 Agent 时**立刻调度** model 链的 aclose
（淘汰发生在 get_or_create 的 await 链内，loop 一定在跑）。
被淘汰的必须关、活着的**不误杀**——后者是这条修复的另一半验收。

不打一次模型：只创建（client 构造不联网），断言 is_closed 状态。
"""
import asyncio
from pathlib import Path

import pytest

from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.settings import load_settings

MAX_SESSIONS = 200
TOTAL = 220  # 超出上限 → 20 个被淘汰


@pytest.fixture()
def registry(tmp_path: Path):
    """真实 model 链（ThrottledChatModel + fallback）的 SessionRegistry。"""
    from app.application.agents.main_agent import MainAgentFactory, SessionRegistry
    from app.application.agents.search_agent import SearchAgentFactory
    from app.application.agents.trade_agent import TradeAgentFactory
    from app.application.harness.confirmation import ConfirmationTracker
    from app.application.harness.loop_detector import LoopDetector
    from app.application.harness.order_provenance import OrderProvenanceTracker
    from app.application.usecases.catalog_search import CatalogSearchUseCase
    from app.application.usecases.order_usecases import (
        CancelOrderUseCase,
        PlaceOrderUseCase,
        QueryOrderUseCase,
    )
    from app.infrastructure.harness_middleware import build_tool_middlewares
    from app.infrastructure.persistence.in_memory_repositories import (
        InMemoryOrderRepository,
        InMemoryProductRepository,
    )
    from app.infrastructure.persistence.json_file_stores import (
        JsonFilePreferenceStore,
        JsonFileSessionStore,
    )
    from app.infrastructure.resilience import CircuitBreakerRegistry
    from app.infrastructure.throttle import GatewayThrottle
    from app.domain.catalog.exchange_rate import ExchangeRateTable
    from app.domain.shipping.tariff_schedule import TariffSchedule

    settings = load_settings()
    bus = TradeEventBus()
    circuit = CircuitBreakerRegistry()
    product_repo, order_repo = InMemoryProductRepository(), InMemoryOrderRepository()

    def middlewares():
        return build_tool_middlewares(
            settings, circuit_registry=circuit, bus=bus,
            loop_detector=LoopDetector(repeat_threshold=3),
            order_provenance=OrderProvenanceTracker(),
            confirmation=ConfirmationTracker(),
        )

    search = SearchAgentFactory(
        settings, CatalogSearchUseCase(product_repo), bus, None, circuit,
        GatewayThrottle(2, 0.0), product_repo=product_repo,
        tariff=TariffSchedule(rates=ExchangeRateTable()), tool_middlewares=middlewares,
    )
    trade = TradeAgentFactory(
        settings, PlaceOrderUseCase(product_repo, order_repo), QueryOrderUseCase(order_repo),
        CancelOrderUseCase(product_repo, order_repo), bus, circuit, GatewayThrottle(2, 0.0),
        tool_middlewares=middlewares,
    )
    pref = JsonFilePreferenceStore(tmp_path)
    store = JsonFileSessionStore(tmp_path)
    main_factory = MainAgentFactory(
        settings, search, trade, bus, pref, circuit, GatewayThrottle(2, 0.0),
        sequencing=None, loop_detector=None, order_provenance=None, confirmation=None,
    )
    return SessionRegistry(main_factory, store, max_sessions=MAX_SESSIONS, on_evict=None)


def _client_closed(model) -> bool:
    """model 链上所有 AsyncOpenAI 是否都已关闭（主 + fallback）。

    注意 openai 的 is_closed 是**方法**（bound method）不是属性——
    直接 getattr 拿到的是恒真的 callable，必须调用后取值。
    """
    clients = [model.client]
    fallback = getattr(model, "_fallback", None)
    if fallback is not None and getattr(fallback, "client", None) is not None:
        clients.append(fallback.client)
    for client in clients:
        value = getattr(client, "is_closed", False)
        value = value() if callable(value) else value
        if not value:
            return False
    return True


async def _settle():
    """让淘汰时调度的 aclose task 跑完：20 个 task、每个内含多层 await，
    sleep(0) 轮次要够 + 给一点真实时间让连接池的关闭 IO 完成。"""
    for _ in range(20):
        await asyncio.sleep(0)
    await asyncio.sleep(0.1)


class TestEvictionClosesModelClients:
    async def test_evicted_sessions_clients_get_closed(self, registry):
        """核心验收：LRU 淘汰后，被淘汰会话的 model 客户端必须被关闭——

        修复前它们只能等 openai `__del__` 的 fire-and-forget task，
        transport/SSLContext 在那之前一直占内存（soak 拐点后斜率不归零的成因）。
        """
        models: dict[str, object] = {}
        for i in range(1, TOTAL + 1):
            agent = await registry.get_or_create(f"diag-close-{i:04d}")
            models[f"diag-close-{i:04d}"] = agent.model
        await _settle()

        evicted = [models[f"diag-close-{i:04d}"] for i in range(1, TOTAL - MAX_SESSIONS + 1)]
        assert len(evicted) == TOTAL - MAX_SESSIONS, "必须有会话被淘汰，否则测的是空集"
        assert all(_client_closed(m) for m in evicted), (
            "被淘汰会话的模型客户端必须被显式关闭"
        )

    async def test_alive_sessions_clients_stay_open(self, registry):
        """另一半验收：不能把还活着的会话的客户端关掉——那是新缺陷。"""
        for i in range(1, TOTAL + 1):
            await registry.get_or_create(f"diag-alive-{i:04d}")
        await _settle()

        cached = registry.cached_sessions()
        assert len(cached) == MAX_SESSIONS
        for sid in cached[:5]:
            agent = await registry.get_or_create(sid)  # 命中缓存，不重建
            assert not _client_closed(agent.model), f"活会话 {sid} 的客户端不得被误杀"

    async def test_close_survives_loop_and_missing_model(self, registry):
        """淘汰回调的健壮性：agent 无 model 属性 / 无 running loop 时不得炸——
        淘汰路径挂了会连累正常对话轮次。"""
        from app.application.agents.main_agent import SessionRegistry

        class _NoModelAgent:
            pass

        reg = registry
        reg._close_agent_model(_NoModelAgent())  # 无 model 属性：静默跳过
        # 同步上下文（无 running loop）下也不得抛
        import threading

        error: list[Exception] = []

        def _worker():
            try:
                reg._close_agent_model(type("M", (), {"aclose": staticmethod(lambda: None)})())
            except Exception as err:  # noqa: BLE001
                error.append(err)

        thread = threading.Thread(target=_worker)
        thread.start()
        thread.join()
        assert not error, "无 running loop 时应优雅降级而不是抛异常"
