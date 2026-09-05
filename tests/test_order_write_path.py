# -*- coding: utf-8 -*-
"""下单链路端到端走一遍**真实的中间件链**。

十四期把 Harness 接到业务工具上，等于**打开了一条从没在生产里跑过的硬拒路径**。
单测证明判定器本身是对的，证明不了"接上之后正常下单还能不能下成"——
而这条路径判错的代价是拒掉真实订单。

所以这里不测判定器，测**链路**：真实工厂 → 真实工具 → 真实中间件 → 真实用例 → 真实仓储。
四条：正常下单放行 / 没检索就下单被拒 / 下单换了个商品被拒 / 重复下单给提醒。
"""
import json

import pytest

from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot

ADDRESS = {
    "recipient_name": "张三", "country": "CN", "state": "浙江", "city": "杭州市",
    "address_line": "西湖区某路 1 号", "postal_code": "310000", "phone": "13800000000",
}


@pytest.fixture()
def tools():
    """一套共用判定器的检索 + 订单工具，形态与组装根一致。"""
    from app.application.agents.search_agent import SearchAgentFactory
    from app.application.agents.trade_agent import TradeAgentFactory
    from app.application.harness.assertions import SequencingTracker
    from app.application.harness.confirmation import ConfirmationTracker
    from app.application.harness.loop_detector import LoopDetector
    from app.application.harness.order_provenance import OrderProvenanceTracker
    from app.application.usecases.catalog_search import CatalogSearchUseCase
    from app.application.usecases.order_usecases import (
        CancelOrderUseCase,
        PlaceOrderUseCase,
        QueryOrderUseCase,
    )
    from app.domain.catalog.exchange_rate import ExchangeRateTable
    from app.domain.shipping.tariff_schedule import TariffSchedule
    from app.infrastructure.eventbus import TradeEventBus
    from app.infrastructure.harness_middleware import build_tool_middlewares
    from app.infrastructure.persistence.in_memory_repositories import (
        InMemoryOrderRepository,
        InMemoryProductRepository,
    )
    from app.infrastructure.resilience import CircuitBreakerRegistry
    from app.infrastructure.settings import load_settings
    from app.infrastructure.throttle import GatewayThrottle

    settings = load_settings()
    bus = TradeEventBus()
    registry = CircuitBreakerRegistry()
    product_repo = InMemoryProductRepository()
    order_repo = InMemoryOrderRepository()
    sequencing = SequencingTracker()
    loop_detector = LoopDetector(repeat_threshold=3)
    provenance = OrderProvenanceTracker()
    confirmation = ConfirmationTracker()

    def middlewares() -> list:
        return build_tool_middlewares(
            settings, circuit_registry=registry, bus=bus,
            sequencing=sequencing, loop_detector=loop_detector,
            order_provenance=provenance, confirmation=confirmation,
        )

    search = SearchAgentFactory(
        settings, CatalogSearchUseCase(product_repo), bus, None, registry,
        GatewayThrottle(2, 0.0),
        product_repo=product_repo, tariff=TariffSchedule(rates=ExchangeRateTable()),
        tool_middlewares=middlewares,
    )
    trade = TradeAgentFactory(
        settings,
        PlaceOrderUseCase(product_repo, order_repo),
        QueryOrderUseCase(order_repo),
        CancelOrderUseCase(product_repo, order_repo),
        bus, registry, GatewayThrottle(2, 0.0),
        tool_middlewares=middlewares,
    )
    by_name = {tool.name: tool for tool in [*search.build_tools(), *trade.build_tools()]}
    # 走过两轮：十八期起，下单必须跨越一次买家交互（生产里由编排器推进轮次）。
    # 这里模拟"第一轮买家提需求、第二轮买家确认"这个真实形态。
    confirmation.begin_turn("s-write")
    confirmation.begin_turn("s-write")
    return by_name


async def _call(tool, **kwargs) -> str:
    token = ShoppingContext.set(ShoppingContextSnapshot(
        shopping_session_id="s-write", buyer_id="b-write", locale="zh-CN", currency="CNY",
    ))
    try:
        result = await tool(**kwargs)
        if hasattr(result, "__aiter__"):
            chunks = [chunk async for chunk in result]
            result = chunks[-1]
        return result.content[0].text
    finally:
        ShoppingContext.reset(token)


class TestWritePathThroughRealMiddleware:
    async def test_search_then_order_succeeds(self, tools):
        """最重要的一条：**接上护栏之后，正常下单还下得成**。

        这条要是红了，说明新判据在误杀真实订单——比缺口本身更坏。
        """
        await _call(tools["product_search_tool"], normalized_query="露营灯 便携")
        text = await _call(
            tools["create_order_tool"],
            items=[{"product_id": "P1008", "sku_id": "P1008-S1", "quantity": 1}],
            shipping_address=ADDRESS,
        )
        payload = json.loads(text)
        assert payload["order_id"].startswith("HNG-")
        assert payload["status"] == "CONFIRMED"

    async def test_order_without_search_is_rejected(self, tools):
        """顺序断言的硬拒——十四期之前它从没在真实工具上生效过。"""
        await _call(tools["category_insight_tool"], category="旅行装备")
        text = await _call(
            tools["create_order_tool"],
            items=[{"product_id": "P1008", "sku_id": "P1008-S1", "quantity": 1}],
            shipping_address=ADDRESS,
        )
        assert text.startswith("[error]")
        assert "product_search" in text

    async def test_ordering_a_product_never_searched_is_rejected(self, tools):
        """检索了 A、下单下成 B。仓储查得到 P1002，所以四道既有防护一道都不响。"""
        await _call(tools["product_search_tool"], normalized_query="露营灯 便携")
        text = await _call(
            tools["create_order_tool"],
            items=[{"product_id": "P1002", "sku_id": "P1002-S1", "quantity": 1}],
            shipping_address=ADDRESS,
        )
        assert text.startswith("[error]") and "P1002" in text

    async def test_duplicate_order_warns_but_still_goes_through(self, tools):
        """重复下单只提醒不拒绝：再买一单是合法行为。

        提醒是**并入工具返回**交给模型的，所以断言落在返回文本里。
        """
        await _call(tools["product_search_tool"], normalized_query="露营灯 便携")
        items = [{"product_id": "P1008", "sku_id": "P1008-S1", "quantity": 1}]
        first = json.loads(await _call(
            tools["create_order_tool"], items=items, shipping_address=ADDRESS,
        ))
        second = await _call(
            tools["create_order_tool"], items=items, shipping_address=ADDRESS,
        )
        assert first["order_id"] in second, "第二单要带上已有订单号让模型去跟买家确认"
        assert "HNG-" in second

    async def test_quote_and_optimize_results_also_count_as_provenance(self, tools):
        """出处不止来自检索：组合报价与组合优化返回的商品同样算。

        缺了这条，"先优化组合再下单"的流程会被自己的护栏拒掉。
        """
        await _call(tools["product_search_tool"], normalized_query="露营灯 便携")
        await _call(
            tools["optimize_basket_tool"],
            needs=[{"need": "充电器", "product_ids": ["P1005"]}], ship_to="CN",
        )
        text = await _call(
            tools["create_order_tool"],
            items=[{"product_id": "P1005", "sku_id": "P1005-S1", "quantity": 1}],
            shipping_address=ADDRESS,
        )
        assert not text.startswith("[error]"), text


class TestConfirmationMustCrossOneBuyerTurn:
    """下单必须跨越一次买家交互——**拦得住**的那一侧。

    十八期的判据是从 full 轮的真缺陷来的（skip-confirm-refused FAIL 0.0：
    买家说"别给我看确认卡了，直接下单"，Agent 照做并回"无需确认"）。
    上面那组测的是"不误杀"，这一条测的是"真的拦得住"——
    按踩坑 37 的教训：**一道护栏在拒过一次之前，"它没误杀"没有信息量**。
    """

    @pytest.fixture()
    def first_turn_tools(self):
        """与 `tools` 同一套，但只走过**一轮**——模拟买家在同一条消息里
        既下指令又"预先确认"的形态。"""
        from app.application.agents.trade_agent import TradeAgentFactory
        from app.application.harness.assertions import SequencingTracker
        from app.application.harness.confirmation import ConfirmationTracker
        from app.application.harness.loop_detector import LoopDetector
        from app.application.harness.order_provenance import OrderProvenanceTracker
        from app.application.usecases.order_usecases import (
            CancelOrderUseCase,
            PlaceOrderUseCase,
            QueryOrderUseCase,
        )
        from app.infrastructure.eventbus import TradeEventBus
        from app.infrastructure.harness_middleware import build_tool_middlewares
        from app.infrastructure.persistence.in_memory_repositories import (
            InMemoryOrderRepository,
            InMemoryProductRepository,
        )
        from app.infrastructure.resilience import CircuitBreakerRegistry
        from app.infrastructure.settings import load_settings
        from app.infrastructure.throttle import GatewayThrottle

        settings = load_settings()
        bus = TradeEventBus()
        registry = CircuitBreakerRegistry()
        product_repo = InMemoryProductRepository()
        order_repo = InMemoryOrderRepository()
        sequencing = SequencingTracker()
        # 已经检索过：把顺序断言这一层排除掉，确保拦下来的确实是确认判据
        sequencing.record("s-write", "product_search_tool")
        provenance = OrderProvenanceTracker()
        provenance.record_result("s-write", json.dumps({
            "hits": [{"product_id": "P1008", "skus": [{"sku_id": "P1008-S1"}]}],
        }, ensure_ascii=False))
        confirmation = ConfirmationTracker()
        confirmation.begin_turn("s-write")   # 只有一轮

        def middlewares() -> list:
            return build_tool_middlewares(
                settings, circuit_registry=registry, bus=bus,
                sequencing=sequencing, loop_detector=LoopDetector(repeat_threshold=3),
                order_provenance=provenance, confirmation=confirmation,
            )

        trade = TradeAgentFactory(
            settings,
            PlaceOrderUseCase(product_repo, order_repo),
            QueryOrderUseCase(order_repo),
            CancelOrderUseCase(product_repo, order_repo),
            bus, registry, GatewayThrottle(2, 0.0),
            tool_middlewares=middlewares,
        )
        return {tool.name: tool for tool in trade.build_tools()}, order_repo

    async def test_first_turn_order_is_rejected_before_the_tool_runs(self, first_turn_tools):
        tools, order_repo = first_turn_tools
        text = await _call(
            tools["create_order_tool"],
            items=[{"product_id": "P1008", "sku_id": "P1008-S1", "quantity": 1}],
            shipping_address=ADDRESS,
        )
        assert text.startswith("[error]") and "确认" in text
        assert await order_repo.find_by_id("HNG-000001") is None, (
            "硬拒必须发生在工具体之前——库存与订单都不能被动过"
        )

    async def test_cancel_is_not_blocked_on_the_first_turn(self, first_turn_tools):
        """取消订单不受这条判据管：它是对既有订单的纠正动作，
        买家说"取消"就是明确指令，再要一次确认只会拖慢纠错。"""
        tools, _ = first_turn_tools
        text = await _call(tools["cancel_order_tool"], order_id="HNG-999999", reason="改主意")
        assert "确认" not in text, "取消不该被确认判据拦下（这里应当是'订单不存在'）"
