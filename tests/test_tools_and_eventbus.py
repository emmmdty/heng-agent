# -*- coding: utf-8 -*-
"""工具层与事件总线单测：工具直调（绕过 LLM）+ EventBus 订阅。"""
import asyncio
import json

import pytest

from app.application.tools.order_tools import build_create_order_tool
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.usecases.order_usecases import PlaceOrderUseCase
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.in_memory_repositories import (
    InMemoryOrderRepository,
    InMemoryProductRepository,
)

ADDRESS = {
    "recipient_name": "张三",
    "country": "CN",
    "state": "浙江",
    "city": "杭州",
    "address_line": "西湖区某路 1 号",
    "postal_code": "310000",
    "phone": "13800000000",
}


class TestTradeEventBus:
    async def test_publish_routes_to_subscriber(self):
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        other = bus.subscribe("s2")
        bus.publish("s1", "final.result", {"text": "done"})

        event = await asyncio.wait_for(queue.get(), timeout=1)
        assert event.type == "final.result"
        assert other.empty(), "事件不能串台到其他会话"

    def test_reject_unknown_event_type(self):
        bus = TradeEventBus()
        with pytest.raises(ValueError, match="未知事件类型"):
            bus.publish("s1", "not.a.type", {})


class TestToolsDirectInvoke:
    async def test_product_search_tool(self):
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)

        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            response = await tool(normalized_query="旅行三件套 抗造")
        finally:
            ShoppingContext.reset(token)

        payload = json.loads(response.content[0].text)
        assert payload["hits"][0]["product_id"] == "P1001"
        # tool.invoke + tool.result 两条事件
        assert queue.qsize() == 2

    async def test_product_search_tool_accepts_numeric_string(self):
        """回归：模型（如 qwen3-max）会把数字参数传成字符串，工具必须接住并强转，
        而不是在 schema 校验层被拒收（实测 price_max_major="300" 曾导致检索全程失败）。"""
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)

        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            response = await tool(normalized_query="旅行三件套 抗造", price_max_major="300", top_k="3")
        finally:
            ShoppingContext.reset(token)

        payload = json.loads(response.content[0].text)
        # 字符串被强转为数字后进检索链路，预算硬约束生效：候选主价均不超过 300
        assert payload["hits"], "传字符串价格上限不应导致检索为空"
        for hit in payload["hits"]:
            assert hit["price_major"] <= 300
        # tool.invoke + tool.result 两条事件
        assert queue.qsize() == 2

    async def test_product_search_tool_rejects_bad_numeric_string(self):
        """非法数字字符串应返回 [error] 而不是抛异常。"""
        bus = TradeEventBus()
        bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)

        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            response = await tool(normalized_query="旅行三件套", price_max_major="不是数字")
        finally:
            ShoppingContext.reset(token)

        assert response.content[0].text.startswith("[error] price_max_major 非法")

    async def test_create_order_tool_and_error_path(self):
        bus = TradeEventBus()
        product_repo = InMemoryProductRepository()
        tool = build_create_order_tool(PlaceOrderUseCase(product_repo, InMemoryOrderRepository()), bus)

        # 买家身份由 ShoppingContext 注入，而非模型入参
        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            ok = await tool(
                items=[{"product_id": "P1001", "sku_id": "P1001-S1", "quantity": 1}],
                shipping_address=ADDRESS,
            )
            snapshot = json.loads(ok.content[0].text)
            assert snapshot["status"] == "CONFIRMED"
            assert snapshot["order_id"].startswith("GBX-")
            assert snapshot["buyer_id"] == "b1"

            bad = await tool(
                items=[{"product_id": "P9999", "sku_id": "X", "quantity": 1}],
                shipping_address=ADDRESS,
            )
            assert bad.content[0].text.startswith("[error]")
        finally:
            ShoppingContext.reset(token)


class TestQuoteBasketDestinationErrors:
    """组合报价的目的国错误必须能让模型自纠。

    九期评测挖出的同一条缝（另一头在 catalog_search）：模型把买家说的"欧盟"
    翻成 DE/FR，工具报的是"P1002（TrailOx 20寸登机行李箱）不可寄往 DE"——
    这句话把锅甩给了商品，模型据此得出"这些箱子不发欧盟"，
    而真相是**规则表根本没有 DE 这个目的国**。

    校验顺序决定了错误信息的对错：规则表支持性要先于商品可达性判断，
    否则那句带支持列表的错误（`TariffSchedule.quote_basket` 里本来就有）
    永远被挡在后面，模型看不到它。
    """

    def _tool(self):
        from app.application.tools.quote_basket_tool import build_quote_basket_tool
        from app.domain.catalog.exchange_rate import ExchangeRateTable
        from app.domain.shipping.tariff_schedule import TariffSchedule

        return build_quote_basket_tool(
            InMemoryProductRepository(),
            TariffSchedule(rates=ExchangeRateTable()),
            TradeEventBus(),
        )

    async def _call(self, tool, **kwargs):
        token = ShoppingContext.set(ShoppingContextSnapshot(
            shopping_session_id="s-quote", buyer_id="b1", locale="zh-CN", currency="CNY",
        ))
        try:
            return await tool(**kwargs)
        finally:
            ShoppingContext.reset(token)

    async def test_unsupported_destination_names_the_supported_ones(self):
        chunk = await self._call(
            self._tool(),
            items=[{"product_id": "P1002", "quantity": 1}],
            ship_to="DE",
        )
        text = chunk.content[0].text
        assert "DE" in text
        assert "EU" in text, "必须给出支持列表，模型才知道该改填什么"

    async def test_does_not_blame_the_product(self):
        """最要命的是把"规则表没有 DE"说成"这件商品不发 DE"——
        模型没有任何办法识破这句话，只能照着它给买家一个错误结论。"""
        chunk = await self._call(
            self._tool(),
            items=[{"product_id": "P1002", "quantity": 1}],
            ship_to="DE",
        )
        assert "不可寄往" not in chunk.content[0].text

    async def test_genuinely_unshippable_product_still_says_so(self):
        """回归：目的国规则表支持、而商品确实不发那儿时，照旧点名商品。

        GlideCase（P1016）的 ships_to 只有 CN，寄美国属于这种情况。
        """
        chunk = await self._call(
            self._tool(),
            items=[{"product_id": "P1016", "quantity": 1}],
            ship_to="US",
        )
        text = chunk.content[0].text
        assert "P1016" in text and "不可寄往" in text

    async def test_supported_destination_still_quotes(self):
        chunk = await self._call(
            self._tool(),
            items=[{"product_id": "P1002", "quantity": 1}],
            ship_to="EU",
        )
        payload = json.loads(chunk.content[0].text)
        assert payload["landed_total_major"] == 974.0


class TestToolDocsMatchTheRuleTable:
    """工具描述里写死的目的国枚举，必须和规则表一致。

    AgentScope 从 docstring 生成 JSON schema，取值只能写死在文字里、
    没法从 `_TARIFF_RATES` 动态生成——那就必然有脱钩的一天：
    规则表加了新目的国，工具描述还是老五个，模型永远不会去用新的。

    这类"忘了同步"的故障外观和"故意不支持"完全一样，没有任何告警
    （七期 BM25 忘接线是同一类）。所以用一条测试把两边钉在一起。
    """

    def _supported(self):
        from app.domain.catalog.exchange_rate import ExchangeRateTable
        from app.domain.shipping.tariff_schedule import TariffSchedule

        return TariffSchedule(rates=ExchangeRateTable()).supported_destinations()

    def test_product_search_tool_lists_every_supported_destination(self):
        import inspect

        from app.application.tools import product_search_tool as module

        source = inspect.getsource(module)
        for code in self._supported():
            assert f'"{code}"' in source, f"工具描述里没提到规则表支持的 {code}"

    def test_quote_basket_tool_lists_every_supported_destination(self):
        import inspect

        from app.application.tools import quote_basket_tool as module

        source = inspect.getsource(module)
        for code in self._supported():
            assert f'"{code}"' in source, f"工具描述里没提到规则表支持的 {code}"
