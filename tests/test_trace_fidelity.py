# -*- coding: utf-8 -*-
"""事件轨迹保真单测

不变式：**tool.result 事件里的内容，必须与同一次调用喂给模型的返回一致。**

为什么单立一个文件锁它：落盘的事件轨迹是事后审计、bad case 回放与
金额出处校验（app/application/harness/number_provenance.py）共同的数据底座。
轨迹缺一块，三样东西一起失真，而且症状离故障点很远——

    实测 eval-tool-error-honesty 那一轮：product_search_tool 命中 0 条、
    filtered_out 里有三个被目的国挡掉的商品，模型如实照抄了它们的价格，
    但事件只发了 hits，于是流水里看起来像"模型凭空报了三个价"。
    同理 category_insight_tool 只发 hit_count，知识片段里的价格区间
    在轨迹里全无出处。两处都不是模型的问题，是轨迹漏发。

漏发在外观上和"这次真的没有"完全一样，没有任何告警——与设计演进记录里
「BM25 索引忘了接线」是同一类失败，所以同样需要一条测试把它钉住。
"""
import json

import pytest

from app.application.tools.product_search_tool import build_product_search_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository


async def _invoke(tool, **kwargs):
    token = ShoppingContext.set(
        ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
    )
    try:
        return await tool(**kwargs)
    finally:
        ShoppingContext.reset(token)


def _drain(queue) -> list[dict]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return [e.payload for e in events if e.type == "tool.result"]


@pytest.fixture()
def search_tool():
    bus = TradeEventBus()
    queue = bus.subscribe("s1")
    tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)
    return tool, queue


class TestProductSearchTraceFidelity:
    async def test_filtered_out_reaches_the_event_trail(self, search_tool):
        """被硬约束挡掉的候选也要进事件轨迹——模型看得到，审计就必须看得到。"""
        tool, queue = search_tool
        response = await _invoke(tool, normalized_query="露营灯", price_max_major=100)

        returned = json.loads(response.content[0].text)
        assert returned.get("filtered_out"), "夹具应能造出被价格上限挡掉的候选"
        (event,) = _drain(queue)
        assert event["filtered_out"] == returned["filtered_out"]

    async def test_event_carries_same_keys_as_model_sees(self, search_tool):
        tool, queue = search_tool
        response = await _invoke(tool, normalized_query="旅行三件套", ship_to="US")

        returned = json.loads(response.content[0].text)
        (event,) = _drain(queue)
        missing = set(returned) - set(event)
        assert not missing, f"事件轨迹漏发了模型能看到的字段：{sorted(missing)}"


    async def test_attribute_mismatch_reaches_the_event_trail(self, search_tool):
        """属性冲突声明也要进事件轨迹。

        这是经验 2 的同一条：`filtered_out` 与 `insights` 当年都是"后来加进返回值、
        忘了同步事件"，症状是审计把有出处的东西判成无出处。
        `attribute_mismatch` 藏在 hits[] 里，顶层 key 比对看不见它，
        所以要单独钉一条——否则"卡片上有、轨迹里没有"不会有任何告警。
        """
        tool, queue = search_tool
        response = await _invoke(tool, normalized_query="主动降噪 耳机", top_k=8)

        returned = json.loads(response.content[0].text)
        flagged = [hit for hit in returned["hits"] if "attribute_mismatch" in hit]
        assert flagged, "夹具应召回显式声明不具备主动降噪的候选，否则这条测不到东西"
        (event,) = _drain(queue)
        assert event["hits"] == returned["hits"], "事件里的商品卡必须与模型看到的逐字一致"


class TestPerSkuLandedPrice:
    async def test_every_sku_carries_its_own_landed_price(self, search_tool):
        """非主 SKU 的到手价必须由工具给出。

        实测 bad case（eval-compare-two / eval-landed-price-us 各两轮）：
        商品卡只给主 SKU 的 landed_price，买家问"月光白多少钱"时模型只能
        自己拿 229 USD × 汇率 + 运费算成 $238.15——数字碰巧对，但它是模型
        算的，一旦跨了免税额度或换了品类费率就会错。
        """
        tool, _ = search_tool
        response = await _invoke(tool, normalized_query="降噪耳机", ship_to="US", target_currency="USD")

        hits = json.loads(response.content[0].text)["hits"]
        multi_sku = [hit for hit in hits if len(hit["skus"]) > 1]
        assert multi_sku, "夹具应含多 SKU 商品，否则这条测试测不到东西"
        for hit in multi_sku:
            for sku in hit["skus"]:
                assert "landed_price" in sku, f"{sku['sku_id']} 缺到手价出处"
                assert "landed_total_major" in sku["landed_price"]

    async def test_sku_landed_price_differs_by_sku_price(self, search_tool):
        tool, _ = search_tool
        response = await _invoke(tool, normalized_query="降噪耳机", ship_to="US", target_currency="USD")

        hits = json.loads(response.content[0].text)["hits"]
        multi_sku = next(hit for hit in hits if len(hit["skus"]) > 1)
        totals = {sku["sku_id"]: sku["landed_price"]["landed_total_major"] for sku in multi_sku["skus"]}
        prices = {sku["sku_id"]: sku["price_major"] for sku in multi_sku["skus"]}
        cheaper, dearer = sorted(prices, key=prices.get)[0], sorted(prices, key=prices.get)[-1]
        assert totals[cheaper] < totals[dearer], "到手价应随 SKU 单价变化，不能是同一个数"

    async def test_no_landed_price_without_ship_to(self, search_tool):
        """不传 ship_to 时不该凭空出现到手价字段。"""
        tool, _ = search_tool
        response = await _invoke(tool, normalized_query="降噪耳机")

        for hit in json.loads(response.content[0].text)["hits"]:
            for sku in hit["skus"]:
                assert "landed_price" not in sku


class TestCrossCurrencyPrice:
    """跨币种展示价必须由工具给出。

    真实 bad case（干净整轮的 long-context-memory，出处校验抓到的）：
    商品原生价 149 USD，买家口径是 CNY，回复写成"$149 USD（约 ¥1080）"——
    正确是 149 × 7.1 = ¥1057.9，模型自己折汇率折错了 2%。

    根因是同一个缝的另一处：**没传 ship_to 时商品卡不含任何目标币种金额**
    （到手价才带折算），模型想给买家看人民币就只能自己乘汇率。
    到手价那条缝已经补上（逐 SKU landed_price），这条也得补。
    """

    async def test_native_currency_price_is_converted_for_the_buyer(self, search_tool):
        tool, _ = search_tool
        response = await _invoke(tool, normalized_query="登山杖", target_currency="CNY")

        hits = json.loads(response.content[0].text)["hits"]
        foreign = [hit for hit in hits if hit["currency"] != "CNY"]
        assert foreign, "夹具应含非人民币计价商品，否则这条测试测不到东西"
        for hit in foreign:
            assert "price_in_target_major" in hit, f"{hit['product_id']} 缺目标币种价格"
            assert hit["target_currency"] == "CNY"

    async def test_no_conversion_field_when_currency_already_matches(self, search_tool):
        """同币种时不该多发一个一模一样的数字。"""
        tool, _ = search_tool
        response = await _invoke(tool, normalized_query="露营灯", target_currency="CNY")

        for hit in json.loads(response.content[0].text)["hits"]:
            if hit["currency"] == "CNY":
                assert "price_in_target_major" not in hit

    async def test_each_sku_carries_its_own_converted_price(self):
        """多 SKU 外币商品：非主 SKU 的折算价同样不能让模型自己乘汇率。"""
        bus = TradeEventBus()
        bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)
        response = await _invoke(tool, normalized_query="降噪耳机", target_currency="CNY")

        hits = json.loads(response.content[0].text)["hits"]
        foreign_multi = [h for h in hits if h["currency"] != "CNY" and len(h["skus"]) > 1]
        assert foreign_multi, "夹具应含多 SKU 外币商品"
        for hit in foreign_multi:
            for sku in hit["skus"]:
                assert "price_in_target_major" in sku, f"{sku['sku_id']} 缺折算价"
            converted = {s["sku_id"]: s["price_in_target_major"] for s in foreign_multi[0]["skus"]}
            assert len(set(converted.values())) > 1, "不同单价的 SKU 折算价不能是同一个数"
