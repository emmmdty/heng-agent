# -*- coding: utf-8 -*-
"""optimize_basket_tool 工具层单测（真实种子商品库，零外部依赖）。

领域层的最优性由 `test_basket_optimizer.py` 保证，这里只钉工具层的三件事：
**入参宽容度、错误信息可自纠、事件发的就是喂给模型的那一份**。
最后一条是八期的教训：`product_search_tool` 曾只发 hits、把 filtered_out 漏在外面，
模型看得到、轨迹看不到，事后审计与金额出处校验一起失真。
"""
import json

import pytest

from app.application.tools.optimize_basket_tool import build_optimize_basket_tool
from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.shipping.tariff_schedule import TariffSchedule
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository


@pytest.fixture()
def bus() -> TradeEventBus:
    return TradeEventBus()


@pytest.fixture()
def tool(bus):
    return build_optimize_basket_tool(
        InMemoryProductRepository(), TariffSchedule(rates=ExchangeRateTable()), bus,
    )


async def _call(tool, **kwargs):
    token = ShoppingContext.set(ShoppingContextSnapshot(
        shopping_session_id="s-opt", buyer_id="b1", locale="zh-CN", currency="CNY",
    ))
    try:
        return await tool(**kwargs)
    finally:
        ShoppingContext.reset(token)


class TestHappyPath:
    async def test_returns_plan_with_budget_arithmetic(self, tool):
        """P1008 露营灯 89 + P1005 充电器 159，寄 CN：
        小计 248 + 运费 25×1.6=40 + 关税 0 = 288；预算 300 还剩 12。

        这三个数（288 / 12 / 分开买省 10）此前全部由模型自己算，
        判据里长期表现为 suspected_sum / suspected_difference。
        """
        chunk = await _call(
            tool,
            needs=[
                {"need": "露营灯", "product_ids": ["P1008"]},
                {"need": "充电器", "product_ids": ["P1005"]},
            ],
            ship_to="CN",
            budget_major=300,
        )
        payload = json.loads(chunk.content[0].text)
        assert payload["landed_total_major"] == 288.0
        assert payload["remaining_major"] == 12.0
        assert payload["all_needs_covered"] is True
        assert payload["separate_purchase_landed_major"] == 298.0   # 114 + 184
        assert payload["combining_saving_major"] == 10.0
        assert [item["product_id"] for item in payload["selection"]] == ["P1008", "P1005"]

    async def test_zero_budget_means_no_constraint(self, tool):
        """`budget_major=0` 是"不设预算"而不是"预算为零"。

        默认值必须是一个模型不会误解的口径：回 remaining=0 会被读成
        "刚好花光"，凭空造出买家没说过的约束。
        """
        chunk = await _call(
            tool, needs=[{"need": "露营灯", "product_ids": ["P1008"]}], ship_to="CN",
        )
        payload = json.loads(chunk.content[0].text)
        assert payload["budget_major"] is None
        assert payload["remaining_major"] is None
        assert payload["all_needs_covered"] is True

    async def test_quantity_accepts_numeric_string(self, tool):
        """模型偶尔把数量当字符串传，与 product_search_tool / quote_basket_tool 同样宽松。"""
        chunk = await _call(
            tool,
            needs=[{"need": "露营灯", "product_ids": ["P1008"], "quantity": "2"}],
            ship_to="CN",
        )
        payload = json.loads(chunk.content[0].text)
        assert payload["quote"]["total_quantity"] == 2

    async def test_budget_shortfall_reports_the_price_tag_of_the_gap(self, tool):
        """配不齐时给"缺的那件最低多少"，Agent 才能有据地说"再加 X 就能配上"。"""
        chunk = await _call(
            tool,
            needs=[
                {"need": "露营灯", "product_ids": ["P1008"]},
                {"need": "行李箱", "product_ids": ["P1002"]},
            ],
            ship_to="CN",
            budget_major=200,
        )
        payload = json.loads(chunk.content[0].text)
        assert payload["all_needs_covered"] is False
        gap = payload["uncovered_needs"][0]
        assert gap["need"] == "行李箱"
        assert gap["reason"] == "over_budget"
        assert gap["cheapest_landed_major"] == 924.0   # 899 + 25


class TestConstraintsAreVisibleNotSilent:
    """被硬约束挡掉的候选要**回传**，不能默默丢掉。

    与 `filtered_out` 同一条：让模型能区分"库里没有"与"有但不满足约束"，
    否则它会把"这件不发美国"答成"没有这个商品"。
    """

    async def test_unshippable_candidate_is_reported_not_fatal(self, tool):
        """P1016 只发 CN。寄美国时它该被排除出候选，而不是让整次调用失败——
        优化器本来就是在候选里做选择，一个候选不可用不等于问题无解。
        """
        chunk = await _call(
            tool,
            needs=[
                {"need": "露营灯", "product_ids": ["P1008"]},
                {"need": "行李箱", "product_ids": ["P1016"]},
            ],
            ship_to="US",
        )
        payload = json.loads(chunk.content[0].text)
        assert payload["landed_total_major"] == 154.0   # 89 + 运费 65，未超免税额度
        excluded = payload["excluded_candidates"]
        assert excluded[0]["product_id"] == "P1016"
        assert excluded[0]["reason"] == "not_shippable"
        assert payload["uncovered_needs"][0]["need"] == "行李箱"

    async def test_all_candidates_excluded_still_returns_a_plan(self, tool):
        chunk = await _call(
            tool, needs=[{"need": "行李箱", "product_ids": ["P1016"]}], ship_to="US",
        )
        payload = json.loads(chunk.content[0].text)
        assert payload["covered_need_count"] == 0
        assert payload["uncovered_needs"][0]["reason"] == "no_candidates"


class TestErrorsCanBeSelfCorrected:
    async def test_unsupported_destination_names_the_supported_ones(self, tool):
        chunk = await _call(
            tool, needs=[{"need": "露营灯", "product_ids": ["P1008"]}], ship_to="DE",
        )
        text = chunk.content[0].text
        assert text.startswith("[error]")
        assert "EU" in text, "必须给出支持列表，模型才知道该改填什么"

    async def test_does_not_blame_the_product_for_a_rule_table_gap(self, tool):
        """十期教训：规则表没有 DE 却报成"这件商品不发 DE"，
        模型没有任何办法识破这句话，只能照着它给买家一个错误结论。"""
        chunk = await _call(
            tool, needs=[{"need": "行李箱", "product_ids": ["P1016"]}], ship_to="DE",
        )
        assert "不可寄往" not in chunk.content[0].text

    async def test_unknown_product_is_named(self, tool):
        chunk = await _call(
            tool, needs=[{"need": "露营灯", "product_ids": ["P9999"]}], ship_to="CN",
        )
        text = chunk.content[0].text
        assert text.startswith("[error]") and "P9999" in text

    async def test_empty_needs_rejected(self, tool):
        chunk = await _call(tool, needs=[], ship_to="CN")
        assert chunk.content[0].text.startswith("[error]")


class TestTraceFidelity:
    async def test_event_carries_the_same_payload_the_model_sees(self, tool, bus):
        """事件发的就是喂给模型的那一份。

        少发一部分的后果不是"少看点东西"：金额出处校验扫的是轨迹，
        模型看得到、轨迹看不到时，正常回复会被判成凭空编数字（八期实测）。
        """
        queue = bus.subscribe("s-opt")
        chunk = await _call(
            tool, needs=[{"need": "露营灯", "product_ids": ["P1008"]}], ship_to="CN",
            budget_major=300,
        )
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        kinds = [event.type for event in events]
        assert kinds == ["tool.invoke", "tool.result"]
        returned = json.loads(chunk.content[0].text)
        published = {k: v for k, v in events[1].payload.items() if k != "tool"}
        assert published == returned

    async def test_error_path_publishes_the_error(self, tool, bus):
        queue = bus.subscribe("s-opt")
        await _call(tool, needs=[{"need": "x", "product_ids": ["P9999"]}], ship_to="CN")
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        assert events[-1].payload["tool"] == "optimize_basket_tool"
        assert "P9999" in events[-1].payload["error"]
