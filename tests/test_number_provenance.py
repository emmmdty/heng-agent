# -*- coding: utf-8 -*-
"""数字出处校验单测（纯函数层，不依赖模型与外部服务）。

被测判据只有一条：**回复里出现的每一个金额，都必须能在工具返回或买家原话里找到出处。**
测试用的样本全部取自 data/conversations 下的真实评测流水（见各用例注释），
不是构造出来的理想输入——这个校验器的价值取决于它在真实回复上的误报率。
"""
from app.application.harness.number_provenance import (
    KIND_DIFFERENCE,
    check_reply,
    collect_sources,
)


def _sources(tool_results=(), buyer_texts=()):
    return collect_sources(tool_results=tool_results, buyer_texts=buyer_texts)


class TestAmountExtraction:
    def test_extracts_prefixed_and_suffixed_amounts(self):
        report = check_reply("到手价 ¥1,619.9，运费 65 元。", _sources())
        assert [item.value for item in report.unsourced] == [1619.9, 65.0]

    def test_ignores_numbers_without_currency_marker(self):
        """续航 40 小时、库存 150 不是金额，不该进校验范围。"""
        report = check_reply("续航 40 小时，库存 150 件，降噪 -45dB。", _sources())
        assert report.total_amounts == 0

    def test_ignores_percentages(self):
        report = check_reply("关税税率 13%，实收 0 元。", _sources())
        assert [item.value for item in report.unsourced] == [0.0]


class TestProvenance:
    def test_amount_returned_by_tool_is_sourced(self):
        tool_result = {"tool": "product_search_tool", "hits": [{"price_major": 89.0}]}
        report = check_reply("LumenGo 露营灯 ¥89。", _sources([tool_result]))
        assert report.clean

    def test_amount_inside_tool_result_string_is_sourced(self):
        """category_insight_tool 的知识片段是纯文本，里面的价格区间同样是工具出处。"""
        tool_result = {"tool": "category_insight_tool", "insights": ["旅行三件套入门档 80-150 元"]}
        report = check_reply("入门档大概 150 元。", _sources([tool_result]))
        assert report.clean

    def test_amount_quoted_from_buyer_is_sourced(self):
        report = check_reply("你的预算 300 元完全够。", _sources(buyer_texts=["帮我找 300 块以内的露营灯"]))
        assert report.clean

    def test_display_rounding_still_counts_as_sourced(self):
        """工具给 1619.9，回复写 ¥1,620 属展示取整，不算无出处。"""
        tool_result = {"landed_total_major": 1619.9}
        report = check_reply("到手 ¥1,620。", _sources([tool_result]))
        assert report.clean

    def test_unrelated_amount_is_unsourced(self):
        tool_result = {"landed_total_major": 89.0}
        report = check_reply("到手 ¥129。", _sources([tool_result]))
        assert not report.clean
        assert report.unsourced[0].value == 129.0


class TestSuspicionKinds:
    def test_sum_of_two_tool_amounts_is_flagged_as_sum(self):
        """真实 bad case（eval-compare-two-6d0690）：¥364 + ¥154 = ¥518，运费被重复计了一次。"""
        tool_result = {"hits": [
            {"landed_price": {"landed_total_major": 364.0}},
            {"landed_price": {"landed_total_major": 154.0}},
        ]}
        report = check_reply("两个一起买到手 ¥518。", _sources([tool_result]))
        assert [item.kind for item in report.unsourced] == ["suspected_sum"]
        assert "364" in report.unsourced[0].explain

    def test_difference_of_two_tool_amounts_is_flagged_as_difference(self):
        """真实 bad case（eval-landed-price-us-045682）：预算 250 − 到手 228.15 ≈ 剩 $22。"""
        tool_result = {"landed_total_major": 228.15}
        report = check_reply("到手 $228.15，预算还剩 $22。", _sources([tool_result], ["预算 250 美元"]))
        assert [item.kind for item in report.unsourced] == ["suspected_difference"]

    def test_zero_is_not_used_as_an_addend(self):
        """关税 0 元遍地都是，把它算进组合等于给任何数字都配一个假解释。"""
        tool_result = {"landed_total_major": 1619.9, "tariff_major": 0.0}
        report = check_reply("到手 ¥1,625.9。", _sources([tool_result]))
        assert report.unsourced[0].kind == "unsourced", "0 + 1619.9 不是有意义的成因"

    def test_rate_fields_are_not_treated_as_money(self):
        """tariff_rate=0.075 是税率不是金额，拿它去凑加数会编出荒唐的成因。"""
        tool_result = {"landed_total_major": 198.15, "tariff_rate": 0.075}
        report = check_reply("到手不到 $200。", _sources([tool_result]))
        assert report.unsourced[0].kind == "unsourced"

    def test_explanation_must_match_closely(self):
        """成因容差只容展示取整（±1），不容"差了 6 块也算"。"""
        tool_result = {"landed_total_major": 1554.9, "freight_major": 65.0}
        report = check_reply("到手 ¥1,625.9。", _sources([tool_result]))
        assert report.unsourced[0].kind == "unsourced", "1554.9+65=1619.9 不是 1625.9 的成因"

    def test_plain_unsourced_amount_has_kind_unsourced(self):
        report = check_reply("到手 ¥777。", _sources([{"landed_total_major": 89.0}]))
        assert report.unsourced[0].kind == "unsourced"


class TestReportShape:
    def test_counts_cover_every_extracted_amount(self):
        tool_result = {"landed_total_major": 89.0}
        report = check_reply("原价 ¥89，另一款 ¥129。", _sources([tool_result]))
        assert report.total_amounts == 2
        assert len(report.unsourced) == 1

    def test_empty_reply_is_clean(self):
        assert check_reply("", _sources()).clean

    def test_error_reply_is_clean(self):
        assert check_reply("[error] 上游超时", _sources()).clean


class TestSessionSourceTracker:
    """出处按会话累积：模型第 3 轮引用第 1 轮检索到的价格是正常行为，不该判无出处。"""

    def test_sources_accumulate_across_turns(self):
        from app.application.harness.number_provenance import SessionSources

        tracker = SessionSources()
        tracker.observe("s1", tool_results=[{"price_major": 89.0}], buyer_texts=["300 块以内"])
        tracker.observe("s1", tool_results=[{"price_major": 129.0}], buyer_texts=["再看看毛巾"])

        report = check_reply("第一轮那款露营灯 ¥89，预算 300 元。", tracker.of("s1"))
        assert report.clean

    def test_sessions_do_not_leak_into_each_other(self):
        from app.application.harness.number_provenance import SessionSources

        tracker = SessionSources()
        tracker.observe("s1", tool_results=[{"price_major": 89.0}])
        assert not check_reply("¥89", tracker.of("s2")).clean

    def test_retained_numbers_are_capped(self):
        from app.application.harness.number_provenance import MAX_RETAINED_NUMBERS, SessionSources

        tracker = SessionSources()
        for value in range(MAX_RETAINED_NUMBERS + 500):
            tracker.observe("s1", tool_results=[{"price_major": float(value)}])
        assert len(tracker.of("s1").numbers) <= MAX_RETAINED_NUMBERS

    def test_reset_drops_session(self):
        from app.application.harness.number_provenance import SessionSources

        tracker = SessionSources()
        tracker.observe("s1", tool_results=[{"price_major": 89.0}])
        tracker.reset("s1")
        assert not check_reply("¥89", tracker.of("s1")).clean


class TestClassificationCost:
    """成因推断是组合搜索，池子大小按立方增长——必须有界，否则长会话每轮都卡。"""

    # 长会话攒下的金额池：每次检索 5 个商品卡各带十几个金额字段，几轮就上百
    _POOL = tuple(round(1.0 + index * 0.37, 2) for index in range(600))

    def test_unexplainable_amount_does_not_blow_up(self):
        """最坏情况：凑不出成因，2 元组搜完还要搜 3 元组（C(600,3) ≈ 3600 万）。"""
        import time

        sources = collect_sources([{"landed_total_major": list(self._POOL)}])
        started = time.perf_counter()
        report = check_reply("到手 ¥999999.99。", sources)
        elapsed = time.perf_counter() - started

        assert elapsed < 1.0, f"600 个金额的池子耗时 {elapsed:.2f}s，运行时每轮都要付这个代价"
        assert report.unsourced[0].kind == "unsourced"

    def test_pruning_still_finds_a_real_two_sum(self):
        sources = collect_sources([{"landed_total_major": list(self._POOL)}])
        target = self._POOL[-1] + self._POOL[-2]
        report = check_reply(f"一起买 ¥{target:.2f}。", sources)
        assert report.unsourced[0].kind == "suspected_sum", "剪枝不能把真正的两数之和剪掉"


class TestTaxableBaseCountsAsMoney:
    """应税基数字段要进金额池，否则由它派生的数字只会被归成 `unsourced`。

    十一期给 `to_dict()` 补了 `taxable_base_major`（超出免税额度、实际计征的部分）。
    出处判定看的是全部数字，所以有没有出处不受影响；但**成因推断**只在金额字段上做，
    字段名不匹配 `_MONEY_FIELD` 时，"3.72 是它自己从 153.72 − 150 减出来的"
    这条线索就丢了——而这类线索正是"判据指向工具该补什么"的依据。
    """

    def test_taxable_base_enters_the_money_pool(self):
        sources = collect_sources([{"taxable_base_major": 29.0, "tariff_rate": 0.12}])
        assert 29.0 in sources.money
        # 费率不是金额：名字里带 rate 的一律挡在池外，否则会编出
        # "0.12 + x" 这种荒唐成因（八期教训）
        assert 0.12 not in sources.money

    def test_difference_from_taxable_base_is_explained(self):
        sources = collect_sources([{"taxable_base_major": 100.0, "subtotal_major": 130.0}])
        report = check_reply("差额是 30 元", sources)
        assert report.unsourced and report.unsourced[0].kind == KIND_DIFFERENCE

    def test_budget_arithmetic_fields_enter_the_money_pool(self):
        """组合优化回的预算算术同样是钱。

        这三个字段（budget / remaining / saving）是十一期 optimize_basket_tool
        专门为"预算还剩多少""一起买省多少"补的出处；名字不匹配金额字段时，
        由它们派生的数会失去成因线索。
        """
        sources = collect_sources([{
            "budget_major": 300.0,
            "remaining_major": 12.0,
            "combining_saving_major": 10.0,
            "considered_combinations": 6,
        }])
        assert {300.0, 12.0, 10.0} <= set(sources.money)
        assert 6 not in sources.money, "枚举次数不是金额"
