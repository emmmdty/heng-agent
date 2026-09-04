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


class TestClosestExplanationWins:
    """成因推断必须报**最贴近**的那个算式，不能被一个擦边命中的候选抢走。

    要防的问题（十九期实测，data/conversations/eval-conflict-budget-spec-9c422d.jsonl）：
    买家说"预算 200 元"，模型列了一张"超出预算"的表，三个差额全部被解释错了——

        ¥99        报 "9 + 89"        真相是 299 − 200
        ¥1,141.90  报 "1341.9 - 199"  真相是 1341.9 − 200
        ¥1,354.90  报 "1554.9 - 199"  真相是 1554.9 − 200

    错的代价不是"少说了一句"，而是**把人指向错误的根因**：读报告的人会去查
    199 那款登山杖为什么进了算式，而真正该看的是"模型拿预算做减法"。
    一个错的解释比没有解释更坏（交接文档第七节经验 9）。

    两处成因叠在一起：
      1. 候选一命中就 return，没有在多个候选里挑误差最小的；
        "1341.9 − 199" 差 1.0，靠 ±1 的绝对容差擦边命中，先被找到就赢了。
      2. 预算 200 只出现在买家原话里，而减数池只取金额字段——精确解
        "1341.9 − 200" 压根不在候选集内，光挑最近的也救不回来。
    修法不能靠调容差：调大只会制造更多错解释，调小会连"$22 是 250 − 228.15
    的展示取整"这类真解释一起砍掉。
    """

    @staticmethod
    def _sources_of_the_real_case():
        """照抄那一轮的 tool.result 与买家原话（数值字段一个不改）。

        自造的均匀样本抓不到这个 bug——本仓踩过这个坑：擦边命中要的是
        199 与 200 这种"真实价目表里恰好差 1"的组合，构造样本时不会想到。
        `total_candidates: 9` 也照抄：那个荒唐的 "9 + 89" 就是从它来的。
        """
        tool_result = {
            "tool": "product_search_tool",
            "hit_count": 5,
            "hits": [
                {"product_id": "P1053", "price_major": 39.0, "score": 0.0143,
                 "skus": [{"price_major": 39.0, "stock": 220}]},
                {"product_id": "P1013", "price_major": 89.0, "score": 0.0001,
                 "skus": [{"price_major": 89.0, "stock": 90}]},
                {"product_id": "P1054", "price_major": 119.0, "score": 0.0001,
                 "skus": [{"price_major": 119.0, "stock": 85}]},
                {"product_id": "P1049", "price_major": 39.0, "score": 0.0001,
                 "skus": [{"price_major": 39.0, "stock": 300}]},
                {"product_id": "P1040", "price_major": 199.0, "score": 0.0,
                 "skus": [{"price_major": 199.0, "stock": 60}]},
            ],
            "total_candidates": 9,
            "filtered_out": [
                {"product_id": "P1004", "price_major": 1554.9, "reason": "over_price_cap"},
                {"product_id": "P1023", "price_major": 1341.9, "reason": "over_price_cap"},
                {"product_id": "P1022", "price_major": 299.0, "reason": "over_price_cap"},
            ],
        }
        return _sources([tool_result], ["预算 200 元，给我来一副顶配的主动降噪耳机。"])

    def test_exact_budget_difference_beats_the_edge_of_tolerance_candidate(self):
        """1341.9 − 200 精确相等，不能报成差 1.0 的 "1341.9 - 199"。"""
        report = check_reply("超出 ¥1,141.90", self._sources_of_the_real_case())
        finding = report.unsourced[0]
        assert finding.kind == KIND_DIFFERENCE
        assert finding.explain == "1341.9 - 200", "擦边候选把精确成因挤掉了"

    def test_the_second_overage_in_the_same_reply_is_also_explained_by_the_budget(self):
        """同一条回复里的第二个差额：1554.9 − 200，不是 "1554.9 - 199"。"""
        report = check_reply("超出 ¥1,354.90", self._sources_of_the_real_case())
        assert report.unsourced[0].explain == "1554.9 - 200"

    def test_an_exact_difference_beats_a_near_sum(self):
        """挑最近的要跨类挑：¥99 的真相是差额 299 − 200，而不是差 1.0 的和 "9 + 89"。

        先搜和、命中就 return 的写法，会让一个凑巧的和永远赢过精确的差。
        """
        report = check_reply("超出 ¥99", self._sources_of_the_real_case())
        finding = report.unsourced[0]
        assert (finding.kind, finding.explain) == (KIND_DIFFERENCE, "299 - 200")

    def test_buyer_stated_budget_can_be_the_subtrahend(self):
        """"到手价 − 预算"这个方向必须成立。

        原来的减数池只取工具的金额字段，只覆盖了"预算 − 到手价 = 还剩多少"
        一个方向；买家自述的预算做减数时（超预算多少），精确解不在候选集里。
        """
        sources = _sources([{"landed_total_major": 1341.9}], ["我预算就 200 元"])
        report = check_reply("超预算 ¥1,141.90", sources)
        assert report.unsourced[0].explain == "1341.9 - 200"

    def test_display_rounding_is_still_explained_when_nothing_closer_exists(self):
        """容差不许被顺手调小：没有更近的候选时，展示取整仍要给出解释。

        真实读数（eval-landed-price-us-045682）：250 − 228.15 = 21.85 被写成 $22。
        """
        sources = _sources([{"landed_total_major": 228.15}], ["预算 250 美元"])
        report = check_reply("预算还剩 $22", sources)
        assert report.unsourced[0].kind == KIND_DIFFERENCE

    def test_the_documented_sum_case_was_itself_being_explained_wrong(self):
        """本模块文档里的招牌例子（¥364 + ¥154 = ¥518，运费重复计了一次），
        在真实会话里报出来的却是 "129 + 388"——差 1.0 的擦边命中。

        实测（eval-compare-two-c9143e）：同一轮里 388 是组合小计（299 + 89）、
        129 是另一款商品的价，两个都在金额字段上，凑出 517 就把精确的 154 + 364 挤掉了。
        单测之所以一直是绿的，是因为它自造的池子里只有 364 和 154——
        擦边候选要真实价目表才凑得出来，这正是"自造样本太均匀"的坑。
        """
        tool_result = {"hits": [
            {"landed_price": {"landed_total_major": 364.0}},
            {"landed_price": {"landed_total_major": 154.0}},
            {"price_major": 129.0},
            {"combined": {"subtotal_major": 388.0}},
        ]}
        report = check_reply("各买一件分别下单一共 ¥518。", _sources([tool_result]))
        assert report.unsourced[0].explain == "154 + 364"

    def test_a_stock_count_does_not_beat_the_budget_on_a_tie(self):
        """精确解不止一个时，取操作数出身更硬的那个。

        实测（eval-landed-price-us-281ec8）：$21.85 有两个精确算式——
        "250 − 228.15"（买家预算 − 到手价，真相）和 "220 − 198.15"，
        而 220 是某个 SKU 的 **库存数**，只是被当成数字扫进了出处池。
        报后者等于让人去查一个根本不存在的 220 元商品。
        """
        tool_result = {
            "landed_total_major": 228.15,
            "hits": [{"price_major": 198.15, "skus": [{"price_major": 39.0, "stock": 220}]}],
        }
        report = check_reply("到手 $228.15，预算还剩 $21.85。", _sources([tool_result], ["预算 250 美元"]))
        assert report.unsourced[0].explain == "250 - 228.15"
