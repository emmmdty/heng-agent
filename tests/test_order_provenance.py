# -*- coding: utf-8 -*-
"""下单参数的出处校验（写路径，纯函数 + 会话隔离）。

判据一句话：**下单的每一个商品，都必须在本会话的工具返回里出现过。**

这是金额出处校验（八期）在写路径上的同一条缝，而且后果更重：
回复里的数字错了买家看得出来，订单错了**库存已经扣了**。

现有四道防护都挡不住它：
    仓储查找        挡编造的 id，挡不住真实存在但买家没看过的 SKU
    Sequencing 断言  挡"完全没检索就下单"，挡不住"检索了 A、下单下成 B"
    幂等键          挡同一句话重复提交，挡不住换个说法再下一单
    权限白名单       挡工具层越权，确认卡内容与下单参数是否一致没人校验

同品牌变体互串在本仓是**已知**的失败形态（`audit_cases.py` 整个脚本就是为它写的）。
"""
import json

from app.application.harness.order_provenance import OrderProvenanceTracker


def _search_result(*products) -> str:
    """一份形如 product_search_tool 返回的 JSON（含 skus 与 filtered_out）。"""
    return json.dumps({
        "hits": [
            {
                "product_id": pid,
                "title": f"商品{pid}",
                "skus": [{"sku_id": f"{pid}-S1", "spec": "标准", "price_major": 99.0}],
            }
            for pid in products
        ],
        "recall_strategy": "hybrid_rerank",
        "filtered_out": [],
    }, ensure_ascii=False)


def _items(*pairs) -> list[dict]:
    return [{"product_id": pid, "sku_id": sku, "quantity": 1} for pid, sku in pairs]


class TestProductProvenance:
    def test_ordering_a_searched_product_passes(self):
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", _search_result("P1008"))
        outcome = tracker.check("s1", _items(("P1008", "P1008-S1")))
        assert not outcome.rejected and not outcome.warnings

    def test_ordering_an_unsearched_product_is_rejected(self):
        """"检索了 A、下单下成 B"——仓储查得到，所以现有四道防护一道都不响。"""
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", _search_result("P1008"))
        outcome = tracker.check("s1", _items(("P1002", "P1002-S1")))
        assert outcome.rejected
        assert "P1002" in outcome.reject_reason

    def test_reject_reason_tells_the_model_what_to_do(self):
        """错误信息要能让模型自纠（十期教训）：光说"不允许"它只会重试同一个动作。"""
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", _search_result("P1008"))
        outcome = tracker.check("s1", _items(("P1002", "P1002-S1")))
        assert "product_search" in outcome.reject_reason

    def test_no_observation_degrades_to_warning(self):
        """会话可能从 AgentState 快照恢复，内存里的观测记录为空。

        此时硬拒会误杀合法下单——沿用 Sequencing 断言那条「有证据才硬拒」的纪律。
        """
        tracker = OrderProvenanceTracker()
        outcome = tracker.check("s-restored", _items(("P1002", "P1002-S1")))
        assert not outcome.rejected
        assert outcome.warnings

    def test_sessions_are_isolated(self):
        """并发多会话不能互相当出处：A 会话检索过的商品，B 会话下单时不算数。"""
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", _search_result("P1008"))
        tracker.record_result("s2", _search_result("P1002"))
        assert tracker.check("s2", _items(("P1008", "P1008-S1"))).rejected

    def test_reset_clears_one_session_only(self):
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", _search_result("P1008"))
        tracker.record_result("s2", _search_result("P1008"))
        tracker.reset("s1")
        assert tracker.check("s2", _items(("P1008", "P1008-S1"))).rejected is False


class TestScopeIsDeliberatelyNarrow:
    """范围收窄的方向一律取"宁可漏报不误报"，与金额出处校验同一条纪律。"""

    def test_sku_without_provenance_never_blocks_the_order(self):
        """sku_id 只警告不硬拒，且**没见过任何 sku 时连警告都不发**。

        `filtered_out` 与 quote/optimize 两个工具的返回里都没有 sku_id
        （实测：filtered_out 只有 product_id/title/category/price/reason）。
        按 sku 硬拒会把"算了就买那个超预算的"这种合法流程当场拒掉；
        而在一个 sku 都没见过的情况下发提醒同样不对——那时我们没有依据。
        商品级替换才是要抓的东西。
        """
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", json.dumps({
            "filtered_out": [{"product_id": "P1016", "reason": "over_price_cap"}],
        }, ensure_ascii=False))
        outcome = tracker.check("s1", _items(("P1016", "P1016-S1")))
        assert not outcome.rejected
        assert not outcome.warnings

    def test_filtered_out_counts_as_provenance(self):
        """被硬约束挡掉的候选**也算出处**：它确实被工具返回、被模型看到过。

        买家完全可能说"算了，超预算那个我也要"。出处校验管的是
        "这个商品是不是模型自己想出来的"，不是"它满不满足约束"。
        """
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", json.dumps({
            "hits": [], "filtered_out": [{"product_id": "P1043", "reason": "over_price_cap"}],
        }, ensure_ascii=False))
        assert not tracker.check("s1", _items(("P1043", "P1043-S1"))).rejected

    def test_error_text_is_not_a_source(self):
        """工具报错文本不算出处：`[error] 商品不存在：P9999` 里的 P9999
        恰恰是**不该**被当成合法候选的那个。"""
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", _search_result("P1008"))
        tracker.record_result("s1", "[error] 商品不存在：P9999")
        assert tracker.check("s1", _items(("P9999", "P9999-S1"))).rejected

    def test_quantity_and_address_are_not_checked(self):
        """数量与地址来自买家原话，工具返回里没有它们的出处。

        硬要校验只能靠猜，而猜错的代价是拒掉合法订单。
        """
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", _search_result("P1008"))
        items = [{"product_id": "P1008", "sku_id": "P1008-S1", "quantity": 99}]
        assert not tracker.check("s1", items).rejected


class TestDuplicateOrderWarning:
    """重复下单只提醒，不拒绝。

    买家再买一份同样的东西是完全合法的行为，拒掉它属于替买家做决定。
    幂等键管的是"同一句话被提交两次"（传输层重复），与"买家真的想再买一单"
    是两回事，不能用同一把锤子。
    """

    def test_second_identical_order_warns_with_the_existing_id(self):
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", _search_result("P1008"))
        items = _items(("P1008", "P1008-S1"))
        assert not tracker.check("s1", items).warnings
        tracker.record_order("s1", items, "GBX-000001")

        outcome = tracker.check("s1", items)
        assert not outcome.rejected
        assert any("GBX-000001" in w for w in outcome.warnings)

    def test_different_quantity_is_not_a_duplicate(self):
        """数量不同说明买家在追加，不是重复提交——按精确匹配判，宁可漏报。"""
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", _search_result("P1008"))
        tracker.record_order("s1", _items(("P1008", "P1008-S1")), "GBX-000001")

        outcome = tracker.check(
            "s1", [{"product_id": "P1008", "sku_id": "P1008-S1", "quantity": 2}],
        )
        assert not outcome.warnings

    def test_item_order_does_not_matter(self):
        """同样两件商品换个排列顺序仍是同一单。"""
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", _search_result("P1008", "P1002"))
        tracker.record_order(
            "s1", _items(("P1008", "P1008-S1"), ("P1002", "P1002-S1")), "GBX-000001",
        )
        outcome = tracker.check(
            "s1", _items(("P1002", "P1002-S1"), ("P1008", "P1008-S1")),
        )
        assert any("GBX-000001" in w for w in outcome.warnings)

    def test_duplicate_across_sessions_is_not_flagged(self):
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", _search_result("P1008"))
        tracker.record_result("s2", _search_result("P1008"))
        tracker.record_order("s1", _items(("P1008", "P1008-S1")), "GBX-000001")
        assert not tracker.check("s2", _items(("P1008", "P1008-S1"))).warnings


class TestMalformedInput:
    def test_items_that_are_not_dicts_do_not_crash(self):
        """模型偶尔传出奇形怪状的入参。校验器不能自己抛异常——
        那会把一次参数错误升级成整轮失败，而工具本身有它自己的报错。"""
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", _search_result("P1008"))
        assert not tracker.check("s1", ["P1008"]).rejected  # type: ignore[list-item]
        assert not tracker.check("s1", []).rejected

    def test_non_json_result_is_ignored(self):
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", "这不是 JSON")
        tracker.record_result("s1", "")
        assert tracker.check("s1", _items(("P1008", "P1008-S1"))).warnings


class TestSkuWarningOnlyWhenWeHaveGrounds:
    """sku 警告只在**见过该商品的 sku** 时才提醒。

    组合报价与组合优化的返回里只有 product_id，没有 sku_id。
    "没见过任何 sku"与"见过别的 sku、唯独没见过这一个"是两回事：
    前者我们没有依据，提醒等于噪声——而模型对反复出现的无效提醒会学会忽略，
    连带把真正有依据的那次也一起忽略掉。
    """

    def test_no_warning_when_no_sku_was_ever_seen_for_that_product(self):
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", json.dumps({
            "selection": [{"product_id": "P1005", "title": "充电器"}],
        }, ensure_ascii=False))
        outcome = tracker.check("s1", _items(("P1005", "P1005-S1")))
        assert not outcome.rejected
        assert not outcome.warnings, "没见过这个商品的任何 sku，就没有依据提醒规格选错"

    def test_warns_when_other_skus_of_the_same_product_were_seen(self):
        """见过 P1008-S1、却下单 P1008-S9：这时提醒是有依据的。"""
        tracker = OrderProvenanceTracker()
        tracker.record_result("s1", _search_result("P1008"))
        outcome = tracker.check("s1", _items(("P1008", "P1008-S9")))
        assert not outcome.rejected
        assert any("P1008-S9" in w for w in outcome.warnings)
