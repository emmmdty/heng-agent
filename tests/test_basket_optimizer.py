# -*- coding: utf-8 -*-
"""预算感知的组合优化（领域层，纯函数，零外部依赖）。

要解决的问题：买家给了预算，Agent 只能一件件查、自己拼、自己减。
"预算还剩 $22""分开买贵 $3.65"这类数字**至今没有工具出处**，
金额出处校验里长期表现为 `suspected_difference`。

为什么这条能力值得做深：60 SPU 的候选空间小到可以**暴力枚举出真最优解**，
也就是说它的 ground truth 是确定性算出来的，不需要 LLM judge 打分。
这与金额出处校验、与给 judge 补规则表是同一条方法论主线——
**能拿回确定性判据的，就别留给 judge。**

目标函数（字典序，刻意选可判定的口径）：
    1. 覆盖的需求组数最多；
    2. 同覆盖数下**优先保住靠前的需求**（needs 按买家陈述顺序传入）；
    3. 再并列时组合**到手价最低**；
    4. 仍并列按 product_id 字典序取定（保证同输入同输出，可作回归基线）。
不按"评分/性价比"最优，是因为那要先定义一个价值模型，而价值模型无法确定性验证，
ground truth 立刻退回给 judge 打分。
"""
import pytest

from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.money import Money
from app.domain.shipping.basket_optimizer import (
    NeedCandidate,
    NeedGroup,
    optimize_basket,
)
from app.domain.shipping.tariff_schedule import TariffSchedule


@pytest.fixture()
def tariff() -> TariffSchedule:
    return TariffSchedule(rates=ExchangeRateTable())


def _candidate(product_id: str, major: float, category: str = "数码配件") -> NeedCandidate:
    return NeedCandidate(
        product_id=product_id,
        title=f"商品{product_id}",
        category=category,
        unit_price=Money.from_major_units(major, "CNY"),
    )


def _two_needs() -> list[NeedGroup]:
    """耳机三选一 + 箱子一选一。CN 口径：基础运费 25、免税额度 5000、数码配件 13%。"""
    return [
        NeedGroup(need="耳机", candidates=[_candidate("P2", 200), _candidate("P1", 300)]),
        NeedGroup(need="箱子", candidates=[_candidate("P3", 500, "旅行装备")]),
    ]


class TestObjective:
    def test_without_budget_picks_cheapest_full_coverage(self, tariff):
        """不传预算等价于"覆盖全部需求的最便宜组合"。"""
        plan = optimize_basket(tariff, _two_needs(), ship_to="CN", target_currency="CNY")
        assert [line.product_id for line in plan.quote.lines] == ["P2", "P3"]
        assert plan.all_needs_covered is True
        assert plan.uncovered == []
        # 小计 700 + 运费 25×1.6=40 + 关税 0（未超 5000 免税额度）
        assert plan.quote.to_dict()["landed_total_major"] == 740.0

    def test_budget_trims_coverage_and_reports_what_is_missing(self, tariff):
        """预算配不齐时不是报错，而是给出**能配的最优解 + 缺口的价签**。

        "缺的那件最低多少"必须一并回传：不给的话，Agent 想说
        "再加 300 就能把箱子配上"就只能自己减——正是要堵的那条缝。
        """
        plan = optimize_basket(
            tariff, _two_needs(), ship_to="CN", target_currency="CNY",
            budget=Money.from_major_units(600, "CNY"),
        )
        assert [line.product_id for line in plan.quote.lines] == ["P2"]
        assert plan.all_needs_covered is False
        assert [item.need for item in plan.uncovered] == ["箱子"]
        # 箱子单独买：500 + 25 = 525
        assert plan.uncovered[0].cheapest_landed_major == 525.0
        assert plan.uncovered[0].cheapest_product_id == "P3"
        assert plan.uncovered[0].reason == "over_budget"
        # "预算再加多少能配上"：并进组合后 700+40=740，减预算 600 → 140。
        # 注意不是单买价 525 减剩余预算（375）——组合运费只多付续件 60%，
        # 拿单买价算出来的缺口会偏大，而这正是模型自己算时会犯的错。
        assert plan.uncovered[0].to_dict()["additional_budget_needed_major"] == 140.0

    def test_budget_is_judged_on_landed_price_not_subtotal(self, tariff):
        """预算按**到手价**判定（含运费关税）。买家说"300 块预算"指的是最终付款额，
        按小计判会给出一个"看着够、结账超"的方案。"""
        groups = [NeedGroup(need="耳机", candidates=[_candidate("P1", 290)])]
        plan = optimize_basket(
            tariff, groups, ship_to="CN", target_currency="CNY",
            budget=Money.from_major_units(300, "CNY"),
        )
        # 290 + 25 = 315 > 300，配不上
        assert plan.all_needs_covered is False
        assert plan.uncovered[0].cheapest_landed_major == 315.0

    def test_empty_candidate_list_is_reported_as_no_candidates(self, tariff):
        """"库里没有"与"有但超预算"必须能被模型区分——同 filtered_out 的思路。"""
        groups = [
            NeedGroup(need="耳机", candidates=[_candidate("P2", 200)]),
            NeedGroup(need="帐篷", candidates=[]),
        ]
        plan = optimize_basket(tariff, groups, ship_to="CN", target_currency="CNY")
        assert plan.uncovered[0].reason == "no_candidates"
        assert plan.uncovered[0].cheapest_landed_major is None
        assert plan.uncovered[0].to_dict()["additional_budget_needed_major"] is None

    def test_earlier_need_wins_when_budget_covers_only_one(self, tariff):
        """同覆盖数下**先保住靠前的需求**，不是取最便宜的那件。

        这条是写评测用例时实算出来的：预算 250 美元、需求「降噪耳机 219 + 充电器 22.39」，
        两件合计 256.04 超预算。原规则（同覆盖数取最便宜）会配一个 31.54 的充电器、
        剩 218 美元——买家的主要需求被丢掉，答案荒唐。
        needs 按买家陈述顺序传入，顺序就是优先级，工具不做优先级猜测。
        """
        groups = [
            NeedGroup(need="降噪耳机", candidates=[_candidate("P1", 1554.9)]),
            NeedGroup(need="充电器", candidates=[_candidate("P2", 159)]),
        ]
        plan = optimize_basket(
            tariff, groups, ship_to="CN", target_currency="CNY",
            budget=Money.from_major_units(1600, "CNY"),
        )
        assert [line.product_id for line in plan.quote.lines] == ["P1"]
        assert [item.need for item in plan.uncovered] == ["充电器"]

    def test_ties_break_on_product_id_for_reproducibility(self, tariff):
        """同价并列时按 product_id 取定：优化器是评测基线，同输入必须同输出。"""
        groups = [NeedGroup(need="耳机", candidates=[_candidate("P9", 200), _candidate("P1", 200)])]
        first = optimize_basket(tariff, groups, ship_to="CN", target_currency="CNY")
        second = optimize_basket(tariff, groups, ship_to="CN", target_currency="CNY")
        assert first.quote.lines[0].product_id == "P1"
        assert first.to_dict() == second.to_dict()


class TestFreightSemantics:
    """组合运费**必须**走一次履约口径，这是本工具最容易踩的坑。

    `_to_card` 里 `combine_hint` 防的就是"把单品到手价相加"，
    新工具内部同样会踩：逐单品各算一次运费再相加等于假设分开发货，会显著高估。
    """

    def test_freight_is_charged_once_for_the_whole_basket(self, tariff):
        plan = optimize_basket(tariff, _two_needs(), ship_to="CN", target_currency="CNY")
        payload = plan.quote.to_dict()
        assert payload["freight_major"] == 40.0          # 25 × (1 + 0.6)
        assert payload["freight_major"] != 50.0          # 不是 25 + 25

    def test_separate_purchase_comparison_has_its_own_number(self, tariff):
        """"分开买 vs 一起买省多少"两个数都要有出处。

        判据一直在报这条：两个到手价都有出处、**差额没有**，
        于是差额被归成 suspected_difference。
        """
        plan = optimize_basket(tariff, _two_needs(), ship_to="CN", target_currency="CNY")
        payload = plan.to_dict()
        assert payload["separate_purchase_landed_major"] == 750.0   # 225 + 525
        assert payload["landed_total_major"] == 740.0
        assert payload["combining_saving_major"] == 10.0

    def test_quantity_counts_into_the_one_shipment_freight(self, tariff):
        groups = [NeedGroup(need="耳机", candidates=[_candidate("P2", 200)], quantity=3)]
        plan = optimize_basket(tariff, groups, ship_to="CN", target_currency="CNY")
        payload = plan.quote.to_dict()
        assert payload["total_quantity"] == 3
        assert payload["freight_major"] == 55.0   # 25 × (1 + 0.6 × 2)


class TestBudgetArithmeticIsSourced:
    def test_remaining_is_returned_not_left_to_the_model(self, tariff):
        """"预算还剩 $22"这个数至今没有工具出处（判据里是 suspected_difference）。"""
        plan = optimize_basket(
            tariff, _two_needs(), ship_to="CN", target_currency="CNY",
            budget=Money.from_major_units(1000, "CNY"),
        )
        payload = plan.to_dict()
        assert payload["budget_major"] == 1000.0
        assert payload["remaining_major"] == 260.0    # 1000 - 740
        assert payload["all_needs_covered"] is True

    def test_no_budget_reports_none_rather_than_zero(self, tariff):
        """不传预算时 remaining 必须是 null 而不是 0——
        回 0 会被模型读成"预算刚好花光"，凭空造出一个买家没说过的约束。"""
        payload = optimize_basket(
            tariff, _two_needs(), ship_to="CN", target_currency="CNY",
        ).to_dict()
        assert payload["budget_major"] is None
        assert payload["remaining_major"] is None


class TestGuardsAndErrors:
    def test_unsupported_destination_names_the_supported_ones(self, tariff):
        """十期教训：错误信息要能让模型自纠，否则它只会把错话转述给买家。"""
        with pytest.raises(ValueError, match="EU"):
            optimize_basket(tariff, _two_needs(), ship_to="DE", target_currency="CNY")

    def test_empty_groups_rejected(self, tariff):
        with pytest.raises(ValueError):
            optimize_basket(tariff, [], ship_to="CN", target_currency="CNY")

    def test_non_positive_quantity_rejected(self, tariff):
        groups = [NeedGroup(need="耳机", candidates=[_candidate("P2", 200)], quantity=0)]
        with pytest.raises(ValueError):
            optimize_basket(tariff, groups, ship_to="CN", target_currency="CNY")

    def test_search_space_is_capped_with_an_actionable_message(self, tariff):
        """枚举是暴力的，规模必须有闸。报错要说清怎么改（收窄候选），
        不能只说"太大了"——同 supported 列表那条。"""
        groups = [
            NeedGroup(need=f"需求{i}", candidates=[_candidate(f"P{i}{j}", 100 + j) for j in range(12)])
            for i in range(5)
        ]
        with pytest.raises(ValueError, match="候选"):
            optimize_basket(tariff, groups, ship_to="CN", target_currency="CNY")

    def test_reports_how_many_combinations_were_considered(self, tariff):
        """枚举数进 payload：它是"这确实是全局最优"的证据，
        也是规模失控时最先能看见的信号。"""
        plan = optimize_basket(tariff, _two_needs(), ship_to="CN", target_currency="CNY")
        # 耳机 3 种选择（两候选 + 不选） × 箱子 2 种 = 6
        assert plan.to_dict()["considered_combinations"] == 6


class TestOptimalityAgainstBruteForce:
    """优化器自己就是暴力枚举，所以这条对照的是**独立写的一遍**：
    随机造 20 组输入，逐一与"另写一遍的枚举"比对到手价。

    为什么值得写：优化器是评测的 ground truth 提供者，它错了，
    以它为基准的用例会一起错，而且错得看不出来。
    """

    def test_matches_an_independent_enumeration(self, tariff):
        import itertools
        import random

        rng = random.Random(20260903)
        for _ in range(20):
            groups = [
                NeedGroup(
                    need=f"n{g}",
                    candidates=[
                        _candidate(f"P{g}{c}", rng.randrange(50, 900))
                        for c in range(rng.randrange(1, 4))
                    ],
                )
                for g in range(rng.randrange(1, 4))
            ]
            budget = Money.from_major_units(rng.randrange(200, 2000), "CNY")
            plan = optimize_basket(
                tariff, groups, ship_to="CN", target_currency="CNY", budget=budget,
            )

            best = None
            options = [[None, *group.candidates] for group in groups]
            for combo in itertools.product(*options):
                picked = [
                    (i, g, c) for i, (g, c) in enumerate(zip(groups, combo)) if c is not None
                ]
                chosen = [(g, c) for _, g, c in picked]
                if not chosen:
                    landed, covered = 0, 0
                else:
                    from app.domain.shipping.tariff_schedule import BasketLine

                    quote = tariff.quote_basket(
                        [
                            BasketLine(c.product_id, c.title, c.category, c.unit_price, g.quantity)
                            for g, c in chosen
                        ],
                        ship_to="CN", target_currency="CNY",
                    )
                    landed = quote.landed_total().amount_in_minor_units
                    covered = len(chosen)
                if landed > budget.amount_in_minor_units:
                    continue
                key = (
                    -covered,
                    tuple(i for i, _, _ in picked),
                    landed,
                    tuple(c.product_id for _, c in chosen),
                )
                if best is None or key < best[0]:
                    best = (key, landed, covered)

            payload = plan.to_dict()
            assert payload["covered_need_count"] == best[2]
            assert payload["landed_total_major"] == pytest.approx(best[1] / 100, abs=0.01)
