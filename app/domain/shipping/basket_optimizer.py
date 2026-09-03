# -*- coding: utf-8 -*-
"""basket_optimizer —— 预算感知的组合优化

买家给了预算、报了几个需求，"在预算内怎么配"这件事此前完全由模型自己拼：
一件件查、自己加、自己减。代价在金额出处校验里长期可见——"预算还剩 $22"
"分开买贵 $3.65"这类数字没有任何工具出处，被归成 `suspected_difference`。

本模块把这件事收进领域层，口径与 `TariffSchedule.quote_basket()` 完全一致
（运费按一次履约计、免税额度按整批判定），并把过程中的每一个数都回传出去。

**目标函数（字典序）**：
    1. 覆盖的需求组数最多；
    2. 同覆盖数下组合**到手价最低**；
    3. 仍并列时按 product_id 字典序取定。

不按"评分/性价比"最优是刻意的：那要先定义一个价值模型，而价值模型无法确定性验证，
ground truth 立刻退回给 LLM judge 打分。60 SPU 的候选空间小到可以暴力枚举出真最优解，
**能拿回确定性判据的，就别留给 judge**。

预算按**到手价**判定（含运费与关税），不是按商品小计：买家说"300 块预算"
指的是最终付款额，按小计判会给出"看着够、结账超"的方案。

需求之间**等权**：配不齐时取覆盖数最多、其次最便宜的那组，不做优先级推断
（"哪个需求更重要"没有可靠判据，猜错等于替买家做主）。缺口连同它的价签一起回传，
让 Agent 有据可依地说"再加 X 就能把 Y 配上"。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from app.domain.catalog.money import Money
from app.domain.shipping.tariff_schedule import BasketLine, BasketQuote, TariffSchedule

# 枚举规模的闸。60 SPU 下典型规模是 3 组 × 8 候选 = 729 种，毫秒级；
# 闸设在这里是防止调用方把整个检索结果一股脑塞进来。
MAX_GROUPS = 4
MAX_CANDIDATES_PER_GROUP = 12
MAX_COMBINATIONS = 20_000

REASON_OVER_BUDGET = "over_budget"
REASON_NO_CANDIDATES = "no_candidates"


@dataclass(frozen=True)
class NeedCandidate:
    """一个需求下的一个候选商品（价格取主 SKU，与商品卡口径一致）。"""

    product_id: str
    title: str
    category: str
    unit_price: Money


@dataclass(frozen=True)
class NeedGroup:
    """一个需求槽位：最多选一件（`quantity` 是选中后买几件）。

    为什么是分组而不是平铺的候选列表：买家说"耳机 + 登机箱，预算 300"是两个槽位，
    平铺列表表达不了"这两个不该重复买"，枚举出来的最优解会是两副耳机。
    """

    need: str
    candidates: list[NeedCandidate]
    quantity: int = 1


@dataclass(frozen=True)
class UncoveredNeed:
    need: str
    reason: str
    cheapest_product_id: str | None = None
    cheapest_landed: Money | None = None

    @property
    def cheapest_landed_major(self) -> float | None:
        return round(self.cheapest_landed.to_major_units(), 2) if self.cheapest_landed else None

    def to_dict(self) -> dict:
        return {
            "need": self.need,
            "reason": self.reason,
            "cheapest_product_id": self.cheapest_product_id,
            # 缺口的价签必须回传：不给的话 Agent 想说"再加 300 就能配上"
            # 就只能自己减，正是这个工具要堵的那条缝
            "cheapest_landed_major": self.cheapest_landed_major,
        }


@dataclass(frozen=True)
class SelectedLine:
    need: str
    candidate: NeedCandidate
    quantity: int


@dataclass(frozen=True)
class BasketPlan:
    ship_to: str
    currency: str
    selection: list[SelectedLine]
    quote: BasketQuote | None
    uncovered: list[UncoveredNeed]
    budget: Money | None
    separate_purchase_landed: Money
    considered_combinations: int

    @property
    def all_needs_covered(self) -> bool:
        return not self.uncovered

    @property
    def landed_total(self) -> Money:
        return self.quote.landed_total() if self.quote else Money.of(0, self.currency)

    @property
    def remaining(self) -> Money | None:
        """预算余额。不传预算时是 None——回 0 会被模型读成"刚好花光"，
        凭空造出一个买家没说过的约束。"""
        if self.budget is None:
            return None
        return Money.of(
            max(0, self.budget.amount_in_minor_units - self.landed_total.amount_in_minor_units),
            self.currency,
        )

    @property
    def combining_saving(self) -> Money:
        """分开买 − 一起买。这个差额此前没有出处（判据里是 suspected_difference）。"""
        return Money.of(
            max(
                0,
                self.separate_purchase_landed.amount_in_minor_units
                - self.landed_total.amount_in_minor_units,
            ),
            self.currency,
        )

    def to_dict(self) -> dict:
        remaining = self.remaining
        return {
            "ship_to": self.ship_to,
            "currency": self.currency,
            "budget_major": (
                round(self.budget.to_major_units(), 2) if self.budget is not None else None
            ),
            "remaining_major": (
                round(remaining.to_major_units(), 2) if remaining is not None else None
            ),
            "landed_total_major": round(self.landed_total.to_major_units(), 2),
            "covered_need_count": len(self.selection),
            "all_needs_covered": self.all_needs_covered,
            "selection": [
                {
                    "need": line.need,
                    "product_id": line.candidate.product_id,
                    "title": line.candidate.title,
                    "quantity": line.quantity,
                    "unit_price_major": line.candidate.unit_price.to_major_units(),
                    "unit_price_currency": line.candidate.unit_price.currency,
                }
                for line in self.selection
            ],
            # 完整报价原样带上：小计/运费/关税/免税额度/应税基数都在里面，
            # 少带一个字段模型就得自己算一个（八期与十一期反复验证过的事）
            "quote": self.quote.to_dict() if self.quote else None,
            "uncovered_needs": [item.to_dict() for item in self.uncovered],
            "separate_purchase_landed_major": round(
                self.separate_purchase_landed.to_major_units(), 2,
            ),
            "combining_saving_major": round(self.combining_saving.to_major_units(), 2),
            # 枚举数是"这确实是全局最优"的证据，也是规模失控时最先看得见的信号
            "considered_combinations": self.considered_combinations,
        }


def _line_of(group: NeedGroup, candidate: NeedCandidate) -> BasketLine:
    return BasketLine(
        product_id=candidate.product_id,
        title=candidate.title,
        category=candidate.category,
        unit_price=candidate.unit_price,
        quantity=group.quantity,
    )


def _validate(tariff: TariffSchedule, groups: list[NeedGroup], ship_to: str) -> None:
    """入口校验。

    目的国规则表支持性放在最前面（十期教训）：顺序反了的话，传 DE 会先撞上
    "某商品不可寄往 DE"，模型据此告诉买家"这些商品不发欧盟"，
    而真相是规则表根本没有 DE——**报的是一句模型无法识破的错话**。
    """
    if ship_to not in tariff.supported_destinations():
        raise ValueError(
            f"计价规则表不支持目的国 {ship_to}（支持 {tariff.supported_destinations()}）。"
            f"若买家说的是欧盟/日本/新加坡等，请改用括号里的代码重试。",
        )
    if not groups:
        raise ValueError("needs 不能为空：至少要有一个需求及其候选商品")
    if len(groups) > MAX_GROUPS:
        raise ValueError(
            f"需求组最多 {MAX_GROUPS} 个（收到 {len(groups)}）：请分批调用，"
            f"或先收窄每个需求的候选商品后重试",
        )
    for group in groups:
        if group.quantity <= 0:
            raise ValueError(f"需求「{group.need}」的 quantity 必须为正整数")
        if len(group.candidates) > MAX_CANDIDATES_PER_GROUP:
            raise ValueError(
                f"需求「{group.need}」的候选商品最多 {MAX_CANDIDATES_PER_GROUP} 个"
                f"（收到 {len(group.candidates)}）：请先按检索排名取前几个候选再调用",
            )

    space = 1
    for group in groups:
        space *= len(group.candidates) + 1
    if space > MAX_COMBINATIONS:
        raise ValueError(
            f"候选组合数 {space} 超过上限 {MAX_COMBINATIONS}：请收窄每个需求的候选商品后重试",
        )


def optimize_basket(
    tariff: TariffSchedule,
    groups: list[NeedGroup],
    ship_to: str,
    target_currency: str,
    budget: Money | None = None,
) -> BasketPlan:
    """在预算内枚举出最优组合。

    暴力枚举，不做启发式剪枝：候选空间已经被入口闸限制在 20000 以内，
    而启发式会让"这是全局最优"这句话失去证据——**本工具的价值正建立在
    结果可被独立复算上**（单测里另写了一遍枚举做对照）。
    """
    _validate(tariff, groups, ship_to)

    budget_target = tariff.rates.convert(budget, target_currency) if budget else None

    def unit_minor(candidate: NeedCandidate) -> int:
        return tariff.rates.convert(candidate.unit_price, target_currency).amount_in_minor_units

    # 候选按（价格, product_id）定序：并列时的取舍必须可复现，优化器是评测基线
    options: list[list[NeedCandidate | None]] = []
    for group in groups:
        ordered = sorted(group.candidates, key=lambda c: (unit_minor(c), c.product_id))
        options.append([None, *ordered])

    def quote_of(chosen: list[tuple[NeedGroup, NeedCandidate]]) -> BasketQuote | None:
        if not chosen:
            return None
        return tariff.quote_basket(
            [_line_of(group, candidate) for group, candidate in chosen],
            ship_to=ship_to,
            target_currency=target_currency,
        )

    best_key: tuple | None = None
    best: tuple[list[tuple[NeedGroup, NeedCandidate]], BasketQuote | None] = ([], None)
    considered = 0
    for combo in itertools.product(*options):
        considered += 1
        chosen = [
            (group, candidate)
            for group, candidate in zip(groups, combo)
            if candidate is not None
        ]
        quote = quote_of(chosen)
        landed_minor = quote.landed_total().amount_in_minor_units if quote else 0
        if budget_target is not None and landed_minor > budget_target.amount_in_minor_units:
            continue
        key = (-len(chosen), landed_minor, tuple(c.product_id for _, c in chosen))
        if best_key is None or key < best_key:
            best_key, best = key, (chosen, quote)

    chosen, quote = best
    chosen_needs = {group.need for group, _ in chosen}

    uncovered: list[UncoveredNeed] = []
    for group in groups:
        if group.need in chosen_needs:
            continue
        if not group.candidates:
            uncovered.append(UncoveredNeed(need=group.need, reason=REASON_NO_CANDIDATES))
            continue
        # 缺口的价签：该需求**单独买**时最便宜的到手价。
        # "库里没有"与"有但超预算"必须能被模型区分——同 filtered_out 的思路。
        cheapest: tuple[int, NeedCandidate] | None = None
        for candidate in group.candidates:
            landed = tariff.quote_basket(
                [_line_of(group, candidate)], ship_to=ship_to, target_currency=target_currency,
            ).landed_total()
            if cheapest is None or landed.amount_in_minor_units < cheapest[0]:
                cheapest = (landed.amount_in_minor_units, candidate)
        uncovered.append(
            UncoveredNeed(
                need=group.need,
                reason=REASON_OVER_BUDGET,
                cheapest_product_id=cheapest[1].product_id,
                cheapest_landed=Money.of(cheapest[0], target_currency),
            ),
        )

    # 分开买的对照：每件各自一次履约。差额（"一起买省多少"）此前没有出处。
    separate_minor = 0
    for group, candidate in chosen:
        separate_minor += tariff.quote_basket(
            [_line_of(group, candidate)], ship_to=ship_to, target_currency=target_currency,
        ).landed_total().amount_in_minor_units

    return BasketPlan(
        ship_to=ship_to,
        currency=target_currency,
        selection=[
            SelectedLine(need=group.need, candidate=candidate, quantity=group.quantity)
            for group, candidate in chosen
        ],
        quote=quote,
        uncovered=uncovered,
        budget=budget_target,
        separate_purchase_landed=Money.of(separate_minor, target_currency),
        considered_combinations=considered,
    )
