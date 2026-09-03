# -*- coding: utf-8 -*-
"""TariffSchedule / ShippingQuote

跨境到手价的规则内核：国家×品类关税费率 + 基础运费 + 免税额度。
纯规则纯函数，可单测；生产替换为关税服务时只需换实现，ShippingQuote 结构不变。

到手价口径（目标币种统一折算后计算）：
    landed = 商品小计 + 运费 + 关税
    关税   = 商品小计超出免税额度部分 × 费率（不足免税额度则为 0）
"""
from __future__ import annotations

from dataclasses import dataclass

from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.money import Money

# 目的国 → 品类 → 关税费率（未列品类走 "*" 兜底）
_TARIFF_RATES: dict[str, dict[str, float]] = {
    "CN": {"数码配件": 0.13, "旅行装备": 0.09, "户外运动": 0.09, "家居生活": 0.09, "*": 0.09},
    "US": {"数码配件": 0.0, "旅行装备": 0.075, "户外运动": 0.075, "家居生活": 0.05, "*": 0.06},
    "EU": {"*": 0.12},
    "JP": {"*": 0.08},
    "SG": {"*": 0.07},
}

# 目的国免税额度的**原生口径**：(金额 major, 币种)。
#
# 为什么不存折算后的 CNY：额度本来就是各国用自己的货币定义的（US 是 800 USD、EU 是 150 EUR），
# 存 CNY 等于把汇率烙进规则表，而且**原生口径就此丢失**——买家语境是美国、目标币种是 CNY 时，
# Agent 想说"美国免税门槛"只能自己反折成 `$800`，这个数没有工具出处。
# 出处校验实测抓到过：它写的是"美国免税门槛 $800（约 ¥5680）"，5680 有出处、800 没有。
# 这已经是同一个 `$800` 第三次从新路径回来（八期堵"凭知识说"、十期堵"工具没被调到"、
# 这次是"调到了但口径不对"）。折算统一交给汇率表，规则表只管规则。
#
# 另：JP 原先写的是 `10_000 * 5`，手写的 0.05 与汇率表里的 JPY=0.048 不一致；
# 改为从汇率表推导后 JP 额度是 480 CNY（原 500），属修正，且消掉了一处重复定义的汇率。
_DE_MINIMIS_NATIVE: dict[str, tuple[float, str]] = {
    "CN": (5_000.0, "CNY"),   # 5000 元（个人物品行邮口径，简化）
    "US": (800.0, "USD"),
    "EU": (150.0, "EUR"),
    "JP": (10_000.0, "JPY"),  # 简化口径
    "SG": (400.0, "SGD"),
}

# 目的国基础运费（CNY 分，单件；多件按 60% 递增，简化的续重逻辑）
_BASE_FREIGHT_CNY_MINOR: dict[str, int] = {
    "CN": 25_00,
    "US": 65_00,
    "EU": 75_00,
    "JP": 45_00,
    "SG": 40_00,
}


def _major(money: Money | None) -> float | None:
    return round(money.to_major_units(), 2) if money is not None else None


def _de_minimis_fields(
    threshold: Money | None,
    threshold_native: Money | None,
    taxable_base: Money | None,
    taxable_base_native: Money | None,
) -> dict:
    """免税额度与应税基数的对外字段。

    单品报价与组合报价必须给出**同一组**字段，否则模型在"一件"和"两件"之间
    切换时会时有时无地失去出处——单品/组合两条路径各写一份是这类缺陷的温床
    （八期就是只修了单品那条，组合那条从另一头漏回来）。
    """
    return {
        "de_minimis_threshold_major": _major(threshold),
        "de_minimis_threshold_native_major": _major(threshold_native),
        "de_minimis_threshold_native_currency": (
            threshold_native.currency if threshold_native is not None else None
        ),
        "taxable_base_major": _major(taxable_base),
        "taxable_base_native_major": _major(taxable_base_native),
    }


@dataclass(frozen=True)
class ShippingQuote:
    ship_to: str
    subtotal: Money
    freight: Money
    tariff: Money
    tariff_rate: float
    de_minimis_applied: bool  # 是否命中免税额度（关税为 0）
    # 本次实际适用的免税额度（目标币种）。必须回给模型：
    # 只给 de_minimis_applied 的话，模型要向买家解释"多少以下免税"就只能凭自己的知识说，
    # 实测它说的是"美国 $800 以下免税"——数字碰巧对，但改了规则表它照样这么说，
    # 而且没有任何东西会报错（出处校验抓到的）。
    de_minimis_threshold: Money = None  # type: ignore[assignment]
    # 免税额度的原生口径（US 800 USD / EU 150 EUR）与应税基数，理由见 `_DE_MINIMIS_NATIVE`
    # 与 `taxable_base` 的注释：这三个数不给，模型要解释关税就只能自己减、自己反折。
    de_minimis_threshold_native: Money = None  # type: ignore[assignment]
    # 应税基数：超出免税额度、**实际参与计征**的那部分。
    # 十期实测缺陷：Agent 写"1,199 × 12% ≈ ¥3.48"——最终数字对（工具给的）、
    # 计税基数错（整单 1199 × 12% = 143.88，差 40 倍）。同一轮出处校验报的两处
    # 无出处金额都是 `€3.72`，正是它自己减出来的这个基数。判据指的地方就是工具该补的地方。
    taxable_base: Money = None  # type: ignore[assignment]
    taxable_base_native: Money = None  # type: ignore[assignment]

    def landed_total(self) -> Money:
        return self.subtotal.add(self.freight).add(self.tariff)

    def to_dict(self) -> dict:
        return {
            "ship_to": self.ship_to,
            "subtotal_major": self.subtotal.to_major_units(),
            "freight_major": self.freight.to_major_units(),
            "tariff_major": self.tariff.to_major_units(),
            "tariff_rate": self.tariff_rate,
            "de_minimis_applied": self.de_minimis_applied,
            **_de_minimis_fields(
                self.de_minimis_threshold, self.de_minimis_threshold_native,
                self.taxable_base, self.taxable_base_native,
            ),
            "landed_total_major": self.landed_total().to_major_units(),
            "currency": self.subtotal.currency,
        }


@dataclass(frozen=True)
class BasketLine:
    """组合报价的一行：一个 SKU 及其件数。"""

    product_id: str
    title: str
    category: str
    unit_price: Money
    quantity: int


@dataclass(frozen=True)
class BasketQuote:
    ship_to: str
    lines: list[BasketLine]
    total_quantity: int
    subtotal: Money
    freight: Money
    tariff: Money
    de_minimis_applied: bool
    de_minimis_threshold: Money = None  # type: ignore[assignment]
    de_minimis_threshold_native: Money = None  # type: ignore[assignment]
    taxable_base: Money = None  # type: ignore[assignment]
    taxable_base_native: Money = None  # type: ignore[assignment]

    def landed_total(self) -> Money:
        return self.subtotal.add(self.freight).add(self.tariff)

    def to_dict(self) -> dict:
        return {
            "ship_to": self.ship_to,
            "total_quantity": self.total_quantity,
            "lines": [
                {
                    "product_id": line.product_id,
                    "title": line.title,
                    "quantity": line.quantity,
                    "unit_price_major": line.unit_price.to_major_units(),
                    "unit_price_currency": line.unit_price.currency,
                }
                for line in self.lines
            ],
            "subtotal_major": self.subtotal.to_major_units(),
            "freight_major": self.freight.to_major_units(),
            "tariff_major": self.tariff.to_major_units(),
            "de_minimis_applied": self.de_minimis_applied,
            **_de_minimis_fields(
                self.de_minimis_threshold, self.de_minimis_threshold_native,
                self.taxable_base, self.taxable_base_native,
            ),
            "landed_total_major": self.landed_total().to_major_units(),
            "currency": self.subtotal.currency,
        }


@dataclass(frozen=True)
class TariffSchedule:
    rates: ExchangeRateTable

    def supported_destinations(self) -> list[str]:
        return sorted(_TARIFF_RATES.keys())

    def de_minimis_native(self, ship_to: str) -> Money:
        """免税额度的原生口径（US → 800 USD、EU → 150 EUR）。

        公开出来是为了让规则表以外的地方（judge 的事实基准）不必再 import 私有常量——
        十期的 `_landed_price_rules()` 就是那么写的，规则表一改结构那边就断。
        """
        if ship_to not in _DE_MINIMIS_NATIVE:
            raise ValueError(f"暂不支持的目的国：{ship_to}（支持 {self.supported_destinations()}）")
        major, currency = _DE_MINIMIS_NATIVE[ship_to]
        return Money.from_major_units(major, currency)

    def de_minimis(self, ship_to: str, target_currency: str) -> Money:
        """免税额度折算到指定币种。折算只从原生口径出发一次，避免二次折算累积误差。"""
        return self.rates.convert(self.de_minimis_native(ship_to), target_currency)

    def quote_basket(
        self,
        lines: list["BasketLine"],
        ship_to: str,
        target_currency: str,
    ) -> "BasketQuote":
        """多件组合的到手价。

        为什么需要它：单品到手价由 `quote()` 算好随商品卡返回，但买家问
        "这两个一起多少钱"时，总价没有任何工具提供，只能由模型自己相加——
        而模型做金额加法会出错（评测 compare-two 实测：$51.26 + $21.69 被算成
        $62.95，正确是 $72.95）。本仓的"计价收敛进工具、不让模型算数字"原则
        原先只覆盖了单品，这里把它补全到组合。

        组合不是各单品报价的简单相加，两处口径必须按**整批**算：
          - 运费按一次履约计：首件全价 + 每件续件 60%，`quantity` 取全篮总件数。
            若逐单品各算一次运费再相加，等于假设分开发货，会显著高估。
          - 免税额度按整批小计判定：海关对一个包裹整体计税，不是逐件计税。
        单行且数量一致时，本方法与 `quote()` 结果完全一致（有单测锁定）。
        """
        if not lines:
            raise ValueError("lines 不能为空")
        if ship_to not in _TARIFF_RATES:
            raise ValueError(f"暂不支持的目的国：{ship_to}（支持 {self.supported_destinations()}）")

        total_quantity = sum(line.quantity for line in lines)
        if total_quantity <= 0:
            raise ValueError("总件数必须为正整数")

        # 小计：各行 单价 × 件数，统一折算到目标币种
        subtotal_target = Money.of(0, target_currency)
        for line in lines:
            line_total = line.unit_price.multiply(line.quantity)
            subtotal_target = subtotal_target.add(self.rates.convert(line_total, target_currency))

        # 运费：整批一次履约
        base_freight_cny = Money.of(_BASE_FREIGHT_CNY_MINOR[ship_to], "CNY")
        freight_minor_cny = round(
            base_freight_cny.amount_in_minor_units * (1 + 0.6 * (total_quantity - 1)),
        )
        freight_target = self.rates.convert(Money.of(freight_minor_cny, "CNY"), target_currency)

        # 关税：免税额度按整批小计判定；超出部分按各行品类费率分摊计征
        rate_table = _TARIFF_RATES[ship_to]
        subtotal_cny_minor = sum(
            self.rates.convert(line.unit_price.multiply(line.quantity), "CNY").amount_in_minor_units
            for line in lines
        )
        de_minimis_cny = self.de_minimis(ship_to, "CNY")
        taxable_total_minor = max(0, subtotal_cny_minor - de_minimis_cny.amount_in_minor_units)
        de_minimis_applied = taxable_total_minor == 0

        tariff_cny_minor = 0
        if not de_minimis_applied and subtotal_cny_minor > 0:
            # 免税额度按各行金额占比分摊，再各按自己的品类费率计征——
            # 直接对整批用单一费率会在混合品类时算错。
            for line in lines:
                line_cny = self.rates.convert(
                    line.unit_price.multiply(line.quantity), "CNY",
                ).amount_in_minor_units
                share = line_cny / subtotal_cny_minor
                line_taxable = taxable_total_minor * share
                line_rate = rate_table.get(line.category, rate_table["*"])
                tariff_cny_minor += round(line_taxable * line_rate)
        tariff_target = self.rates.convert(Money.of(tariff_cny_minor, "CNY"), target_currency)

        return BasketQuote(
            ship_to=ship_to,
            lines=list(lines),
            total_quantity=total_quantity,
            subtotal=subtotal_target,
            freight=freight_target,
            tariff=tariff_target,
            de_minimis_applied=de_minimis_applied,
            **self._de_minimis_snapshot(ship_to, taxable_total_minor, target_currency),
        )

    def _de_minimis_snapshot(
        self, ship_to: str, taxable_cny_minor: int, target_currency: str,
    ) -> dict:
        """免税额度与应税基数的两种口径，单品与组合两条路径共用一份。

        共用是刻意的：这两条路径各写一份的话，"一件"有出处、"两件"没出处
        （八期就只修了单品那条）。原生口径同时给出，是因为买家语境的币种
        常常不等于 target_currency，Agent 想跨币种表述就只能自己反折。
        """
        native = self.de_minimis_native(ship_to)
        taxable_cny = Money.of(taxable_cny_minor, "CNY")
        return {
            "de_minimis_threshold": self.rates.convert(native, target_currency),
            "de_minimis_threshold_native": native,
            "taxable_base": self.rates.convert(taxable_cny, target_currency),
            "taxable_base_native": self.rates.convert(taxable_cny, native.currency),
        }

    def quote(self, subtotal: Money, category: str, ship_to: str, quantity: int, target_currency: str) -> ShippingQuote:
        """按目的国规则计算到手价三要素，全部折算为 target_currency。"""
        if ship_to not in _TARIFF_RATES:
            raise ValueError(f"暂不支持的目的国：{ship_to}（支持 {self.supported_destinations()}）")
        if quantity <= 0:
            raise ValueError("quantity 必须为正整数")

        subtotal_target = self.rates.convert(subtotal, target_currency)

        # 运费：首件全价 + 续件 60%
        base_freight_cny = Money.of(_BASE_FREIGHT_CNY_MINOR[ship_to], "CNY")
        freight_minor_cny = round(base_freight_cny.amount_in_minor_units * (1 + 0.6 * (quantity - 1)))
        freight_target = self.rates.convert(Money.of(freight_minor_cny, "CNY"), target_currency)

        # 关税：小计（CNY 口径）超出免税额度部分 × 费率
        rate_table = _TARIFF_RATES[ship_to]
        tariff_rate = rate_table.get(category, rate_table["*"])
        subtotal_cny = self.rates.convert(subtotal, "CNY")
        de_minimis_cny = self.de_minimis(ship_to, "CNY")
        taxable_minor_cny = max(
            0, subtotal_cny.amount_in_minor_units - de_minimis_cny.amount_in_minor_units,
        )
        de_minimis_applied = taxable_minor_cny == 0
        tariff_cny = Money.of(round(taxable_minor_cny * tariff_rate), "CNY")
        tariff_target = self.rates.convert(tariff_cny, target_currency)

        return ShippingQuote(
            ship_to=ship_to,
            subtotal=subtotal_target,
            freight=freight_target,
            tariff=tariff_target,
            tariff_rate=tariff_rate,
            de_minimis_applied=de_minimis_applied,
            **self._de_minimis_snapshot(ship_to, taxable_minor_cny, target_currency),
        )
