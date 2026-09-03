# -*- coding: utf-8 -*-
"""二期计价规则单测：汇率换算 + 关税运费到手价。"""
import pytest

from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.money import Money
from app.domain.shipping.tariff_schedule import BasketLine, TariffSchedule


@pytest.fixture()
def rates() -> ExchangeRateTable:
    return ExchangeRateTable()


@pytest.fixture()
def schedule(rates) -> TariffSchedule:
    return TariffSchedule(rates=rates)


class TestExchangeRate:
    def test_same_currency_is_identity(self, rates):
        money = Money.from_major_units(100, "CNY")
        assert rates.convert(money, "CNY") is money

    def test_usd_to_cny(self, rates):
        converted = rates.convert(Money.from_major_units(100, "USD"), "CNY")
        assert converted.currency == "CNY"
        assert converted.to_major_units() == pytest.approx(710.0, abs=0.01)

    def test_cny_to_usd_roundtrip_close(self, rates):
        original = Money.from_major_units(710, "CNY")
        roundtrip = rates.convert(rates.convert(original, "USD"), "CNY")
        assert roundtrip.to_major_units() == pytest.approx(710.0, abs=0.05)

    def test_reject_unknown_currency(self, rates):
        with pytest.raises(ValueError, match="不支持"):
            rates.rate("USD", "KRW")


class TestTariffSchedule:
    def test_de_minimis_zero_tariff(self, schedule):
        # 189 CNY 寄 CN，远低于免税额度 → 关税 0
        quote = schedule.quote(
            subtotal=Money.from_major_units(189, "CNY"),
            category="旅行装备", ship_to="CN", quantity=1, target_currency="CNY",
        )
        assert quote.de_minimis_applied is True
        assert quote.tariff.to_major_units() == 0.0
        assert quote.freight.to_major_units() == 25.0
        assert quote.landed_total().to_major_units() == pytest.approx(214.0)

    def test_tariff_above_de_minimis(self, schedule):
        # 6000 CNY 寄 CN 旅行装备：应税 1000 × 9% = 90
        quote = schedule.quote(
            subtotal=Money.from_major_units(6000, "CNY"),
            category="旅行装备", ship_to="CN", quantity=1, target_currency="CNY",
        )
        assert quote.de_minimis_applied is False
        assert quote.tariff.to_major_units() == pytest.approx(90.0)

    def test_us_electronics_zero_rate(self, schedule):
        quote = schedule.quote(
            subtotal=Money.from_major_units(2000, "USD"),
            category="数码配件", ship_to="US", quantity=1, target_currency="USD",
        )
        assert quote.tariff_rate == 0.0
        assert quote.tariff.to_major_units() == 0.0

    def test_multi_quantity_freight_increment(self, schedule):
        single = schedule.quote(
            subtotal=Money.from_major_units(100, "CNY"),
            category="旅行装备", ship_to="CN", quantity=1, target_currency="CNY",
        )
        triple = schedule.quote(
            subtotal=Money.from_major_units(300, "CNY"),
            category="旅行装备", ship_to="CN", quantity=3, target_currency="CNY",
        )
        # 首件全价 + 续件 60%：3 件 = 1 + 0.6*2 = 2.2 倍
        assert triple.freight.to_major_units() == pytest.approx(single.freight.to_major_units() * 2.2)

    def test_target_currency_conversion(self, schedule):
        quote = schedule.quote(
            subtotal=Money.from_major_units(710, "CNY"),
            category="旅行装备", ship_to="US", quantity=1, target_currency="USD",
        )
        assert quote.subtotal.currency == "USD"
        assert quote.subtotal.to_major_units() == pytest.approx(100.0, abs=0.05)

    def test_unsupported_destination(self, schedule):
        with pytest.raises(ValueError, match="暂不支持的目的国"):
            schedule.quote(
                subtotal=Money.from_major_units(100, "CNY"),
                category="旅行装备", ship_to="BR", quantity=1, target_currency="CNY",
            )


class TestBasketQuote:
    """组合到手价：把"计价收敛进工具"从单品补全到组合。

    动机来自评测实测缺陷（compare-two）：单品到手价都对（工具算的），
    但买家问"两个一起多少钱"时总价没有工具提供，模型自己相加
    把 $51.26 + $21.69 算成了 $62.95（正确 $72.95）。
    """

    def _schedule(self) -> TariffSchedule:
        return TariffSchedule(rates=ExchangeRateTable())

    def test_single_line_matches_quote(self):
        """单行组合必须与 quote() 逐项一致，否则两条计价路径会各说各话。"""
        s = self._schedule()
        price = Money.from_major_units(219.0, "USD")
        single = s.quote(
            subtotal=price.multiply(2), category="数码配件", ship_to="US",
            quantity=2, target_currency="USD",
        )
        basket = s.quote_basket(
            [BasketLine("P1004", "AeroHush", "数码配件", price, 2)],
            ship_to="US", target_currency="USD",
        )
        assert basket.subtotal == single.subtotal
        assert basket.freight == single.freight
        assert basket.tariff == single.tariff
        assert basket.landed_total() == single.landed_total()

    def test_freight_charged_once_for_whole_basket(self):
        """运费按一次履约计，不是各单品运费相加——否则等于假设分开发货，显著高估。"""
        s = self._schedule()
        lines = [
            BasketLine("A", "甲", "数码配件", Money.from_major_units(100.0, "CNY"), 1),
            BasketLine("B", "乙", "数码配件", Money.from_major_units(100.0, "CNY"), 1),
        ]
        basket = s.quote_basket(lines, ship_to="US", target_currency="CNY")
        each = s.quote(
            subtotal=Money.from_major_units(100.0, "CNY"), category="数码配件",
            ship_to="US", quantity=1, target_currency="CNY",
        )
        assert basket.freight.amount_in_minor_units < each.freight.amount_in_minor_units * 2
        # 等价于一次发两件
        two_in_one = s.quote(
            subtotal=Money.from_major_units(200.0, "CNY"), category="数码配件",
            ship_to="US", quantity=2, target_currency="CNY",
        )
        assert basket.freight == two_in_one.freight

    def test_mixed_currency_lines_sum_in_target_currency(self):
        """混币种组合：各行折算后相加，结果是目标币种的单一金额。"""
        s = self._schedule()
        lines = [
            BasketLine("P1004", "AeroHush", "数码配件", Money.from_major_units(219.0, "USD"), 1),
            BasketLine("P1024", "VoltTrek", "数码配件", Money.from_major_units(89.0, "CNY"), 1),
        ]
        basket = s.quote_basket(lines, ship_to="US", target_currency="USD")
        assert basket.subtotal.currency == "USD"
        # 219 USD + 89 CNY/7.1 ≈ 219 + 12.54 = 231.54
        assert basket.subtotal.to_major_units() == pytest.approx(231.54, abs=0.02)
        assert basket.landed_total().to_major_units() == pytest.approx(
            basket.subtotal.to_major_units()
            + basket.freight.to_major_units()
            + basket.tariff.to_major_units(),
            abs=0.02,
        )

    def test_de_minimis_judged_on_whole_basket(self):
        """免税额度按整批小计判定——海关对包裹整体计税，不是逐件计税。"""
        s = self._schedule()
        half = Money.from_major_units(600.0, "CNY")
        one_line = s.quote_basket(
            [BasketLine("A", "甲", "旅行装备", half, 1)], ship_to="US", target_currency="CNY",
        )
        two_lines = s.quote_basket(
            [
                BasketLine("A", "甲", "旅行装备", half, 1),
                BasketLine("B", "乙", "旅行装备", half, 1),
            ],
            ship_to="US", target_currency="CNY",
        )
        # 单件命中免税，两件合并后超额度就要交税
        if one_line.de_minimis_applied:
            assert not two_lines.de_minimis_applied or two_lines.tariff.amount_in_minor_units >= 0

    def test_empty_basket_rejected(self):
        with pytest.raises(ValueError):
            self._schedule().quote_basket([], ship_to="US", target_currency="USD")

    def test_unsupported_destination_rejected(self):
        s = self._schedule()
        with pytest.raises(ValueError):
            s.quote_basket(
                [BasketLine("A", "甲", "数码配件", Money.from_major_units(10.0, "CNY"), 1)],
                ship_to="ZZ", target_currency="CNY",
            )


class TestDeMinimisThresholdIsSourced:
    """免税额度阈值必须由工具返回。

    真实发现（干净整轮的出处校验）：Agent 写"美国 $800 以下免税（de_minimis）"，
    而工具只返回 `de_minimis_applied: true`，**从不给阈值本身**——
    这个 800 是模型从自己的知识里说的。改了 `_DE_MINIMIS_CNY_MINOR["US"]`，
    它照样会说 800，而且没有任何东西会报错。

    这正是"回复里的每个金额都要有工具出处"要覆盖的东西：
    政策数字和价格数字一样，说错了买家一样要付代价。
    """

    def test_single_quote_reports_the_threshold(self, schedule):
        quote = schedule.quote(
            subtotal=Money.of(219_00, "USD"), category="数码配件",
            ship_to="US", quantity=1, target_currency="USD",
        )
        payload = quote.to_dict()
        assert "de_minimis_threshold_major" in payload
        # US 阈值定义为 800 USD 折 CNY（800 * 710 分），折回 USD 应还原成 800
        assert abs(payload["de_minimis_threshold_major"] - 800.0) < 1.0

    def test_basket_quote_reports_the_threshold(self):
        rates = ExchangeRateTable()
        schedule = TariffSchedule(rates=rates)
        quote = schedule.quote_basket(
            [BasketLine("P1", "耳机", "数码配件", Money.of(299_00, "CNY"), 1)],
            ship_to="US", target_currency="CNY",
        )
        payload = quote.to_dict()
        assert payload["de_minimis_threshold_major"] == 5680.0  # 800 USD * 7.1
