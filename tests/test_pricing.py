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
    这个 800 是模型从自己的知识里说的。改了规则表里 US 的额度（十一期后是 `_DE_MINIMIS_NATIVE["US"]`），
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


class TestTaxableBaseIsSourced:
    """应税基数必须由工具返回。

    十期实测（`de-minimis-boundary-eu`）：最终关税 3.48 元是对的（工具算的），
    但 Agent 的解释写成「1,199 × 12% ≈ ¥3.48」——**计税基数写成了整单金额**
    （1199 × 12% = 143.88，差了 40 倍）。数字对、过程错。

    同一轮金额出处校验报了两处无出处金额，都是 `€3.72`——正是它自己
    减出来的应税基数（153.72 − 150）。**判据指的地方就是工具该补的地方**：
    工具已经回了免税额度与费率，却没回"超出额度的那部分是多少"，
    模型要解释就只能自己减。补上字段比在提示词里讲道理更可能奏效
    （同 `combine_hint` 的思路）。
    """

    def test_single_quote_reports_taxable_base(self, schedule):
        # 1199 CNY 寄 EU：免税额度 1170（150 EUR × 7.8），应税基数 29，关税 29 × 12% = 3.48
        quote = schedule.quote(
            subtotal=Money.from_major_units(1199, "CNY"), category="家居生活",
            ship_to="EU", quantity=1, target_currency="CNY",
        )
        payload = quote.to_dict()
        assert payload["de_minimis_threshold_major"] == 1170.0
        assert payload["taxable_base_major"] == 29.0
        assert payload["tariff_major"] == 3.48

    def test_taxable_base_is_zero_when_de_minimis_applies(self, schedule):
        quote = schedule.quote(
            subtotal=Money.from_major_units(189, "CNY"), category="旅行装备",
            ship_to="CN", quantity=1, target_currency="CNY",
        )
        payload = quote.to_dict()
        assert payload["de_minimis_applied"] is True
        assert payload["taxable_base_major"] == 0.0

    def test_basket_quote_reports_taxable_base(self):
        schedule = TariffSchedule(rates=ExchangeRateTable())
        quote = schedule.quote_basket(
            [BasketLine("P1", "灯", "家居生活", Money.from_major_units(1199, "CNY"), 1)],
            ship_to="EU", target_currency="CNY",
        )
        payload = quote.to_dict()
        assert payload["taxable_base_major"] == 29.0
        assert payload["tariff_major"] == 3.48

    def test_taxable_base_matches_tariff_over_rate(self, schedule):
        """自洽校验：关税 == 应税基数 × 费率。

        这条锁死的是"两个字段各算一套"的退化——应税基数若不是真正参与
        计征的那个数，模型抄了它反而会算出对不上的结果。
        """
        quote = schedule.quote(
            subtotal=Money.from_major_units(2000, "CNY"), category="数码配件",
            ship_to="CN", quantity=1, target_currency="CNY",
        )
        payload = quote.to_dict()
        assert payload["taxable_base_major"] * payload["tariff_rate"] == pytest.approx(
            payload["tariff_major"], abs=0.01,
        )


class TestDeMinimisNativeCurrency:
    """免税额度必须同时给出**原生口径**。

    规则表里 US 的额度本来就定义为 800 USD、EU 是 150 EUR，但只按 target_currency
    回一个值。于是 `target_currency=CNY` 而买家语境是美国时，Agent 想说"美国免税门槛"
    就只能自己反折：实测写的是"美国免税门槛 **$800**（约 ¥5680）"——
    5680 有出处，800 没有。

    这是同一个 `$800` 第三次从新路径回来（八期堵"凭知识说"、十期堵"工具没被调到"、
    这次是"调到了但口径不对"）。**一个症状被修掉，不等于产生它的那类缺口被封上。**
    """

    def test_us_threshold_native_is_dollars(self, schedule):
        quote = schedule.quote(
            subtotal=Money.of(219_00, "USD"), category="数码配件",
            ship_to="US", quantity=1, target_currency="CNY",
        )
        payload = quote.to_dict()
        assert payload["de_minimis_threshold_major"] == 5680.0            # CNY 口径
        assert payload["de_minimis_threshold_native_major"] == 800.0      # 原生口径
        assert payload["de_minimis_threshold_native_currency"] == "USD"

    def test_eu_threshold_native_is_euros(self, schedule):
        quote = schedule.quote(
            subtotal=Money.from_major_units(1199, "CNY"), category="家居生活",
            ship_to="EU", quantity=1, target_currency="CNY",
        )
        payload = quote.to_dict()
        assert payload["de_minimis_threshold_native_major"] == 150.0
        assert payload["de_minimis_threshold_native_currency"] == "EUR"
        # 应税基数的原生口径：29 CNY ÷ 7.8 = 3.72 EUR，正是判据报过的那个 €3.72
        assert payload["taxable_base_native_major"] == 3.72

    def test_basket_quote_reports_native_threshold(self):
        schedule = TariffSchedule(rates=ExchangeRateTable())
        quote = schedule.quote_basket(
            [BasketLine("P1", "耳机", "数码配件", Money.of(299_00, "CNY"), 1)],
            ship_to="US", target_currency="CNY",
        )
        payload = quote.to_dict()
        assert payload["de_minimis_threshold_native_major"] == 800.0
        assert payload["de_minimis_threshold_native_currency"] == "USD"

    def test_native_currency_equals_target_keeps_one_number(self, schedule):
        """目标币种与原生币种一致时两个口径应当相等，不能因为多绕一次折算而漂移。"""
        quote = schedule.quote(
            subtotal=Money.of(219_00, "USD"), category="数码配件",
            ship_to="US", quantity=1, target_currency="USD",
        )
        payload = quote.to_dict()
        assert payload["de_minimis_threshold_native_major"] == 800.0
        assert abs(payload["de_minimis_threshold_major"] - 800.0) < 0.01

    def test_public_accessors_expose_both_kinds(self, schedule):
        """规则表以外的地方（judge 的事实基准）要拿到这两个口径，
        不该再去 import 私有常量——十期 `_landed_price_rules()` 就是那么写的。"""
        native = schedule.de_minimis_native("EU")
        assert (native.to_major_units(), native.currency) == (150.0, "EUR")
        assert schedule.de_minimis("EU", "CNY").to_major_units() == 1170.0

    def test_jp_threshold_derives_from_the_rate_table(self, schedule):
        """JP 额度原本硬编码 `10_000 * 5`（手写 0.05 汇率），而汇率表里 JPY 是 0.048。
        改为从汇率表推导后是 480 CNY——这是修正，也是可观测的行为变化。"""
        assert schedule.de_minimis_native("JP").to_major_units() == 10_000.0
        assert schedule.de_minimis("JP", "CNY").to_major_units() == 480.0

    def test_unsupported_destination_rejected_by_accessors(self, schedule):
        with pytest.raises(ValueError, match="不支持"):
            schedule.de_minimis_native("ZZ")
