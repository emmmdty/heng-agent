# -*- coding: utf-8 -*-
"""种子商品库的不变量。

商品库是**所有评测用例的事实基准**：judge 拿它核对价格与库存，
`audit_cases.py` 拿它判断品牌指代是否唯一，40 多条用例的 P0 里写死的数字
全部来自它。它自己出问题，上面这一整层判据会一起失真——
而失真的方式是"分数看着正常，只是评的东西不对了"。

这里只钉**结构性不变量**（重复 id、口径不一致），不钉具体商品——
钉具体商品等于每加一款货就要改一次测试，那种测试很快会被人删掉。
"""
from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.shipping.tariff_schedule import TariffSchedule
from app.infrastructure.persistence.seed_products import build_seed_products


def _products():
    return build_seed_products()


class TestIdentifiersAreUnique:
    def test_product_ids_do_not_repeat(self):
        ids = [p.product_id for p in _products()]
        assert len(ids) == len(set(ids)), "product_id 重复会让检索与下单指向不同商品"

    def test_sku_ids_are_globally_unique(self):
        """sku_id 全局唯一，不只是商品内唯一。

        下单出处校验（十四期）与订单行都按 sku_id 认商品；
        两个商品共用一个 sku_id 时，"下单的是不是检索到的那个"就判不出来了。
        """
        sku_ids = [sku.sku_id for p in _products() for sku in p.skus]
        duplicates = sorted({s for s in sku_ids if sku_ids.count(s) > 1})
        assert not duplicates, f"sku_id 重复：{duplicates}"

    def test_every_product_has_at_least_one_sku(self):
        empty = [p.product_id for p in _products() if not p.skus]
        assert not empty, f"没有 sku 的商品下不了单：{empty}"


class TestPricingIsQuotable:
    """每个商品都要能算出到手价——算不出的商品会在评测里变成一条噪声。"""

    def test_every_currency_is_convertible(self):
        rates = ExchangeRateTable()
        for product in _products():
            for sku in product.skus:
                rates.convert(sku.price, "CNY")  # 抛异常即失败

    def test_ships_to_only_uses_known_destination_codes(self):
        """`ships_to` 里的国家码必须是规则表认识的。

        规则表不认识的目的国会走"计价规则表不支持"这条路（十期修的那条），
        而买家看到的是"这件商品不发那儿"——一句错话。
        """
        supported = set(TariffSchedule(rates=ExchangeRateTable()).supported_destinations())
        unknown = sorted({
            code for product in _products() for code in product.ships_to if code not in supported
        })
        assert not unknown, f"商品声明了规则表不认识的目的国：{unknown}"

    def test_every_product_ships_somewhere(self):
        nowhere = [p.product_id for p in _products() if not p.ships_to]
        assert not nowhere, f"哪儿都不发的商品无法参与任何用例：{nowhere}"

    def test_prices_are_positive(self):
        zero = [
            f"{p.product_id}/{sku.sku_id}"
            for p in _products() for sku in p.skus
            if sku.price.amount_in_minor_units <= 0
        ]
        assert not zero, f"零价商品会让到手价判据失去意义：{zero}"


class TestCatalogIsBigEnoughToDiscriminate:
    def test_at_least_sixty_products(self):
        """六期把商品库从 10 扩到 60，是因为 10 个 SPU 下 Recall@10 恒等于 1、
        指标没有区分度。缩回去等于让整套召回评测失去意义。"""
        assert len(_products()) >= 60
