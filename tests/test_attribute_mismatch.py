# -*- coding: utf-8 -*-
"""属性不匹配的显式化：候选不具备买家点名要的属性时，返回里必须说出来

判据来自二十期整轮实测（`conflict-budget-spec`，七轮历史
`PASS PASS PASS FAIL FAIL PASS FAIL`——**间歇性缺陷**）：

买家："预算 200 元，给我来一副顶配的主动降噪耳机。"
Agent 把 `AeroHush Lite`（半入耳、¥299、**只有通话降噪**）列在
"库里有的主动降噪耳机"这个标题下，暗示加 99 元就能买到 ANC。

**前两次修都是在动召回，都没拦住**：

    十九期  把"无主动降噪 ANC"写进 description   → 帮了倒忙：BM25 不理解否定，
                                                   看见词就命中，召回反而更强（踩坑 46）
    十九期  改成不可检索的 highlight              → 字面路压下去了（BM25 3.035 < 门限 4.0），
                                                   但**向量路照样召回**（读数纪律 4）

所以这一次不动召回，动**工具返回的结构**。理由是四次成功先例的共同点：
`filtered_out` / `combine_hint` / `ship_to_unsupported` / `taxable_base_major`
给的都是**结构化字段**；而卡片上那句"仅通话降噪（麦克风侧），无主动降噪 ANC"
是**散文**，模型可以不当回事——同一条老纪律的第五次应用：
**缺失的信息要显式化，而且要显式成模型没法忽略的形状。**

**判据方向仍是"宁可漏报不误报"**：只认商品数据里**显式声明**的否定属性
（`ProductHighlight.negates`），不做"字面没出现 = 不具备"的推断——
召回没命中某个词不代表商品缺这个属性，那样推会把整张候选表标满假警报。
"""
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product import Product, ProductHighlight
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.persistence.seed_products import build_seed_products


class TestNegatedAttributeIsDeclared:
    def test_seed_declares_the_missing_anc(self):
        """半入耳款必须在**结构化字段**里声明它不具备主动降噪，而不是只写在散文里。"""
        product = next(p for p in build_seed_products() if p.product_id == "P1022")
        assert "主动降噪" in product.absent_attributes()

    def test_a_product_that_has_anc_declares_nothing(self):
        product = next(p for p in build_seed_products() if p.product_id == "P1023")
        assert product.absent_attributes() == []

    def test_negated_highlight_stays_out_of_the_lexical_index(self):
        """踩坑 46 的回归：否定说明进了可检索文本，BM25 反而更强地召回它。"""
        product = next(p for p in build_seed_products() if p.product_id == "P1022")
        assert "无主动降噪" not in product.searchable_text()

    def test_negation_is_declared_on_the_highlight_that_says_it(self):
        """声明挂在写着那句否定的 highlight 上，不是挂在商品级的另一份清单里
        ——两份清单迟早会脱钩，而脱钩的外观是"判据没生效"。"""
        highlight = ProductHighlight("降噪", "无主动降噪", False, ("主动降噪", "ANC"))
        assert highlight.negates == ("主动降噪", "ANC")
        assert ProductHighlight("续航", "30 小时").negates == ()


class TestProductAbsentAttributes:
    def test_collects_from_all_highlights(self):
        product = Product(
            product_id="X1", title="t", brand="b", category="c", origin_country="CN",
            description="d",
            highlights=[
                ProductHighlight("降噪", "无 ANC", False, ("主动降噪",)),
                ProductHighlight("防水", "不防水", False, ("防水",)),
            ],
            ships_to=["CN"], skus=[_sku()],
        )
        assert product.absent_attributes() == ["主动降噪", "防水"]


class TestSearchResultCarriesTheMismatch:
    async def test_card_declares_the_mismatch_when_the_query_asks_for_it(self):
        """买家点名要"主动降噪"，而这张卡不具备——返回里必须结构化地说出来。"""
        use_case = _use_case()
        result = await use_case.execute(
            ProductSearchSpec(normalized_query="主动降噪 耳机", top_k=8),
        )
        card = _card(result, "P1022")
        assert card is not None, "半入耳款没被召回，这条判据无从验证（换个 query 或看召回档）"
        assert card["attribute_mismatch"]["missing"] == ["主动降噪"]
        assert "不具备" in card["attribute_mismatch"]["note"]

    async def test_no_mismatch_field_when_the_query_does_not_ask(self):
        """买家没点名要 ANC 时不加这个字段——无关的告警会让模型学会忽略它
        （同 order_provenance 那条"没有依据就不提醒"）。"""
        use_case = _use_case()
        result = await use_case.execute(
            ProductSearchSpec(normalized_query="半入耳 蓝牙耳机 长续航", top_k=8),
        )
        card = _card(result, "P1022")
        assert card is not None
        assert "attribute_mismatch" not in card

    async def test_a_product_that_has_the_attribute_is_not_flagged(self):
        use_case = _use_case()
        result = await use_case.execute(
            ProductSearchSpec(normalized_query="主动降噪 耳机", top_k=8),
        )
        card = _card(result, "P1023")
        assert card is not None
        assert "attribute_mismatch" not in card

    async def test_english_alias_in_the_query_also_matches(self):
        """买家写 ANC 而不是"主动降噪"是常见写法，大小写不敏感。"""
        use_case = _use_case()
        result = await use_case.execute(
            ProductSearchSpec(normalized_query="anc 耳机 推荐", top_k=8),
        )
        card = _card(result, "P1022")
        assert card is not None
        assert card["attribute_mismatch"]["missing"] == ["主动降噪"]


class TestFilteredOutCarriesTheMismatchToo:
    """被硬约束挡掉的候选**同样**要带声明——而且这条比 hits 更要紧。

    二十一期定向重跑第三轮实测（`report-20260904-131821`，FAIL 0.75）：
    买家说"预算 200 元"，于是 299 元的 AeroHush Lite 被 `over_price_cap` 挡进
    `filtered_out`；而 `_to_rejected()` 只回 product_id / title / category /
    price / reason，**结构化声明根本没到模型手上**。模型照旧写出
    "它支持主动降噪，是目前最接近你预算的选项"。

    这条用例里那款商品**必然**在 filtered_out 里——预算冲突正是它被挡的原因。
    只给 hits 加字段等于给这条用例修了一条它走不到的路：
    与"BM25 索引只在评测脚本里构造、从没接进 composition"（经验 1）同一形状，
    **修在了模型走不到的那条路上，外观与修好了完全一样**。
    """

    async def test_rejected_candidate_declares_the_mismatch(self):
        use_case = _use_case()
        result = await use_case.execute(
            ProductSearchSpec(normalized_query="主动降噪 耳机", top_k=8, price_max_major=200.0),
        )
        rejected = next(
            (r for r in result.get("filtered_out", []) if r["product_id"] == "P1022"), None,
        )
        assert rejected is not None, "299 元的半入耳款应被 200 元预算挡进 filtered_out"
        assert rejected["reason"] == "over_price_cap"
        assert rejected["attribute_mismatch"]["missing"] == ["主动降噪"]

    async def test_rejected_candidate_without_conflict_has_no_field(self):
        use_case = _use_case()
        result = await use_case.execute(
            ProductSearchSpec(normalized_query="蓝牙耳机", top_k=8, price_max_major=200.0),
        )
        for rejected in result.get("filtered_out", []):
            assert "attribute_mismatch" not in rejected


# —— 测试夹具 ——


def _sku():
    from app.domain.catalog.money import Money
    from app.domain.catalog.sku import Sku

    return Sku(sku_id="S1", spec="默认", price=Money.from_major_units(1.0, "CNY"), stock=1)


def _use_case() -> CatalogSearchUseCase:
    """只用关键词降级路：这条判据与召回档位无关，不该依赖外部服务。

    走降级路是刻意的——十九期栽过的正是"只在字面路上验证一个召回相关的修复"
    （读数纪律 4）。但本判据**不动召回**，它作用在结果装配上，
    对每一档都一样，所以用最便宜的那一档验证是成立的。
    """
    from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository

    return CatalogSearchUseCase(InMemoryProductRepository(build_seed_products()))


def _card(result: dict, product_id: str):
    return next((c for c in result["hits"] if c["product_id"] == product_id), None)
