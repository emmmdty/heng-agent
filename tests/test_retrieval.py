# -*- coding: utf-8 -*-
"""二期检索链路单测：二阶段召回 / 降级链 / 价格硬约束 / 到手价内联。

embedding 用确定性桩实现（关键词特征轴 + 余弦），向量索引用 Qdrant 本地嵌入模式，
全程不依赖外部服务与 LLM。
"""
import pytest

from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.ports.retrieval_ports import EmbeddingClient, Reranker
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.infrastructure.settings import Settings
from app.infrastructure.vector.index_bootstrap import bootstrap_product_index
from app.infrastructure.vector.qdrant_product_index import QdrantProductIndex

# 特征轴词表：文本命中即该维置 1，余弦相似度即可反映关键词重合度。
#
# 词表必须覆盖测试 query 里出现的**每一个**概念，否则多概念 query 会被投影到同一根轴上，
# 不同商品的向量变成同向（cosine 恒为 1.0），排序退化为向量库的任意顺序。
# 商品库由 10 SPU 扩到 60 SPU 后这个坑真实发生过：query「露营灯 抗造」缺 `抗造` 轴时，
# P1008（便携露营灯）与 P1039（太阳能营地灯，描述里也含"露营灯"）无法区分。
_FEATURE_TERMS = (
    "露营灯", "登山杖", "毛巾", "睡袋", "行李箱", "耳机", "充电器", "三件套", "背包", "茶具",
    "抗造",
)


class AxisEmbeddingClient(EmbeddingClient):
    """确定性桩：按特征词命中构造向量。"""

    async def embed(self, text: str) -> list[float]:
        return [1.0 if term in text else 0.0 for term in _FEATURE_TERMS]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class BrokenEmbeddingClient(EmbeddingClient):
    async def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding 服务不可用")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding 服务不可用")


class ReverseReranker(Reranker):
    """确定性桩：把候选顺序整体反转（分数与原顺序相反）。"""

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [float(i) for i in range(len(documents))]


def _settings(tmp_path) -> Settings:
    base = Settings(
        llm_base_url="", llm_api_key="", llm_model="", port=8000, log_level="info",
        embedding_base_url="", embedding_api_key="", embedding_model="", embedding_dim=8,
        qdrant_url="", qdrant_collection="test_products",
        reranker_base_url="", reranker_model="", tavily_api_key="",
        otlp_endpoint="", data_dir=tmp_path,
        category_kb_collection="test_category_kb",
        context_size=128000, tool_result_limit=20000, reply_token_budget=0,
        tool_failure_threshold=3, tool_circuit_reset_seconds=60.0,
        cors_origins=["http://localhost:5173"],
    )
    return base


@pytest.fixture()
async def indexed(tmp_path):
    """已建库的（repo, embedder, index）三元组。"""
    repo = InMemoryProductRepository()
    embedder = AxisEmbeddingClient()
    index = QdrantProductIndex(_settings(tmp_path))
    ok = await bootstrap_product_index(repo, embedder, index)
    assert ok, "本地嵌入模式建库应成功"
    yield repo, embedder, index
    await index.close()


class TestTwoStageRecall:
    async def test_embedding_recall_ranks_camping_light_first(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(ProductSearchSpec(normalized_query="露营灯 抗造"))
        assert result["recall_strategy"] == "embedding_only"
        assert result["rerank_applied"] is False
        assert result["hits"][0]["product_id"] == "P1008", "露营灯应排第一"

    async def test_rerank_applied_changes_order(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(
            repo, embedder=embedder, vector_index=index, reranker=ReverseReranker(),
        )
        result = await usecase.execute(ProductSearchSpec(normalized_query="露营灯"))
        assert result["recall_strategy"] == "embedding_rerank"
        assert result["rerank_applied"] is True
        # 反转桩生效：露营灯不再是第一位
        assert result["hits"][0]["product_id"] != "P1008"

    async def test_degrade_to_keyword_when_embedding_broken(self, indexed):
        repo, _, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=BrokenEmbeddingClient(), vector_index=index)
        result = await usecase.execute(ProductSearchSpec(normalized_query="露营灯 抗造"))
        assert result["recall_strategy"] == "keyword_2gram"
        assert result["hits"], "关键词降级仍应有召回"
        assert result["hits"][0]["product_id"] == "P1008"

    async def test_price_cap_hard_filter(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="行李箱 登机", price_max_major=500.0),
        )
        # P1002 行李箱 899 CNY 超预算，必须被结构化过滤
        assert all(hit["product_id"] != "P1002" for hit in result["hits"])

    async def test_over_price_cap_candidate_reported_in_filtered_out(self, indexed):
        """超预算候选必须如实回传，否则模型会把"有但超预算"答成"没有这个商品"
        （三期评测 long-context-memory 曾暴露此缺陷）。"""
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="行李箱 登机", price_max_major=500.0),
        )
        rejected = {item["product_id"]: item for item in result["filtered_out"]}
        assert "P1002" in rejected, "被价格上限挡掉的候选必须在 filtered_out 里可见"
        assert rejected["P1002"]["reason"] == "over_price_cap"
        # 价格按目标币种给出，模型才能直接告知买家超了多少
        assert rejected["P1002"]["currency"] == "CNY"
        assert rejected["P1002"]["price_major"] > 500.0

    async def test_unshippable_candidate_reported_in_filtered_out(self, indexed):
        """商品自身不发这个目的国 → filtered_out 里如实标注。

        例子用 JP：规则表支持它，而 LumenGo 系列的 ships_to 是 CN/US/EU。
        （原先这条用的是 BR——那是**规则表根本不支持**的目的国，属于下面
        TestUnsupportedDestination 的情形，两者混用会把真正的缺陷藏住。）
        """
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="露营灯 抗造", ship_to="JP"),
        )
        reasons = {item["reason"] for item in result["filtered_out"]}
        assert reasons == {"ship_to_unavailable"}, "不可达目的国应标注为 ship_to_unavailable"

    async def test_no_filtered_out_key_without_hard_constraints(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(ProductSearchSpec(normalized_query="露营灯"))
        assert "filtered_out" not in result, "无硬约束时不应污染工具返回"

    async def test_landed_price_inlined_with_ship_to(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="露营灯", ship_to="US", target_currency="USD"),
        )
        top = result["hits"][0]
        assert "landed_price" in top
        landed = top["landed_price"]
        assert landed["currency"] == "USD"
        assert landed["landed_total_major"] == pytest.approx(
            landed["subtotal_major"] + landed["freight_major"] + landed["tariff_major"],
            abs=0.02,
        )

    async def test_no_landed_price_without_ship_to(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(ProductSearchSpec(normalized_query="露营灯"))
        assert "landed_price" not in result["hits"][0]

    async def test_tool_event_carries_hits_for_frontend(self, indexed):
        """tool.result 事件必须带可渲染的商品卡，否则前端商品卡区域一片空白
        （三期浏览器验证曾暴露此缺陷：事件只带 hit_count 没带 hits）。

        断言锁的是**商品卡契约**而不是具体命中哪个商品：本用例要防的回归是
        "事件丢字段"，排序质量由召回评测（`scripts/eval/run_product_recall.py`）负责。
        早先这里钉死 `hits[0] == "P1008"`，商品库扩容后就因为一个与本用例无关的
        排序变化而失败——测试该测自己那一件事。
        """
        from app.application.tools.product_search_tool import build_product_search_tool
        from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
        from app.infrastructure.eventbus import TradeEventBus

        repo, embedder, index = indexed
        bus = TradeEventBus()
        queue = bus.subscribe("s-cards")
        tool = build_product_search_tool(
            CatalogSearchUseCase(repo, embedder=embedder, vector_index=index),
            bus,
        )
        token = ShoppingContext.set(
            ShoppingContextSnapshot(
                shopping_session_id="s-cards", buyer_id="b", locale="zh-CN", currency="CNY",
            ),
        )
        try:
            await tool(normalized_query="露营灯", ship_to="US", target_currency="USD")
        finally:
            ShoppingContext.reset(token)

        queue.get_nowait()  # tool.invoke
        result_event = queue.get_nowait()
        assert result_event.type == "tool.result"
        hits = result_event.payload["hits"]
        assert hits, "tool.result 必须带 hits，否则前端无数据可渲染"

        card = hits[0]
        # 前端 ProductCards.tsx 渲染所依赖的字段，缺一个卡片就残
        for field in ("product_id", "title", "brand", "price_major", "currency", "skus"):
            assert field in card, f"商品卡缺字段 {field}，前端渲染会残"
        assert card["landed_price"]["currency"] == "USD", "传了 ship_to 就必须内联到手价"


class TestUnsupportedDestination:
    """规则表不认识的目的国，必须和"这件商品不发那儿"区分开。

    九期评测实测挖出：买家问"寄到欧盟"，Agent 合理地把它翻成具体国家码
    试了 DE 和 FR（工具参数当时写的是"收货国家二位码，如 CN、US"，
    正是这么诱导的）。规则表只认 CN/US/EU/JP/SG，于是**每一个商品**都被
    逐个标成 `ship_to_unavailable`，Agent 看到的是"所有 TrailOx 都不发欧盟"，
    只好放弃到手价、转而凭自己的知识说"欧盟免税额度约 €150"——
    正是八期补 `de_minimis_threshold_major` 要防的行为，从另一条路径又回来了。

    两处都要修：
      1. 语义分清。系统不认识 DE，不代表商品不发德国——据此过滤商品是错的判断。
      2. 返回要能自纠。错误里必须带上支持的目的国，否则模型无从知道该改填什么，
         只能凭知识硬答。这与 `combine_hint` 是同一条原则：**工具返回值要能自证边界**。
    """

    async def test_unsupported_destination_is_named_at_top_level(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="行李箱 铝框", ship_to="DE"),
        )
        assert "ship_to_unsupported" in result, "规则表不支持的目的国必须在顶层说明"
        assert result["ship_to_unsupported"]["given"] == "DE"

    async def test_supported_list_is_returned_so_the_model_can_retry(self, indexed):
        """不给支持列表，模型就只能猜——实测它猜了 DE 又猜 FR，然后放弃。"""
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="行李箱 铝框", ship_to="DE"),
        )
        supported = result["ship_to_unsupported"]["supported"]
        assert "EU" in supported, "买家说的欧盟对应 EU，必须能从返回里看出来"
        assert "DE" not in supported

    async def test_products_are_not_mislabeled_as_unshippable(self, indexed):
        """这是最要命的一条：把"系统不认识这个目的国"表述成"这些商品都不发那儿"，
        是一句**事实错误**，而且模型完全没有办法识破它。"""
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="行李箱 铝框", ship_to="DE"),
        )
        reasons = {item["reason"] for item in result.get("filtered_out", [])}
        assert "ship_to_unavailable" not in reasons
        assert result["hits"], "商品本身照常返回——买家问的商品信息不该因为目的国填错而消失"

    async def test_no_landed_price_is_fabricated_for_unsupported_destination(self, indexed):
        """拿不到规则就不能给到手价，一个数字都不能有。"""
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="行李箱 铝框", ship_to="DE"),
        )
        assert result["hits"], "哨兵：没有商品卡的话，下面的断言会平凡通过（踩坑 29）"
        for hit in result["hits"]:
            assert hit.get("landed_price") is None, "目的国不支持时不得内联到手价"
            for sku in hit.get("skus", []):
                assert sku.get("landed_price") is None

    async def test_supported_destination_is_untouched(self, indexed):
        """回归：支持的目的国照旧过滤 + 内联到手价，行为一点不变。"""
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="行李箱 铝框", ship_to="EU"),
        )
        assert "ship_to_unsupported" not in result
        assert result["hits"][0]["landed_price"]["landed_total_major"] > 0


class TestPriceCapLooksAtAllSkus:
    """价格上限按**最便宜的规格**判，不是按主规格。

    今天还不会出事：库里只有两件商品的规格之间有差价（P1001 / P1004），
    而它们的主规格恰好就是最便宜的那个。但这是个陷阱——
    以后谁加一件"主规格贵、另有便宜规格"的商品，符合预算的那个规格会被
    连着整件商品一起挡掉，模型看到的是 `over_price_cap`，
    于是把"库里有你买得起的规格"答成"没有符合预算的商品"。

    这类缺陷的特点是**加一条数据就会突然出现**，而那时没有任何判据会响。
    """

    def _product_with_cheap_variant(self):
        from app.domain.catalog.money import Money
        from app.domain.catalog.product import Product, ProductHighlight
        from app.domain.catalog.sku import Sku

        return Product(
            product_id="PX01",
            title="TestBrand 双规格测试品",
            brand="TestBrand",
            category="旅行装备",
            origin_country="CN",
            description="主规格贵、另有便宜规格",
            highlights=[ProductHighlight(label="材质", detail="测试")],
            ships_to=["CN"],
            skus=[
                Sku(sku_id="PX01-S1", spec="豪华版", price=Money.from_major_units(500, "CNY"), stock=5),
                Sku(sku_id="PX01-S2", spec="基础版", price=Money.from_major_units(150, "CNY"), stock=5),
            ],
        )

    async def _search(self, price_max):
        from app.application.usecases.catalog_search import CatalogSearchUseCase
        from app.domain.catalog.product_search_spec import ProductSearchSpec
        from app.infrastructure.persistence.in_memory_repositories import (
            InMemoryProductRepository,
        )

        repo = InMemoryProductRepository([self._product_with_cheap_variant()])
        usecase = CatalogSearchUseCase(repo)
        return await usecase.execute(
            ProductSearchSpec(normalized_query="双规格测试品", price_max_major=price_max),
        )

    async def test_cheap_variant_keeps_the_product_in(self):
        result = await self._search(price_max=200)
        assert [h["product_id"] for h in result["hits"]] == ["PX01"], (
            "基础版 150 元在预算内，整件商品不该被挡掉"
        )

    async def test_all_variants_over_cap_is_still_filtered(self):
        """反向也要成立：所有规格都超预算时照旧挡掉并给出 reason，
        否则这条修改就变成了"价格上限失效"。"""
        result = await self._search(price_max=100)
        assert not result["hits"]
        assert result["filtered_out"][0]["reason"] == "over_price_cap"
