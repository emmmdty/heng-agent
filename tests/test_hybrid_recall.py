# -*- coding: utf-8 -*-
"""混合召回（BM25 + 向量，RRF 融合）单测

全部确定性、零外部依赖：embedding 用特征轴桩，向量索引用 Qdrant 本地嵌入模式，
BM25 是纯本地实现。本文件锁三件事：
    1. BM25 本身算对了（IDF 压常见词、长度归一惩罚长文档、结果确定可复现）
    2. RRF 融合的数学与确定性
    3. 混合档的降级语义：向量路挂了要如实标 bm25_only，而不是谎报 hybrid_rrf
"""
import math

import pytest

from app.application.usecases.catalog_search import CatalogSearchUseCase, tokenize_terms
from app.domain.catalog.money import Money
from app.domain.catalog.product import Product, ProductHighlight
from app.domain.catalog.ports.retrieval_ports import EmbeddingClient
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.catalog.sku import Sku
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.infrastructure.retrieval.bm25_index import Bm25LexicalIndex
from app.infrastructure.settings import Settings
from app.infrastructure.vector.index_bootstrap import bootstrap_product_index
from app.infrastructure.vector.qdrant_product_index import QdrantProductIndex

_FEATURE_TERMS = ("露营灯", "登山杖", "毛巾", "睡袋", "行李箱", "耳机", "充电器", "三件套", "背包", "茶具", "抗造")


class AxisEmbeddingClient(EmbeddingClient):
    async def embed(self, text: str) -> list[float]:
        return [1.0 if term in text else 0.0 for term in _FEATURE_TERMS]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class BrokenEmbeddingClient(EmbeddingClient):
    async def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding 服务不可用")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding 服务不可用")


def _settings(tmp_path) -> Settings:
    return Settings(
        llm_base_url="", llm_api_key="", llm_model="", port=8000, log_level="info",
        embedding_base_url="", embedding_api_key="", embedding_model="", embedding_dim=11,
        qdrant_url="", qdrant_collection="test_hybrid",
        reranker_base_url="", reranker_model="", tavily_api_key="",
        otlp_endpoint="", data_dir=tmp_path,
        category_kb_collection="test_hybrid_kb",
        context_size=128000, tool_result_limit=20000, reply_token_budget=0,
        tool_failure_threshold=3, tool_circuit_reset_seconds=60.0,
        cors_origins=["http://localhost:5173"],
    )


def _product(pid: str, title: str, description: str) -> Product:
    return Product(
        product_id=pid, title=title, brand="B", category="户外运动", origin_country="CN",
        description=description, highlights=[ProductHighlight("x", "y")],
        ships_to=["CN", "US"],
        skus=[Sku(sku_id=f"{pid}-S1", spec="默认", price=Money.from_major_units(100.0, "CNY"), stock=10)],
    )


class TestBm25:
    def test_idf_downweights_ubiquitous_terms(self):
        """全库都含的词几乎不贡献分数，独有词才有区分度。

        这正是原 `_keyword_score`（命中词数等权）做不到的事。
        """
        index = Bm25LexicalIndex()
        index.index([
            _product("A", "甲 露营 记忆棉", "露营 记忆棉"),
            _product("B", "乙 露营", "露营"),
            _product("C", "丙 露营", "露营"),
            _product("D", "丁 露营", "露营"),
        ])
        # "露营"四篇全含 → IDF≈0；"记忆棉"仅 A 含 → IDF 高
        ubiquitous = index.search("露营", top_n=10)
        rare = index.search("记忆棉", top_n=10)
        assert rare[0].product_id == "A"
        assert rare[0].score > max((h.score for h in ubiquitous), default=0.0)

    def test_length_normalization_penalizes_padding(self):
        """同样命中一次，描述灌水的长文档不该赢过短文档。"""
        index = Bm25LexicalIndex()
        index.index([
            _product("SHORT", "短", "记忆棉"),
            _product("LONG", "长", "记忆棉 " + " ".join(f"词{i}" for i in range(60))),
        ])
        hits = {h.product_id: h.score for h in index.search("记忆棉", top_n=10)}
        assert hits["SHORT"] > hits["LONG"], "长度归一应惩罚靠字数堆命中的文档"

    def test_empty_index_returns_nothing(self):
        assert Bm25LexicalIndex().search("露营灯", top_n=5) == []

    def test_ties_broken_deterministically(self):
        """同分必须按 id 定序：否则同一标注集两次跑出不同 MRR，评测失去判据资格。"""
        index = Bm25LexicalIndex()
        index.index([_product("P2", "乙", "记忆棉"), _product("P1", "甲", "记忆棉")])
        ids = [h.product_id for h in index.search("记忆棉", top_n=10)]
        assert ids == sorted(ids)
        assert ids == [h.product_id for h in index.search("记忆棉", top_n=10)]

    def test_tokenize_terms_keeps_frequency(self):
        """BM25 依赖词频，分词器必须保留重复项（tokenize() 的 set 语义会丢掉）。"""
        terms = tokenize_terms("露营 露营")
        assert terms.count("露营") == 2, "两字词不该因 2-gram 与自身重合而被重复计数"
        # 三字及以上才补 2-gram，且原词保留
        assert tokenize_terms("记忆棉") == ["记忆棉", "记忆", "忆棉"]


class TestRrfFusion:
    def _usecase(self) -> CatalogSearchUseCase:
        return CatalogSearchUseCase(InMemoryProductRepository(), rrf_k=60)

    def test_rrf_score_matches_formula(self):
        """两路都排第一的文档得分应等于 2/(k+1)，别的名次同理。"""
        uc = self._usecase()
        fused = uc._rrf_fuse([["X", "Y"], ["X", "Z"]])
        assert fused[0] == "X", "两路都投的候选应升到首位"
        assert set(fused) == {"X", "Y", "Z"}, "只出现在单路里的候选也要保留"

    def test_single_channel_top1_can_be_outranked_by_consensus(self):
        """RRF 的核心行为：单路第一 < 两路共识的第二。

        这也是它的代价——向量路高置信度的第一名会被字面路的附议改写。
        代价是否划算由消融评测回答，不由直觉回答。
        """
        uc = self._usecase()
        fused = uc._rrf_fuse([["ONLY_A", "SHARED"], ["OTHER", "SHARED"]])
        assert fused[0] == "SHARED"

    def test_fusion_is_deterministic_on_ties(self):
        uc = self._usecase()
        first = uc._rrf_fuse([["A", "B"], ["B", "A"]])
        assert first == uc._rrf_fuse([["A", "B"], ["B", "A"]])
        assert first == sorted(first), "同分按 id 定序"

    def test_smaller_k_is_more_aggressive(self):
        """k 越小，头部名次的权重差越大——消融要扫的就是这个旋钮。"""
        head_gap_small = 1 / (10 + 1) - 1 / (10 + 2)
        head_gap_large = 1 / (60 + 1) - 1 / (60 + 2)
        assert head_gap_small > head_gap_large


@pytest.fixture()
async def hybrid_ready(tmp_path):
    repo = InMemoryProductRepository()
    embedder = AxisEmbeddingClient()
    index = QdrantProductIndex(_settings(tmp_path))
    assert await bootstrap_product_index(repo, embedder, index)
    lexical = Bm25LexicalIndex()
    lexical.index(await repo.list_all())
    yield repo, embedder, index, lexical
    await index.close()


class TestHybridStrategyLadder:
    async def test_hybrid_reports_its_own_strategy(self, hybrid_ready):
        repo, embedder, index, lexical = hybrid_ready
        uc = CatalogSearchUseCase(
            repo, embedder=embedder, vector_index=index, lexical_index=lexical,
        )
        result = await uc.execute(ProductSearchSpec(normalized_query="露营灯 抗造"))
        assert result["recall_strategy"] == "hybrid_rrf"
        assert result["hits"], "混合召回应有结果"

    async def test_vector_failure_degrades_to_bm25_and_says_so(self, hybrid_ready):
        """向量路挂掉时必须如实标 bm25_only。

        `recall_strategy` 是给模型看的事实：谎报成 hybrid_rrf 会让模型以为
        语义召回仍在生效，进而对"没搜到"给出过度自信的解释。
        """
        repo, _, index, lexical = hybrid_ready
        uc = CatalogSearchUseCase(
            repo, embedder=BrokenEmbeddingClient(), vector_index=index, lexical_index=lexical,
        )
        result = await uc.execute(ProductSearchSpec(normalized_query="露营灯 抗造"))
        assert result["recall_strategy"] == "bm25_only"
        assert result["hits"], "字面路是纯本地的，向量挂了仍必须有召回"

    async def test_without_lexical_index_behaviour_is_unchanged(self, hybrid_ready):
        """不注入 lexical_index 时行为与四期完全一致——混合召回是加法不是改写。"""
        repo, embedder, index, _ = hybrid_ready
        uc = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await uc.execute(ProductSearchSpec(normalized_query="露营灯 抗造"))
        assert result["recall_strategy"] == "embedding_only"

    async def test_hard_filters_still_apply_on_hybrid(self, hybrid_ready):
        """融合改的是排序，不能绕过硬约束过滤与 filtered_out 可观测性。"""
        repo, embedder, index, lexical = hybrid_ready
        uc = CatalogSearchUseCase(
            repo, embedder=embedder, vector_index=index, lexical_index=lexical,
        )
        result = await uc.execute(
            ProductSearchSpec(normalized_query="行李箱 登机", price_max_major=500.0),
        )
        assert all(h["price_major"] <= 500.0 for h in result["hits"])
        if "filtered_out" in result:
            assert all("reason" in item for item in result["filtered_out"])


class TestBm25OnlyIsReachable:
    """纯 BM25 档必须真的走 BM25。

    回归的是一个真实缺陷：`execute()` 的入口条件曾写成
    `embedder is not None and vector_index is not None`，只注入字面索引时整块被跳过，
    实际落回旧的 keyword_2gram 打分。表现是评测里 bm25_only 与 keyword_2gram
    跑出**逐位相同**的指标（0.871/0.841/0.821）——消融的"字面基线"测的是另一套东西。
    """

    async def test_lexical_only_uses_bm25_not_legacy_keyword(self):
        repo = InMemoryProductRepository()
        lexical = Bm25LexicalIndex()
        lexical.index(await repo.list_all())
        uc = CatalogSearchUseCase(repo, lexical_index=lexical)
        result = await uc.execute(ProductSearchSpec(normalized_query="露营灯 抗造"))
        assert result["recall_strategy"] == "bm25_only"
        assert result["hits"]

    async def test_bm25_and_legacy_keyword_rank_differently(self):
        """两套打分若结果永远一致，说明其中一条根本没被执行。"""
        repo = InMemoryProductRepository()
        lexical = Bm25LexicalIndex()
        lexical.index(await repo.list_all())
        spec = ProductSearchSpec(normalized_query="轻便 结实 户外 照明", top_k=8)

        bm25 = await CatalogSearchUseCase(repo, lexical_index=lexical).execute(spec)
        legacy = await CatalogSearchUseCase(repo).execute(spec)
        assert bm25["recall_strategy"] == "bm25_only"
        assert legacy["recall_strategy"] == "keyword_2gram"
        assert [h["product_id"] for h in bm25["hits"]] != [
            h["product_id"] for h in legacy["hits"]
        ], "IDF + 长度归一必须让 BM25 的排序区别于命中词数打分"


class TestLexicalConfidenceGate:
    """字面路置信度门控：BM25 自己没把握时不参与融合。

    动机来自实测而非直觉：无条件融合下语义类 query 的 MRR 由 0.875 掉到 0.778，
    因为 BM25 在这类 query 上自身 MRR 仅 0.350（约等于噪声），而 RRF 只看名次
    不看置信度，噪声被平权投票混进头部。
    """

    async def test_low_confidence_lexical_is_excluded_and_reported(self, hybrid_ready):
        """描述性 query（库里无字面可匹配）应被判为低置信度，走纯向量并如实标注。"""
        repo, embedder, index, lexical = hybrid_ready
        uc = CatalogSearchUseCase(
            repo, embedder=embedder, vector_index=index, lexical_index=lexical,
            lexical_gate=1e9,  # 门限拉到不可能达到，强制走门控分支
        )
        result = await uc.execute(ProductSearchSpec(normalized_query="露营灯 抗造"))
        assert result["recall_strategy"] == "hybrid_gated_vector", (
            "门控生效时必须如实标注，不能谎报 hybrid_rrf——"
            "recall_strategy 是给模型看的事实，也是评测分辨两条路的依据"
        )

    async def test_gate_zero_restores_unconditional_fusion(self, hybrid_ready):
        """gate=0 即关闭门控，退回无条件融合——消融要有可比的对照组。"""
        repo, embedder, index, lexical = hybrid_ready
        uc = CatalogSearchUseCase(
            repo, embedder=embedder, vector_index=index, lexical_index=lexical, lexical_gate=0.0,
        )
        result = await uc.execute(ProductSearchSpec(normalized_query="露营灯 抗造"))
        assert result["recall_strategy"] == "hybrid_rrf"

    async def test_empty_lexical_result_does_not_crash_gate(self, hybrid_ready):
        """字面路一条都没召回时门控要安全地判低置信度，而不是索引越界。"""
        repo, embedder, index, lexical = hybrid_ready
        uc = CatalogSearchUseCase(
            repo, embedder=embedder, vector_index=index, lexical_index=lexical,
        )
        result = await uc.execute(ProductSearchSpec(normalized_query="zzzz不存在的词zzzz"))
        assert result["recall_strategy"] in ("hybrid_gated_vector", "hybrid_rrf")


class TestLandedPriceCarriesCombineConstraint:
    """到手价必须自带「不可简单相加」的约束说明。

    动机是实测缺陷（评测 compare-two）：系统提示词里已经写了"多件总价必须调
    quote_basket_tool、不得自行相加"，模型照样在写回复时把两个 landed_total 加起来，
    把一次履约的运费算了两次（¥364 + ¥154 = ¥518，正确组合价是 ¥492：
    小计 388 + 运费 104 + 关税 0）。
    结论是**约束离数字越近越管用**——写在系统提示词里隔着几千 token，
    写在它正要读的那个数字旁边才拦得住。同 filtered_out 的设计思路。
    """

    async def test_landed_price_includes_combine_hint(self, hybrid_ready):
        repo, embedder, index, lexical = hybrid_ready
        uc = CatalogSearchUseCase(
            repo, embedder=embedder, vector_index=index, lexical_index=lexical,
        )
        result = await uc.execute(
            ProductSearchSpec(normalized_query="露营灯", ship_to="US", target_currency="USD"),
        )
        card = result["hits"][0]
        assert "landed_price" in card
        hint = card["landed_price"].get("combine_hint", "")
        assert "quote_basket_tool" in hint, "约束必须点名替代工具，否则模型不知道该改调什么"
        assert "相加" in hint, "必须明确禁止相加这个具体动作"
        # 到手价按 quantity=1 算，买 3 件同一商品同样不能用"单价 × 3 + 运费"推——
        # 运费是首件全价 + 续件 60%，乘法推不出来。措辞里必须把这一路也堵上，
        # 否则"多件合并购买"会被读成只指"多个不同商品"。
        assert "1 件" in hint or "单件" in hint, "必须说清这个价只对应 1 件"
        assert "同一商品" in hint, "同一商品买多件也要走 quote_basket_tool"

    async def test_no_hint_when_ship_to_absent(self, hybrid_ready):
        """没传 ship_to 就没有到手价，也就不该有这条提示污染返回值。"""
        repo, embedder, index, lexical = hybrid_ready
        uc = CatalogSearchUseCase(
            repo, embedder=embedder, vector_index=index, lexical_index=lexical,
        )
        result = await uc.execute(ProductSearchSpec(normalized_query="露营灯"))
        assert "landed_price" not in result["hits"][0]
