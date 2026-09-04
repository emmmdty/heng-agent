# -*- coding: utf-8 -*-
"""CatalogSearchUseCase

商品检索核心 UseCase，对齐参考实现五步流程：
    1. EmbeddingClient 把 normalized_query 向量化
    2. ProductVectorIndex.search(top_n) 拿候选 product_id（Qdrant，COSINE）
    3. ProductRepository.find_by_ids 还原 Product 聚合
    4. Reranker 精排取 top_k；失败/未配置降级按向量分排序（rerank_applied=false）
    5. 组装商品卡 JSON；命中 ship_to 时内联到手价（小计+运费+关税，统一目标币种）

降级链（recall_strategy 如实标注）：
    embedding_rerank → embedding_only → keyword_2gram（embedding 服务异常时兜底）

计价收敛设计：到手价在检索链路内联计算（TariffSchedule 规则内核），
不给 Agent 单独暴露比价/运费工具，减少不必要的工具调用轮次。

过滤可观测：被 ship_to / price_max_major 硬约束挡掉的候选以 filtered_out 摘要回传，
让模型能区分"库里没有这个商品"与"有但不满足约束"，不致于给出误导性结论。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional

from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.catalog.ports.retrieval_ports import (
    EmbeddingClient,
    LexicalIndex,
    ProductVectorIndex,
    Reranker,
)
from app.domain.catalog.product import Product
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.shipping.tariff_schedule import TariffSchedule
from app.infrastructure.transient import describe_error

logger = logging.getLogger(__name__)

# 一阶段向量召回候选数。
#
# 必须 > 业务 top_k，否则精排在数学上不可能提升 Recall@K——它只能重排既有候选，
# 不能凭空补召回。这条曾以一个很隐蔽的形式出现在评测里：`embedding_rerank` 与
# `embedding_only` 的 Recall 逐位相同（.935），看着像"精排没用"，实际是
# 一阶段深度 8 == K 8 把精排的召回贡献封死了；同一份数据上混合档
# （融合深度 16 > K）的精排就实打实抬了 2.2pt 召回。
# 取 16 与 `_FUSION_DEPTH` 对齐，让两条通路给精排的空间一致，档位间才可比。
_RECALL_TOP_N = 16

# 被硬约束挡掉的候选回传条数上限（只回摘要，避免上下文膨胀）
_FILTERED_OUT_LIMIT = 3

# RRF 融合常数：score = Σ 1/(k + rank)。
# k 越大越"温和"（各路排名差异被压平），越小越激进（只有头部几名有话语权）。
# 60 是 Cormack 等人原文的缺省值，本仓把它做成可配 + 可消融，默认值等评测出结论再定。
_DEFAULT_RRF_K = 60

# 融合时每一路各取多少候选进入排名。取得比 _RECALL_TOP_N 大：
# RRF 只看名次，某一路的第 9~12 名有可能因为另一路也投它而被抬进最终 top_k，
# 截得太短会把这部分收益直接砍掉。
_FUSION_DEPTH = 16

# 一阶段档位 → 精排后的档位名。
# 精排是 cross-encoder 打分，对任何候选集都成立，因此三条一阶段通路都可接；
# `keyword_2gram` 不在其中：它是"所有召回基建都挂了"的兜底，此时 reranker
# 大概率也不可用，硬试只会白等一次超时。
_RERANKABLE = {
    "embedding_only": "embedding_rerank",
    "hybrid_rrf": "hybrid_rerank",
    "hybrid_gated_vector": "hybrid_gated_rerank",
    "bm25_only": "bm25_rerank",
}

# 字面路参与融合的最低置信度（BM25 top-1 分数）。
#
# 由 67 条标注实测标定，不是拍的：字面类 query 的 BM25 top-1 分布是
# [3.13, 35.68]（中位 17.94），语义类是 [0.00, 3.96]（中位 3.23，p25 为 0）。
# 两类几乎不重叠，重叠带只有 3.1~4.0。
#
# 为什么需要这道门：RRF 只看名次不看置信度，字面路在语义类 query 上自身
# MRR 只有 0.350（约等于噪声），却和向量路等权投票，把向量路高置信度的
# 第一名挤下去。实测代价是语义类 MRR 0.875 → 0.778。
#
# 取值 4.0 来自门限扫描实测（gate ∈ {0,3,4,5,8,12}，67 条标注）：
#   gate=0（无条件融合）Recall .974 / MRR .948 / NDCG .928，语义类 MRR .778
#   gate=4              Recall .974 / MRR .966 / NDCG .940，语义类 MRR .875
# gate=4 在每一项上都不劣于 gate=0，即"保住字面类的召回增益，同时不牺牲语义类排序"。
#
# 两点如实记录的局限：
#   1. 两类分布**有重叠**（字面类最低 3.13，语义类最高 3.96），完美分离不存在。
#      4.0 会把「行李秤」「颈枕」这类超短字面 query 判为语义类走纯向量——
#      实测无损（字面类 Recall 仍 0.986），但这是运气好，不是设计保证。
#   2. 语义类只有 12 条，在这个样本量上取扫描最优点有过拟合风险。
#      gate=5 的读数几乎相同（.970/.963/.939），选 4 还是 5 并无实质差别；
#      真要拿它做决策，先把语义类标注扩到 50 条以上。
#
# 阈值是**语料相关**的：商品库或分词规则一变就必须用
# `scripts/eval/run_product_recall.py --sweep-lexical-gate` 重新标定。
_LEXICAL_GATE_MIN_SCORE = 4.0


@dataclass(frozen=True)
class ProductCard:
    product_id: str
    title: str
    brand: str
    category: str
    origin_country: str
    price_major: float
    currency: str
    highlights: list[str]
    skus: list[dict]
    score: float
    landed_price: Optional[dict]  # ship_to 命中时的到手价明细，未命中为 None
    # 目标币种折算价：仅当商品原生币种 != 买家口径币种时才有，同币种不重复发
    price_in_target_major: Optional[float] = None
    target_currency: str = ""
    # 买家点名要、而本商品**显式声明不具备**的属性。None = 无冲突（不发这个字段）
    attribute_mismatch: Optional[dict] = None

    def to_dict(self) -> dict:
        card = {
            "product_id": self.product_id,
            "title": self.title,
            "brand": self.brand,
            "category": self.category,
            "origin_country": self.origin_country,
            "price_major": self.price_major,
            "currency": self.currency,
            "highlights": self.highlights,
            "skus": self.skus,
            "score": round(self.score, 4),
        }
        if self.price_in_target_major is not None:
            card["price_in_target_major"] = self.price_in_target_major
            card["target_currency"] = self.target_currency
        if self.landed_price is not None:
            card["landed_price"] = self.landed_price
        if self.attribute_mismatch is not None:
            card["attribute_mismatch"] = self.attribute_mismatch
        return card


def tokenize_terms(text: str) -> list[str]:
    """极简分词：空格切词 + 中文连续段落的 2-gram。

    返回 list 而非 set，因为 BM25 需要词频（tf）。`tokenize()` 保留 set 语义供
    原关键词降级路使用——两条路共用同一套切词规则是刻意的：否则"评测里的字面路"
    与"线上兜底的字面路"会是两种分词，评测结论无法迁移。
    """
    terms: list[str] = []
    for chunk in text.lower().split():
        terms.append(chunk)
        # 对含 CJK 的 chunk 补 2-gram，缓解中文无空格问题
        if any("\u4e00" <= ch <= "\u9fff" for ch in chunk) and len(chunk) > 2:
            # len == 2 时 2-gram 就是 chunk 本身，再补一次会把两字词的 tf 凭空翻倍，
            # BM25 会因此系统性偏袒两字词。set 语义的 tokenize() 看不见这个问题。
            terms.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return terms


def tokenize(text: str) -> set[str]:
    """极简分词的去重视图（关键词降级召回用）。"""
    return set(tokenize_terms(text))


class CatalogSearchUseCase:
    def __init__(
        self,
        product_repo: ProductRepository,
        embedder: Optional[EmbeddingClient] = None,
        vector_index: Optional[ProductVectorIndex] = None,
        reranker: Optional[Reranker] = None,
        tariff_schedule: Optional[TariffSchedule] = None,
        lexical_index: Optional[LexicalIndex] = None,
        rrf_k: int = _DEFAULT_RRF_K,
        lexical_gate: float = _LEXICAL_GATE_MIN_SCORE,
    ) -> None:
        self._product_repo = product_repo
        self._embedder = embedder
        self._vector_index = vector_index
        self._reranker = reranker
        self._tariff = tariff_schedule or TariffSchedule(rates=ExchangeRateTable())
        # lexical_index 为 None 即退回四期行为（纯向量 + 关键词兜底），
        # 混合召回是加法而不是改写：既有降级链一行未动。
        self._lexical_index = lexical_index
        self._rrf_k = rrf_k
        # 0 = 关闭门控（退回无条件融合），用于消融对照
        self._lexical_gate = lexical_gate

    @property
    def lexical_gate(self) -> float:
        """实际生效的字面门限。对外只读，供 /health 自报配置——
        阈值是语料相关的可调量，报告里不记它，跨轮次的读数就无法归因。"""
        return self._lexical_gate

    async def execute(self, spec: ProductSearchSpec) -> dict:
        scored: list[tuple[float, Product]] = []
        recall_strategy = "keyword_2gram"
        rerank_applied = False

        vector_ready = self._embedder is not None and self._vector_index is not None
        if vector_ready or self._lexical_index is not None:
            try:
                if vector_ready and self._lexical_index is not None:
                    # 混合召回：两路各自出名次 → RRF 融合。
                    # 向量路失败时整体降级到纯字面路（而不是整轮失败），
                    # 因为字面路是纯本地的，永远可用。
                    scored, recall_strategy = await self._hybrid_recall(spec)
                elif vector_ready:
                    scored = await self._vector_recall(spec)
                    recall_strategy = "embedding_only"
                else:
                    # 只注入了字面索引：BM25 独立成档，而不是掉回旧的 keyword_2gram。
                    # 这条路是混合档的对照基线，必须真的走 BM25，否则消融对比的
                    # 「字面基线」测的是另一套打分（曾因此让两档跑出完全相同的数字）。
                    scored = await self._bm25_recall(spec)
                    recall_strategy = "bm25_only"
            except Exception as err:  # noqa: BLE001 —— 召回基建异常必须降级而非失败
                logger.warning("向量召回不可用，降级关键词召回：%s", describe_error(err))
                scored = []
                recall_strategy = "keyword_2gram"

        if scored and recall_strategy in _RERANKABLE:
            # 二阶段精排；失败降级为按一阶段召回分排序
            try:
                scored = await self._rerank(spec, scored)
                recall_strategy = _RERANKABLE[recall_strategy]
                rerank_applied = True
            except Exception as err:  # noqa: BLE001
                logger.warning("rerank 不可用，按一阶段召回分排序：%s", describe_error(err))
        elif not scored:
            scored = await self._keyword_recall(spec)
            recall_strategy = "keyword_2gram"

        # 规则表不认识的目的国（买家说"欧盟"，模型填了 DE/FR）：**不能据此过滤商品**。
        # 系统没有德国的计价规则，不代表这些箱子不发德国——按 ships_to 逐个标
        # ship_to_unavailable 会给出一句事实错误的结论（"所有 TrailOx 都不发欧盟"），
        # 而模型没有任何办法识破它，只能放弃到手价、凭自己的知识硬答免税额度。
        # 正确做法是把这个目的国当作没传（照常返回商品、不内联到手价），
        # 并在顶层如实说明 + 给出支持列表，让模型能自己改填重试。
        unsupported_ship_to: Optional[str] = None
        if spec.ship_to and spec.ship_to not in self._tariff.supported_destinations():
            unsupported_ship_to = spec.ship_to
            spec = replace(spec, ship_to=None)

        # ship_to / 价格硬约束过滤 + top_k 截断（硬约束走结构化过滤，不交给模型）
        filtered: list[tuple[float, Product]] = []
        filtered_out: list[dict] = []
        for score, product in scored:
            reason = self._reject_reason(product, spec)
            if reason is None:
                filtered.append((score, product))
            elif len(filtered_out) < _FILTERED_OUT_LIMIT:
                filtered_out.append(self._to_rejected(product, spec, reason))

        hits = [self._to_card(score, product, spec) for score, product in filtered[: spec.top_k]]
        result = {
            "hits": [card.to_dict() for card in hits],
            "total_candidates": len(filtered),
            "recall_strategy": recall_strategy,
            "rerank_applied": rerank_applied,
        }
        if filtered_out:
            # 如实告知"召回到了但被硬约束挡掉"，否则模型分不清"库里没有"与"被过滤"，
            # 会把超预算商品答成"没有这个商品"
            result["filtered_out"] = filtered_out
        if unsupported_ship_to is not None:
            # 支持列表必须一起回：只说"不支持"，模型无从知道该改填什么，
            # 实测它会接着猜（DE 猜完猜 FR），猜不中就凭自身知识作答。
            # 同 combine_hint 的原则——工具返回值要能自证边界。
            result["ship_to_unsupported"] = {
                "given": unsupported_ship_to,
                "supported": self._tariff.supported_destinations(),
                "note": (
                    f"计价规则表不支持目的国 {unsupported_ship_to}，本次未按目的国过滤、"
                    f"也未计算到手价（商品信息本身照常返回）。"
                    f"若买家说的是欧盟/日本/新加坡等，请改用 supported 里的代码重试；"
                    f"确实不在支持范围内时，如实告知买家无法计算到手价，不要自行估算运费或关税。"
                ),
            }
        return result

    def _reject_reason(self, product: Product, spec: ProductSearchSpec) -> Optional[str]:
        """返回硬约束拒绝原因，None 表示通过。"""
        if spec.ship_to and spec.ship_to not in product.ships_to:
            return "ship_to_unavailable"
        if not self._within_price_cap(product, spec):
            return "over_price_cap"
        return None

    def _to_rejected(self, product: Product, spec: ProductSearchSpec, reason: str) -> dict:
        primary_in_target = self._tariff.rates.convert(product.primary_sku().price, spec.target_currency)
        return {
            "product_id": product.product_id,
            "title": product.title,
            "category": product.category,
            "price_major": round(primary_in_target.to_major_units(), 2),
            "currency": spec.target_currency,
            "reason": reason,
        }

    def _within_price_cap(self, product: Product, spec: ProductSearchSpec) -> bool:
        """按**最便宜的规格**判，不是按主规格。

        按主规格判会把"主规格贵、另有便宜规格"的商品整件挡掉，
        模型看到 `over_price_cap`，于是把"库里有你买得起的规格"
        答成"没有符合预算的商品"。

        今天的商品库里还碰不到（只有两件商品规格间有差价，且主规格恰好最便宜），
        但那是数据的偶然——加一条数据就会突然出事，而那时没有任何判据会响。
        商品卡本来就逐个列出每个 SKU 的价格与到手价，模型挑得出符合预算的那个。
        """
        if spec.price_max_major is None:
            return True
        cheapest = min(
            self._tariff.rates.convert(sku.price, spec.target_currency).to_major_units()
            for sku in product.skus
        )
        return cheapest <= spec.price_max_major

    # ---- 一阶段（混合）：两路召回 + RRF 融合 ----

    async def _hybrid_recall(
        self, spec: ProductSearchSpec,
    ) -> tuple[list[tuple[float, Product]], str]:
        """向量路 + BM25 字面路，按 RRF 融合。

        返回 (融合结果, 实际档位)。向量路失败时返回纯字面路结果并如实标注档位——
        `recall_strategy` 是给模型看的事实，不能因为"我们有兜底"就谎报成混合档。

        为什么用 RRF 而不是加权分数融合：两路分数不同量纲（cosine 有界 [0,1]，
        BM25 无上界且随语料变化），要相加必须先归一化，而归一化系数得按 query
        逐条标定，标定不准反而引入新的偏置。RRF 只用名次，天然免疫量纲问题。
        代价是丢掉了分数的"置信度"信息：向量路以 0.95 的高分锁定的第一名，
        与它以 0.31 勉强排出的第一名，在 RRF 眼里一样重。这个代价是否划算，
        由消融评测回答（scripts/eval/run_product_recall.py --compare-strategies）。
        """
        lexical_hits = self._lexical_index.search(spec.normalized_query, top_n=_FUSION_DEPTH)
        lexical_ranking = [hit.product_id for hit in lexical_hits]
        # 字面路的自评置信度：top-1 的 BM25 分数。低于门限说明这条 query 在库里
        # 没有字面可匹配的东西（典型是"睡觉怕光想把眼睛遮起来"这类描述性问法），
        # 此时它的排名是噪声，让它参与投票只会污染向量路的结果。
        lexical_confident = bool(lexical_hits) and lexical_hits[0].score >= self._lexical_gate

        try:
            vector_hits = await self._vector_search_ids(spec, top_n=_FUSION_DEPTH)
        except Exception as err:  # noqa: BLE001
            logger.warning("混合召回的向量路不可用，本轮退化为纯 BM25：%s", describe_error(err))
            fused_ids = lexical_ranking
            strategy = "bm25_only"
        else:
            if lexical_confident:
                fused_ids = self._rrf_fuse([vector_hits, lexical_ranking])
                strategy = "hybrid_rrf"
            else:
                # 如实标注成"混合链路判定后走了纯向量"，而不是谎报 hybrid_rrf：
                # recall_strategy 是给模型看的事实，也是评测能分辨两条路的依据。
                fused_ids = vector_hits
                strategy = "hybrid_gated_vector"

        products = await self._product_repo.find_by_ids(fused_ids)
        by_id = {product.product_id: product for product in products}
        # 融合分用「名次倒数」本身，保持与排序一致；商品卡上的 score 因此是可比的相对量
        scored = [
            (1.0 / (self._rrf_k + rank), by_id[pid])
            for rank, pid in enumerate(fused_ids, start=1)
            if pid in by_id
        ]
        return scored, strategy

    async def _bm25_recall(self, spec: ProductSearchSpec) -> list[tuple[float, Product]]:
        """纯 BM25 召回（无向量路时的独立档位）。"""
        hits = self._lexical_index.search(spec.normalized_query, top_n=_FUSION_DEPTH)
        products = await self._product_repo.find_by_ids([hit.product_id for hit in hits])
        by_id = {product.product_id: product for product in products}
        return [(hit.score, by_id[hit.product_id]) for hit in hits if hit.product_id in by_id]

    def _rrf_fuse(self, rankings: list[list[str]]) -> list[str]:
        """Reciprocal Rank Fusion：score(d) = Σ_r 1/(k + rank_r(d))，只出现在部分路里也计分。

        同分按 product_id 排序保证确定性——否则同一份标注集两次跑出不同 MRR，
        评测就失去了作为判据的资格。
        """
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, product_id in enumerate(ranking, start=1):
                scores[product_id] = scores.get(product_id, 0.0) + 1.0 / (self._rrf_k + rank)
        return [pid for pid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]

    async def _vector_search_ids(self, spec: ProductSearchSpec, top_n: int) -> list[str]:
        embedding = await self._embedder.embed(spec.normalized_query)
        hits = await self._vector_index.search(embedding, top_n=top_n)
        return [hit.product_id for hit in hits]

    # ---- 一阶段：向量召回 ----

    async def _vector_recall(self, spec: ProductSearchSpec) -> list[tuple[float, Product]]:
        embedding = await self._embedder.embed(spec.normalized_query)
        vector_hits = await self._vector_index.search(embedding, top_n=_RECALL_TOP_N)
        products = await self._product_repo.find_by_ids([hit.product_id for hit in vector_hits])
        by_id = {product.product_id: product for product in products}
        return [
            (hit.score, by_id[hit.product_id])
            for hit in vector_hits
            if hit.product_id in by_id
        ]

    # ---- 二阶段：精排 ----

    async def _rerank(
        self,
        spec: ProductSearchSpec,
        scored: list[tuple[float, Product]],
    ) -> list[tuple[float, Product]]:
        if self._reranker is None:
            raise RuntimeError("Reranker 未配置")
        documents = [product.searchable_text() for _, product in scored]
        rerank_scores = await self._reranker.rerank(spec.normalized_query, documents)
        reranked = [
            (rerank_scores[i], product)
            for i, (_, product) in enumerate(scored)
        ]
        reranked.sort(key=lambda pair: pair[0], reverse=True)
        return reranked

    # ---- 兜底：关键词召回 ----

    async def _keyword_recall(self, spec: ProductSearchSpec) -> list[tuple[float, Product]]:
        query_terms = tokenize(spec.normalized_query)
        candidates: list[tuple[float, Product]] = []
        for product in await self._product_repo.list_all():
            score = self._keyword_score(query_terms, product, spec)
            if score > 0:
                candidates.append((score, product))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return candidates

    @staticmethod
    def _keyword_score(query_terms: set[str], product: Product, spec: ProductSearchSpec) -> float:
        doc_terms = tokenize(product.searchable_text())
        matched = query_terms & doc_terms
        if not matched:
            return 0.0
        score = float(len(matched))
        # 品类槽位命中加权，让"槽位过滤"优于全文命中
        if spec.category and spec.category in product.category:
            score += 3.0
        return score

    # ---- 商品卡组装（含到手价内联）----

    def _to_sku_dict(self, sku, product: Product, spec: ProductSearchSpec) -> dict:
        card_sku = {
            "sku_id": sku.sku_id,
            "spec": sku.spec,
            "price_major": sku.price.to_major_units(),
            "currency": sku.price.currency,
            "stock": sku.stock,
        }
        # 逐 SKU 折算价：与卡级同一个理由，非主 SKU 的折算同样不能交给模型。
        # 同币种时不发，避免把一模一样的数字重复一遍。
        if sku.price.currency != spec.target_currency:
            card_sku["price_in_target_major"] = round(
                self._tariff.rates.convert(sku.price, spec.target_currency).to_major_units(), 2,
            )
            card_sku["target_currency"] = spec.target_currency
        landed = self._sku_landed_price(sku, product, spec)
        if landed is not None:
            card_sku["landed_price"] = landed
        return card_sku

    def _sku_landed_price(self, sku, product: Product, spec: ProductSearchSpec) -> Optional[dict]:
        """单个 SKU 的到手价（只留金额分项，不重复 ship_to/税率等卡级信息）。

        为什么每个 SKU 都要算：商品卡原先只给主 SKU 的 landed_price，买家问
        "月光白多少钱"时，非主 SKU 的到手价没有任何工具出处，模型只能自己
        拿 229 USD × 汇率 + 运费凑（实测 eval-compare-two / eval-landed-price-us
        四轮里都出现了这个 $238.15）。数字碰巧对，但推导过程是模型做的：
        一旦该 SKU 的小计跨过免税额度，或与主 SKU 不同品类费率，就会算错。

        只带金额四项（小计/运费/关税/合计）+ 币种：ship_to、tariff_rate、
        de_minimis_applied 与主 SKU 完全一致，逐 SKU 重复一遍只会白涨上下文。
        """
        if not spec.ship_to:
            return None
        try:
            quote = self._tariff.quote(
                subtotal=sku.price,
                category=product.category,
                ship_to=spec.ship_to,
                quantity=1,
                target_currency=spec.target_currency,
            )
        except ValueError:
            return None  # 目的国不支持时卡级 landed_price 已如实标注原因，此处不重复
        detail = quote.to_dict()
        return {
            key: detail[key]
            for key in ("subtotal_major", "freight_major", "tariff_major", "landed_total_major", "currency")
        }

    def _to_card(self, score: float, product: Product, spec: ProductSearchSpec) -> ProductCard:
        primary = product.primary_sku()
        landed_price: Optional[dict] = None
        if spec.ship_to:
            try:
                quote = self._tariff.quote(
                    subtotal=primary.price,
                    category=product.category,
                    ship_to=spec.ship_to,
                    quantity=1,
                    target_currency=spec.target_currency,
                )
                landed_price = quote.to_dict()
                # 把「不可简单相加」这条约束贴在**数字旁边**，而不是只写在系统提示词里。
                # 实测：提示词里写了"必须调 quote_basket_tool 不得自行相加"，模型照样
                # 在写回复时把两个 landed_total 加起来（把一次履约的运费算了两次）。
                # 约束离数字越近越管用——同 filtered_out 的思路：工具返回值要能自证边界。
                landed_price["combine_hint"] = (
                    "本价格对应「本商品单独下单、数量 1 件」。任何其他组合的总价"
                    "——多个不同商品一起买、或同一商品买多件——都必须调 "
                    "quote_basket_tool 取得：不能把各商品的 landed_total_major 相加，"
                    "也不能用单件到手价乘以件数。运费按一次履约计"
                    "（首件全价 + 每件续件 60%），相加或相乘都会把运费算错"
                )
            except ValueError as err:
                # 目的国不在规则表内：如实标注，不编造数字
                landed_price = {"unavailable_reason": str(err)}
        return ProductCard(
            product_id=product.product_id,
            title=product.title,
            brand=product.brand,
            category=product.category,
            origin_country=product.origin_country,
            price_major=primary.price.to_major_units(),
            currency=primary.price.currency,
            highlights=[f"{h.label}：{h.detail}" if h.detail else h.label for h in product.highlights],
            skus=[
                self._to_sku_dict(sku, product, spec)
                for sku in product.skus
            ],
            score=score,
            landed_price=landed_price,
            # 原生币种与买家口径不一致时，折算价必须由这里给出。
            # 不给的话模型会自己乘汇率——实测折错过（149 USD 写成"约 ¥1080"，
            # 正确 ¥1057.9，差 2%）。到手价那条缝已补，这是同一条缝的另一头。
            price_in_target_major=(
                round(self._tariff.rates.convert(primary.price, spec.target_currency).to_major_units(), 2)
                if primary.price.currency != spec.target_currency
                else None
            ),
            target_currency=spec.target_currency,
            attribute_mismatch=self._attribute_mismatch(product, spec),
        )

    @staticmethod
    def _attribute_mismatch(product: Product, spec: ProductSearchSpec) -> Optional[dict]:
        """买家点名要、而本商品显式声明不具备的属性——**结构化地**说出来。

        来源（二十期整轮实测 `conflict-budget-spec`）：买家要"顶配的主动降噪耳机"，
        模型把只有通话降噪的半入耳款列在"库里有的主动降噪耳机"标题下。
        卡片上那句"仅通话降噪（麦克风侧），无主动降噪 ANC"**当时就在**——
        它是散文，模型可以不当回事。

        前两次修都在动召回，都没拦住（写进 description 反而让 BM25 召回更强；
        改成不可检索的 highlight 压住了字面路，向量路照样召回）。
        这一次动的是返回结构，理由是四次成功先例的共同点——`filtered_out`、
        `combine_hint`、`ship_to_unsupported`、`taxable_base_major` 给的都是
        **结构化字段**。同一条老纪律的第五次应用：缺失的信息要显式化，
        而且要显式成模型没法忽略的形状。

        note 里带上"该怎么办"而不只是"是什么"：只说"不具备"，模型仍可能把它
        当成一个可以商量的次优选项——十期那次它拿着一句"这些商品不发欧盟"
        直接告诉了买家（工具的错误信息要能让模型自纠）。
        """
        missing = product.missing_attributes_for(spec.normalized_query)
        if not missing:
            return None
        joined = "、".join(missing)
        return {
            "missing": missing,
            "note": (
                f"买家点名要的「{joined}」，本商品**明确不具备**。"
                f"不得把它列为具备该属性的候选，也不得暗示加价/换规格就能获得；"
                f"要推荐它，必须同时说明它不具备{joined}。"
            ),
        }
