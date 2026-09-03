# -*- coding: utf-8 -*-
"""BM25 字面索引（纯本地，零外部依赖）

为什么是 BM25 而不是沿用原有的 `_keyword_score`：
    原实现是"命中词数 + 品类命中加 3 分"，两个缺陷在 60 SPU 商品库上会放大——
      1. 无 IDF：「露营」这种到处都有的词与「记忆棉」这种独有词等权，
         导致长尾区分度差；
      2. 无长度归一：描述写得长的商品天然命中更多词，凭字数取胜。
    BM25 两项都有：IDF 压常见词，b=0.75 的长度归一惩罚长文档。

分词沿用 `catalog_search.tokenize`（空格切词 + CJK 2-gram），刻意不引入 jieba：
    多一个模型依赖换来的收益无法在现有 67 条标注上验证，而 2-gram 对中文商品名
    （品牌 + 品类 + 属性的短串）已经够用。若后续标注扩到 200+ 条且证明分词是瓶颈，
    再换不迟——那时也有判据可用。

索引是进程内内存结构：商品目录目前是种子数据 + 内存仓储（60 SPU），建索引耗时
可忽略。商品目录一旦落库并支持增量变更，这里要换成可增量更新的实现。
"""
from __future__ import annotations

import math
from collections import Counter

from app.domain.catalog.ports.retrieval_ports import LexicalHit, LexicalIndex
from app.domain.catalog.product import Product

# BM25 标准参数：k1 控制词频饱和速度，b 控制长度归一强度。
# 取 Robertson 等人的经典缺省值；商品短文本场景没有调参依据，
# 先用缺省值 + 评测量化，不凭直觉调。
_K1 = 1.2
_B = 0.75


class Bm25LexicalIndex(LexicalIndex):
    def __init__(self, k1: float = _K1, b: float = _B) -> None:
        self._k1 = k1
        self._b = b
        self._doc_terms: dict[str, Counter[str]] = {}
        self._doc_len: dict[str, int] = {}
        self._avg_len: float = 0.0
        self._doc_freq: Counter[str] = Counter()
        self._doc_count: int = 0

    def index(self, products: list[Product]) -> None:
        # 延迟导入避免 application → domain 的反向依赖：分词器住在 UseCase 模块里，
        # 两边共用同一套切词规则才能保证"评测里的字面路"和"线上的字面路"一致。
        from app.application.usecases.catalog_search import tokenize_terms

        self._doc_terms.clear()
        self._doc_len.clear()
        self._doc_freq.clear()

        for product in products:
            terms = tokenize_terms(product.searchable_text())
            counts = Counter(terms)
            self._doc_terms[product.product_id] = counts
            self._doc_len[product.product_id] = sum(counts.values())
            for term in counts:  # DF 按"出现过的文档数"计，不是总词频
                self._doc_freq[term] += 1

        self._doc_count = len(self._doc_terms)
        total_len = sum(self._doc_len.values())
        self._avg_len = (total_len / self._doc_count) if self._doc_count else 0.0

    def search(self, query: str, top_n: int) -> list[LexicalHit]:
        from app.application.usecases.catalog_search import tokenize_terms

        if not self._doc_count:
            return []

        query_terms = set(tokenize_terms(query))
        scored: list[LexicalHit] = []
        for product_id, counts in self._doc_terms.items():
            score = sum(
                self._term_score(term, counts.get(term, 0), self._doc_len[product_id])
                for term in query_terms
            )
            if score > 0.0:
                scored.append(LexicalHit(product_id=product_id, score=score))

        # 同分时按 product_id 排序，保证结果确定可复现（评测与单测都依赖这一点）
        scored.sort(key=lambda hit: (-hit.score, hit.product_id))
        return scored[:top_n]

    def _term_score(self, term: str, tf: int, doc_len: int) -> float:
        if tf == 0:
            return 0.0
        df = self._doc_freq.get(term, 0)
        # 带 +0.5 平滑的 IDF；全库都含的词 idf 趋近 0，但用 max 兜住不让它变负，
        # 否则一个 stop-word 级的词会倒扣掉真实命中的分数。
        idf = max(0.0, math.log(1.0 + (self._doc_count - df + 0.5) / (df + 0.5)))
        norm = 1.0 - self._b + self._b * (doc_len / self._avg_len if self._avg_len else 1.0)
        return idf * (tf * (self._k1 + 1.0)) / (tf + self._k1 * norm)
