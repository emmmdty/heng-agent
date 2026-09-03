# -*- coding: utf-8 -*-
"""检索基础设施端口：EmbeddingClient / ProductVectorIndex / Reranker / LexicalIndex

Domain 不关心实现：Infrastructure 提供 OpenAI 兼容 embedding、Qdrant 索引、HTTP reranker、
本地 BM25 字面索引。UseCase 通过这些端口完成召回，任一环节不可用时由 UseCase 负责降级。

两路召回的分工（由 `eval/product_recall.jsonl` 的分类别读数确定，不是拍脑袋）：
    向量路  语义类 query 上不可替代（字面不含商品词时仍能召回）
    字面路  字面类 query 上略胜向量路，且零外部依赖、永远可用
两路互有胜负，因此值得融合而不是二选一；融合收益由消融评测量化。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.domain.catalog.product import Product


class EmbeddingClient(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class VectorHit:
    product_id: str
    score: float


class ProductVectorIndex(ABC):
    @abstractmethod
    async def ensure_ready(self, vector_dim: int) -> None:
        """确保 collection 存在（幂等）。"""

    @abstractmethod
    async def upsert_products(self, products: list[Product], embeddings: list[list[float]]) -> None:
        ...

    @abstractmethod
    async def search(self, embedding: list[float], top_n: int) -> list[VectorHit]:
        ...


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """返回与 documents 等长的精排分数；失败抛异常，由调用方降级。"""


@dataclass(frozen=True)
class LexicalHit:
    product_id: str
    score: float


class LexicalIndex(ABC):
    """字面召回端口（BM25）。

    与 `ProductVectorIndex` 并列而非从属：它不是"向量挂了才用的兜底"，而是
    混合召回里一条独立的、语义正交的通路。降级兜底只是它的附带用途。
    """

    @abstractmethod
    def index(self, products: list[Product]) -> None:
        """建立/重建索引（幂等，全量替换）。"""

    @abstractmethod
    def search(self, query: str, top_n: int) -> list[LexicalHit]:
        ...
