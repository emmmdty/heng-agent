# -*- coding: utf-8 -*-
"""Product 聚合根

Globex 把跨境商品建模为 Product（SPU）+ Sku（多个），携带品牌、产地、亮点等结构化属性。
SearchAgent 召回的"候选集"传递的就是 Product 卡片，TradeAgent 创建订单时再以 Sku 粒度结算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.domain.catalog.sku import Sku


@dataclass(frozen=True)
class ProductHighlight:
    label: str
    detail: str = ""
    # 是否进入字面检索索引。**否定型属性必须置 False**：
    # "无主动降噪 ANC" 写进可检索文本后，BM25 反而会在"主动降噪"这个 query 上
    # 更强地召回它——字面检索不理解否定，看见词就算命中。
    # 实测代价：给半入耳款加了这条说明之后，它反而排进了"主动降噪耳机"的候选表，
    # 模型据此把它描述成"有主动降噪功能"，比不加还糟。
    # 卡片照常展示（模型要靠它做判断），只是不参与匹配。
    searchable: bool = True


@dataclass
class Product:
    product_id: str
    title: str
    brand: str
    category: str
    origin_country: str
    description: str
    highlights: list[ProductHighlight] = field(default_factory=list)
    ships_to: list[str] = field(default_factory=list)
    skus: list[Sku] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.product_id:
            raise ValueError("Product.product_id required")
        if not self.skus:
            raise ValueError(f"Product 至少要有一个 Sku：{self.product_id}")

    def primary_sku(self) -> Sku:
        return self.skus[0]

    def find_sku(self, sku_id: str) -> Optional[Sku]:
        return next((s for s in self.skus if s.sku_id == sku_id), None)

    def searchable_text(self) -> str:
        """召回用的可检索文本：标题 + 品牌 + 品类 + 描述 + 亮点。"""
        highlight_text = " ".join(
            f"{h.label} {h.detail}" for h in self.highlights if h.searchable
        )
        return " ".join(
            [self.title, self.brand, self.category, self.origin_country, self.description, highlight_text],
        )
