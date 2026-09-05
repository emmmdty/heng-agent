# -*- coding: utf-8 -*-
"""Product 聚合根

「衡 · Heng」把跨境商品建模为 Product（SPU）+ Sku（多个），携带品牌、产地、亮点等结构化属性。
SearchAgent 召回的"候选集"传递的就是 Product 卡片，TradeAgent 创建订单时再以 Sku 粒度结算。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.domain.catalog.sku import Sku


def _mentions(text: str, alias: str) -> bool:
    """query 里有没有点名这个属性。

    纯 ASCII 别名按词边界匹配：`ANC` 直接做子串会命中 balanced / advanced 之类，
    一个错的告警比没有告警更坏（经验 9）。中文没有词边界，按子串。
    """
    alias = alias.lower().strip()
    if not alias:
        return False
    if alias.isascii():
        return re.search(rf"\b{re.escape(alias)}\b", text) is not None
    return alias in text


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
    # 这条亮点**显式否定**了哪些属性，以及它们的别名（首个为规范名）。
    # 例：("主动降噪", "ANC")。
    #
    # 为什么要一个结构化字段，而不是让检索侧去解析"无主动降噪 ANC"这句散文：
    # 散文的解析规则与文案会脱钩，而脱钩的外观是"判据没生效"——本仓栽过的
    # docstring 与规则表脱钩（十期）是同一形状。声明挂在写着那句否定的亮点上，
    # 而不是商品级的另一份清单里，理由也是这条：两份清单迟早对不上。
    negates: tuple[str, ...] = ()


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

    def absent_attributes(self) -> list[str]:
        """本商品**显式声明**不具备的属性（规范名，按亮点顺序）。

        只认显式声明，不做"可检索文本里没这个词 = 不具备"的推断：
        召回没命中某个词不代表商品缺这个属性，那样推会把整张候选表标满假警报，
        而模型对反复出现的无效告警会学会忽略——连带把真正有依据的那次一起忽略。
        """
        return [h.negates[0] for h in self.highlights if h.negates]

    def missing_attributes_for(self, query: str) -> list[str]:
        """query 里点名要、而本商品显式声明不具备的属性。

        买家没点名要就不报：无关的告警是噪声（同 order_provenance 那条
        "没有依据就不提醒"）。
        """
        text = (query or "").lower()
        return [
            h.negates[0]
            for h in self.highlights
            if h.negates and any(_mentions(text, alias) for alias in h.negates)
        ]

    def searchable_text(self) -> str:
        """召回用的可检索文本：标题 + 品牌 + 品类 + 描述 + 亮点。"""
        highlight_text = " ".join(
            f"{h.label} {h.detail}" for h in self.highlights if h.searchable
        )
        return " ".join(
            [self.title, self.brand, self.category, self.origin_country, self.description, highlight_text],
        )
