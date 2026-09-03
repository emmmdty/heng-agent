# -*- coding: utf-8 -*-
"""order_provenance —— 下单参数的出处校验（写路径）

判据一句话：**下单的每一个商品，都必须在本会话的工具返回里出现过。**

这是金额出处校验（`number_provenance.py`）在写路径上的同一条缝，
而后果更重：回复里的数字错了买家看得出来，订单错了**库存已经扣了**。

现有四道防护都挡不住它：

    仓储查找         挡编造的 id，挡不住真实存在但买家从没看过的 SKU
    Sequencing 断言   挡"完全没检索就下单"，挡不住"检索了 A、下单下成 B"
    幂等键           挡同一句话重复提交，挡不住换个说法再下一单
    权限白名单        挡工具层越权；确认卡内容与下单参数是否一致没有任何代码在校验

同品牌变体互串在本仓是**已知**的失败形态——`scripts/eval/audit_cases.py` 整个脚本
就是为它写的，`conflict-budget-spec` 用例抓的也是"悄悄把 Lite 当降噪推给你"。

**范围刻意收窄，方向一律取"宁可漏报不误报"**（与金额出处校验同一条纪律）：

    1. `product_id` 硬拒，`sku_id` 只警告。`filtered_out` 与 quote/optimize
       两个工具的返回里都**没有** sku_id（实测 filtered_out 只有
       product_id/title/category/price/reason），按 sku 硬拒会把
       "算了就买那个超预算的"这种合法流程当场拒掉。商品级替换才是要抓的东西。
    2. 数量与地址不校验：它们来自买家原话，工具返回里没有出处，
       硬要校验只能靠猜，而猜错的代价是拒掉合法订单。
    3. `filtered_out` 里的候选**算出处**：它确实被工具返回过、被模型看到过。
       出处校验管的是"这商品是不是模型自己想出来的"，不是"它满不满足约束"。
    4. 报错文本不算出处：`[error] 商品不存在：P9999` 里的 P9999
       恰恰是**不该**被当成合法候选的那个。所以只从能解析成 JSON 的返回里抽。
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.application.harness.assertions import AssertionOutcome

# 工具返回里"这个字段装的是商品标识"的键名。按精确键名匹配而不是正则捞 `P\\d{4}`：
# 正则会把 highlights 里的型号文字、地址里的邮编一并捞进来，
# 而出处集合一旦被污染，这条判据就等于永远通过。
_PRODUCT_KEY = "product_id"
_SKU_KEY = "sku_id"


def _walk(node: Any, products: set[str], skus: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == _PRODUCT_KEY and isinstance(value, str):
                products.add(value)
            elif key == _SKU_KEY and isinstance(value, str):
                skus.add(value)
            else:
                _walk(value, products, skus)
    elif isinstance(node, list):
        for value in node:
            _walk(value, products, skus)


def extract_identifiers(text: str) -> tuple[set[str], set[str]]:
    """从一份工具返回里抽出商品与 SKU 标识。

    解析不了 JSON 就返回空集——报错文本（`[error] ...`）走的正是这条路，
    它里面的 id 不该成为出处。
    """
    products: set[str] = set()
    skus: set[str] = set()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return products, skus
    _walk(payload, products, skus)
    return products, skus


def _normalize_items(items: Any) -> list[tuple[str, str, int]]:
    """把下单入参整理成 (product_id, sku_id, quantity) 三元组。

    形状不对的元素直接跳过而不是抛异常：校验器自己抛会把一次参数错误
    升级成整轮失败，而工具本身有它自己的报错路径。
    """
    normalized: list[tuple[str, str, int]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        if not isinstance(item, dict) or _PRODUCT_KEY not in item:
            continue
        try:
            quantity = int(item.get("quantity", 1) or 1)
        except (TypeError, ValueError):
            quantity = 1
        normalized.append((str(item[_PRODUCT_KEY]), str(item.get(_SKU_KEY, "")), quantity))
    return normalized


def _fingerprint(items: list[tuple[str, str, int]]) -> tuple:
    """一单的指纹：排序后的三元组。

    排序是为了让"同样两件商品换个排列顺序"仍判为同一单；
    带上 quantity 是刻意的——数量不同说明买家在追加而不是重复提交，
    按精确匹配判，宁可漏报。
    """
    return tuple(sorted(items))


@dataclass
class OrderProvenanceTracker:
    """按会话累积出处，并记录本会话已创建的订单。

    与 `SequencingTracker` 同一形态（按 shopping_session_id 分桶）：
    并发多会话若共用一份记录，A 会话检索过的商品会成为 B 会话的出处，
    出处校验当场失效。
    """

    _products: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _skus: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _orders: dict[str, list[tuple[tuple, str]]] = field(
        default_factory=lambda: defaultdict(list),
    )

    def record_result(self, session_id: str, text: str) -> None:
        products, skus = extract_identifiers(text)
        if products:
            self._products[session_id] |= products
        if skus:
            self._skus[session_id] |= skus

    def record_order(self, session_id: str, items: Any, order_id: str) -> None:
        normalized = _normalize_items(items)
        if normalized:
            self._orders[session_id].append((_fingerprint(normalized), order_id))

    def reset(self, session_id: str) -> None:
        self._products.pop(session_id, None)
        self._skus.pop(session_id, None)
        self._orders.pop(session_id, None)

    def check(self, session_id: str, items: Any) -> AssertionOutcome:
        outcome = AssertionOutcome()
        normalized = _normalize_items(items)
        if not normalized:
            return outcome

        seen_products = self._products.get(session_id, set())
        if not seen_products:
            # 没有任何观测记录：会话可能是从 AgentState 快照恢复的（进程重启后
            # 内存里的记录为空）。此时硬拒会误杀合法下单——沿用 Sequencing 断言
            # 那条「有证据才硬拒」的纪律。
            outcome.warnings.append(
                "注意：本会话没有可用的检索记录，无法核对下单商品的出处。"
                "请确认这些商品确实来自本轮检索结果，而不是凭印象填写的。",
            )
            return outcome

        unknown_products = sorted(
            {product_id for product_id, _, _ in normalized if product_id not in seen_products},
        )
        if unknown_products:
            outcome.reject_reason = (
                f"拒绝下单：{'、'.join(unknown_products)} 没有在本会话的工具返回里出现过。"
                f"订单商品必须来自 product_search_tool 的检索结果，不能凭印象填写。"
                f"请先用 product_search_tool 检索这些商品，确认 product_id 与 sku_id 后再下单。"
            )
            return outcome

        seen_skus = self._skus.get(session_id, set())
        unknown_skus = sorted(
            {sku_id for _, sku_id, _ in normalized if sku_id and sku_id not in seen_skus},
        )
        if unknown_skus:
            # 只警告：部分工具返回（filtered_out、quote_basket、optimize_basket）
            # 本来就不带 sku_id，硬拒会误杀合法流程
            outcome.warnings.append(
                f"注意：{'、'.join(unknown_skus)} 未在本会话的工具返回里出现过。"
                f"请确认规格选对了（同一商品的不同规格 sku_id 不同）。",
            )

        duplicate = self._find_duplicate(session_id, normalized)
        if duplicate:
            outcome.warnings.append(
                f"注意：本会话已为完全相同的商品与数量创建过订单 {duplicate}。"
                f"请先向买家确认这是不是重复下单，确认要再买一单再继续。",
            )
        return outcome

    def _find_duplicate(
        self, session_id: str, normalized: list[tuple[str, str, int]],
    ) -> str:
        fingerprint = _fingerprint(normalized)
        for existing, order_id in self._orders.get(session_id, []):
            if existing == fingerprint:
                return order_id
        return ""
