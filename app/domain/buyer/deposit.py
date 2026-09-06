# -*- coding: utf-8 -*-
"""MemoryDeposit 值对象 + 确定性验证器（#12 任务 B，M0-c）。

与 BuyerPreference 的关系：偏好是**服务侧**记忆的注入单元（一句话陈述），
沉淀是**评测侧**记忆的记账单元——它把"这条偏好改变了回放里的哪个行为"
钉成可机检的断言。没有沉淀的偏好照样能注入，但它对行为的影响不可对账；
不可对账的"自进化"等于不可信的进化（任务 B 范围约束）。

验证器是纯文本判定、零 LLM：对照两份回放 transcript（注入开/关）断言
行为差异。assertion 字段是人读的断言描述，verifier_spec 是它的机检形态
——两者都必须有，只有 assertion 没有 verifier_spec 的条目写入时被拒。
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from app.domain.buyer.preference import VALID_KINDS

# 影响的行为面：封闭枚举。开放字符串会让"影响的行为面"退化成自由文本，
# 沉淀也就退化回它被禁止成为的东西。
BEHAVIOR_SURFACES = ("recommendation", "order", "clarification", "price_quote")

_REQUIRED_TEXT_FIELDS = (
    "buyer_id", "statement", "trigger_session_id", "trigger_query",
    "precondition", "assertion",
)


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    detail: str  # 过了什么 / 为什么没过——失败必须留名（二十三期纪律）


class DepositVerifier(ABC):
    """确定性验证器端口：对照注入开/关两份 transcript 断言行为差异。"""

    @abstractmethod
    def check(self, transcript_on: str, transcript_off: str) -> VerificationResult:
        ...


def _contains(transcript: str, needle: str) -> bool:
    return needle in transcript


@dataclass(frozen=True)
class ProductPresenceVerifier(DepositVerifier):
    """商品在/不在推荐结果里——偏好遵从率的确定性判定（结果口径，非声称口径）。

    require_contrast=True 时还要求两臂在该商品上**不同**：两臂都含（或都不含）
    说明注入没有改变这个行为面，这条沉淀就回答不了"它改变了哪个行为"。
    """

    product: str
    expect_on: bool  # True=注入开后应出现；False=注入开后不得出现
    require_contrast: bool = True

    def check(self, transcript_on: str, transcript_off: str) -> VerificationResult:
        on_has = _contains(transcript_on, self.product)
        off_has = _contains(transcript_off, self.product)
        if on_has != self.expect_on:
            return VerificationResult(
                ok=False,
                detail=(
                    f"注入开{'未包含' if self.expect_on else '仍包含'}「{self.product}」——"
                    f"断言要求注入开后{'出现' if self.expect_on else '不出现'}"
                ),
            )
        if self.require_contrast and on_has == off_has:
            return VerificationResult(
                ok=False,
                detail=(
                    f"两臂{'都包含' if on_has else '都不包含'}「{self.product}」——"
                    "注入没有改变该行为，无对比不作数"
                ),
            )
        return VerificationResult(
            ok=True,
            detail=f"注入开{'包含' if on_has else '不含'}「{self.product}」、注入关{'包含' if off_has else '不含'}",
        )


@dataclass(frozen=True)
class PreferenceMentionVerifier(DepositVerifier):
    """注入开后回复中体现偏好（任一关键词命中即可）、注入关不体现。

    两臂都提到 = 偏好从对话上下文等其他渠道泄漏，注入没有产生差异。"""

    keywords: tuple[str, ...]

    def check(self, transcript_on: str, transcript_off: str) -> VerificationResult:
        on_hit = next((k for k in self.keywords if _contains(transcript_on, k)), None)
        off_hit = next((k for k in self.keywords if _contains(transcript_off, k)), None)
        if on_hit is None:
            return VerificationResult(
                ok=False,
                detail=f"注入开的回复未体现偏好（关键词 {'、'.join(self.keywords)} 均未命中）",
            )
        if off_hit is not None:
            return VerificationResult(
                ok=False,
                detail=f"注入关的回复也提到了偏好（命中「{off_hit}」）——注入未产生差异",
            )
        return VerificationResult(ok=True, detail=f"仅注入开体现偏好（命中「{on_hit}」）")


_VERIFIER_KINDS = {
    "product_presence": lambda spec: ProductPresenceVerifier(
        product=_non_empty_text(spec["product"], "product"),
        expect_on=bool(spec["expect_on"]),
        require_contrast=bool(spec.get("require_contrast", True)),
    ),
    "preference_mention": lambda spec: PreferenceMentionVerifier(
        keywords=_non_empty_keywords(spec["keywords"]),
    ),
}


def _non_empty_text(value, name: str) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} 不能为空——空匹配串对任何回复都恒命中，是假验证")
    return token


def _non_empty_keywords(keywords) -> tuple[str, ...]:
    tokens = tuple(str(item).strip() for item in keywords if str(item).strip())
    if not tokens:
        raise ValueError("keywords 不能为空——空关键词的验证器对任何回复都判定成功，是假验证")
    return tokens


def build_verifier(spec: dict) -> DepositVerifier:
    """verifier_spec → 验证器。构造不出来的 spec 在这里报错（写入门的闸芯）。"""
    kind = spec.get("kind") or ""
    if kind not in _VERIFIER_KINDS:
        raise ValueError(
            f"未知验证器 kind：{kind!r}（可用：{'、'.join(sorted(_VERIFIER_KINDS))}）"
            "——没有确定性验证器的沉淀不可信，不许写入"
        )
    builder = _VERIFIER_KINDS[kind]
    try:
        return builder(spec)
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(
            f"验证器 spec 不完整或非法（kind={kind!r}）：{err}——不可验证 = 不许写入"
        ) from err


def _deposit_id(buyer_id: str, statement: str, trigger_session_id: str) -> str:
    payload = f"{buyer_id}|{statement}|{trigger_session_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class MemoryDeposit:
    """一条结构化沉淀：能回答"回放里它改变了哪个行为"，否则不许存在。"""

    buyer_id: str
    kind: str  # like / dislike（与 BuyerPreference 对齐）
    statement: str  # 偏好断言，与注入链的 BuyerPreference.statement 同文
    trigger_session_id: str  # 触发会话
    trigger_query: str  # 触发轮的买家原话
    behavior_surface: str  # 影响的行为面（BEHAVIOR_SURFACES 封闭枚举）
    precondition: str  # 生效条件
    assertion: str  # 可验证断言（人读）
    verifier_spec: dict  # 可验证断言（机检：build_verifier 可构造）
    deposit_id: str = ""  # 缺省按 (buyer, statement, trigger_session) 确定性派生
    created_at: str = ""

    def __post_init__(self) -> None:
        for name in _REQUIRED_TEXT_FIELDS:
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"MemoryDeposit.{name} required（沉淀字段缺一不可）")
        if self.kind not in VALID_KINDS:
            raise ValueError(f"MemoryDeposit.kind 必须是 {VALID_KINDS}：{self.kind}")
        if self.behavior_surface not in BEHAVIOR_SURFACES:
            raise ValueError(
                f"MemoryDeposit.behavior_surface 必须是 {BEHAVIOR_SURFACES}：{self.behavior_surface}"
            )
        if not isinstance(self.verifier_spec, dict) or not self.verifier_spec:
            raise ValueError("MemoryDeposit.verifier_spec required（没有机检断言的沉淀不许写入）")
        # 构造时即验证 spec 可构造：脏 spec 活不过构造，而不是躺在库里等人踩
        build_verifier(self.verifier_spec)
        if not self.deposit_id:
            object.__setattr__(
                self, "deposit_id",
                _deposit_id(self.buyer_id, self.statement, self.trigger_session_id),
            )
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "deposit_id": self.deposit_id,
            "buyer_id": self.buyer_id,
            "kind": self.kind,
            "statement": self.statement,
            "trigger_session_id": self.trigger_session_id,
            "trigger_query": self.trigger_query,
            "behavior_surface": self.behavior_surface,
            "precondition": self.precondition,
            "assertion": self.assertion,
            "verifier_spec": self.verifier_spec,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MemoryDeposit":
        return cls(**payload)
