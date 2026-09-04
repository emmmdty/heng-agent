# -*- coding: utf-8 -*-
"""knowledge_provenance —— 知识库出处的校验

判据只有一条：**回复里声称"来自知识库 / 品类洞察"的内容，本会话必须真的
有过一次成功的知识库返回。**

来源（交接文档"第一点五优先"）：二十期分诊 `category-insight` 时确认，
judge 结构上看不到工具返回（经验 5），"知识库当时可不可用"它判不了。
判据只能写 judge 判得了的事，所以"出处属不属实"这半必须搬给确定性判据——
就是本模块。数值对不对那半已由 judge 侧写死的区间覆盖。

**与另外几条轮末判据的分工**：

    number_provenance    金额从哪来
    arithmetic_check     过程算不算得通
    contact_provenance   买家的个人信息从哪来
    knowledge_provenance 选购常识的出处从哪来（本模块）

买家问"这个品类怎么挑"时，Agent 会先讲判断标准再给商品清单；
判断标准应当来自品类知识库（RAG）。模型把自己的常识安上"知识库"的
出处说出去，买家无从分辨，评测里也难以归因——这正是要抓的缝。

**范围刻意收窄，方向一律取"宁可漏报不误报"**：

    1. 只认**显式**出处声明（"知识库 / 品类洞察"字样）。
    2. **诚实降级不算声明**：工具报错后"知识库暂时不可用，我先按常识讲"
       是唯一正确的行为，同一行里出现降级措辞就不判——反过来罚它，
       下一次它就会编一个出处。
    3. 反方向（工具返回了、回复没引用）不是错。
    4. 按行判定：降级说明与出处声明常出现在同一回复的不同行。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# 显式出处声明：只认这两个词。更宽的（"选购指南""避坑点"）会撞上
# 商品文案与通用建议，误报会把判据逼成摆设。
_CLAIM_WORD = re.compile(r"知识库|品类洞察")
# 归因构型：光提到"知识库"不算声明，要把它**当出处用**才算。
# 校准来源：120 份真实流水上的 15 处声明行逐条分诊——
# "我可以提供品类洞察"（能力提议）不是归因，不能抓。
_ATTRIBUTION = re.compile(
    r"(?:来自|从|根据|按照|结合|依据)(?:品类)?(?:知识库|品类洞察)"
    r"|(?:品类)?知识库(?:里|中)"
    r"|(?:品类)?(?:知识库|品类洞察)(?:显示|说|提到|指出|表明|拉回来|整理|返回|给出|公认|足以)"
    r"|品类洞察(?:中|里)"
    r"|品类知识库"
    r"|（(?:来自)?品类洞察）"
    r"|\((?:来自)?品类洞察\)",
)
# 诚实降级：工具报错后如实告知"不可用"是唯一正确的行为，反过来罚它，
# 下一次它就会编一个出处。
_DEGRADATION = re.compile(r"不可用|无法|出错|失败|繁忙|暂时|稍后|恢复|重试|未就绪")
# 缺失观察与反向免责："知识库里这条数据不完整""暂无专项指南""非知识库数据"
# ——这些说的是知识库**没有什么**，不是把内容归因给它。
# "知识库里没有"是缺失（知识库是"没有"的主语）；"知识库里说这款没有快充"
# 是归因（"没有"属于商品）——前者排除、后者保留，中间隔着的字必须连续才判缺失。
# 已知缺口（不补）：无知识库返回的会话里编造"知识库里没有 X"会漏过；
# 宁可漏报不误报，缺口的代价小于误报。
_ABSENCE = re.compile(
    r"非知识库|不来自知识库|暂无|不完整|未收录|没有收录|查不到"
    r"|(?:品类)?知识库(?:里|中)?没有",
)


@dataclass(frozen=True)
class KnowledgeClaim:
    """回复里断言出来的一处知识库出处。"""

    raw: str


@dataclass(frozen=True)
class KnowledgeSources:
    """本会话的知识库出处状态：有没有过一次**非空**的成功返回。"""

    available: bool = False


@dataclass
class KnowledgeReport:
    claims: int = 0
    unsourced: list[KnowledgeClaim] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.unsourced

    def to_dict(self) -> dict:
        return {
            "claims": self.claims,
            "unsourced": [{"raw": item.raw} for item in self.unsourced],
        }


def _payload(payload: Any) -> Any:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return None
    return payload if isinstance(payload, dict) else None


def collect_knowledge_sources(tool_results: Iterable[Any] = ()) -> KnowledgeSources:
    """判定会话内是否存在可作依据的知识库返回。

    只认 `category_insight_tool` 的**成功且非空**返回：报错与零命中
    都给不出"知识库里说"的出处。其余工具的返回一概不算。
    """
    for payload in tool_results:
        data = _payload(payload)
        if data and data.get("tool") == "category_insight_tool" and data.get("insights"):
            return KnowledgeSources(available=True)
    return KnowledgeSources(available=False)


def check_knowledge(reply: str, sources: KnowledgeSources) -> KnowledgeReport:
    """校验一条回复里的知识库出处声明。

    按行判定：降级说明与出处声明经常分处同一回复的不同行
    （"知识库查询失败了。\n不过知识库里说……"），按整条判会两头失真。
    """
    report = KnowledgeReport()
    for line in (reply or "").splitlines():
        if not _CLAIM_WORD.search(line):
            continue
        if not _ATTRIBUTION.search(line):
            continue
        if _DEGRADATION.search(line):
            continue
        if _ABSENCE.search(line):
            continue
        report.claims += 1
        if not sources.available:
            report.unsourced.append(KnowledgeClaim(raw=line.strip()[:80]))
    return report


@dataclass
class SessionKnowledgeSources:
    """按会话累积出处状态，与 `SessionSources` 同一形态（按 shopping_session_id 分桶）。

    跨轮累积是必要的：第 1 轮查过知识库，第 3 轮引用它的结论是正常行为。
    """

    _available: dict[str, bool] = field(default_factory=dict)

    def observe(self, session_id: str, tool_results: Iterable[Any] = ()) -> None:
        if collect_knowledge_sources(tool_results).available:
            self._available[session_id] = True

    def of(self, session_id: str) -> KnowledgeSources:
        return KnowledgeSources(available=self._available.get(session_id, False))

    def reset(self, session_id: str) -> None:
        self._available.pop(session_id, None)
