# -*- coding: utf-8 -*-
"""number_provenance —— 金额出处校验

判据只有一条：**回复里出现的每一个金额，都必须能在工具返回或买家原话里找到出处。**

为什么要做成一个确定性判据，而不是继续靠 rubric 里的 P0 条目：

    评测里"数字必须来自工具"由 LLM judge 判，而 judge 对"自洽即通过"的宽容
    恰好放过了本项目最典型的一类错误——模型把两个工具算出来的到手价相加。
    实测 eval-compare-two 的一轮：¥364 + ¥154 = ¥518，judge 判 PASS，
    因为每个加数都对；错的是"运费按一次履约计"这条模型不可能自己推出来的口径。
    同一轮里模型还自行算了非主 SKU 的到手价（229 USD × 汇率 + 运费 = $238.15）。
    这两处都不是"编造"，是"自行推导"，语义判据抓不住，算术判据抓得住。

    LLM 判据还有运行间差异：单次 PASS 不构成修复证据（设计演进记录里已栽过）。
    确定性判据每次跑出同一个数，才能当回归基线用。

**范围是刻意收窄的**，收窄的方向一律取"宁可漏报不误报"：

    1. 只看**带货币标记**的数字（¥ $ € £ / 元 块 美元 CNY USD ...）。
       表格里裸写的 "| 65 | 0 |" 会漏掉——所以本判据给出的是无出处金额的**下界**。
       放宽到所有数字会把"续航 40 小时""库存 150 件"全扫进来，判据立刻失去可用性。
    2. 百分数不算金额（"关税 13%" 对应 payload 里的 0.13，形态对不上，
       强行换算只会制造误报）。
    3. 出处只按**数值**比对，不比币种。"$219" 与 payload 里的 219.0 CNY 会算命中。
       这是已知的放松：它换来的是对"工具返回 USD、回复折成 CNY 展示"这类
       正常情形的零误报。
    4. 出处按**会话**累积，不按轮次：模型引用上一轮检索结果里的价格是正常行为。

误报来源在实测中只有一类，且根因不在本模块：工具 tool.result 事件没把
喂给模型的返回**完整**发出来（product_search_tool 曾漏发 filtered_out，
category_insight_tool 曾只发 hit_count）。轨迹与模型所见不一致时，
本判据会把有出处的数字判成无出处——这正是它顺带暴露出来的问题。
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# 金额数字的字面形态：允许千分位与小数
_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
# 货币前缀（¥1,619.9 / $228.15）与后缀（65 元 / 219 USD）
_CURRENCY_PREFIXED = re.compile(r"[¥￥$€£]\s*(" + _NUMBER + r")")
_CURRENCY_SUFFIXED = re.compile(
    r"(" + _NUMBER + r")\s*(?:元|块钱|块|美元|欧元|日元|新元|人民币|CNY|USD|EUR|JPY|SGD)",
)
_ANY_NUMBER = re.compile(_NUMBER)
# payload 里"这个字段装的是钱"的判据；组合搜索只在这些数上做，避免在
# score/stock/hit_count 里凑出巧合的和。
# 排除 *_rate：tariff_rate=0.075 名字里带 tariff 但装的是税率不是金额，
# 放进加数池会编出 "0.075 + 198.15 ≈ 200" 这种荒唐成因。
_MONEY_FIELD = re.compile(r"(price|subtotal|freight|tariff|landed|amount|total)", re.IGNORECASE)
_RATE_FIELD = re.compile(r"rate", re.IGNORECASE)

# 组合搜索的元数上限。2 覆盖"两件相加""预算减到手价"，3 覆盖"小计+运费+关税"；
# 再往上加，凑出巧合的概率比抓到真错的概率涨得快。
_MAX_COMBINATION = 3

# 参与组合搜索的加数上限。开销按 C(n, 3) 走，是立方级：一次检索返回 5 个商品卡、
# 每张卡十几个金额字段，几轮下来池子就到几百，600 个实测要 3.6s——运行时每轮都付
# 这个代价不可接受。取最大的 60 个：正数相加只会变大，最可能凑出目标的是靠近它的
# 那些数，砍掉的是量级差得远、本来也凑不出来的小额。
_MAX_CLASSIFY_POOL = 60

# 出处比对容差：相对 0.2%，绝对不超过 1.0。
# 相对项容"¥1,620 展示 1619.9"这类取整，绝对上限防止大额下容差被放大到
# 能盖住两个不同商品的差价。
_RELATIVE_TOLERANCE = 0.002
_ABSOLUTE_TOLERANCE_CAP = 1.0

# 成因推断（kind）的容差：只放宽到能容下展示取整（实测 250 − 228.15 = 21.85
# 被写成 "$22"），不按比例放宽。按 1% 放宽试过，大额上会把 1619.9 认成 1625.9
# 的成因——一个错的解释比没有解释更坏，它会把人引向错误的根因。
_CLASSIFY_ABSOLUTE_TOLERANCE = 1.0

# 无出处金额的分类。kind 是给人看的诊断线索，不参与通过与否的判定。
KIND_UNSOURCED = "unsourced"
KIND_SUM = "suspected_sum"
KIND_DIFFERENCE = "suspected_difference"


def _parse(literal: str) -> float:
    return float(literal.replace(",", ""))


def _matches(value: float, source: float) -> bool:
    tolerance = min(_ABSOLUTE_TOLERANCE_CAP, max(0.01, abs(source) * _RELATIVE_TOLERANCE))
    return abs(value - source) <= tolerance


def _roughly(value: float, expected: float) -> bool:
    return abs(value - expected) <= _CLASSIFY_ABSOLUTE_TOLERANCE


@dataclass(frozen=True)
class AmountSources:
    """本会话内一切可作为出处的数字。

    numbers  工具返回与买家原话里出现过的**所有**数字，用于判定"有没有出处"
    money    其中落在金额字段上的那部分，用于推断"是不是模型自己加减出来的"
    """

    numbers: tuple[float, ...] = ()
    money: tuple[float, ...] = ()

    def has(self, value: float) -> bool:
        return any(_matches(value, source) for source in self.numbers)


@dataclass(frozen=True)
class UnsourcedAmount:
    value: float
    raw: str      # 回复中的原文片段，便于人工复核
    kind: str
    explain: str = ""


@dataclass
class ProvenanceReport:
    total_amounts: int = 0
    unsourced: list[UnsourcedAmount] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.unsourced

    def to_dict(self) -> dict:
        return {
            "total_amounts": self.total_amounts,
            "unsourced": [
                {"value": item.value, "raw": item.raw, "kind": item.kind, "explain": item.explain}
                for item in self.unsourced
            ],
        }


def _walk(node: Any, field_name: str, numbers: list[float], money: list[float]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _walk(value, key, numbers, money)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _walk(value, field_name, numbers, money)
    elif isinstance(node, bool):
        pass  # bool 是 int 的子类，必须在数值分支之前挡掉
    elif isinstance(node, (int, float)):
        numbers.append(float(node))
        if _MONEY_FIELD.search(field_name) and not _RATE_FIELD.search(field_name):
            money.append(float(node))
    elif isinstance(node, str):
        # 工具返回里的自由文本（知识库片段、错误信息）同样是出处
        for match in _ANY_NUMBER.finditer(node):
            numbers.append(_parse(match.group(0)))


def collect_sources(
    tool_results: Iterable[Any] = (),
    buyer_texts: Iterable[str] = (),
) -> AmountSources:
    """把本会话的工具返回与买家原话汇总成出处集合。

    买家原话必须算出处：买家自己说"预算 300 块"，Agent 复述 300 元不是编造。
    """
    numbers: list[float] = []
    money: list[float] = []
    for payload in tool_results:
        _walk(payload, "", numbers, money)
    for text in buyer_texts:
        for match in _ANY_NUMBER.finditer(text or ""):
            numbers.append(_parse(match.group(0)))
    return AmountSources(numbers=tuple(numbers), money=tuple(sorted(set(money))))


def extract_amounts(text: str) -> list[tuple[float, str]]:
    """抽取带货币标记的金额，返回 [(数值, 原文片段)]，按出现位置排序。"""
    found: list[tuple[int, float, str]] = []
    for pattern in (_CURRENCY_PREFIXED, _CURRENCY_SUFFIXED):
        for match in pattern.finditer(text or ""):
            found.append((match.start(), _parse(match.group(1)), match.group(0).strip()))
    found.sort(key=lambda item: item[0])
    return [(value, raw) for _, value, raw in found]


def _classify(value: float, sources: AmountSources) -> tuple[str, str]:
    """给无出处金额找一个最可能的成因，找不到就是纯无出处。

    0 不进任何池：免税额度内关税为 0，几乎每条商品卡上都有一个 0，
    留着它等于给任何数字都配得出一个 "0 + x" 的假解释，把真成因盖掉。

    加数池另外按目标值剪枝——正金额相加只会变大，比目标还大的数不可能是加数。
    减数池**不能**同样剪：差额的减数本来就比差值大（250 − 228.15 = 21.85）。
    """
    positive = tuple(item for item in sources.money if item > 0.0)

    addends = tuple(item for item in positive if item <= value + _CLASSIFY_ABSOLUTE_TOLERANCE)
    addends = addends[-_MAX_CLASSIFY_POOL:]
    for size in range(2, _MAX_COMBINATION + 1):
        for combo in itertools.combinations(addends, size):
            if _roughly(value, sum(combo)):
                return KIND_SUM, " + ".join(f"{item:g}" for item in combo)

    # 差额：买家预算减到手价这类。被减数取全部出处数字（预算来自买家原话，
    # 不在金额字段里），减数只取金额字段。
    subtrahends = positive[-_MAX_CLASSIFY_POOL:]
    for minuend in dict.fromkeys(sources.numbers):
        for subtrahend in subtrahends:
            if minuend > subtrahend and _roughly(value, minuend - subtrahend):
                return KIND_DIFFERENCE, f"{minuend:g} - {subtrahend:g}"
    return KIND_UNSOURCED, ""


def check_reply(reply: str, sources: AmountSources) -> ProvenanceReport:
    """校验一条最终回复里的金额出处。

    工具失败时回复是 "[error] ..." 文本，其中不含金额，天然 clean——
    不给它开特例，是因为特例会让"错误回复里编数字"也一并被放过。
    """
    report = ProvenanceReport()
    for value, raw in extract_amounts(reply):
        report.total_amounts += 1
        if sources.has(value):
            continue
        kind, explain = _classify(value, sources)
        report.unsourced.append(UnsourcedAmount(value=value, raw=raw, kind=kind, explain=explain))
    return report


# 单会话保留的出处数字上限。长会话每轮都会往里塞一批检索结果，无上限会稳步涨；
# 超出后丢最早的——旧轮次的价格越久越不可能被引用，而近几轮的必须留住。
MAX_RETAINED_NUMBERS = 4000


@dataclass
class SessionSources:
    """按会话累积出处，与 SequencingTracker 同一形态（按 shopping_session_id 分桶）。

    为什么必须跨轮累积：多轮会话里模型引用第 1 轮检索到的价格是正常行为
    （long-context-memory 用例最后一轮就要求它复述首轮那款露营灯的价格）。
    只按单轮判定会把这类正确行为全判成无出处。
    """

    _numbers: dict[str, list[float]] = field(default_factory=dict)
    _money: dict[str, list[float]] = field(default_factory=dict)

    def observe(
        self,
        session_id: str,
        tool_results: Iterable[Any] = (),
        buyer_texts: Iterable[str] = (),
    ) -> None:
        fresh = collect_sources(tool_results=tool_results, buyer_texts=buyer_texts)
        numbers = self._numbers.setdefault(session_id, [])
        money = self._money.setdefault(session_id, [])
        numbers.extend(fresh.numbers)
        money.extend(fresh.money)
        if len(numbers) > MAX_RETAINED_NUMBERS:
            del numbers[: len(numbers) - MAX_RETAINED_NUMBERS]
        if len(money) > MAX_RETAINED_NUMBERS:
            del money[: len(money) - MAX_RETAINED_NUMBERS]

    def of(self, session_id: str) -> AmountSources:
        return AmountSources(
            numbers=tuple(self._numbers.get(session_id, ())),
            money=tuple(sorted(set(self._money.get(session_id, ())))),
        )

    def reset(self, session_id: str) -> None:
        self._numbers.pop(session_id, None)
        self._money.pop(session_id, None)
