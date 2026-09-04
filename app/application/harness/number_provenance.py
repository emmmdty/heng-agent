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
# `taxable`（应税基数）、`budget`/`remaining`/`saving`（组合优化返回的预算算术）
# 同样是钱。不纳入的话，由它们派生的数只会被归成 unsourced，
# 而"3.72 = 153.72 − 150"这类线索正是指向"工具该补哪个字段"的依据。
_MONEY_FIELD = re.compile(
    r"(price|subtotal|freight|tariff|landed|amount|total|taxable|budget|remaining|saving)",
    re.IGNORECASE,
)
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

# "算得上精确"的门槛：只用来提前收工，不参与判定。取 1e-9 是因为浮点相减必有残差
# （1341.9 - 200 = 1141.9000000000001），拿 == 0 比会永远早停不了。
_EXACT_EPSILON = 1e-9

# 无出处金额的分类。kind 是给人看的诊断线索，不参与通过与否的判定。
# basket_misadd 例外：它不是线索，是**确定性违规**——四个条件同时成立才升级
# （无出处、≥2 个 landed 值相加、金额所在行带组合语境且不带分开语境、
# 会话内存在 quote_basket 报价且组合总价与该金额不符），见 check_reply。
KIND_UNSOURCED = "unsourced"
KIND_SUM = "suspected_sum"
KIND_DIFFERENCE = "suspected_difference"
KIND_BASKET_MISADD = "basket_misadd"

# 组合语境 / 分开语境：金额**所在行**的措辞决定"相加当总价"是不是缺陷。
# "两件分开买合计：¥518"（eval-compare-two-1b9144）是合法用法——分开买
# 本来就是各付各的运费，加法恰好是对的；"一起下单的组合到手价 ¥518"
# 才是运费重复计。一行里两种语境都出现时按分开理解（宁漏报不误报）。
_COMBINED_CONTEXT = re.compile(
    r"组合|合并|合单|一起下单|一起买|合起来|一同下单|同时下单|同一包裹|装在一起|一起结算",
)
_SEPARATE_CONTEXT = re.compile(r"分开|分别|各买|各自|单独下单|单独买")


def _parse(literal: str) -> float:
    return float(literal.replace(",", ""))


def _matches(value: float, source: float) -> bool:
    tolerance = min(_ABSOLUTE_TOLERANCE_CAP, max(0.01, abs(source) * _RELATIVE_TOLERANCE))
    return abs(value - source) <= tolerance


def _is_exact(key: tuple[float, int, int]) -> bool:
    """已经是两个硬出身操作数的精确算式——没有候选能再赢它，可以停止搜索。

    留这个早停是为了别把"挑最贴近的"变成"每个无出处金额都跑满 C(60,3)"：
    组合搜索在运行时每轮都要付钱。
    三项都要满足才停：只看误差会在还有更短、出身更硬的候选时提前收工。
    """
    error, size, rank = key
    return error <= _EXACT_EPSILON and size == 2 and rank == 0


@dataclass(frozen=True)
class AmountSources:
    """本会话内一切可作为出处的数字。

    numbers  工具返回与买家原话里出现过的**所有**数字，用于判定"有没有出处"
    money    其中落在金额字段上的那部分，用于推断"是不是模型自己加减出来的"
    stated   买家原话里的数字。**预算只在这里**——它不在任何工具字段上，
             而"超预算多少""还剩多少"这两类算式的操作数正是预算。
             十九期之前它只进 numbers，于是精确成因根本不在候选集里
             （见 `_classify` 的注释）。
    landed   金额字段里名字带 landed 的那部分（单品到手价）。
             basket_misadd 的加数必须全部出自这里——"拿单品到手价相加当组合总价"
             是特定缺陷形状，价格+价格相加不在此列。
    basket   quote_basket_tool 报价里的全部金额字段。会话内存在组合报价，
             "相加当组合总价"才有了 ground truth；没有它只有线索，不定罪。
    """

    numbers: tuple[float, ...] = ()
    money: tuple[float, ...] = ()
    stated: tuple[float, ...] = ()
    landed: tuple[float, ...] = ()
    basket: tuple[float, ...] = ()

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


def _walk(
    node: Any,
    field_name: str,
    numbers: list[float],
    money: list[float],
    landed: list[float],
    basket: list[float],
    in_basket: bool = False,
) -> None:
    if isinstance(node, dict):
        # quote_basket_tool 的返回整体是"组合报价的 ground truth"：
        # 组合总价、分开买对照、节省额都算。判定靠工具名认，不靠字段名猜。
        child_basket = in_basket or node.get("tool") == "quote_basket_tool"
        for key, value in node.items():
            _walk(value, key, numbers, money, landed, basket, child_basket)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _walk(value, field_name, numbers, money, landed, basket, in_basket)
    elif isinstance(node, bool):
        pass  # bool 是 int 的子类，必须在数值分支之前挡掉
    elif isinstance(node, (int, float)):
        numbers.append(float(node))
        if _MONEY_FIELD.search(field_name) and not _RATE_FIELD.search(field_name):
            money.append(float(node))
            if in_basket:
                basket.append(float(node))
            if "landed" in field_name.lower():
                landed.append(float(node))
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
    stated: list[float] = []
    landed: list[float] = []
    basket: list[float] = []
    for payload in tool_results:
        _walk(payload, "", numbers, money, landed, basket)
    for text in buyer_texts:
        for match in _ANY_NUMBER.finditer(text or ""):
            value = _parse(match.group(0))
            numbers.append(value)
            stated.append(value)
    return AmountSources(
        numbers=tuple(numbers),
        money=tuple(sorted(set(money))),
        stated=tuple(sorted(set(stated))),
        landed=tuple(sorted(set(landed))),
        basket=tuple(sorted(set(basket))),
    )


def extract_amounts(text: str) -> list[tuple[float, str]]:
    """抽取带货币标记的金额，返回 [(数值, 原文片段)]，按出现位置排序。"""
    found: list[tuple[int, float, str]] = []
    for pattern in (_CURRENCY_PREFIXED, _CURRENCY_SUFFIXED):
        for match in pattern.finditer(text or ""):
            found.append((match.start(), _parse(match.group(1)), match.group(0).strip()))
    found.sort(key=lambda item: item[0])
    return [(value, raw) for _, value, raw in found]


def _subtrahends(sources: AmountSources) -> tuple[float, ...]:
    """能当减数的数：工具的金额字段 + 买家自述的数字，去重后从小到大。

    买家自述必须进：**预算不在任何工具字段上**，而"超出预算多少"这类算式
    正是拿预算当减数。十九期之前减数池只取金额字段，于是
    "1341.9 − 200（买家预算）" 这个精确成因压根不在候选集里，
    剩下的候选里最"像"的就成了擦边命中的 "1341.9 − 199"——
    这种情况下光会挑最贴近的也救不回来，候选集本身就是缺的。

    0 不进：免税额度内关税为 0，几乎每条商品卡上都有一个 0，
    留着它等于给任何数字都配得出一个 "x − 0" 的假解释，把真成因盖掉。
    """
    pool = {item for item in sources.money if item > 0.0}
    pool.update(item for item in sources.stated if item > 0.0)
    return tuple(sorted(pool))


def _classify(value: float, sources: AmountSources) -> tuple[str, str, tuple[float, ...]]:
    """给无出处金额找一个最贴近的成因，找不到就是纯无出处。

    返回值第三项是和类成因的加数（非和类为空元组）——basket_misadd
    的升级判定需要知道加数是谁，全来自 landed 字段才谈得上"到手价相加"。

    **必须把候选找全再挑最贴近的，不能一命中就返回。**实测代价
    （eval-conflict-budget-spec-9c422d，买家自述预算 200）：

        ¥1,141.90  报成 "1341.9 - 199"（差 1.0，靠容差上限擦边命中）
                   真相是 1341.9 - 200（精确相等）
        ¥99        报成 "9 + 89"（差 1.0）
                   真相是 299 - 200——所以"先搜和、命中就返回"也不行，
                   挑最贴近的必须**跨类**挑，不能让和天然赢过差。

    容差是留给展示取整的（250 − 228.15 = 21.85 被写成 "$22"），
    不是给"差一块钱也算"用的。调容差治不了这个：调大只会制造更多错解释，
    调小会连真的取整解释一起砍掉——该改的是在候选里挑最贴近的那个。

    候选排序按 (误差, 操作数个数, 操作数出身)：误差小的赢；误差相同取算式更短的；
    再相同取操作数出身更硬的（金额字段/买家自述 > 自由文本里解析出的数字）；
    最后按发现顺序（和在前、差在后），保证同一份输入每次给同一个解释。

    第三项不是锦上添花：**精确解经常不止一个**。$21.85 那轮里
    "250 − 228.15"（买家预算 − 到手价，真相）与 "220 − 198.15" 都精确成立，
    而 220 只是自由文本里解析出来的一个数——按发现顺序挑就会挑中后者，
    读报告的人会去查一个根本不存在的 220 元商品。

    0 不进加数池：理由同 `_subtrahends`。
    加数池另外按目标值剪枝——正金额相加只会变大，比目标还大的数不可能是加数。
    减数池**不能**同样剪：差额的减数本来就比差值大（250 − 228.15 = 21.85）。
    被减数取全部出处数字，不收窄到金额字段：实测收窄会掉解释——
    long-context-memory 那轮的 ¥27 真相是 "72 − 45"，而 72 只出现在
    工具返回的自由文本里；把它剪掉之后，赢下来的是差 1.0 的 "39 − 13"。
    """
    positive = tuple(item for item in sources.money if item > 0.0)
    addends = tuple(item for item in positive if item <= value + _CLASSIFY_ABSOLUTE_TOLERANCE)
    addends = addends[-_MAX_CLASSIFY_POOL:]
    strong = _subtrahends(sources)
    subtrahends = strong[-_MAX_CLASSIFY_POOL:]
    strong_set = frozenset(strong)

    best_key: tuple[float, int, int] | None = None
    best: tuple[str, str] = (KIND_UNSOURCED, "")
    best_addends: tuple[float, ...] = ()

    # 加数一律取自金额字段，出身分档恒为 0
    for size in range(2, _MAX_COMBINATION + 1):
        for combo in itertools.combinations(addends, size):
            error = abs(value - sum(combo))
            if error > _CLASSIFY_ABSOLUTE_TOLERANCE:
                continue
            key = (error, size, 0)
            if best_key is None or key < best_key:
                best_key = key
                best = (KIND_SUM, " + ".join(f"{item:g}" for item in combo))
                best_addends = combo
                if _is_exact(key):
                    return best + (best_addends,)

    # 差额：买家预算减到手价（还剩多少）、到手价减预算（超出多少），两个方向都要覆盖。
    for minuend in dict.fromkeys(sources.numbers):
        rank = 0 if minuend in strong_set else 1
        for subtrahend in subtrahends:
            if minuend <= subtrahend:
                continue
            error = abs(value - (minuend - subtrahend))
            if error > _CLASSIFY_ABSOLUTE_TOLERANCE:
                continue
            key = (error, 2, rank)
            if best_key is None or key < best_key:
                best_key = key
                best = (KIND_DIFFERENCE, f"{minuend:g} - {subtrahend:g}")
                best_addends = ()
                if _is_exact(key):
                    return best + (best_addends,)
    return best + (best_addends,)


def _line_containing(reply: str, position: int) -> str:
    """金额所在行的原文——组合/分开语境的锚点按行取，不按整条回复取。

    整条回复里两种语境几乎总会同时出现（组合价一节旁边就是分开买对照），
    按整条取会让锚点永远失真；按行取才能对上真实流水的排版。
    """
    start = reply.rfind("\n", 0, position) + 1
    end = reply.find("\n", position)
    return reply[start:end if end != -1 else len(reply)]


def _is_basket_misadd(value: float, addends: tuple[float, ...], sources: AmountSources, line: str) -> bool:
    """升级为确定性违规的四个条件，缺一不可（宁可漏报不误报）：

    1. 成因是 ≥2 个加数的和（调用方已保证 kind == suspected_sum）；
    2. 加数**全部**来自 landed 字段——"单品到手价相加"是特定缺陷形状；
    3. 金额所在行带组合语境、且不带分开语境（"分开买合计"是合法用法）；
    4. 会话内存在 quote_basket 报价，且没有哪个报价金额与该值相符
       ——相符即有出处，根本到不了这里；再比一次是防 separate 对照
       恰好等于该值的情形被误判。
    """
    if not sources.basket or not addends:
        return False
    landed = frozenset(sources.landed)
    if any(addend not in landed for addend in addends):
        return False
    if _SEPARATE_CONTEXT.search(line):
        return False
    if not _COMBINED_CONTEXT.search(line):
        return False
    return not any(_matches(value, quote) for quote in sources.basket)


def check_reply(reply: str, sources: AmountSources) -> ProvenanceReport:
    """校验一条最终回复里的金额出处。

    工具失败时回复是 "[error] ..." 文本，其中不含金额，天然 clean——
    不给它开特例，是因为特例会让"错误回复里编数字"也一并被放过。
    """
    report = ProvenanceReport()
    occurrences: list[tuple[int, float, str]] = []
    for pattern in (_CURRENCY_PREFIXED, _CURRENCY_SUFFIXED):
        for match in pattern.finditer(reply or ""):
            occurrences.append((match.start(), _parse(match.group(1)), match.group(0).strip()))
    occurrences.sort(key=lambda item: item[0])
    for position, value, raw in occurrences:
        report.total_amounts += 1
        if sources.has(value):
            continue
        kind, explain, addends = _classify(value, sources)
        if kind == KIND_SUM and _is_basket_misadd(value, addends, sources, _line_containing(reply, position)):
            kind = KIND_BASKET_MISADD
            explain = f"{explain}（quote_basket 已报组合总价：单品到手价相加会把运费重复计一次）"
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
    _stated: dict[str, list[float]] = field(default_factory=dict)
    _landed: dict[str, list[float]] = field(default_factory=dict)
    _basket: dict[str, list[float]] = field(default_factory=dict)

    def observe(
        self,
        session_id: str,
        tool_results: Iterable[Any] = (),
        buyer_texts: Iterable[str] = (),
    ) -> None:
        fresh = collect_sources(tool_results=tool_results, buyer_texts=buyer_texts)
        numbers = self._numbers.setdefault(session_id, [])
        money = self._money.setdefault(session_id, [])
        stated = self._stated.setdefault(session_id, [])
        landed = self._landed.setdefault(session_id, [])
        basket = self._basket.setdefault(session_id, [])
        numbers.extend(fresh.numbers)
        money.extend(fresh.money)
        stated.extend(fresh.stated)
        landed.extend(fresh.landed)
        basket.extend(fresh.basket)
        for bucket in (numbers, money, stated, landed, basket):
            if len(bucket) > MAX_RETAINED_NUMBERS:
                del bucket[: len(bucket) - MAX_RETAINED_NUMBERS]

    def of(self, session_id: str) -> AmountSources:
        return AmountSources(
            numbers=tuple(self._numbers.get(session_id, ())),
            money=tuple(sorted(set(self._money.get(session_id, ())))),
            stated=tuple(sorted(set(self._stated.get(session_id, ())))),
            landed=tuple(sorted(set(self._landed.get(session_id, ())))),
            basket=tuple(sorted(set(self._basket.get(session_id, ())))),
        )

    def reset(self, session_id: str) -> None:
        self._numbers.pop(session_id, None)
        self._money.pop(session_id, None)
        self._stated.pop(session_id, None)
        self._landed.pop(session_id, None)
        self._basket.pop(session_id, None)
