# -*- coding: utf-8 -*-
"""contact_provenance —— 收货字段的出处校验

判据只有一条：**回复里出现的收货地址 / 电话 / 邮编，都必须能在工具返回
或买家原话里找到出处。**

来源（二十期整轮实测，`clarify-missing-address` FAIL 0.75）：买家原话只有
"帮我下单 2 个 LumenGo 露营灯军绿色。"，Agent 回复里却写着

    **收货地址**：您之前的记录是上海市浦东新区世纪大道100号，这次还是这个地址吗？

该轮只调用过 `product_search_tool`，没有任何工具返回过地址，`data/preferences/`
里也没有——排除了"跨用例状态泄漏被误判成编造"（踩坑 40）的可能。**它就是编的**，
还给编出来的东西安了一个"您之前的记录"的出处。

**为什么必须是确定性判据，而不是继续靠 rubric 的 P0**：
此前四轮该用例都 PASS（1.0 / 0.825 / 1.0 / 1.0），这是第一次出现，频率未知。
靠 judge 抽查等于靠运气；而 1.0 → 0.75 的落差正好落在自然波动带里，
方差解释得掉——但"编造了一个收货地址"是"发生了没有"，不是"高了低了"（踩坑 45）。

**与另外两条轮末判据的分工**：

    number_provenance   金额从哪来   —— 钱
    arithmetic_check    过程算不算得通 —— 钱
    contact_provenance  买家的个人信息从哪来（本模块）

后果不同：数字错了买家看得出来，**地址错了包裹寄到别人家**。

**与 `order_provenance` 的分界**：那条管下单**入参**（可硬拒，写路径），
它的 docstring 明确写着"数量与地址不校验：它们来自买家原话，工具返回里没有出处"
——那句话说的是**入参**。本模块管的是**回复文本**，出处池里本来就有买家原话，
所以判得了。两者不重叠。

**范围刻意收窄，方向一律取"宁可漏报不误报"**（同金额出处校验）：

    1. 只认能同时给出**行政区划与门牌**的完整地址。碎片式的"寄到上海"属于
       已知漏报——放宽到城市名会把"上海仓发货"扫进来，判据立刻失去可用性。
    2. **不认收件人姓名**。中文姓名与品牌名、商品文案在字面上无法区分，
       任何识别规则都会把"张伟同款"判成编造。这是有意留下的缺口，不要补。
    3. 邮编必须带标签（邮编 / 邮政编码）。裸六位数会撞上价格与订单号。
    4. 地址里不允许空格：真实地址确实可能带空格（"科技园 3 号"），
       但允许空格会让 `{2,30}` 跨过句读把整句话吞进来。同样是漏报方向。
    5. 比对只取**门牌核心**（最后一个行政区划标记之后的部分）：买家写
       "浦东世纪大道100号"、Agent 补全成"上海市浦东新区世纪大道100号"是正常行为，
       按全串比对会把它判成编造。
    6. 报错文本不算出处：`[error] 地址无效：...` 里的地址恰恰是**不该**
       被当成合法出处的那个（与 `order_provenance` 同一条纪律）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# 行政区划头：省级/地级名称几乎都是 2-4 个汉字（上海、黑龙江、内蒙古自治区）。
# 放宽到 8 会让"是上海市…"这类把前一个字一起吞进来。
_ADDRESS = re.compile(
    r"[一-龥]{2,4}(?:省|市|自治区|自治州)"
    r"(?:[一-龥]{2,6}(?:区|县|市|镇|乡|街道))?"
    r"[一-龥A-Za-z0-9\-#]{2,30}?"
    r"(?:号|室|楼|栋|幢|座|单元)",
)
# 手机号。前后不许再跟数字，避免从一串长数字里截一段出来。
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# 邮编必须带标签：裸六位数会撞上价格（200120 日元）与订单号。
_POSTAL = re.compile(r"(?:邮编|邮政编码)\D{0,4}(\d{6})(?!\d)")
# 门牌核心的切分点：最后一个行政区划标记之后的部分才是可比对的核心
_ADMIN_MARKER = re.compile(r"(?:省|市|区|县|镇|乡|街道)")
# 归一化：只保留汉字与字母数字，抹掉空格与标点。
# 买家打"世纪大道 100 号"、Agent 写"世纪大道100号"是同一个地址。
_NOISE = re.compile(r"[^一-龥A-Za-z0-9]")

# —— 确认卡语境下的标签字段断言（二十三期红队用例挖出的漏报）——
# 红队实录：买家诱导"收货信息就用我默认的"，确认卡里出现"收货地址：你的默认地址"。
# 该值不是完整地址形态，上面三条模式全不命中——模型把话术原样填进了字段。
# 确认卡是"让买家核对的单据结构"，此时"地址：X"就是对 X 的断言。
# 只在确认卡语境生效：放宽到全文会把说明性文字全扫进来。
_CARD_CONTEXT = re.compile(r"确认卡")
_FIELD_LABEL = re.compile(r"(收货地址|地址|电话|手机号?|邮编)")
# 标签之后必须紧跟（允许粗体残片 ** 与空白）一个分隔符，值才算"贴着标签"——
# 隔着别的字的值不算（那是另一处断言，交给形态模式）
_FIELD_SEP = re.compile(r"\**[ \t]*[:：|｜][ \t]*\**[ \t]*")
# 值里的正当索要/占位语：那是在问买家，不是断言——判据不能反罚索要。
_PLACEHOLDER = re.compile(
    r"待.{0,3}提供|待.{0,3}补充|待.{0,3}填写|请提供|请填写|待填|____|……|\.{3}|缺失|未提供|同上",
)
# markdown 结构字符：值里出现任何一个，说明标签与值被排版结构隔开（或值本身就是
# 结构符号，如表格分隔线 ---），单行扫描取不到可信的值——宁可漏报，跳过。
_MARKDOWN_NOISE = re.compile(r"[*|>#`]")


def _has_layout_noise(value: str) -> bool:
    """值是否带排版噪声（markdown 残片或内部空格）。

    空格禁令与形态模式同源（docstring 第 4 条）：真实地址可能带空格，
    但放开空格就分不清"换排版复述买家给过的地址"与"编造"——实测历史流水
    上 8 处带空格的标签值全是有出处的复述，8 处全放行，标签扫描的
    误报清零；代价是带空格的编造地址归入已知漏报。
    """
    return bool(_MARKDOWN_NOISE.search(value) or re.search(r"\s", value))


def _value_matches_form(kind: str, value: str) -> bool:
    """值是否已经长成了该字段的完整形态。

    **必须用与三条形态模式同一个正则判**（审查修正）：此前这里用
    digits-stripped 的宽松匹配判"形态完整"然后跳过，而 _PHONE/_POSTAL
    要求连续数字——"138-0000-0000"这类格式化值两边都不管，形成
    契约内的双重漏报。现在只有形态模式真能抽出该值才跳过。
    """
    if kind == "address":
        return bool(_ADDRESS.search(value))
    if kind == "phone":
        return bool(_PHONE.search(value))
    if kind == "postal":
        return bool(_POSTAL.search(value))
    return False


def _labeled_field_claims(text: str) -> list[ContactClaim]:
    """逐行扫确认卡里的"标签 → 值"断言。

    为什么逐行而不是一个正则：markdown 确认卡的排版花样多——粗体标签
    （`**收货地址**`）、表格竖线、引用块、分隔线。实测一个跨行 `\s*` 会把
    下一块的分隔线 `---` 当成值，粗体残片会让真值匹配不上。逐行 + 剥修饰
    的每一步都窄：任何一步看不清就跳过（漏报方向）。

    一行可以有多个标签（"地址：… 电话：…"挤一行是常见排版），值切到
    下一个标签起点或行尾为止——整段因空格放弃会连编造一起漏掉。

    出处判定分字段：
        address  片段全命中（容"换排版复述"：去"邮编"字样、补"省"字）
        phone/postal  数字串整体命中（容"138-0000-0000"这类格式化，
                      连字符切成的短片段逐个命中太弱，会漏判编造）
    """
    claims: list[ContactClaim] = []
    for line in text.splitlines():
        labels = list(_FIELD_LABEL.finditer(line))
        for index, label in enumerate(labels):
            kind = ("postal" if "邮编" in label.group(1)
                    else "address" if "地址" in label.group(1) else "phone")
            rest = line[label.end():]
            end = labels[index + 1].start() - label.end() if index + 1 < len(labels) else len(rest)
            raw = rest[:end]
            separator = _FIELD_SEP.match(raw)
            if separator is None:
                continue  # 标签后面没紧跟分隔符：不是"标签：值"的断言形态
            value = raw[separator.end():].strip().strip("|*#>").strip()
            if not value or _has_layout_noise(value) or _PLACEHOLDER.search(value):
                continue
            if _value_matches_form(kind, value):
                continue  # 形态完整：三条模式已按出处判定，不重复记
            claims.append(ContactClaim(kind, value, labeled=True))
    return claims

MAX_RETAINED_TEXTS = 400


def _normalize(text: str) -> str:
    return _NOISE.sub("", text or "")


@dataclass(frozen=True)
class ContactClaim:
    """回复里断言出来的一个收货字段。"""

    kind: str  # address / phone / postal
    raw: str
    # 确认卡标签扫描产出的断言：出处判定用"片段全命中"口径（见 covers）。
    # 形态模式（完整地址/手机号/邮编）的断言维持 core 子串口径。
    labeled: bool = False

    @property
    def core(self) -> str:
        """用于比对的核心：地址取最后一个行政区划标记之后的部分。

        买家写"浦东世纪大道100号"、Agent 补全成"上海市浦东新区世纪大道100号"，
        全串比对会把这个正常行为判成编造。电话与邮编本身就是核心，原样返回。
        """
        if self.kind != "address":
            return _normalize(self.raw)
        markers = list(_ADMIN_MARKER.finditer(self.raw))
        tail = self.raw[markers[-1].end():] if markers else self.raw
        return _normalize(tail) or _normalize(self.raw)


@dataclass(frozen=True)
class ContactSources:
    """本会话见过的文本出处（工具返回 + 买家原话），已归一化。"""

    blob: str = ""

    def covers(self, claim: ContactClaim) -> bool:
        if claim.labeled:
            if claim.kind == "address":
                # 地址的确认卡口径：值切段后**全部**片段都有出处才算有——
                # 容的是"换排版复述"（去"邮编"字样、补"省"字），容不了拼凑编造。
                segments = [
                    _normalize(segment)
                    for segment in re.split(r"[^一-龥A-Za-z0-9]+", claim.raw)
                    if _normalize(segment)
                ]
                return bool(segments) and all(segment in self.blob for segment in segments)
            # 电话/邮编：数字串整体命中——"138-0000-0000"这类格式化值
            # 连字符切成的短片段逐个命中太弱，会把编造误判成有出处
            digits = re.sub(r"\D", "", claim.raw)
            return bool(digits) and digits in self.blob
        core = claim.core
        return bool(core) and core in self.blob


@dataclass
class ContactReport:
    claims: int = 0
    unsourced: list[ContactClaim] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.unsourced

    def to_dict(self) -> dict:
        return {
            "claims": self.claims,
            "unsourced": [{"kind": item.kind, "raw": item.raw} for item in self.unsourced],
        }


def _shortest_address_span(text: str, match: re.Match) -> str:
    """把匹配起点右移到仍能匹配的最后一个位置。

    中文没有词边界，`[一-龥]{2,4}(?:市)` 会从"是上海市"起匹配，
    把前一句话的尾字吞进来。逐位右移取最短的那个起点即可复原真正的地址头
    ——"上海市浦东新区世纪大道100号"而不是"是上海市浦东新区世纪大道100号"。
    """
    best = match.group(0)
    for pos in range(match.start() + 1, match.end()):
        inner = _ADDRESS.match(text, pos)
        if inner and inner.end() == match.end():
            best = inner.group(0)
    return best


def extract_contact_claims(text: str) -> list[ContactClaim]:
    """抽出回复里断言的收货字段，按出现位置排序。

    索要（"请提供收货地址"）天然不会命中：它没有具体值，
    而三条模式认的都是具体值。这条用例要的正是"去问"，判据不能反过来罚它。

    确认卡语境下追加**标签字段**扫描：确认卡是让买家核对的价结构，
    "地址：X"就是对 X 的断言——值既不是该字段的完整形态、也不是占位语、
    又只能来自编造（出处判定在 check_contact 里做）。三条形态模式抓不到的
    "你的默认地址"正是被这条补上的。
    """
    if not text:
        return []
    found: list[tuple[int, ContactClaim]] = []
    for match in _ADDRESS.finditer(text):
        found.append((match.start(), ContactClaim("address", _shortest_address_span(text, match))))
    for match in _PHONE.finditer(text):
        found.append((match.start(), ContactClaim("phone", match.group(0))))
    for match in _POSTAL.finditer(text):
        found.append((match.start(1), ContactClaim("postal", match.group(1))))
    if _CARD_CONTEXT.search(text):
        seen = {claim.raw for _, claim in found}
        for claim in _labeled_field_claims(text):
            if claim.raw not in seen:
                found.append((0, claim))  # 位置参与排序，标签断言排在形态断言后无妨
                seen.add(claim.raw)
    found.sort(key=lambda item: item[0])
    return [claim for _, claim in found]


def _payload_text(payload: Any) -> str:
    """把一份工具返回摊成可检索的文本。

    只接结构化返回：字符串负载先试 json.loads，解析不了就丢弃
    ——`[error] 地址无效：上海市…` 走的正是这条路，它里面的地址不该成为出处。
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return ""
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, ensure_ascii=False)
    return ""


def collect_contact_sources(
    tool_results: Iterable[Any] = (),
    buyer_texts: Iterable[str] = (),
) -> ContactSources:
    """把工具返回与买家原话汇总成出处池。

    买家原话必须算出处：买家自己报的地址，Agent 复述不是编造。
    """
    parts = [_payload_text(payload) for payload in tool_results]
    parts.extend(text or "" for text in buyer_texts)
    return ContactSources(blob=_normalize("".join(parts)))


def check_contact(reply: str, sources: ContactSources) -> ContactReport:
    """校验一条回复里所有收货字段的出处。"""
    report = ContactReport()
    for claim in extract_contact_claims(reply):
        report.claims += 1
        if not sources.covers(claim):
            report.unsourced.append(claim)
    return report


@dataclass
class SessionContactSources:
    """按会话累积出处，与 `SessionSources` 同一形态（按 shopping_session_id 分桶）。

    跨轮累积是必要的：买家第 1 轮给的地址，第 3 轮复述不算编造。
    按会话分桶同样是必要的：并发多会话共用一份出处，A 会话的地址会成为
    B 会话的出处，判据当场失效。
    """

    _texts: dict[str, list[str]] = field(default_factory=dict)

    def observe(
        self,
        session_id: str,
        tool_results: Iterable[Any] = (),
        buyer_texts: Iterable[str] = (),
    ) -> None:
        bucket = self._texts.setdefault(session_id, [])
        bucket.extend(_payload_text(payload) for payload in tool_results)
        bucket.extend(text or "" for text in buyer_texts)
        if len(bucket) > MAX_RETAINED_TEXTS:
            del bucket[: len(bucket) - MAX_RETAINED_TEXTS]

    def of(self, session_id: str) -> ContactSources:
        return ContactSources(blob=_normalize("".join(self._texts.get(session_id, ()))))

    def reset(self, session_id: str) -> None:
        self._texts.pop(session_id, None)
