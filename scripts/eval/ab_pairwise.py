# -*- coding: utf-8 -*-
"""成对比较器（任务 A 第 1 项）：transcript 对 → {winner, tie, 判词}。

指标口径冻结在交接文档「五之一」：A/B 两臂各采 k 次，longcat-2.0 盲判成对比较
（judge 不知道版本身份），报 win/tie/loss——方法论对齐 Chatbot Arena / AlpacaEval。

本模块只含确定性部分，LLM 调用通过 judge_call 注入（测试用假 judge，
真实适配器在 ab_run.py）。两个设计决定必须知道：

**盲判是读数有效性的前提。** 提示词只给"回复1 / 回复2"，臂名与版本身份
进不了 judge 的视野——否则量到的是自我偏好而不是提示词差异。

**脏输出绝不静默塌缩。** rejudge 的教训（二十三期）：脏判词塌缩成少计条目，
读数看着正常、其实不可比。这里塌缩的形态更隐蔽——把"裁决: 3"当平局、
把"裁决: 12"截成 1，胜率会悄悄偏向某一侧。所以解析不了的输出一律
VerdictParseError 向上抛，由调用方记 error 行（带用例名），宁可少一对，
不进一条假读数。矛盾值（一处 1 一处 2）同理：这是 judge 降级信号，
不是"取第一个"就能糊弄过去的事。
"""
from __future__ import annotations

import re
from typing import Awaitable, Callable

# 裁决行：标签（裁决/胜者/WINNER/VERDICT）+ 值（1/2/平局/tie，全半角等价，
# 允许"回复1"式前缀，允许值后跟同一行的自由文本）。值后面必须不是数字，
# 否则 "12" 会被截成 "1" 静默通过。
_VERDICT_RE = re.compile(
    r"(?:裁决|胜者|winner|verdict)\s*[:：]\s*(?:回复\s*)?([12１２]|平局|平|tie)",
    re.IGNORECASE,
)
# 值后面紧跟着数字（全半角都算）→ 这次匹配不是完整值，视为脏
_AFTER_VALUE_RE = re.compile(r"[0-9０-９]")
_RATIONALE_RE = re.compile(r"(?:理由|rationale|reason)\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)

# 理由段内的裁决行：只认**行首**标签。行中出现的（"我本来想写 裁决: 2，
# 但更正为回复1"）是引用不是裁决，忽略；独占一行的（理由后又判了一次且
# 值不同）是真矛盾，要抓——遮蔽不能把真矛盾一起吞掉。
_VERDICT_LINE_RE = re.compile(
    r"^\s*(?:裁决|胜者|winner|verdict)\s*[:：]\s*(?:回复\s*)?([12１２]|平局|平|tie)",
    re.IGNORECASE | re.MULTILINE,
)
_TIE_TOKENS = {"平局", "平", "tie"}


class VerdictParseError(ValueError):
    """判词脏数据。向上抛、由调用方记 error 行留名，不许在本层塌缩成读数。"""


def build_pair_prompt(
    case_prompt_text: str,
    transcript_left: str,
    transcript_right: str,
    ground_truth: str = "",
    prior_context: str = "",
) -> str:
    """构造盲判提示词：judge 只看到"回复1/回复2"，看不到任何版本身份。

    平局口径：两份回复质量相当、没有一方明显更好——不设倾向性描述，
    描述写得太具体会替 judge 做决定。

    ground_truth（build_ground_truth 的产物）与会话前置事实是**实现决策不是
    口径变更**（授权文档 M1）：judge 要判"哪份回复的事实更可靠"，手上必须有
    工具口径的事实基准——本仓 rubric judge 一贯喂事实表，成对比较没有理由
    更穷。盲判约束只针对**版本身份**，不针对事实表；不喂事实表的成对比较
    只能比文笔。prior_context 防的是另一类误判：memory-recall 应用历史偏好
    会被当成无据编造（eval_regression 的 JUDGE_SYSTEM_PROMPT 同一条先例）。
    两段为空就整段省略，不给 judge 留"事实表：未知"的空段落。
    """
    ground_block = f"【商品库事实表】\n{ground_truth}\n\n" if ground_truth else ""
    prior_block = f"【会话前置事实】\n{prior_context}\n\n" if prior_context else ""
    return (
        "你是电商购物助手的评测评审。下面是同一位买家的同一请求，"
        "以及两个助手回复（回复1 与 回复2）。请判断哪个回复更好。\n\n"
        f"{ground_block}"
        f"{prior_block}"
        "【买家请求】\n"
        f"{case_prompt_text}\n\n"
        "【回复1】\n"
        f"{transcript_left}\n\n"
        "【回复2】\n"
        f"{transcript_right}\n\n"
        "评判标准（按重要性排序）：事实正确性（价格/库存/订单数字必须可靠）、"
        "是否如实说明系统能力边界、是否回答了买家的实际问题、表达是否清楚。\n"
        "两者质量相当、没有一方明显更好时判平局。\n\n"
        "输出格式（严格遵守，共两行）：\n"
        "裁决: 1|2|平局\n"
        "理由: <一句话，说明裁决依据>"
    )


def parse_verdict(raw: str | None) -> dict:
    """judge 原始输出 → {"winner": "1"|"2"|"tie", "rationale": str}。

    容错：大小写、全半角冒号与数字、"回复1"前缀、值后跟自由文本、
    裁决行不在末尾、同值多行。
    脏输出一律 VerdictParseError：空输出 / 无标签 / 值非法 / 值互相矛盾 /
    理由缺失。错误信息带原文，方便直接进 error 行留名。
    """
    if raw is None or not raw.strip():
        raise VerdictParseError(f"判词为空（judge 原始输出：{raw!r}）")

    # 先定位理由段：理由里**引用**裁决字样（"我本来想写 裁决: 2，但更正为
    # 回复1"）不是矛盾，直接全文找裁决行会误丢对——误丢不进 error 行也看
    # 不出来，decisive pairs 无端缩水。策略：理由段之外全文找；理由段之内
    # 只认行首裁决行（引用在行中，真矛盾独占一行）。
    masked = raw
    rationale_match = _RATIONALE_RE.search(raw)
    rationale_span = ""
    if rationale_match is not None:
        rationale_span = rationale_match.group(0)
        masked = (
            raw[:rationale_match.start()]
            + " " * (rationale_match.end() - rationale_match.start())
            + raw[rationale_match.end():]
        )

    def _normalize(token: str) -> str:
        return "tie" if token in _TIE_TOKENS else token.translate(str.maketrans("１２", "12"))

    matches = []
    for match in _VERDICT_RE.finditer(masked):
        token = match.group(1)
        # "12" 这种连续数字不能被截成 "1"——匹配后紧跟数字即为脏
        tail = masked[match.end():match.end() + 1]
        if tail and _AFTER_VALUE_RE.match(tail):
            raise VerdictParseError(f"裁决值不完整或非法：{token}…（judge 原始输出：{raw!r}）")
        matches.append(_normalize(token))
    matches.extend(_normalize(m.group(1)) for m in _VERDICT_LINE_RE.finditer(rationale_span))

    if not matches:
        raise VerdictParseError(f"找不到裁决行（需要『裁决: 1|2|平局』）：{raw[:120]!r}")
    if len(set(matches)) > 1:
        raise VerdictParseError(f"裁决值互相矛盾（{matches}）——judge 降级信号，不计入读数：{raw[:120]!r}")

    if rationale_match is None or not rationale_match.group(1).strip():
        raise VerdictParseError(f"理由缺失或为空——只回一个数字不给理由是降级输出：{raw[:120]!r}")

    return {"winner": matches[0], "rationale": rationale_match.group(1).strip()}


def map_winner(winner: str, order: tuple[str, str]) -> str:
    """把 judge 视角的 "1"/"2"/"tie" 映射成臂名。order=(左位臂名, 右位臂名)。

    位置互换（同一对样本正反各判一次）就靠这个映射换序：互换后同样的
    原始裁决映射成相反臂名，一致率在此基础上比对。
    """
    if winner == "tie":
        return "tie"
    if winner == "1":
        return order[0]
    if winner == "2":
        return order[1]
    raise ValueError(f"非法 winner：{winner!r}（应为 '1'/'2'/'tie'）")


def build_pairs(samples_a: list, samples_b: list, mode: str = "diagonal") -> list[tuple]:
    """两臂样本 → 成对比较列表 [(transcript_a, transcript_b, pair_index)]。

    mode="diagonal"（默认）：zip 对齐，k=2 两臂各采 2 次得 2 对/用例；
    mode="cross"：笛卡尔积（AlpacaEval 式全组合，judge 成本按 k² 增长）。
    """
    if not samples_a or not samples_b:
        raise ValueError("两臂样本都不能为空——空臂配不出对，静默返回空表是假绿")
    if mode == "diagonal":
        if len(samples_a) != len(samples_b):
            raise ValueError(f"diagonal 配对要求两臂等长，收到 {len(samples_a)} vs {len(samples_b)}")
        return [(a, b, i) for i, (a, b) in enumerate(zip(samples_a, samples_b))]
    if mode == "cross":
        return [
            (a, b, index)
            for index, (a, b) in enumerate((x, y) for x in samples_a for y in samples_b)
        ]
    raise ValueError(f"未知 mode：{mode!r}（应为 diagonal/cross）")


async def judge_pair(
    judge_call: Callable[[str], Awaitable[str]],
    case_prompt_text: str,
    transcript_a: str,
    transcript_b: str,
    order: tuple[str, str] = ("a", "b"),
    ground_truth: str = "",
    prior_context: str = "",
) -> dict:
    """跑一次成对比较：提示词 → judge → 解析 → 映射臂名。

    judge_call 抛的异常（网络/限流）与解析抛的 VerdictParseError 都**原样向上抛**，
    由调用方记 error 行——本层不做任何静默兜底。
    """
    prompt = build_pair_prompt(
        case_prompt_text, transcript_a, transcript_b,
        ground_truth=ground_truth, prior_context=prior_context,
    )
    raw = await judge_call(prompt)
    verdict = parse_verdict(raw)
    return {
        "winner": map_winner(verdict["winner"], order),
        "rationale": verdict["rationale"],
        "raw": raw,
    }
