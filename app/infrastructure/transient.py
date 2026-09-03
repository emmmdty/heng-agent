# -*- coding: utf-8 -*-
"""transient

上游瞬时故障判定。模型层与编排层共用同一份判据，避免两处标记表各自漂移。

为什么按 message 特征匹配而不是按异常类型：OpenAI 兼容网关把限流写在 SSE 流中间时，
抛出的是笼统的 openai.APIError，类型上无法与真实业务错误区分，只能看文案；
同一套判据还要覆盖 httpx 超时与网关 5xx。
"""
from __future__ import annotations

import re

# 可重试的 HTTP 状态码：429 限流 + 5xx 网关侧故障。
# 400/401/403/404 一律不重试——它们是请求本身的问题，重试只会掩盖真实原因。
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# OpenAI SDK 异常的 str() 形态就是 "Error code: 502"，**没有** "Bad Gateway" 字样。
# 这一条是踩坑换来的：判据表原先只有 "bad gateway" / "service unavailable" 这类
# 文案标记，看着覆盖了 5xx，实际永远匹配不到 SDK 抛出的真实异常。
_STATUS_IN_MESSAGE = re.compile(r"error code:\s*(\d{3})")

# 全小写匹配。"throttling" 来自实测：网关限流返回 code=Throttling.Concurrency
_TRANSIENT_ERROR_MARKERS = (
    "too many concurrent",
    "rate limit",
    "request rate",
    "too many requests",
    "throttling",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "internal server error",
    "bad gateway",
    "connection reset",
    "connection error",
)


def is_transient_error(error: BaseException) -> bool:
    """判断异常是否属于可重试的上游瞬时故障。

    三级判据，从最可靠到最兜底：
      1. 异常自带的 `status_code`（httpx / openai 异常都有），最准；
      2. 消息里的 "Error code: NNN"——OpenAI SDK 的 str() 就长这样，
         纯数字、无文案，所以必须单独匹配；
      3. 文案标记，兜住网关把限流写进 SSE 流中间那类没有状态码的场景。
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS_CODES:
        return True

    message = str(error).lower()
    found = _STATUS_IN_MESSAGE.search(message)
    if found and int(found.group(1)) in _TRANSIENT_STATUS_CODES:
        return True

    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


def describe_error(err: BaseException) -> str:
    """把异常渲染成"看得出是什么"的一行。

    存在的理由：降级路径普遍写成 `logger.warning("...：%s", err)`，而
    httpx 的连接类异常 str() 往往是空串，日志就成了 `降级关键词召回：` 后面什么都没有。
    降级本身是设计好的行为，**但降级的原因必须留下**——否则一条静默降级
    会直接污染评测读数而查无实据（实测：一条 query 降级让 Recall@8 掉 1pt，
    日志说不出是超时还是连接重置）。
    """
    message = str(err).strip()
    name = type(err).__name__
    return f"{name}: {message}" if message else name
