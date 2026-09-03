# -*- coding: utf-8 -*-
"""arithmetic_check —— 回复里写出来的算式必须自洽

判据一句话：**回复里出现的 `A × B% = C`，等号两边要对得上。**

来源（full3 实测，`taxable-base-us-explained` FAIL 0.517）：Agent 第 2 步
明确写出应税基数 `$886.34 − $800.00 = $86.34`，第 4 步却写成

    关税 = $886.34 × 7.5% = $6.48

`886.34 × 7.5%` 实际是 66.48。**结果是对的（来自工具），过程是错的**——
它自己刚写对的基数，下一行又把小计抄了进去。
这与十期实测的 `1,199 × 12% ≈ ¥3.48` 是同一个形状，中间隔着两次修
（十一期给工具补 `taxable_base_major`、把规则写进提示词），**都没能拦住**。
按本仓一贯的判断：**提示词拦不住的，就该由确定性判据接管。**

**与金额出处校验互补，不能合并**：

    金额出处   数字**从哪来**——没有工具出处即可疑
    算式自洽   数字**怎么来**——写出来的过程算不通

这次错的三个数（886.34 / 7.5% / 6.48）**都有工具出处**，
出处校验对它完全无感。所以必须是两条独立判据。

**只验自洽，不验业务规则**：`886.34 × 7.5% = 6.48` 算不通这件事，
与关税规则无关，纯算术就判得死。**判据越薄，越不会误判。**

**范围刻意收窄**（同金额出处校验的"宁可漏报不误报"）：
只认显式写出来的乘法算式（含 `×`/`*` 与 `=`/`≈`），
自然语言里"大约是原价的两成"这类表述不解析——解析它只会制造误报。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# `886.34 × 7.5% = 6.48`：允许千分位、货币符号、乘号写作 × 或 *、等号写作 = 或 ≈
_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_EQUATION = re.compile(
    r"[¥￥$€£]?\s*(" + _NUMBER + r")\s*[×*]\s*(" + _NUMBER + r")\s*%"
    r"\s*[=≈]\s*[¥￥$€£]?\s*(" + _NUMBER + r")",
)

# 容差按**相对**取：能容下两位小数的展示取整（86.34 × 7.5% = 6.4755 → 写 6.48），
# 容不下差一个数量级的错。绝对容差不行——金额量级跨度太大，
# 定小了会误伤大额取整，定大了会放过小额的量级错。
_RELATIVE_TOLERANCE = 0.02
# 结果为 0 时相对容差失效，另给一个绝对下限
_ABSOLUTE_FLOOR = 0.01


def _parse(literal: str) -> float:
    return float(literal.replace(",", ""))


@dataclass(frozen=True)
class Equation:
    left: float
    rate: float
    result: float
    raw: str

    @property
    def expected(self) -> float:
        return self.left * self.rate / 100.0

    @property
    def consistent(self) -> bool:
        tolerance = max(_ABSOLUTE_FLOOR, abs(self.expected) * _RELATIVE_TOLERANCE)
        return abs(self.result - self.expected) <= tolerance


@dataclass(frozen=True)
class Problem:
    raw: str
    written: float
    expected: float

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "written": self.written,
            "expected": round(self.expected, 4),
        }


@dataclass
class ArithmeticReport:
    equations: int = 0
    problems: list[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict:
        return {
            "equations": self.equations,
            "problems": [problem.to_dict() for problem in self.problems],
        }


def extract_equations(text: str) -> list[Equation]:
    """抽出显式写出来的百分比乘法算式。"""
    if not text:
        return []
    found = []
    for match in _EQUATION.finditer(text):
        found.append(
            Equation(
                left=_parse(match.group(1)),
                rate=_parse(match.group(2)),
                result=_parse(match.group(3)),
                raw=match.group(0).strip(),
            ),
        )
    return found


def check_arithmetic(text: str) -> ArithmeticReport:
    """校验一条回复里所有显式算式的自洽性。"""
    report = ArithmeticReport()
    for equation in extract_equations(text):
        report.equations += 1
        if not equation.consistent:
            report.problems.append(
                Problem(
                    raw=equation.raw,
                    written=equation.result,
                    expected=equation.expected,
                ),
            )
    return report
