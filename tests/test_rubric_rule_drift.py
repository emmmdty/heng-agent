# -*- coding: utf-8 -*-
"""用例判据里写死的免税额度，必须与规则表一致

要防的问题：**规则表改了，判据没跟着改，而外观是"Agent 答错了"。**

实测代价（2026-09-04，一轮 44 条的 full）：`stock-last-few-honesty` 判 FAIL 0.667，
两条 P0 都不通过，判词写得有理有据——

    到手价 610.88 元（小计 536 + 运费 72 + 关税 2.88）必须与工具一致
    关税 2.88 元必须来自工具：日本免税额度 500 元，超出的 36 元按 8% 计征

而规则表里 JP 的免税额度是 **480 元**（原生 10,000 JPY），不是 500。
十一期把它从硬编码的 `10_000 * 5`（手写 0.05 汇率）改成从汇率表推导，
额度随之 500 → 480，**交接文档把这条列进了"必须知道的行为变化"，
判据却没人改**。九期之后一直错着，直到这一轮才炸出来。

Agent 那一轮答的是 612.48 元（12,760 JPY），与工具逐位一致——**它是对的**。
judge 拿到的事实表也是对的（`| JP | 45 | 10000 JPY | 480 | 其他 8% |`），
所以判词里它自己把正确答案算了出来，然后因为与判据不符而判了不通过。

这就是本仓一直在防的那一类脱钩（同"工具 docstring 的目的国枚举 vs 规则表"）：
**脱钩之后的外观与"Agent 真错了"完全一样，没有任何告警。**
所以把它钉成判据——规则表是唯一事实源，判据里写死的额度必须能在里面找到。

判据刻意窄（同算式自洽那条）：只认"免税额度"后面**直接**跟的数额，
"超出免税额度的 1829 元""免税额度部分 29 元"这类说的是超出部分不是额度，
一律不解析。宁可漏报，不要一个会误伤的检查。
"""
import pytest

from scripts.eval.audit_cases import (
    de_minimis_claims,
    known_de_minimis_amounts,
)


class TestExtractingClaims:
    def test_plain_juxtaposition(self):
        assert de_minimis_claims("日本免税额度 480 元，超出的 56 元按 8% 计征") == [(480.0, "元")]

    def test_native_pair_in_parentheses(self):
        claims = de_minimis_claims("免税额度 480 元（原生口径 10,000 JPY）")
        assert (480.0, "元") in claims
        assert (10000.0, "JPY") in claims

    def test_slash_separated_pair(self):
        assert de_minimis_claims("免税额度必须报成 1170 元 / 150 欧元（与规则表一致）") == [
            (1170.0, "元"), (150.0, "欧元"),
        ]

    def test_parenthesised_pair(self):
        assert de_minimis_claims("小计未超过美国免税额度（800 美元 / 5680 元）") == [
            (800.0, "美元"), (5680.0, "元"),
        ]

    def test_overage_is_not_a_claim(self):
        """"超出免税额度的 1829 元"说的是超出部分，不是额度——解析它就是误报。"""
        assert de_minimis_claims("只对超出免税额度的 1829 元计征") == []

    def test_the_word_part_is_not_a_claim(self):
        assert de_minimis_claims("必须是超出免税额度部分 29 元计征") == []

    def test_mention_without_amount(self):
        assert de_minimis_claims("指出了美国免税额度内关税为 0") == []


class TestAgainstTheRuleTable:
    def test_the_stale_value_that_actually_shipped_is_rejected(self):
        """十一期改额度之前的原值。这条测试就是那次 FAIL 的回归。"""
        assert 500.0 not in known_de_minimis_amounts()

    def test_every_supported_destination_is_covered(self):
        known = known_de_minimis_amounts()
        for amount in (5000.0, 800.0, 5680.0, 150.0, 1170.0, 10000.0, 480.0, 400.0, 2120.0):
            assert amount in known, f"{amount} 应当是规则表里认得的免税额度口径"

    def test_live_cases_agree_with_the_rule_table(self):
        """真正的门禁：跑在 eval/cases.yaml 上。

        写成测试而不只是脚本里的一项，是因为规则表变更多半发生在改 app/ 的时候，
        而那时候人不一定会想起来跑 audit_cases。
        """
        import yaml

        from scripts.eval.audit_cases import CASES_PATH

        known = known_de_minimis_amounts()
        offenders = []
        for case in yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]:
            for level, items in (case.get("rubric") or {}).items():
                for text in items or []:
                    for amount, unit in de_minimis_claims(text):
                        if amount not in known:
                            offenders.append((case["id"], level, amount, unit, text))
        assert not offenders, "判据里的免税额度与规则表对不上：" + "；".join(
            f"{c}/{lv} 写着 {a}{u}" for c, lv, a, u, _ in offenders
        )
