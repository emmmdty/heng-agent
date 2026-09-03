# -*- coding: utf-8 -*-
"""算式自洽校验：回复里写出来的算式，等号两边必须对得上。

判据来自 full3 实测（`taxable-base-us-explained` FAIL 0.517）：
Agent 第 2 步明确写出应税基数 `$886.34 − $800.00 = $86.34`，
第 4 步却写成 `关税 = $886.34 × 7.5% = $6.48`——**乘错了数，而结果是对的**
（6.48 来自工具）。886.34 × 7.5% 实际是 66.48。

与十期实测的 `1,199 × 12% ≈ ¥3.48` 是同一个形状，中间隔着两次修
（十一期补 taxable_base_major 字段、写进提示词），**都没能拦住**。
提示词拦不住的，就该由确定性判据接管。

**与金额出处校验互补**：这次错的三个数（886.34 / 7.5% / 6.48）**都有工具出处**，
出处校验对它完全无感。出处管"数字从哪来"，这条管"写出来的过程算不算得通"。
"""
import pytest

from app.application.harness.arithmetic_check import check_arithmetic, extract_equations


class TestExtraction:
    def test_percentage_multiplication(self):
        eqs = extract_equations("关税 = $886.34 × 7.5% = $6.48")
        assert len(eqs) == 1
        assert eqs[0].left == 886.34 and eqs[0].rate == 7.5 and eqs[0].result == 6.48

    def test_accepts_asterisk_and_approx(self):
        assert extract_equations("1,199 * 12% ≈ ¥3.48")[0].left == 1199.0

    def test_ignores_prose_without_an_equation(self):
        """自然语言里的"大约两成"不解析——解析它只会制造误报
        （同金额出处校验"宁可漏报不误报"）。"""
        assert extract_equations("这大概是原价的两成，算下来挺划算") == []

    def test_multiple_equations_in_one_reply(self):
        text = "第一件 100 × 10% = 10 元；第二件 200 × 10% = 20 元"
        assert len(extract_equations(text)) == 2


class TestConsistency:
    def test_correct_equation_passes(self):
        report = check_arithmetic("应税基数 86.34 × 7.5% = 6.48 美元")
        assert report.ok and not report.problems

    def test_the_real_defect_is_caught(self):
        """full3 实测的那一行。"""
        report = check_arithmetic("关税 = $886.34 × 7.5% = $6.48")
        assert not report.ok
        assert report.problems[0].expected == pytest.approx(66.4755, abs=0.01)

    def test_the_phase_ten_defect_is_caught(self):
        """十期实测的那一行——同一个形状，隔了九期还在。"""
        report = check_arithmetic("关税 1,199 × 12% ≈ ¥3.48")
        assert not report.ok

    def test_rounding_is_tolerated(self):
        """86.34 × 7.5% = 6.4755，写成 6.48 是正常取整。"""
        assert check_arithmetic("86.34 × 7.5% = 6.48").ok

    def test_tolerance_does_not_swallow_magnitude_errors(self):
        """容差按相对 2% 取：容得下两位小数取整，容不下差一个数量级。"""
        assert not check_arithmetic("100 × 10% = 100").ok

    def test_zero_result_is_handled(self):
        assert check_arithmetic("0 × 7.5% = 0").ok
        assert not check_arithmetic("100 × 7.5% = 0").ok

    def test_empty_text_is_ok(self):
        assert check_arithmetic("").ok
        assert check_arithmetic(None).ok  # type: ignore[arg-type]


class TestProblemDescription:
    def test_problem_says_what_it_should_have_been(self):
        """报错要能直接指出正确值，否则读的人还得自己算一遍。"""
        problem = check_arithmetic("关税 = $886.34 × 7.5% = $6.48").problems[0]
        assert "886.34" in problem.raw and "66.4" in f"{problem.expected:.2f}"

    def test_to_dict_is_serialisable(self):
        report = check_arithmetic("100 × 10% = 5")
        payload = report.to_dict()
        assert payload["problems"][0]["written"] == 5.0


class TestWiring:
    """接线判据：写好了没人调用等于没写（踩坑 37 / 42 两次教训）。"""

    def test_orchestrator_calls_it_at_turn_end(self):
        import inspect

        from app.application.agents import orchestrator

        source = inspect.getsource(orchestrator)
        assert "self._check_arithmetic(intent, final_text, events)" in source
        assert "arith.inconsistent" in source

    def test_event_type_is_registered(self):
        """事件类型没注册的话 publish 会抛 ValueError，把整轮打挂。"""
        from app.infrastructure.eventbus import EVENT_TYPES

        assert "arith.inconsistent" in EVENT_TYPES

    def test_cache_hit_turns_are_skipped(self):
        """缓存命中的轮次跳过：那是上一次已校验过的回复在重放。"""
        import inspect

        from app.application.agents import orchestrator

        source = inspect.getsource(orchestrator)
        block = source[source.index("if not cache_hit:"):]
        assert block.index("_check_arithmetic") < block.index("_record_conversation")


class TestAgainstRealReplies:
    """拿真实回复验一遍：既看抓得住，也看误报率（同验 L3/L4 的方法）。

    实测（346 条历史回复）：抽出 4 个显式算式、命中 2 处不自洽、**零误报**。
    两处都是真错，其中一处来自 judge 判 PASS 的用例——
    **确定性判据当场抓到了 judge 漏掉的错**，这正是"能拿回确定性判据的
    就别留给 judge"这条主线的又一个实证。
    """

    def test_the_two_real_defects(self):
        """两处实测原文。写成测试是为了将来改容差时能立刻看出会不会放过它们。"""
        for text, expected in (
            ("关税 1,199 × 12% ≈ ¥3.48", 143.88),
            ("€384.49 × 12% = €28.14", 46.1388),
        ):
            report = check_arithmetic(text)
            assert not report.ok, text
            assert report.problems[0].expected == pytest.approx(expected, abs=0.01)

    def test_narrow_by_design(self):
        """判据刻意窄：只认显式写出来的百分比乘法。

        346 条回复里只抽出 4 个算式——**漏报很多，但误报为零**。
        这与金额出处校验"给出的是下界"是同一条纪律：
        判据的可信度比覆盖率重要，误报会让人不再看它。
        """
        prose = "到手价约为原价的九成，运费另计，关税按超出部分收"
        assert check_arithmetic(prose).equations == 0
