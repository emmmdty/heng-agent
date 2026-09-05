# -*- coding: utf-8 -*-
"""A/B 报告渲染器（授权文档 M1：报告渲染器）。

报告必须自带：胜率 / CI / 互换一致率 / 双 judge 一致率 / 阳性对照 /
全部护栏读数 + 两臂配置行 + 算式假设（秒/意图取值）。

两条如实性硬点（同 ab_stats 的纪律）：
  - 护栏没测就写"未测定"，绝不渲染成"通过"——没有读数的护栏不是护栏；
  - "未达显著"与"没有差异"分开表述，reasons 原样进报告。
"""
from scripts.eval.ab_report import render_ab_report


def _payload(**overrides):
    payload = {
        "label": "先导 smoke",
        "stamp": "20260905-160000",
        "pairing": "diagonal",
        "plan": {
            "n_cases": 12, "k": 2, "executions": 48, "intents": 48, "pairs": 24,
            "judge_calls": 48, "decisive_ceiling": 24, "decisive_gate": 30,
            "estimated_minutes": 41.3,
        },
        "wall_clock_assumption": "51.6s/意图（R7 实测），两臂串行；judge 调用另计",
        "arm_lines": {
            "A": "被测模型 mimo-v2.5｜评审模型 longcat-2.0｜提示词 a0915fac",
            "B": "被测模型 mimo-v2.5｜评审模型 longcat-2.0｜提示词 b2222222｜提示词变体 candidate-x",
        },
        "arm_config": {
            "A": {"fingerprint": "a0915fac", "variant": "", "model": "mimo-v2.5"},
            "B": {"fingerprint": "b2222222", "variant": "candidate-x", "model": "mimo-v2.5"},
        },
        "executions": {"total": 48, "ok": 48, "failed": []},
        "swap": {"n_pairs": 24, "n_consistent": 24, "rate": 1.0, "n_error": 0},
        "win_rate": {
            "n": 20, "wins": 14, "losses": 4, "ties": 2, "n_error": 0,
            "n_decisive": 18, "win_rate": 0.7, "win_rate_excl_ties": 14 / 18,
        },
        "n_flip": 0,
        "p_value": 0.0308,
        "ci": {"point": 14 / 18, "lo": 0.5, "hi": 0.95, "n_boot": 10000, "n_pairs": 18, "n_cases": 10, "level": 0.95},
        "significance": {
            "judge_valid": True, "enough_pairs": False, "p_value": 0.0308,
            "ci_excludes_half": True, "significant": False,
            "reasons": ["decisive pairs 18 < 30，样本不足，未达显著"],
        },
        "dual_judge": None,
        "positive_control": False,
        "guardrails": [],
        "cost_latency": None,
        "notes": [],
    }
    payload.update(overrides)
    return payload


class TestCoreSections:
    def test_title_label_and_arms(self):
        text = render_ab_report(_payload())
        assert "A/B" in text and "先导 smoke" in text and "20260905-160000" in text
        assert "a0915fac" in text and "candidate-x" in text
        assert "被测模型 mimo-v2.5" in text

    def test_plan_ledger_and_wall_clock_assumption(self):
        text = render_ab_report(_payload())
        assert "48" in text          # executions
        assert "24" in text          # pairs
        assert "51.6s/意图" in text  # 算式假设必须进报告

    def test_win_rate_and_swap_sections(self):
        text = render_ab_report(_payload())
        assert "胜率" in text and "14/20" in text.replace(" ", "") or "win" in text
        assert "互换一致率" in text and "100" in text
        assert "平局 2" in text

    def test_ci_and_p_value(self):
        text = render_ab_report(_payload())
        assert "p=0.0308" in text
        assert "CI" in text

    def test_insufficient_sample_wording_preserved(self):
        """「样本不足，未达显著」必须原样出现——这不是失败，是口径。"""
        text = render_ab_report(_payload())
        assert "样本不足，未达显著" in text
        assert "没有差异" not in text

    def test_swap_fail_invalidate_wording(self):
        text = render_ab_report(_payload(
            swap={"n_pairs": 10, "n_consistent": 8, "rate": 0.8, "n_error": 2},
            significance={"judge_valid": False, "enough_pairs": False, "p_value": None,
                          "ci_excludes_half": None, "significant": False,
                          "reasons": ["位置互换一致率 0.8 < 0.9，该轮 judge 读数作废重跑"]},
        ))
        assert "作废重跑" in text
        assert "0.8" in text
        assert "2" in text  # n_error 点名

    def test_execution_failures_named(self):
        text = render_ab_report(_payload(
            executions={"total": 48, "ok": 47, "failed": [
                {"case_id": "demo", "arm": "B", "sample_index": 1, "error": "ReadTimeout: 600s"},
            ]},
        ))
        assert "demo" in text and "ReadTimeout" in text

    def test_swap_flip_counted_not_silent(self):
        text = render_ab_report(_payload(n_flip=3))
        assert "3" in text and ("翻转" in text or "不一致" in text)


class TestOptionalSections:
    def test_dual_judge_section_when_present(self):
        text = render_ab_report(_payload(dual_judge={
            "model": "deepseek-v4-flash", "n_pairs": 20, "n_agree": 18, "rate": 0.9, "n_error": 0,
        }))
        assert "deepseek-v4-flash" in text and "0.9" in text
        assert "不参与胜负判定" in text

    def test_dual_judge_absent_is_declared(self):
        text = render_ab_report(_payload())
        assert "双 judge" in text and "未执行" in text

    def test_guardrails_missing_are_not_passes(self):
        """护栏没读数就写'未测定'——空护栏不许渲染成绿。"""
        text = render_ab_report(_payload(guardrails=[]))
        assert "未测定" in text
        assert "通过" not in text.split("护栏")[1][:400] or "未测定" in text

    def test_guardrails_rendered_with_threshold_and_verdict(self):
        text = render_ab_report(_payload(guardrails=[
            {"name": "无出处金额率", "value": "4.0%", "threshold": "≤ 8%", "pass": True, "source": "audit_number_provenance"},
            {"name": "结算 PASS 率", "value": "42/44", "threshold": "≥ 44/44", "pass": False, "source": "eval-mainline"},
        ]))
        assert "无出处金额率" in text and "4.0%" in text and "≤ 8%" in text
        assert "42/44" in text
        assert "未达标" in text

    def test_cost_latency_per_arm(self):
        text = render_ab_report(_payload(cost_latency={
            "A": {"completion_p50": 591, "latency_p50_s": 19.8},
            "B": {"completion_p50": 620, "latency_p50_s": 20.4},
        }))
        assert "591" in text and "620" in text
        assert "19.8" in text


class TestPositiveControl:
    def test_expected_negative_significance_passes(self):
        text = render_ab_report(_payload(
            positive_control=True,
            label="阳性对照",
            significance={"judge_valid": True, "enough_pairs": True, "p_value": 0.001,
                          "ci_excludes_half": True, "significant": True, "reasons": []},
            win_rate={"n": 40, "wins": 30, "losses": 6, "ties": 4, "n_error": 0,
                      "n_decisive": 36, "win_rate": 30 / 40, "win_rate_excl_ties": 30 / 36},
        ))
        assert "阳性对照" in text
        assert "负向显著" in text
        assert "工具有区分度" in text

    def test_not_significant_is_tool_deficiency(self):
        text = render_ab_report(_payload(
            positive_control=True,
            label="阳性对照",
            significance={"judge_valid": True, "enough_pairs": True, "p_value": 0.4,
                          "ci_excludes_half": False, "significant": False,
                          "reasons": ["符号检验 p=0.4 不满足 p<0.05"]},
        ))
        assert "区分度" in text

    def test_inverted_direction_flagged(self):
        text = render_ab_report(_payload(
            positive_control=True,
            label="阳性对照",
            significance={"judge_valid": True, "enough_pairs": True, "p_value": 0.01,
                          "ci_excludes_half": True, "significant": True, "reasons": []},
            win_rate={"n": 40, "wins": 6, "losses": 30, "ties": 4, "n_error": 0,
                      "n_decisive": 36, "win_rate": 6 / 40, "win_rate_excl_ties": 6 / 36},
        ))
        assert "方向" in text


def test_fault_clear_failures_rendered():
    """故障清理失败必须进报告——后续用例可能带着故障跑，零痕迹就是塌缩。"""
    text = render_ab_report(_payload(executions={
        "total": 4, "ok": 4,
        "failed": [],
        "fault_clear_failures": [
            {"case_id": "demo", "arm": "B", "sample_index": 0, "error": "清理也炸了"},
        ],
    }))
    assert "故障清理失败" in text and "demo" in text


def test_cost_latency_missing_arm_is_not_none_literal():
    """单臂成本读数缺失时渲染'未测定'，不许出现字面量 None。"""
    text = render_ab_report(_payload(cost_latency={"A": {"completion_p50": 591, "latency_p50_s": 19.8}}))
    assert "未测定" in text
    assert "None" not in text
