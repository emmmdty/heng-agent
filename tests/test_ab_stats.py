# -*- coding: utf-8 -*-
"""A/B 统计函数（任务 A 第 2 项）：位置互换一致率 / 胜率 / 符号检验 / bootstrap CI。

口径全部冻结在交接文档「五之一」指标表里，本模块是口径的编码：
  - 位置互换自一致率 ≥ 90%，不达标该轮 judge 读数作废重跑（MT-Bench 标准做法）；
  - 显著性 = 去平局符号检验 p < 0.05 且按用例重采样的 bootstrap 95% CI 不含 0.5；
  - decisive pairs ≥ 30 是最低门槛，达不到就如实输出"样本不足"，
    **不许把"未达显著"表述成"没有差异"**——significance() 的 reasons
    就是给上游留的如实表述通道。

全部纯函数、零 LLM 成本、可复现（bootstrap 固定 seed）。
"""
import pytest

from scripts.eval.ab_stats import (
    bootstrap_ci_win_rate,
    decisive_pairs_gate,
    position_swap_consistency,
    significance,
    sign_test_p,
    win_rate_summary,
)


class TestPositionSwapConsistency:
    def test_all_consistent(self):
        rows = [
            {"case_id": "c1", "pair_index": 0, "verdict_ab": "a", "verdict_ba": "a"},
            {"case_id": "c1", "pair_index": 1, "verdict_ab": "b", "verdict_ba": "b"},
            {"case_id": "c2", "pair_index": 0, "verdict_ab": "tie", "verdict_ba": "tie"},
        ]
        out = position_swap_consistency(rows)
        assert out == {"n_pairs": 3, "n_consistent": 3, "rate": 1.0, "n_error": 0}

    def test_all_inconsistent(self):
        rows = [
            {"case_id": "c1", "pair_index": 0, "verdict_ab": "a", "verdict_ba": "b"},
            {"case_id": "c1", "pair_index": 1, "verdict_ab": "a", "verdict_ba": "tie"},
        ]
        out = position_swap_consistency(rows)
        assert out["n_pairs"] == 2 and out["n_consistent"] == 0 and out["rate"] == 0.0

    def test_mixed_rate(self):
        rows = [{"case_id": f"c{i}", "pair_index": 0, "verdict_ab": "a", "verdict_ba": "a" if i < 9 else "b"}
                for i in range(10)]
        out = position_swap_consistency(rows)
        assert out["rate"] == pytest.approx(0.9)

    def test_error_rows_excluded_from_denominator_but_counted(self):
        """None（judge 失败）不得静默进分母，也不得凭空消失——error 行要留数。"""
        rows = [
            {"case_id": "c1", "pair_index": 0, "verdict_ab": "a", "verdict_ba": None},
            {"case_id": "c2", "pair_index": 0, "verdict_ab": None, "verdict_ba": None},
            {"case_id": "c3", "pair_index": 0, "verdict_ab": "a", "verdict_ba": "a"},
        ]
        out = position_swap_consistency(rows)
        assert out == {"n_pairs": 1, "n_consistent": 1, "rate": 1.0, "n_error": 2}

    def test_empty_rows_rate_is_none_not_zero(self):
        """空样本报 None（无从判定），不伪造 0% 或 100%。"""
        assert position_swap_consistency([]) == {"n_pairs": 0, "n_consistent": 0, "rate": None, "n_error": 0}

    def test_tie_counts_as_consistent(self):
        rows = [{"case_id": "c1", "pair_index": 0, "verdict_ab": "tie", "verdict_ba": "tie"}]
        assert position_swap_consistency(rows)["n_consistent"] == 1


class TestWinRateSummary:
    def test_mixed(self):
        out = win_rate_summary(["a", "b", "a", "tie", None])
        # win_rate 的分母只含有效裁决（error 行不进分母）：2 胜 / 4 有效 = 0.5
        assert out == {
            "n": 5, "wins": 2, "losses": 1, "ties": 1, "n_error": 1,
            "n_decisive": 3, "win_rate": 0.5, "win_rate_excl_ties": pytest.approx(2 / 3),
        }

    def test_all_ties(self):
        out = win_rate_summary(["tie", "tie"])
        assert out["n_decisive"] == 0 and out["win_rate_excl_ties"] is None and out["win_rate"] == 0.0

    def test_all_errors(self):
        out = win_rate_summary([None, None])
        assert out["n_decisive"] == 0 and out["win_rate"] is None and out["win_rate_excl_ties"] is None

    def test_empty(self):
        out = win_rate_summary([])
        assert out["n"] == 0 and out["win_rate"] is None

    def test_invalid_element_raises_not_collapses(self):
        """拼错的臂名静默进分类 = 塌缩的另一种形态，必须报错。"""
        with pytest.raises(ValueError):
            win_rate_summary(["a", "A"])
        with pytest.raises(ValueError):
            win_rate_summary(["win"])


class TestSignTestP:
    def test_known_values(self):
        assert sign_test_p(5, 0) == pytest.approx(0.0625)
        assert sign_test_p(6, 0) == pytest.approx(0.03125)
        assert sign_test_p(5, 5) == pytest.approx(1.0)
        assert sign_test_p(8, 1) == pytest.approx(2 * 10 / 512)

    def test_symmetry(self):
        assert sign_test_p(8, 1) == sign_test_p(1, 8)

    def test_result_bounded(self):
        for wins in range(0, 12):
            for losses in range(0, 12):
                if wins + losses == 0:
                    continue
                p = sign_test_p(wins, losses)
                assert 0.0 <= p <= 1.0

    def test_zero_decisive_raises(self):
        with pytest.raises(ValueError):
            sign_test_p(0, 0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            sign_test_p(-1, 5)

    def test_exact_binomial_not_normal_approx(self):
        """小样本必须走精确二项：(3,0) 的双侧 p = 2/8 = 0.25，正态近似会低估。"""
        assert sign_test_p(3, 0) == pytest.approx(0.25)


class TestBootstrapCI:
    def _pairs_by_case(self, spec: dict[str, int]) -> list[tuple[str, int]]:
        return [(case, v) for case, v in spec.items()]

    def test_reproducible_same_seed(self):
        pairs = [("c1", 1)] * 5 + [("c2", 0)] * 3 + [("c3", 1)] * 4
        a = bootstrap_ci_win_rate(pairs, n_boot=500, seed=42)
        b = bootstrap_ci_win_rate(pairs, n_boot=500, seed=42)
        assert (a["lo"], a["hi"], a["point"]) == (b["lo"], b["hi"], b["point"])

    def test_all_wins_degenerate_ci(self):
        pairs = [("c1", 1), ("c2", 1), ("c3", 1)]
        out = bootstrap_ci_win_rate(pairs, n_boot=200)
        assert out["point"] == 1.0 and out["lo"] == 1.0 and out["hi"] == 1.0

    def test_resamples_by_case_not_by_pair(self):
        """按用例重采样是冻结口径：case 内的多对必须整组进出。

        构造 2 组：c1 全胜（10 对）、c2 全败（10 对）。按组重采样时
        只抽到 c1 的复制胜率是 1.0、只抽到 c2 的是 0.0——CI 必然覆盖 0 与 1；
        若错误地按对独立重采样，每组 10 对会把复制均值压向 0.5，CI 明显更窄。
        """
        pairs = [("c1", 1)] * 10 + [("c2", 0)] * 10
        out = bootstrap_ci_win_rate(pairs, n_boot=2000, seed=7)
        assert out["lo"] == 0.0 and out["hi"] == 1.0
        assert out["n_cases"] == 2 and out["n_pairs"] == 20

    def test_point_is_full_sample_mean(self):
        pairs = [("c1", 1), ("c2", 0), ("c3", 1), ("c3", 0)]
        out = bootstrap_ci_win_rate(pairs, n_boot=100)
        assert out["point"] == pytest.approx(0.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci_win_rate([])

    def test_invalid_indicator_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci_win_rate([("c1", 0.5)])

    def test_bad_params_raise(self):
        pairs = [("c1", 1)]
        with pytest.raises(ValueError):
            bootstrap_ci_win_rate(pairs, n_boot=0)
        with pytest.raises(ValueError):
            bootstrap_ci_win_rate(pairs, level=1.5)


class TestSignificance:
    def _swap(self, rate, n_error=0):
        return {"n_pairs": 10, "n_consistent": round(rate * 10) if rate is not None else 0,
                "rate": rate, "n_error": n_error}

    def _summary(self, wins, losses, ties=0, n_error=0):
        return win_rate_summary(["a"] * wins + ["b"] * losses + ["tie"] * ties + [None] * n_error)

    def test_significant_when_all_gates_pass(self):
        summary = self._summary(35, 10)
        swap = self._swap(0.95)
        ci = {"lo": 0.62, "hi": 0.88}
        out = significance(summary, swap, p_value=0.001, ci=ci)
        assert out["significant"] is True and out["judge_valid"] is True

    def test_decisive_below_30_blocks_even_with_tiny_p(self):
        """decisive < 30 不得下结论——哪怕 p 再小也不许判显著（不许凑显著）。"""
        summary = self._summary(12, 2)
        out = significance(summary, self._swap(0.95), p_value=0.001, ci={"lo": 0.7, "hi": 0.95})
        assert out["significant"] is False
        assert out["enough_pairs"] is False
        assert any("样本不足" in r for r in out["reasons"])

    def test_swap_below_90_vetoes(self):
        summary = self._summary(35, 10)
        out = significance(summary, self._swap(0.8), p_value=0.001, ci={"lo": 0.6, "hi": 0.9})
        assert out["judge_valid"] is False and out["significant"] is False

    def test_swap_unknown_vetoes(self):
        out = significance(self._summary(35, 10), self._swap(None), p_value=0.001, ci={"lo": 0.6, "hi": 0.9})
        assert out["judge_valid"] is False and out["significant"] is False

    def test_ci_containing_half_blocks(self):
        out = significance(self._summary(35, 10), self._swap(0.95), p_value=0.001, ci={"lo": 0.4, "hi": 0.8})
        assert out["ci_excludes_half"] is False and out["significant"] is False

    def test_insufficient_sample_reports_honestly(self):
        """n_decisive == 0：p 与 CI 无从计算，输出 None + 原因，不伪造读数。"""
        out = significance(self._summary(0, 0, ties=5), self._swap(0.9), p_value=None, ci=None)
        assert out["significant"] is False
        assert out["p_value"] is None and out["ci_excludes_half"] is None
        assert any("样本不足" in r for r in out["reasons"])

    def test_exactly_30_decisive_is_enough(self):
        summary = self._summary(30, 0)
        out = significance(summary, self._swap(0.95), p_value=0.001, ci={"lo": 0.9, "hi": 1.0})
        assert out["enough_pairs"] is True


class TestDecisivePairsGate:
    def test_pass_and_fail(self):
        assert decisive_pairs_gate({"n_decisive": 30})["sufficient"] is True
        assert decisive_pairs_gate({"n_decisive": 29})["sufficient"] is False

    def test_custom_threshold(self):
        assert decisive_pairs_gate({"n_decisive": 5}, min_pairs=5)["sufficient"] is True

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            decisive_pairs_gate({})
