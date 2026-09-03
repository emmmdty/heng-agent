# -*- coding: utf-8 -*-
"""跑测方差工具。

**没有方差就没有显著性**："这次 0.973、上次 0.95，是改好了还是抖了一下"——
不量方差只能靠感觉回答，而靠感觉回答的结果是：
真退化被当成抖动放过，抖动被当成退化去"修"。

工具第一版自己就栽过一次：它把"改判据之前"和"改判据之后"的读数混在一起，
算出 0.325 的"波动"，而那其实是判据被改对了。
**rubric 本身就是配置的一部分。**
"""
from scripts.eval.variance import collect_scores, config_key, summarize


def _report(run: str, results: list[dict]) -> dict:
    return {"run": run, "results": results}


def _result(case_id: str, score: float, rubric="r1", verdict="PASS") -> dict:
    return {"id": case_id, "score": score, "verdict": verdict, "rubric_fingerprint": rubric}


class TestGrouping:
    def test_same_config_and_rubric_group_together(self):
        scores = collect_scores([
            _report("配置A", [_result("c1", 1.0)]),
            _report("配置A", [_result("c1", 0.8)]),
        ])
        assert list(scores.values()) == [[1.0, 0.8]]

    def test_different_config_does_not_group(self):
        """模型/提示词/精排任一不同，分数就不可比（十期换 ground_truth 的教训）。"""
        scores = collect_scores([
            _report("配置A", [_result("c1", 1.0)]),
            _report("配置B", [_result("c1", 0.8)]),
        ])
        assert sorted(len(v) for v in scores.values()) == [1, 1]

    def test_different_rubric_does_not_group(self):
        """判据改了就是换了把尺子——这正是工具第一版栽的地方。"""
        scores = collect_scores([
            _report("配置A", [_result("c1", 0.675, rubric="old")]),
            _report("配置A", [_result("c1", 1.0, rubric="new")]),
        ])
        assert sorted(len(v) for v in scores.values()) == [1, 1]

    def test_error_runs_are_excluded(self):
        """ERROR 没有判分，把 0.0 混进去会把"跑挂了"算成"分数低"。"""
        scores = collect_scores([
            _report("配置A", [_result("c1", 1.0)]),
            _report("配置A", [_result("c1", 0.0, verdict="ERROR")]),
        ])
        assert list(scores.values()) == [[1.0]]


class TestSummary:
    def test_reports_spread_and_sorts_worst_first(self):
        rows = summarize(collect_scores([
            _report("A", [_result("stable", 1.0), _result("shaky", 0.6)]),
            _report("A", [_result("stable", 1.0), _result("shaky", 1.0)]),
        ]), min_runs=2)
        assert [row["case_id"] for row in rows] == ["shaky", "stable"]
        assert rows[0]["spread"] == 0.4 and rows[1]["spread"] == 0.0

    def test_single_run_cases_are_skipped(self):
        """跑过一轮的用例算不出方差——不该混进表里假装有结论。"""
        rows = summarize(collect_scores([_report("A", [_result("c1", 1.0)])]), min_runs=2)
        assert rows == []

    def test_config_key_falls_back_when_missing(self):
        assert config_key({}) == "未知配置"


class TestRequireFingerprint:
    """只统计带判据指纹的报告——要拿到干净的自然波动读数就得排除旧报告。"""

    def test_filters_out_unfingerprinted_results(self):
        reports = [
            {"run": "A", "results": [{"id": "c1", "score": 1.0, "verdict": "PASS"}]},
            _report("A", [_result("c1", 0.5)]),
        ]
        assert collect_scores(reports, require_fingerprint=True) == {
            ("A｜判据 r1", "c1"): [0.5],
        }

    def test_default_keeps_everything(self):
        """默认不过滤：旧报告也有参考价值，只是工具会警告不能当自然波动。"""
        reports = [
            {"run": "A", "results": [{"id": "c1", "score": 1.0, "verdict": "PASS"}]},
            _report("A", [_result("c1", 0.5)]),
        ]
        assert sum(len(v) for v in collect_scores(reports).values()) == 2


class TestRunLevelMeans:
    """整轮均分的波动——人们引用的就是这个数，而单条散布回答不了它。"""

    def test_groups_by_config_and_size(self):
        from scripts.eval.variance import run_level_means

        means = run_level_means([
            _report("A", [_result("c1", 1.0), _result("c2", 0.8)]),
            _report("A", [_result("c1", 1.0), _result("c2", 1.0)]),
        ])
        assert list(means.values()) == [[0.9, 1.0]]

    def test_smoke_and_full_do_not_mix(self):
        """12 条的均分与 44 条的均分本来就不可比。"""
        from scripts.eval.variance import run_level_means

        means = run_level_means([
            _report("A", [_result("c1", 1.0)]),
            _report("A", [_result("c1", 1.0), _result("c2", 1.0)]),
        ])
        assert len(means) == 2

    def test_error_only_report_is_skipped(self):
        from scripts.eval.variance import run_level_means

        assert run_level_means([_report("A", [_result("c1", 0.0, verdict="ERROR")])]) == {}
