# -*- coding: utf-8 -*-
"""judge 运行间一致性工具（二十三期清单 3）

没有人工时，"judge 可信吗"唯一能自动回答的办法是：**同一批已判过的
transcripts 让同一个 judge 重判一遍**，量它自己的判分波动。

工具的确定部分（选样、指纹核对、比较、聚合）在这里测；真判的那一步
（call_judge）复用 eval_regression 的实现，测试里用假 judge 注入，
不打一次模型。

一个必须钉死的口径：**指纹不匹配的用例不进样本**。报告存的是当轮判据的
指纹，cases.yaml 是现在的——判据改过的用例，"judge 不一致"和"判据变了"
会混在一起，量出来的就不是 judge 波动。
"""
from __future__ import annotations


import pytest

from scripts.eval.rejudge_transcripts import (
    compare_result,
    pick_cases,
    rejudge_all,
    render,
    summarize_comparisons,
)


from scripts.eval_regression import rubric_fingerprint


def _case(case_id: str, criterion: str = "数字对") -> dict:
    return {"id": case_id, "queries": ["q"], "rubric": {"p0": [{"criterion": criterion, "expect": "x"}]}}


def _fps(cases: dict[str, dict]) -> dict[str, str]:
    """cases.yaml 现算指纹——报告里存的就是它（一致才可重判）。"""
    return {case_id: rubric_fingerprint(case) for case_id, case in cases.items()}


def _judged(p0: list[bool], p1: list[bool] | None = None, p2: list[bool] | None = None) -> dict:
    def items(flags: list[bool], tag: str) -> list[dict]:
        return [{"criterion": f"{tag}-{i}", "pass": flag, "reason": "r"} for i, flag in enumerate(flags)]

    return {
        "p0": items(p0, "p0"),
        "p1": items(p1 or [], "p1"),
        "p2": items(p2 or [], "p2"),
    }


def _rubric_for(judged: dict) -> dict:
    """与 judged 同构的 rubric：判了哪些档，判据就声明哪些档。"""
    return {
        level: [{"criterion": item["criterion"], "expect": "x"} for item in judged.get(level) or []]
        or [{"criterion": "占位", "expect": "x"}] if judged.get(level) else []
        for level in ("p0", "p1", "p2")
    }


def _result(case_id: str, judged: dict, fp: str = "fp-1") -> dict:
    """模拟报告里的一条 result：score/verdict 按 eval_regression 的口径造。"""
    from scripts.eval_regression import score_case

    score, p0_pass = score_case(judged)
    return {
        "id": case_id,
        "verdict": "PASS" if p0_pass and score >= 0.7 else "FAIL",
        "score": score,
        "judged": judged,
        "rubric_fingerprint": fp,
        "transcript": "[买家] q\n[Agent] a",
    }


_RUBRIC = {
    "p0": [{"criterion": "p0-0", "expect": "x"}],
    "p1": [{"criterion": "p1-0", "expect": "x"}],
    "p2": [{"criterion": "p2-0", "expect": "x"}],
}


class TestPickCases:
    def test_errors_and_fingerprint_mismatches_are_excluded(self):
        cases = {"a": _case("a"), "b": _case("b"), "c": _case("c"), "d": _case("d")}
        fps = _fps(cases)
        results = [
            _result("a", _judged([True]), fps["a"]),
            _result("b", _judged([True]), fps["b"]),
            {**_result("c", _judged([True]), fps["c"]), "verdict": "ERROR"},
            {**_result("d", _judged([True]), fps["d"]), "rubric_fingerprint": "fp-old"},
        ]
        picked = pick_cases(results, cases, sample=10)
        assert [r["id"] for r in picked] == ["a", "b"]

    def test_missing_case_in_yaml_is_excluded(self):
        cases = {"a": _case("a")}
        fps = _fps(cases)
        results = [_result("a", _judged([True]), fps["a"]), _result("ghost", _judged([True]), fps["a"])]
        picked = pick_cases(results, cases, sample=10)
        assert [r["id"] for r in picked] == ["a"]

    def test_sample_takes_evenly_spaced_deterministic_subset(self):
        cases = {f"c{i}": _case(f"c{i}") for i in range(10)}
        fps = _fps(cases)
        results = [_result(f"c{i}", _judged([True]), fps[f"c{i}"]) for i in range(10)]
        first = pick_cases(results, cases, sample=3)
        second = pick_cases(results, cases, sample=3)
        assert [r["id"] for r in first] == [r["id"] for r in second]
        # 钉死精确结果：均匀铺开必须含首尾（n=10, k=3 → 索引 0, 4, 9）——
        # 取前 N（c0,c1,c2）这种退化实现必须红
        assert [r["id"] for r in first] == ["c0", "c4", "c9"]

    def test_sample_one_with_large_pool_does_not_divide_by_zero(self):
        cases = {f"c{i}": _case(f"c{i}") for i in range(5)}
        fps = _fps(cases)
        results = [_result(f"c{i}", _judged([True]), fps[f"c{i}"]) for i in range(5)]
        picked = pick_cases(results, cases, sample=1)
        assert len(picked) == 1

    def test_sample_below_one_is_rejected(self):
        cases = {"a": _case("a")}
        results = [_result("a", _judged([True]), _fps(cases)["a"])]
        with pytest.raises(SystemExit):
            pick_cases(results, cases, sample=0)

    def test_only_overrides_sample(self):
        cases = {f"c{i}": _case(f"c{i}") for i in range(5)}
        fps = _fps(cases)
        results = [_result(f"c{i}", _judged([True]), fps[f"c{i}"]) for i in range(5)]
        picked = pick_cases(results, cases, sample=20, only="c2")
        assert [r["id"] for r in picked] == ["c2"]

    def test_sample_larger_than_pool_takes_all(self):
        cases = {"a": _case("a"), "b": _case("b")}
        fps = _fps(cases)
        results = [_result("a", _judged([True]), fps["a"]), _result("b", _judged([True]), fps["b"])]
        assert len(pick_cases(results, cases, sample=20)) == 2


class TestFingerprintGate:
    def test_result_without_fingerprint_is_not_silently_admitted(self):
        """旧报告没有指纹 = 无法验证尺子，比指纹不符更该拦——不许静默放行。"""
        cases = {"a": _case("a")}
        results = [{**_result("a", _judged([True])), "rubric_fingerprint": None}]
        assert pick_cases(results, cases, sample=10) == []


class TestDirtyJudged:
    def test_missing_criterion_is_an_error_row_not_a_crash(self):
        """criterion 缺失的判词是脏数据：该行记 error 留名，不让 TypeError
        裸奔，更不能静默塌缩成少计的一致条目。"""
        cases = {"a": _case("a")}
        dirty = {"p0": [{"criterion": None, "pass": True}], "p1": [], "p2": []}
        results = [_result("a", dirty, _fps(cases)["a"])]

        async def fake_judge(transcript, rubric, ground_truth, prior_context):
            return dirty

        rows = _run_rejudge(results, cases, fake_judge)
        assert rows[0]["error"], "脏判词必须留名"

    def test_empty_judged_against_declared_level_is_an_error(self):
        """judge 返回空档位按 score_case 语义是满分——在 rejudge 语境下
        这是脏读数：rubric 声明了 p0，judged 就必须有 p0。"""
        cases = {"a": _case("a")}
        results = [_result("a", _judged([True]), _fps(cases)["a"])]

        async def fake_judge(transcript, rubric, ground_truth, prior_context):
            return {}  # 结构性合法但为空

        rows = _run_rejudge(results, cases, fake_judge)
        assert rows[0]["error"], "空 judged 不能被记成满分 PASS"


def _run_rejudge(results, cases, judge):
    import asyncio

    from scripts.eval.rejudge_transcripts import rejudge_all as _rejudge

    return asyncio.run(_rejudge(results, cases, judge=judge, ground_truth="GT"))


class TestCompareResult:
    def test_identical_judging_is_full_agreement(self):
        judged = _judged([True, True], [True], [False])
        row = compare_result({"id": "a", "first": judged, "second": judged, "rubric": _RUBRIC})
        assert row["delta_score"] == 0.0
        assert row["flip"] is False
        assert row["agree"] == 4 and row["disagree"] == 0

    def test_score_and_verdict_flip_are_reported(self):
        first = _judged([True, True], [True], [True])    # 1.0
        second = _judged([False, True], [True], [True])  # p0 掉一档 → FAIL
        row = compare_result({"id": "a", "first": first, "second": second, "rubric": _RUBRIC})
        assert row["flip"] is True
        assert row["first_verdict"] == "PASS" and row["second_verdict"] == "FAIL"
        assert row["disagree"] == 1 and row["agree"] == 3

    def test_disagreement_names_the_criterion(self):
        first = _judged([True])
        second = _judged([False])
        row = compare_result({"id": "a", "first": first, "second": second,
                              "rubric": {"p0": [{"criterion": "p0-0", "expect": "x"}]}})
        assert row["disagreed_criteria"] == ["p0:p0-0"]

    def test_single_sided_criterion_counts_as_disagree(self):
        """judge 重判时多 hallucinate 了一条 criterion：单边出现的记
        disagree（保守口径），分数仍按各自判定算。"""
        first = {"p0": [{"criterion": "共有条目", "pass": True}]}
        second = {
            "p0": [
                {"criterion": "共有条目", "pass": True},
                {"criterion": "重判时多出的条目", "pass": False},
            ],
        }
        rubric = {"p0": [{"criterion": "共有条目"}]}
        row = compare_result({"id": "a", "first": first, "second": second, "rubric": rubric})
        assert row["agree"] == 1
        assert row["disagree"] == 1
        assert "p0:重判时多出的条目" in row["disagreed_criteria"]


class TestSummarize:
    def test_aggregates_waveband_and_flips(self):
        rows = [
            {"id": "a", "delta_score": 0.0, "flip": False, "agree": 4, "disagree": 0},
            {"id": "b", "delta_score": 0.175, "flip": True, "agree": 3, "disagree": 1},
            {"id": "c", "delta_score": 0.05, "flip": False, "agree": 4, "disagree": 0},
        ]
        summary = summarize_comparisons(rows)
        assert summary["n"] == 3
        assert summary["max_abs_delta"] == 0.175
        assert summary["flips"] == ["b"]
        assert summary["mean_abs_delta"] == round((0.0 + 0.175 + 0.05) / 3, 4)
        assert summary["item_agreement"] == round(11 / 12, 4)

    def test_empty_rows_give_zeros(self):
        summary = summarize_comparisons([])
        assert summary["n"] == 0 and summary["max_abs_delta"] == 0.0


class TestRejudgeAll:
    async def test_calls_judge_once_per_case_and_builds_rows(self):
        results = [_result("a", _judged([True]))]
        cases = {"a": _case("a")}
        calls: list[str] = []

        async def fake_judge(transcript: str, rubric: dict, ground_truth: str, prior_context: str):
            calls.append(transcript)
            return _judged([True])

        rows = await rejudge_all(results, cases, judge=fake_judge, ground_truth="GT")
        assert len(rows) == 1 and len(calls) == 1
        assert rows[0]["second_score"] == rows[0]["first_score"]

    async def test_judge_failure_marks_the_row_not_the_whole_run(self, tmp_path):
        """一条判挂了不该报废整批——但必须留名，不能静默。"""
        results = [
            _result("a", _judged([True])),
            {**_result("b", _judged([True])), "transcript": "[买家] q\n[Agent] 不同的回复"},
        ]
        cases = {"a": _case("a"), "b": _case("b")}

        async def fake_judge(transcript: str, rubric: dict, ground_truth: str, prior_context: str):
            if "[Agent] a" in transcript:
                raise RuntimeError("503")
            return _judged([True])

        rows = await rejudge_all(results, cases, judge=fake_judge, ground_truth="GT")
        assert [row["id"] for row in rows] == ["a", "b"]
        assert rows[0]["error"] and not rows[1]["error"]


class TestRender:
    def test_render_names_waveband_and_flips(self):
        rows = [{
            "id": "b", "delta_score": 0.175, "flip": True,
            "first_verdict": "PASS", "second_verdict": "FAIL",
            "first_score": 1.0, "second_score": 0.825,
            "agree": 3, "disagree": 1, "disagreed_criteria": ["p0-1"], "error": "",
        }]
        text = render(summarize_comparisons(rows), rows)
        assert "0.175" in text and "b" in text
        assert "PASS → FAIL" in text
        assert "p0-1" in text
