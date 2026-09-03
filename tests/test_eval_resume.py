# -*- coding: utf-8 -*-
"""整轮评测的抗中断能力（增量落盘 + 续跑）。

要防的是一种很贵的失败：40 条用例 80-120 分钟真金白银，
结果**只在整轮跑完后才落盘**——第 39 条崩了、网络抖一下、Ctrl-C 一次，
前面 38 条的结果全部丢失，只能从头再烧一遍。

续跑的难点不在"跳过已完成的"，在**用例之间不独立**：
`memory-recall` 依赖 `memory-write` 先把偏好写进去。跳过前置直接跑后继，
评的是一个不成立的前提，而分数看上去完全正常。
"""
import json

import pytest

from scripts.eval_regression import (
    load_partial,
    merge_results,
    plan_resume,
    write_partial,
)


def _case(case_id: str, **extra) -> dict:
    return {"id": case_id, "description": case_id, **extra}


def _result(case_id: str, verdict: str = "PASS") -> dict:
    return {"id": case_id, "verdict": verdict, "score": 1.0, "judged": {}, "transcript": ""}


class TestPartialPersistence:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "partial-x.json"
        write_partial(path, "配置行", [_result("a"), _result("b")])
        loaded = load_partial(path)
        assert loaded["run_line"] == "配置行"
        assert [r["id"] for r in loaded["results"]] == ["a", "b"]

    def test_overwrites_rather_than_appends(self, tmp_path):
        """全量覆盖写，不是 append。

        JSON 数组 append 要么写坏格式、要么得自己维护括号；
        而每条用例的结果本来就只有几百 KB，整份重写的代价可以忽略。
        """
        path = tmp_path / "partial-x.json"
        write_partial(path, "l", [_result("a")])
        write_partial(path, "l", [_result("a"), _result("b")])
        assert len(json.loads(path.read_text(encoding="utf-8"))["results"]) == 2

    def test_missing_file_reports_actionably(self, tmp_path):
        with pytest.raises(SystemExit, match="不存在"):
            load_partial(tmp_path / "nope.json")


class TestResumePlan:
    def test_skips_completed(self):
        cases = [_case("a"), _case("b"), _case("c")]
        todo, skipped = plan_resume(cases, {"a", "b"})
        assert [c["id"] for c in todo] == ["c"]
        assert skipped == ["a", "b"]

    def test_reruns_prerequisites_of_pending_cases(self):
        """待跑用例的前置若已完成，也要**一并重跑**。

        跳过 memory-write 直接跑 memory-recall，评的是一个不成立的前提，
        而分数看上去完全正常——这类错误没有任何东西会报警。
        """
        cases = [_case("memory-write"), _case("other"), _case("memory-recall", requires=["memory-write"])]
        todo, skipped = plan_resume(cases, {"memory-write", "other"})
        assert [c["id"] for c in todo] == ["memory-write", "memory-recall"]
        assert skipped == ["other"]

    def test_keeps_original_order(self):
        """顺序必须与 cases.yaml 一致：用例之间的依赖靠顺序执行保证。"""
        cases = [_case("a"), _case("b"), _case("c")]
        todo, _ = plan_resume(cases, {"b"})
        assert [c["id"] for c in todo] == ["a", "c"]

    def test_nothing_to_do_is_not_an_error(self):
        todo, skipped = plan_resume([_case("a")], {"a"})
        assert todo == [] and skipped == ["a"]


class TestMergeResults:
    def test_new_results_replace_old_ones(self):
        """重跑过的用例取新结果，没跑的沿用旧结果。"""
        cases = [_case("a"), _case("b"), _case("c")]
        merged = merge_results(
            cases,
            previous=[_result("a", "FAIL"), _result("b"), _result("c", "FAIL")],
            fresh=[_result("a", "PASS"), _result("c", "PASS")],
        )
        assert [(r["id"], r["verdict"]) for r in merged] == [
            ("a", "PASS"), ("b", "PASS"), ("c", "PASS"),
        ]

    def test_orders_by_case_order_not_completion_order(self):
        cases = [_case("a"), _case("b")]
        merged = merge_results(cases, previous=[_result("b")], fresh=[_result("a")])
        assert [r["id"] for r in merged] == ["a", "b"]

    def test_drops_results_of_cases_no_longer_selected(self):
        """用 --tag 缩小范围后续跑：旧 partial 里多出来的用例不该混进报告，
        否则总览的分母是错的。"""
        merged = merge_results([_case("a")], previous=[_result("a"), _result("zzz")], fresh=[])
        assert [r["id"] for r in merged] == ["a"]


class TestRequiresAudit:
    def test_unknown_requirement_is_reported(self):
        from scripts.eval.audit_cases import _dangling_requires

        found = _dangling_requires([_case("a", requires=["nope"])])
        assert found == [("a", ["nope"])]

    def test_known_requirement_passes(self):
        from scripts.eval.audit_cases import _dangling_requires

        assert _dangling_requires([_case("a"), _case("b", requires=["a"])]) == []

    def test_real_cases_file_has_no_dangling_requires(self):
        """真实用例集也要过这条——续跑时才暴露拼错的 id 就太晚了。"""
        from pathlib import Path

        import yaml

        from scripts.eval.audit_cases import _dangling_requires

        path = Path(__file__).resolve().parents[1] / "eval" / "cases.yaml"
        cases = yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]
        assert _dangling_requires(cases) == []


class TestBaseUrlOverride:
    def test_defaults_to_local_8000(self, monkeypatch):
        import importlib

        monkeypatch.delenv("EVAL_BASE_URL", raising=False)
        import scripts.eval_regression as module

        importlib.reload(module)
        assert module.BASE_URL == "http://127.0.0.1:8000"

    def test_env_override_and_trailing_slash(self, monkeypatch):
        """尾斜杠要削掉：拼出 `//health` 时 FastAPI 会 404，
        而报错看起来像"服务没起来"，排查方向立刻跑偏。"""
        import importlib

        monkeypatch.setenv("EVAL_BASE_URL", "http://127.0.0.1:8011/")
        import scripts.eval_regression as module

        importlib.reload(module)
        assert module.BASE_URL == "http://127.0.0.1:8011"
        monkeypatch.delenv("EVAL_BASE_URL", raising=False)
        importlib.reload(module)
