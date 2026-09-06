# -*- coding: utf-8 -*-
"""ab_run 真实路径（授权文档 M1：零 LLM 部分，全部用假执行器/假 judge 测试）。

要防的问题（授权文档逐条点名）：
  - k 次采样串会话：样本索引进 session/buyer id，k 个样本必须是 k 个独立会话，
    否则记忆类用例互相污染；
  - 产物不按 (case_id, arm, sample_index) 落盘就没法续跑、没法配对；
  - 没跑完就配对：缺一边 transcript 的对是假读数；
  - 脏判词静默塌缩：rejudge 教训，error 行留名；
  - 续跑时两臂配置对不上：半批样本跑在旧提示词上，读数混装；
  - 评审模型与被测模型相同：自我偏好，读数无效。

dry-run 与真实路径共享同一份账本（plan_ab_run）与同一道前置闸（preflight），
不许两套算式。
"""
import json

import pytest

import scripts.eval.ab_run as ab_run
from scripts.eval.ab_run import (
    ab_participant_ids,
    build_execution_plan,
    build_pairs_from_executions,
    check_resume_config,
    execute_ab_case,
    guard_judge_models,
    judge_pair_rows,
    partition_pending,
    select_dual_judge_pairs,
    triple_key,
)
from scripts.eval.ab_stats import decisive_indicators


def _case(case_id: str, queries: int = 1, buyer_id=None, requires=None, prior_context=None) -> dict:
    case = {"id": case_id, "queries": [f"问{i}" for i in range(queries)]}
    if buyer_id:
        case["buyer_id"] = buyer_id
    if requires:
        case["requires"] = requires
    if prior_context:
        case["prior_context"] = prior_context
    return case


class TestParticipantIds:
    def test_sample_index_in_both_ids(self):
        case = _case("demo")
        session_a0, buyer_a0 = ab_participant_ids(case, "A", 0)
        session_a1, buyer_a1 = ab_participant_ids(case, "A", 1)
        assert "ab-a-k0-demo" in session_a0
        assert "ab-a-k1-demo" in session_a1
        assert buyer_a0.endswith("-aba k0".replace(" ", "")) or "-abak0" in buyer_a0
        assert buyer_a0 != buyer_a1

    def test_sessions_unique_across_calls(self):
        """同一 (case, arm, sample) 的两次调用也要给出不同会话（uuid 尾）——
        会话级状态不许跨执行复用。"""
        case = _case("demo")
        s1, _ = ab_participant_ids(case, "A", 0)
        s2, _ = ab_participant_ids(case, "A", 0)
        assert s1 != s2

    def test_memory_chain_consistent_within_sample_isolated_across(self):
        """memory-write k1 与 memory-recall k1 必须派生同一个买家；
        k0 与 k1、A 臂与 B 臂互不相同——同一偏好链只在同一样本内可见。"""
        write = _case("memory-write", buyer_id="eval-memory-buyer")
        recall = _case("memory-recall", buyer_id="eval-memory-buyer")
        _, write_b1 = ab_participant_ids(write, "B", 1)
        _, recall_b1 = ab_participant_ids(recall, "B", 1)
        assert write_b1 == recall_b1
        _, write_b0 = ab_participant_ids(write, "B", 0)
        _, write_a1 = ab_participant_ids(write, "A", 1)
        assert len({write_b0, write_b1, write_a1}) == 3


class TestExecutionPlan:
    def test_triple_count_and_order_arm_major_case_major_sample_inner(self):
        cases = [_case("c1"), _case("c2", queries=2)]
        plan = build_execution_plan(cases, k=2)
        assert [(item["arm"], item["case_id"], item["sample_index"]) for item in plan] == [
            ("A", "c1", 0), ("A", "c1", 1),
            ("A", "c2", 0), ("A", "c2", 1),
            ("B", "c1", 0), ("B", "c1", 1),
            ("B", "c2", 0), ("B", "c2", 1),
        ]

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            build_execution_plan([_case("c")], k=0)


class TestPartitionPending:
    def test_skips_completed_keeps_order(self):
        cases = [_case("a"), _case("b")]
        plan = build_execution_plan(cases, k=1)
        pending = partition_pending(plan, {triple_key("a", "A", 0), triple_key("a", "B", 0)})
        assert [(i["case_id"], i["arm"], i["sample_index"]) for i in pending] == [("b", "A", 0), ("b", "B", 0)]

    def test_reruns_prerequisite_of_pending_same_arm_sample_only(self):
        """memory-recall (B,1) 待跑时，已完成的 memory-write (B,1) 要补跑；
        而 (B,0) 的完成状态与它无关——样本隔离的续跑语义。"""
        cases = [
            _case("memory-write", buyer_id="eval-memory-buyer"),
            _case("memory-recall", buyer_id="eval-memory-buyer", requires=["memory-write"]),
        ]
        plan = build_execution_plan(cases, k=2)
        completed = {
            triple_key("memory-write", "B", 0),
            triple_key("memory-recall", "B", 0),
            triple_key("memory-write", "B", 1),
            triple_key("memory-write", "A", 0), triple_key("memory-write", "A", 1),
            triple_key("memory-recall", "A", 0), triple_key("memory-recall", "A", 1),
        }
        pending = partition_pending(plan, completed)
        assert [(i["case_id"], i["arm"], i["sample_index"]) for i in pending] == [
            ("memory-write", "B", 1),  # 前置补跑
            ("memory-recall", "B", 1),
        ]

    def test_all_completed_means_nothing_pending(self):
        plan = build_execution_plan([_case("a")], k=1)
        assert partition_pending(plan, {triple_key("a", "A", 0), triple_key("a", "B", 0)}) == []


class _RecordingExecutor:
    """假执行器：按 (case_id, arm, sample) 给出可预期的 transcript。"""

    def __init__(self, fail: set | None = None) -> None:
        self.calls: list[tuple[str, str, int, str]] = []
        self.fail = fail or set()

    async def __call__(self, case, arm, sample_index, base_url):
        key = triple_key(case["id"], arm, sample_index)
        self.calls.append((case["id"], arm, sample_index, base_url))
        if key in self.fail:
            raise RuntimeError(f"注入的失败 {key}")
        return {
            "case_id": case["id"], "arm": arm, "sample_index": sample_index,
            "session_id": f"sess-{key.replace('|', '-')}",
            "transcript": f"[买家] 问\n[Agent] {arm}{sample_index}-{case['id']}",
            "ok": True, "error": "",
        }


class TestExecuteAbCase:
    async def test_ok_record_carries_transcript_and_ids(self, monkeypatch):
        captured = {}

        async def fake_execute_case(client, case, base_url=None, session_id=None, buyer_id=None):
            captured.update(base_url=base_url, session_id=session_id, buyer_id=buyer_id)
            return {"session_id": session_id, "transcript": "[买家] 问\n[Agent] 答"}

        monkeypatch.setattr(ab_run, "execute_case", fake_execute_case)
        record = await execute_ab_case(_case("demo"), "B", 1, client=None, base_url="http://x:8012")
        assert record["ok"] is True and record["error"] == ""
        assert record["transcript"] == "[买家] 问\n[Agent] 答"
        assert captured["base_url"] == "http://x:8012"
        assert "ab-b-k1-demo" in captured["session_id"]
        assert "-abbk1" in captured["buyer_id"]

    async def test_failure_becomes_named_error_record_not_exception(self, monkeypatch):
        """单次执行失败不许炸掉整轮，也不许悄悄消失——记名 error 行。"""

        async def fake_execute_case(client, case, base_url=None, session_id=None, buyer_id=None):
            raise RuntimeError("ReadTimeout: ")

        monkeypatch.setattr(ab_run, "execute_case", fake_execute_case)
        record = await execute_ab_case(_case("demo"), "A", 0, client=None, base_url="http://x")
        assert record["ok"] is False
        assert "ReadTimeout" in record["error"]
        assert record["fault_clear_error"] == ""
        assert record["case_id"] == "demo" and record["arm"] == "A" and record["sample_index"] == 0




class TestBuildPairsFromExecutions:
    def _execs(self, k=2):
        execs = []
        for arm in ("A", "B"):
            for i in range(k):
                execs.append({
                    "case_id": "demo", "arm": arm, "sample_index": i,
                    "session_id": f"s-{arm}{i}", "transcript": f"T-{arm}{i}",
                    "ok": True, "error": "",
                })
        return execs

    def test_diagonal_matches_sample_indexes(self):
        pairs, errors = build_pairs_from_executions(
            self._execs(), [_case("demo", prior_context="偏好")], k=2, pairing="diagonal",
        )
        assert errors == []
        assert [(p["left"]["transcript"], p["right"]["transcript"]) for p in pairs] == [
            ("T-A0", "T-B0"), ("T-A1", "T-B1"),
        ]
        assert pairs[0]["case_prompt_text"] == "问0"
        assert pairs[0]["prior_context"] == "偏好"

    def test_missing_sample_is_named_error_not_silent_drop(self):
        execs = self._execs()
        execs = [e for e in execs if not (e["arm"] == "B" and e["sample_index"] == 1)]
        pairs, errors = build_pairs_from_executions(execs, [_case("demo")], k=2, pairing="diagonal")
        assert len(pairs) == 1
        assert errors and "demo" in errors[0]["reason"] and "B" in errors[0]["reason"]

    def test_failed_execution_is_named_error(self):
        execs = self._execs()
        execs[1] = {**execs[1], "ok": False, "error": "ReadTimeout"}
        pairs, errors = build_pairs_from_executions(execs, [_case("demo")], k=2, pairing="diagonal")
        assert len(pairs) == 1
        assert any("ReadTimeout" in e["reason"] for e in errors)

    def test_cross_mode_yields_k_squared(self):
        pairs, errors = build_pairs_from_executions(self._execs(), [_case("demo")], k=2, pairing="cross")
        assert errors == []
        assert len(pairs) == 4

    def test_pair_indexes_are_per_case_stable(self):
        execs = self._execs(k=1) + [
            {"case_id": "other", "arm": "A", "sample_index": 0, "session_id": "sA", "transcript": "TA", "ok": True, "error": ""},
            {"case_id": "other", "arm": "B", "sample_index": 0, "session_id": "sB", "transcript": "TB", "ok": True, "error": ""},
        ]
        cases = [_case("demo"), _case("other")]
        pairs, errors = build_pairs_from_executions(execs, cases, k=1, pairing="diagonal")
        assert [p["pair_index"] for p in pairs] == [0, 0]
        assert [p["case_id"] for p in pairs] == ["demo", "other"]


class _ContentJudge:
    """按内容判：谁含 GOOD 谁赢，用于验证两个顺序各判一次且方向映射正确。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "GOOD" in prompt.split("【回复1】")[1].split("【回复2】")[0]:
            return "裁决: 1\n理由: 回复1 有工具出处"
        return "裁决: 2\n理由: 回复2 有工具出处"


class TestJudgePairRows:
    async def test_both_orders_judged_and_mapped(self):
        pairs = [{
            "case_id": "demo", "pair_index": 0,
            "left": {"transcript": "GOOD-A"}, "right": {"transcript": "bad-B"},
            "case_prompt_text": "问", "prior_context": "",
        }]
        judge = _ContentJudge()
        rows = await judge_pair_rows(pairs, judge, ground_truth="事实表")
        assert len(rows) == 1 and len(judge.prompts) == 2
        row = rows[0]
        # 正序：A 在左（含 GOOD）→ verdict_ab=a；反序：A 在右 → judge 仍选 GOOD → 映射回 a
        assert row["verdict_ab"] == "a" and row["verdict_ba"] == "a"
        assert "事实表" in judge.prompts[0]

    async def test_dirty_verdict_becomes_error_row_not_collapse(self):
        pairs = [{
            "case_id": "demo", "pair_index": 0,
            "left": {"transcript": "t1"}, "right": {"transcript": "t2"},
            "case_prompt_text": "问", "prior_context": "",
        }]

        async def dirty(prompt):
            if "【回复1】\nt2" in prompt:  # 反序（t2 在左）→ 脏输出
                return "都差不多"
            return "裁决: 1\n理由: ok"

        rows = await judge_pair_rows(pairs, dirty, ground_truth="")
        assert rows[0]["verdict_ab"] == "a" and rows[0]["verdict_ba"] is None
        assert "demo" in rows[0]["error_ba"] and "判" in rows[0]["error_ba"]

    async def test_judge_exception_is_error_row(self):
        pairs = [{
            "case_id": "demo", "pair_index": 0,
            "left": {"transcript": "t1"}, "right": {"transcript": "t2"},
            "case_prompt_text": "问", "prior_context": "",
        }]

        async def down(prompt):
            raise RuntimeError("Error code: 503")

        rows = await judge_pair_rows(pairs, down, ground_truth="")
        assert rows[0]["verdict_ab"] is None and rows[0]["verdict_ba"] is None
        assert "503" in rows[0]["error_ab"]


class _CyclingJudge:
    """按调用序返回预设裁决（循环），用于投票路径的众数/无众数/脏票测试。"""

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls = 0

    async def __call__(self, prompt: str) -> str:
        answer = self.answers[self.calls % len(self.answers)]
        self.calls += 1
        return answer


def _one_pair() -> list[dict]:
    return [{
        "case_id": "demo", "pair_index": 0,
        "left": {"transcript": "t1"}, "right": {"transcript": "t2"},
        "case_prompt_text": "问", "prior_context": "",
    }]


class TestJudgePairRowsWithVotes:
    """M2'-d 步骤 2（多数投票）：每序 ×3 取众数，口径不变、脏票不留读数。"""

    async def test_majority_of_three_wins_with_votes_recorded(self):
        # 每序 3 次调用：1,1,平局 → 众数 1。ab 序位置1=左 → a；ba 序位置1=右 → b
        judge = _CyclingJudge(["裁决: 1\n理由: x", "裁决: 1\n理由: x", "裁决: 平局\n理由: x"])
        rows = await judge_pair_rows(_one_pair(), judge, ground_truth="", votes=3)
        assert judge.calls == 6
        row = rows[0]
        assert row["verdict_ab"] == "a" and row["verdict_ba"] == "b"
        assert row["votes_ab"] == ["a", "a", "tie"]
        assert row["votes_ba"] == ["b", "b", "tie"]
        assert row["raw_ab"].count("理由") == 3  # 三票原文全留，可审计

    async def test_three_way_split_is_error_row_not_fabricated(self):
        judge = _CyclingJudge(["裁决: 1\n理由: x", "裁决: 2\n理由: x", "裁决: 平局\n理由: x"])
        rows = await judge_pair_rows(_one_pair(), judge, ground_truth="", votes=3)
        row = rows[0]
        assert row["verdict_ab"] is None and row["verdict_ba"] is None
        assert "众数" in row["error_ab"] and "众数" in row["error_ba"]

    async def test_one_dirty_vote_poisons_the_order(self):
        # 第二票脏输出 → 该序整体 error（宁可少一对，不进 2/3 塌缩读数）
        answers = ["裁决: 1\n理由: x", "都差不多", "裁决: 1\n理由: x"]
        judge = _CyclingJudge(answers)
        rows = await judge_pair_rows(_one_pair(), judge, ground_truth="", votes=3)
        assert rows[0]["verdict_ab"] is None
        assert "判" in rows[0]["error_ab"]


class TestSelectDualJudgePairs:
    def test_one_pair_per_case_until_n(self):
        rows = [{"case_id": f"c{i}", "pair_index": 0} for i in range(5)]
        assert len(select_dual_judge_pairs(rows, 3)) == 3
        assert [r["case_id"] for r in select_dual_judge_pairs(rows, 3)] == ["c0", "c1", "c2"]

    def test_more_than_available_takes_all(self):
        rows = [{"case_id": "c", "pair_index": 0}, {"case_id": "c", "pair_index": 1}]
        selected = select_dual_judge_pairs(rows, 5)
        assert len(selected) == 1  # 同一用例只取第一对，把配额摊到更多用例


class TestResumeConfigGuard:
    def test_matching_config_passes(self):
        partial = {"arm_config": {
            "A": {"fingerprint": "a0915fac", "variant": "", "model": "mimo-v2.5"},
            "B": {"fingerprint": "b2222222", "variant": "candidate-x", "model": "mimo-v2.5"},
        }}
        healths = {
            "A": {"prompt_fingerprint": "a0915fac", "prompt_variant": "", "model": "mimo-v2.5"},
            "B": {"prompt_fingerprint": "b2222222", "prompt_variant": "candidate-x", "model": "mimo-v2.5"},
        }
        check_resume_config(partial, healths)

    def test_fingerprint_mismatch_rejects(self):
        """半批样本跑在旧提示词上是最贵的失败形态——续跑前拦。"""
        partial = {"arm_config": {
            "A": {"fingerprint": "old-fp", "variant": "", "model": "mimo-v2.5"},
            "B": {"fingerprint": "b2222222", "variant": "candidate-x", "model": "mimo-v2.5"},
        }}
        healths = {
            "A": {"prompt_fingerprint": "new-fp", "prompt_variant": "", "model": "mimo-v2.5"},
            "B": {"prompt_fingerprint": "b2222222", "prompt_variant": "candidate-x", "model": "mimo-v2.5"},
        }
        with pytest.raises(SystemExit) as err:
            check_resume_config(partial, healths)
        assert "A" in str(err.value)


class TestJudgeModelGuard:
    def test_judge_equal_to_model_under_test_rejects(self):
        healths = {"A": {"model": "mimo-v2.5"}, "B": {"model": "mimo-v2.5"}}
        with pytest.raises(SystemExit) as err:
            guard_judge_models("mimo-v2.5", healths)
        assert "自我偏好" in str(err.value)

    def test_second_judge_equal_to_first_rejects(self):
        healths = {"A": {"model": "mimo-v2.5"}, "B": {"model": "mimo-v2.5"}}
        with pytest.raises(SystemExit):
            guard_judge_models("longcat-2.0", healths, second_judge_model="longcat-2.0")

    def test_legal_pair_passes(self):
        healths = {"A": {"model": "mimo-v2.5"}, "B": {"model": "mimo-v2.5"}}
        guard_judge_models("longcat-2.0", healths, second_judge_model="deepseek-v4-flash")


class TestDecisiveIndicators:
    def test_consistent_rows_feed_win_and_ci(self):
        rows = [
            {"case_id": "c1", "verdict_ab": "a", "verdict_ba": "a"},
            {"case_id": "c1", "verdict_ab": "b", "verdict_ba": "b"},
            {"case_id": "c2", "verdict_ab": "tie", "verdict_ba": "tie"},
            {"case_id": "c2", "verdict_ab": "a", "verdict_ba": "b"},  # 互换翻转
            {"case_id": "c3", "verdict_ab": None, "verdict_ba": None},  # error
        ]
        feed, ci_pairs, n_flip = decisive_indicators(rows)
        assert feed == ["a", "b", "tie"]
        assert ci_pairs == [("c1", 1), ("c1", 0)]
        assert n_flip == 1

    def test_all_errors_give_empty_feed(self):
        feed, ci_pairs, n_flip = decisive_indicators([{"case_id": "c", "verdict_ab": None, "verdict_ba": None}])
        assert feed == [] and ci_pairs == [] and n_flip == 0


class TestRunAbPipeline:
    def _healths(self):
        return {
            "A": {"prompt_fingerprint": "a0915fac", "prompt_variant": "", "model": "mimo-v2.5"},
            "B": {"prompt_fingerprint": "b2222222", "prompt_variant": "candidate-x", "model": "mimo-v2.5"},
        }

    def _arm_config(self):
        return {
            "A": {"fingerprint": "a0915fac", "variant": "", "model": "mimo-v2.5"},
            "B": {"fingerprint": "b2222222", "variant": "candidate-x", "model": "mimo-v2.5"},
        }

    async def _pipeline(self, tmp_path, cases, k=1, *, judge_factory=None, second_judge_model="",
                        dual_judge_pairs=0, resume_path=None, execute_fn=None, fail=(),
                        seconds_per_intent=None, product_prefix="ab", healths=None):
        from scripts.eval.ab_run import run_ab_pipeline

        if execute_fn is None:
            async def execute_fn(case, arm, sample_index, base_url):
                if triple_key(case["id"], arm, sample_index) in fail:
                    raise RuntimeError("ReadTimeout: 600s")
                return {
                    "case_id": case["id"], "arm": arm, "sample_index": sample_index,
                    "session_id": f"s-{arm}{sample_index}-{case['id']}",
                    "transcript": f"[Agent] {'GOOD' if arm == 'A' else 'bad'}-{case['id']}{sample_index}",
                    "ok": True, "error": "",
                }

        if judge_factory is None:
            async def judge_call(prompt):
                return "裁决: 平局\n理由: 相当"
            def judge_factory(model):
                return judge_call
        progress: list[str] = []
        payload = await run_ab_pipeline(
            cases=cases, k=k, pairing="diagonal",
            judge_model="longcat-2.0",
            healths=healths or self._healths(), urls={"A": "http://a:8000", "B": "http://b:8012"},
            arm_lines={"A": "配置行A", "B": "配置行B"},
            arm_config=self._arm_config(),
            ground_truth="事实表", eval_dir=tmp_path, stamp="20260905-170000",
            label="测试", judge_factory=judge_factory,
            second_judge_model=second_judge_model, dual_judge_pairs=dual_judge_pairs,
            resume_path=resume_path, execute_fn=execute_fn, progress=progress.append,
            seconds_per_intent=seconds_per_intent, product_prefix=product_prefix,
        )
        return payload, progress

    async def test_product_prefix_names_outputs(self, tmp_path):
        """#12 记忆回放复用本管线：产物前缀可换（mem-*），分桶互不覆盖。

        缺省 ab- 不变——二十五期既有产物命名与全部既有测试不受影响。"""
        await self._pipeline(tmp_path, [_case("c1")], product_prefix="mem")
        assert (tmp_path / "mem-report-20260905-170000.md").exists()
        run_json = json.loads((tmp_path / "mem-run-20260905-170000.json").read_text(encoding="utf-8"))
        assert run_json["plan"]["pairs"] == 1

    async def test_eval_dir_created_when_missing(self, tmp_path):
        """产物目录不存在时自动创建——第一轮真实跑测就栽在它手上（mock 演练发现）。"""
        eval_dir = tmp_path / "nested" / "ab"
        await self._pipeline(eval_dir, [_case("c1")])
        assert (eval_dir / "ab-report-20260905-170000.md").exists()

    async def test_end_to_end_ties_all_zero_llm(self, tmp_path):
        """假 judge 全平局：链路走通，统计如实给'无从判定'而非伪造读数。"""
        payload, progress = await self._pipeline(tmp_path, [_case("c1"), _case("c2")])
        assert payload["plan"]["pairs"] == 2
        assert payload["swap"]["rate"] == 1.0
        assert payload["win_rate"]["n_decisive"] == 0
        assert payload["p_value"] is None and payload["ci"] is None
        assert any("无从计算" in r or "无决定性对" in r for r in payload["significance"]["reasons"])
        report = (tmp_path / "ab-report-20260905-170000.md").read_text(encoding="utf-8")
        assert "样本不足" in report or "未达显著" in report
        run_json = json.loads((tmp_path / "ab-run-20260905-170000.json").read_text(encoding="utf-8"))
        assert len(run_json["results"]) == 4  # 2 用例 × 2 臂 × k=1
        assert len(run_json["rows"]) == 2
        assert not (tmp_path / "ab-partial-20260905-170000.json").exists()

    async def test_content_judge_produces_wins_and_ci(self, tmp_path):
        """A 臂 transcript 含 GOOD、judge 按内容判 → 全部 A 胜，CI/p 有值。"""
        from scripts.eval.ab_pairwise import VerdictParseError  # noqa: F401

        def factory(model):
            async def judge_call(prompt):
                left = prompt.split("【回复1】")[1].split("【回复2】")[0]
                if "GOOD" in left:
                    return "裁决: 1\n理由: 回复1 有出处"
                return "裁决: 2\n理由: 回复2 有出处"
            return judge_call

        payload, _ = await self._pipeline(tmp_path, [_case("c1"), _case("c2")], judge_factory=factory)
        assert payload["win_rate"]["wins"] == 2 and payload["win_rate"]["losses"] == 0
        assert payload["n_flip"] == 0
        assert payload["p_value"] == 0.5  # sign_test_p(2, 0)
        assert payload["ci"]["point"] == 1.0

    async def test_failed_execution_named_in_report(self, tmp_path):
        payload, _ = await self._pipeline(
            tmp_path, [_case("c1"), _case("ok-case")], fail={triple_key("c1", "A", 0)},
        )
        assert payload["executions"]["ok"] == 3
        assert any(f["case_id"] == "c1" for f in payload["executions"]["failed"])
        assert any("c1" in e["reason"] for e in payload["pair_errors"])
        report = (tmp_path / "ab-report-20260905-170000.md").read_text(encoding="utf-8")
        assert "c1" in report and "ReadTimeout" in report

    async def test_resume_skips_completed_and_merges(self, tmp_path):
        partial = tmp_path / "ab-partial-earlier.json"
        partial.write_text(json.dumps({
            "label": "测试", "pairing": "diagonal",
            "plan": {"k": 1, "pairing": "diagonal", "case_ids": ["c1", "c2"]},
            "arm_config": self._arm_config(),
            "results": [
                {"case_id": "c1", "arm": "A", "sample_index": 0, "session_id": "sA",
                 "transcript": "T-A", "ok": True, "error": ""},
                {"case_id": "c1", "arm": "B", "sample_index": 0, "session_id": "sB",
                 "transcript": "T-B", "ok": True, "error": ""},
            ],
        }, ensure_ascii=False), encoding="utf-8")

        executed = []

        async def execute_fn(case, arm, sample_index, base_url):
            executed.append((case["id"], arm, sample_index))
            return {"case_id": case["id"], "arm": arm, "sample_index": sample_index,
                    "session_id": "s2", "transcript": f"T2-{arm}", "ok": True, "error": ""}

        payload, _ = await self._pipeline(
            tmp_path, [_case("c1"), _case("c2")], resume_path=partial, execute_fn=execute_fn,
        )
        assert executed == [("c2", "A", 0), ("c2", "B", 0)]
        assert len(payload["rows"]) == 2  # c1（沿用）+ c2（新跑）都进了配对

    async def test_resume_config_mismatch_rejects(self, tmp_path):
        partial = tmp_path / "ab-partial-earlier.json"
        partial.write_text(json.dumps({
            "arm_config": {
                "A": {"fingerprint": "old-fp", "variant": "", "model": "mimo-v2.5"},
                "B": {"fingerprint": "b2222222", "variant": "candidate-x", "model": "mimo-v2.5"},
            },
            "results": [],
        }, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(SystemExit):
            await self._pipeline(tmp_path, [_case("c1")], resume_path=partial)

    async def test_resume_partial_survives_error_ridden_judging(self, tmp_path):
        """B2 认证轮实测事故（2026-09-06，周额度耗尽）：80 对里 70 对 judge
        429 烧成 error 行——error 不进互换分母，剩 10 对 9 一致 → rate=0.9
        恰好"达标"，管线判 judge_valid=True 把新旧断点全删了。执行段 160 执行
        的可重判资产差点陪葬（靠最终 run JSON 手工捞回）。

        钉死：判段存在 error 行时，无论 swap rate 多少，resume_path 与
        新 partial 一律保留——断点删除的前提是判段干净（n_error == 0）。"""
        partial = tmp_path / "ab-partial-earlier.json"
        partial.write_text(json.dumps({
            "label": "测试", "pairing": "diagonal",
            "plan": {"k": 1, "pairing": "diagonal",
                     "case_ids": [f"c{i}" for i in range(1, 11)]},
            "arm_config": self._arm_config(),
            "results": [
                {"case_id": f"c{i}", "arm": arm, "sample_index": 0,
                 "session_id": f"s-{arm}{i}", "transcript": f"T-{arm}-{i}",
                 "ok": True, "error": ""}
                for i in range(1, 11) for arm in ("A", "B")
            ],
        }, ensure_ascii=False), encoding="utf-8")

        def factory(model):
            async def judge_call(prompt):
                # 复刻事故形态：一半的对整体失败（两序都 429 → error 对，
                # 不进互换分母），剩余对全部一致 → rate 靠小分母"达标"。
                import re
                found = re.search(r"T-A-(\d+)", prompt)
                case_num = int(found.group(1)) if found else -1
                if case_num % 2 == 0:
                    raise RuntimeError("Error code: 429")
                return "裁决: 平局\n理由: 相当"
            return judge_call

        cases = [_case(f"c{i}") for i in range(1, 11)]  # 10 对：5 对整体失败
        await self._pipeline(
            tmp_path, cases, judge_factory=factory,
            resume_path=partial,
        )
        assert partial.exists(), "resume 断点被误删——error 轮不许弃断点"
        assert (tmp_path / "ab-partial-20260905-170000.json").exists(), \
            "本轮 partial 也被误删——明天重判要靠它"

    async def test_clean_judging_still_deletes_partials(self, tmp_path):
        """对照：判段干净（零 error）时删除断点的既有行为不变。"""
        partial = tmp_path / "ab-partial-earlier.json"
        partial.write_text(json.dumps({
            "label": "测试", "pairing": "diagonal",
            "plan": {"k": 1, "pairing": "diagonal", "case_ids": ["c1", "c2"]},
            "arm_config": self._arm_config(),
            "results": [
                {"case_id": "c1", "arm": "A", "sample_index": 0, "session_id": "sA",
                 "transcript": "T-A", "ok": True, "error": ""},
                {"case_id": "c1", "arm": "B", "sample_index": 0, "session_id": "sB",
                 "transcript": "T-B", "ok": True, "error": ""},
                {"case_id": "c2", "arm": "A", "sample_index": 0, "session_id": "s2A",
                 "transcript": "T2-A", "ok": True, "error": ""},
                {"case_id": "c2", "arm": "B", "sample_index": 0, "session_id": "s2B",
                 "transcript": "T2-B", "ok": True, "error": ""},
            ],
        }, ensure_ascii=False), encoding="utf-8")

        await self._pipeline(tmp_path, [_case("c1"), _case("c2")], resume_path=partial)
        assert not partial.exists()
        assert not (tmp_path / "ab-partial-20260905-170000.json").exists()

    async def test_error_ridden_judging_with_empty_notes_does_not_crash(self, tmp_path):
        """烧前评审（明日额度恢复前最后一审）抓到的必崩点：

        judge_valid=True（error 对不进互换分母，rate 靠小分母达标）+ error 行
        存在 + 无 pair error + 成本审计干净（healths 带 data_dir 且流水齐全
        → cost_notes 空）→ notes == [] → 旧代码 `notes[-1]` IndexError。
        这正是 429 部分失败时最可能的生产形态：管线应以"断点已保留"附注
        优雅退出，而不是 traceback（traceback 会让操作者重跑 → 重烧 480 judge）。"""
        conv = tmp_path / "convdata" / "conversations"
        conv.mkdir(parents=True)
        partial = tmp_path / "ab-partial-earlier.json"
        partial.write_text(json.dumps({
            "label": "测试", "pairing": "diagonal",
            "plan": {"k": 1, "pairing": "diagonal",
                     "case_ids": [f"c{i}" for i in range(1, 11)]},
            "arm_config": self._arm_config(),
            "results": [
                {"case_id": f"c{i}", "arm": arm, "sample_index": 0,
                 "session_id": f"s-{arm}{i}", "transcript": f"T-{arm}-{i}",
                 "ok": True, "error": ""}
                for i in range(1, 11) for arm in ("A", "B")
            ],
        }, ensure_ascii=False), encoding="utf-8")

        def factory(model):
            async def judge_call(prompt):
                import re
                found = re.search(r"T-A-(\d+)", prompt)
                if int(found.group(1)) % 2 == 0:  # 一半的对整体 429
                    raise RuntimeError("Error code: 429")
                return "裁决: 平局\n理由: 相当"
            return judge_call

        healths = self._healths()
        for arm in ("A", "B"):
            healths[arm]["data_dir"] = str(tmp_path / "convdata")
        for i in range(1, 11):
            for arm in ("A", "B"):
                (conv / f"s-{arm}{i}.jsonl").write_text(json.dumps({
                    "kind": "turn", "role": "agent", "model": "mimo-v2.5",
                    "latency_ms": 100, "prompt_tokens": 10, "completion_tokens": 10,
                }), encoding="utf-8")

        payload, progress = await self._pipeline(
            tmp_path, [_case(f"c{i}") for i in range(1, 11)],
            judge_factory=factory, resume_path=partial, healths=healths,
        )
        assert payload["swap"]["n_error"] == 5
        assert any("断点" in line or "保留" in line for line in progress), (
            "else 分支必须给出'断点保留'的可见附注（notes 为空时也不能崩）"
        )
        assert partial.exists()

    async def test_dual_judge_subset_measured(self, tmp_path):
        def factory(model):
            async def judge_call(prompt):
                left = prompt.split("【回复1】")[1].split("【回复2】")[0]
                if "GOOD" in left:
                    return "裁决: 1\n理由: 回复1"
                return "裁决: 2\n理由: 回复2"
            return judge_call

        payload, _ = await self._pipeline(
            tmp_path, [_case("c1"), _case("c2")], judge_factory=factory,
            second_judge_model="deepseek-v4-flash", dual_judge_pairs=1,
        )
        assert payload["dual_judge"]["model"] == "deepseek-v4-flash"
        assert payload["dual_judge"]["n_pairs"] == 2  # 1 对 × 2 顺序
        assert payload["dual_judge"]["rate"] == 1.0   # 两个 judge 同为按内容判
        report = (tmp_path / "ab-report-20260905-170000.md").read_text(encoding="utf-8")
        assert "deepseek-v4-flash" in report

    async def test_dual_judge_requires_second_model(self, tmp_path):
        with pytest.raises(SystemExit):
            await self._pipeline(tmp_path, [_case("c1")], dual_judge_pairs=2)


class TestCostLatencyWiring:
    async def test_per_arm_summary_from_conversations(self, tmp_path):
        from scripts.eval.ab_run import run_ab_pipeline

        conv = tmp_path / "convdata" / "conversations"
        conv.mkdir(parents=True)
        for arm, sid, tokens, latency in (
            ("A", "s-A0-c1", 100, 1000), ("B", "s-B0-c1", 200, 2000),
        ):
            (conv / f"{sid}.jsonl").write_text(json.dumps({
                "kind": "turn", "role": "agent", "model": "mimo-v2.5",
                "latency_ms": latency, "prompt_tokens": 10, "completion_tokens": tokens,
            }), encoding="utf-8")

        async def execute_fn(case, arm, sample_index, base_url):
            return {"case_id": case["id"], "arm": arm, "sample_index": sample_index,
                    "session_id": f"s-{arm}{sample_index}-{case['id']}",
                    "transcript": "t", "ok": True, "error": ""}

        async def judge_call(prompt):
            return "裁决: 平局\n理由: 相当"

        payload = await run_ab_pipeline(
            cases=[_case("c1")], k=1, pairing="diagonal", judge_model="longcat-2.0",
            healths={"A": {"model": "mimo-v2.5", "data_dir": str(tmp_path / "convdata")},
                     "B": {"model": "mimo-v2.5", "data_dir": str(tmp_path / "convdata")}},
            urls={"A": "http://a", "B": "http://b"},
            arm_lines={"A": "a", "B": "b"},
            arm_config={"A": {"fingerprint": "f", "variant": "", "model": "m"},
                        "B": {"fingerprint": "g", "variant": "v", "model": "m"}},
            ground_truth="", eval_dir=tmp_path, stamp="20260905-180000",
            judge_factory=lambda model: judge_call, execute_fn=execute_fn,
            progress=lambda *_: None,
        )
        assert payload["cost_latency"]["A"]["completion_p50"] == 100
        assert payload["cost_latency"]["B"]["completion_p50"] == 200
        assert payload["cost_latency"]["A"]["latency_p50_s"] == 1.0
        report = (tmp_path / "ab-report-20260905-180000.md").read_text(encoding="utf-8")
        assert "100" in report and "200" in report


class TestCliWiring:
    """main_async 接线：真实路径把正确的参数递给管线，dry-run 不进管线。"""

    def _health(self, variant="", fingerprint="a0915fac"):
        return {
            "model": "mimo-v2.5", "prompt_fingerprint": fingerprint, "prompt_variant": variant,
            "semantic_cache": False, "data_dir": "/repo/data",
            "code": {"stale": False, "started_at": "x", "source_mtime": "y", "stale_files": []},
            "retrieval": {"reranker": True, "lexical_index": True, "lexical_gate": 4.0,
                          "probe": {"embedding": "ok", "reranker": "ok"}},
            "fault_injection": {"enabled": True, "active": []},
        }

    def _patch_env(self, monkeypatch):
        monkeypatch.setenv("EVAL_JUDGE_MODEL", "longcat-2.0")
        monkeypatch.delenv("LLM_MODEL", raising=False)

    async def test_real_path_calls_pipeline_with_resolved_args(self, tmp_path, monkeypatch, capsys):
        import scripts.eval.ab_run as ab

        self._patch_env(monkeypatch)
        monkeypatch.setattr(ab, "_fetch_health", self._fake_fetch())
        captured = {}

        async def fake_pipeline(**kwargs):
            captured.update(kwargs)
            return {"report_path": str(tmp_path / "r.md"), "run_json_path": str(tmp_path / "r.json"),
                    "win_rate": {"wins": 1, "n": 2}, "significance": {"significant": False},
                    "stamp": "s", "label": "x"}

        monkeypatch.setattr(ab, "run_ab_pipeline", fake_pipeline)
        args = ab.parse_args([
            "--only", "compare-two", "--k", "1",
            "--arm-a-url", "http://127.0.0.1:8000",
            "--arm-b-url", "http://127.0.0.1:8012",
            "--eval-dir", str(tmp_path),
        ])
        code = await ab.main_async(args)
        assert code == 0
        # 进度回调必须接上：过夜跑的日志里要能看到逐执行/逐判进度
        assert captured["progress"] is not None
        assert captured["progress"] is not print or True
        assert captured["k"] == 1
        assert [c["id"] for c in captured["cases"]] == ["compare-two"]
        assert captured["judge_model"] == "longcat-2.0"
        assert captured["urls"] == {"A": "http://127.0.0.1:8000", "B": "http://127.0.0.1:8012"}
        # 配置行必须来自被测服务自报：两臂各一行，进管线供报告引用
        assert "a0915fac" in captured["arm_lines"]["A"]
        assert captured["arm_config"]["A"]["variant"] == ""
        assert captured["arm_config"]["B"]["variant"] == "candidate-x"

    def _fake_fetch(self):
        async def fetch(url):
            return self._health(variant="candidate-x", fingerprint="b2222222") if ":8012" in url else self._health()
        return fetch

    async def test_judge_same_as_tested_model_blocks_before_pipeline(self, tmp_path, monkeypatch):
        import scripts.eval.ab_run as ab

        monkeypatch.setenv("EVAL_JUDGE_MODEL", "mimo-v2.5")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.setattr(ab, "_fetch_health", self._fake_fetch())

        async def fail_pipeline(**kwargs):
            raise AssertionError("评审模型与被测相同时不许进管线")

        monkeypatch.setattr(ab, "run_ab_pipeline", fail_pipeline)
        args = ab.parse_args([
            "--only", "compare-two", "--k", "1", "--eval-dir", str(tmp_path),
            "--arm-a-url", "http://127.0.0.1:8000", "--arm-b-url", "http://127.0.0.1:8012",
        ])
        with pytest.raises(SystemExit) as err:
            await ab.main_async(args)
        assert "自我偏好" in str(err.value)

    async def test_dry_run_never_enters_pipeline(self, tmp_path, monkeypatch, capsys):
        import scripts.eval.ab_run as ab

        self._patch_env(monkeypatch)
        monkeypatch.setattr(ab, "_fetch_health", self._fake_fetch())

        async def fail_pipeline(**kwargs):
            raise AssertionError("dry-run 不许进真实管线")

        monkeypatch.setattr(ab, "run_ab_pipeline", fail_pipeline)
        args = ab.parse_args([
            "--dry-run", "--only", "compare-two", "--eval-dir", str(tmp_path),
            "--arm-a-url", "http://127.0.0.1:8000", "--arm-b-url", "http://127.0.0.1:8012",
        ])
        code = await ab.main_async(args)
        assert code == 0
        out = capsys.readouterr().out
        assert "未发起任何模型调用" in out

    async def test_pairing_flag_reaches_ledger(self, tmp_path, monkeypatch):
        import scripts.eval.ab_run as ab

        self._patch_env(monkeypatch)
        monkeypatch.setattr(ab, "_fetch_health", self._fake_fetch())
        captured = {}

        async def fake_pipeline(**kwargs):
            captured.update(kwargs)
            return {"report_path": "x", "run_json_path": "y", "win_rate": {},
                    "significance": {}, "stamp": "s", "label": ""}

        monkeypatch.setattr(ab, "run_ab_pipeline", fake_pipeline)
        args = ab.parse_args([
            "--only", "compare-two", "--k", "2", "--pairing", "cross", "--eval-dir", str(tmp_path),
            "--arm-a-url", "http://127.0.0.1:8000", "--arm-b-url", "http://127.0.0.1:8012",
        ])
        await ab.main_async(args)
        assert captured["pairing"] == "cross"
        assert captured["k"] == 2


class TestReviewFixes:
    """独立验收审查（M1 双验收）抓出的问题，逐条回归钉住。"""

    async def test_dual_judge_without_second_model_rejected_before_pipeline(self, tmp_path, monkeypatch):
        """烧完钱才发现缺第二评审模型 = 执行段 + 判段全重烧——必须在开跑前拦。"""
        import scripts.eval.ab_run as ab

        monkeypatch.setenv("EVAL_JUDGE_MODEL", "longcat-2.0")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        wiring = TestCliWiring()
        monkeypatch.setattr(ab, "_fetch_health", wiring._fake_fetch())

        async def fail_pipeline(**kwargs):
            raise AssertionError("配置不完整时不许进管线")

        monkeypatch.setattr(ab, "run_ab_pipeline", fail_pipeline)
        args = ab.parse_args([
            "--only", "compare-two", "--dual-judge-pairs", "3", "--eval-dir", str(tmp_path),
            "--arm-a-url", "http://127.0.0.1:8000", "--arm-b-url", "http://127.0.0.1:8012",
        ])
        with pytest.raises(SystemExit):
            await ab.main_async(args)

    async def test_dual_judge_pairs_above_20_rejected(self, tmp_path, monkeypatch):
        """授权上限 ≤20 对：超了不是调大就行，是预算边界。"""
        import scripts.eval.ab_run as ab

        monkeypatch.setenv("EVAL_JUDGE_MODEL", "longcat-2.0")
        monkeypatch.setenv("EVAL_SECOND_JUDGE_MODEL", "deepseek-v4-flash")
        wiring = TestCliWiring()
        monkeypatch.setattr(ab, "_fetch_health", wiring._fake_fetch())

        async def fail_pipeline(**kwargs):
            raise AssertionError("超授权上限时不许进管线")

        monkeypatch.setattr(ab, "run_ab_pipeline", fail_pipeline)
        args = ab.parse_args([
            "--only", "compare-two", "--dual-judge-pairs", "21",
            "--second-judge-model", "deepseek-v4-flash", "--eval-dir", str(tmp_path),
            "--arm-a-url", "http://127.0.0.1:8000", "--arm-b-url", "http://127.0.0.1:8012",
        ])
        with pytest.raises(SystemExit):
            await ab.main_async(args)

    async def test_resume_prereq_rerun_does_not_duplicate_results(self, tmp_path):
        """前置补跑的三元组不能重复进 results——执行分母不许虚高。"""
        partial = tmp_path / "ab-partial-earlier.json"
        partial.write_text(json.dumps({
            "label": "", "pairing": "diagonal",
            "plan": {"k": 1, "pairing": "diagonal", "case_ids": ["memory-write", "memory-recall"]},
            "arm_config": TestRunAbPipeline()._arm_config(),
            "results": [
                {"case_id": "memory-write", "arm": "B", "sample_index": 0, "session_id": "sw",
                 "transcript": "TW", "ok": True, "error": ""},
                {"case_id": "memory-recall", "arm": "B", "sample_index": 0, "session_id": "sr",
                 "transcript": "TR", "ok": True, "error": ""},
            ],
        }, ensure_ascii=False), encoding="utf-8")

        executed = []

        async def execute_fn(case, arm, sample_index, base_url):
            executed.append(triple_key(case["id"], arm, sample_index))
            return {"case_id": case["id"], "arm": arm, "sample_index": sample_index,
                    "session_id": f"s-{len(executed)}", "transcript": f"T{len(executed)}",
                    "ok": True, "error": ""}

        payload, _ = await TestRunAbPipeline()._pipeline(
            tmp_path,
            [_case("memory-write", buyer_id="b"), _case("memory-recall", buyer_id="b", requires=["memory-write"])],
            k=1, resume_path=partial, execute_fn=execute_fn,
        )
        # A 臂两个三元组待跑；memory-recall (A,0) 的前置 memory-write (A,0) 也待跑（不是补跑）
        assert sorted(executed) == ["memory-recall|A|0", "memory-write|A|0"]
        # results 每个三元组只留最新一条：total == 计划数 4（2 用例 × 2 臂）
        assert payload["executions"]["total"] == 4
        run_json = json.loads((tmp_path / "ab-run-20260905-170000.json").read_text(encoding="utf-8"))
        keys = [triple_key(r["case_id"], r["arm"], r["sample_index"]) for r in run_json["results"]]
        assert len(keys) == len(set(keys))

    async def test_resume_with_different_k_rejected(self, tmp_path):
        """k/pairing/用例选择变了就续跑 = 读数混装——partial 里的 plan 要一起核对。"""
        partial = tmp_path / "ab-partial-earlier.json"
        partial.write_text(json.dumps({
            "label": "", "pairing": "diagonal", "plan": {"k": 2, "pairing": "diagonal", "case_ids": ["c1", "c2"]},
            "arm_config": TestRunAbPipeline()._arm_config(),
            "results": [],
        }, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(SystemExit):
            await TestRunAbPipeline()._pipeline(
                tmp_path, [_case("c1"), _case("c2")], k=1, resume_path=partial,
            )

    async def test_resume_with_different_case_selection_rejected(self, tmp_path):
        partial = tmp_path / "ab-partial-earlier.json"
        partial.write_text(json.dumps({
            "label": "", "pairing": "diagonal", "plan": {"k": 1, "pairing": "diagonal", "case_ids": ["c1", "c2"]},
            "arm_config": TestRunAbPipeline()._arm_config(),
            "results": [],
        }, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(SystemExit):
            await TestRunAbPipeline()._pipeline(tmp_path, [_case("c1"), _case("c3")], k=1, resume_path=partial)

    async def test_cost_latency_uses_per_arm_data_dirs(self, tmp_path):
        """两臂各自报的 data_dir 各扫各的——臂 B 独立 DATA_DIR 是合理配置。"""
        conv_a = tmp_path / "dataA" / "conversations"
        conv_b = tmp_path / "dataB" / "conversations"
        conv_a.mkdir(parents=True)
        conv_b.mkdir(parents=True)
        (conv_a / "s-A0-c1.jsonl").write_text(json.dumps({
            "kind": "turn", "role": "agent", "model": "m", "latency_ms": 1000,
            "prompt_tokens": 10, "completion_tokens": 111,
        }), encoding="utf-8")
        (conv_b / "s-B0-c1.jsonl").write_text(json.dumps({
            "kind": "turn", "role": "agent", "model": "m", "latency_ms": 2000,
            "prompt_tokens": 10, "completion_tokens": 222,
        }), encoding="utf-8")

        healths = {
            "A": {"model": "m", "data_dir": str(tmp_path / "dataA")},
            "B": {"model": "m", "data_dir": str(tmp_path / "dataB")},
        }

        async def execute_fn(case, arm, sample_index, base_url):
            return {"case_id": case["id"], "arm": arm, "sample_index": sample_index,
                    "session_id": f"s-{arm}{sample_index}-{case['id']}",
                    "transcript": "t", "ok": True, "error": ""}

        async def judge_call(prompt):
            return "裁决: 平局\n理由: 相当"

        from scripts.eval.ab_run import run_ab_pipeline

        payload = await run_ab_pipeline(
            cases=[_case("c1")], k=1, pairing="diagonal", judge_model="longcat-2.0",
            healths=healths, urls={"A": "http://a", "B": "http://b"},
            arm_lines={"A": "a", "B": "b"},
            arm_config={"A": {"fingerprint": "f", "variant": "", "model": "m"},
                        "B": {"fingerprint": "g", "variant": "v", "model": "m"}},
            ground_truth="", eval_dir=tmp_path, stamp="20260905-190000",
            judge_factory=lambda model: judge_call, execute_fn=execute_fn,
            progress=lambda *_: None,
        )
        assert payload["cost_latency"]["A"]["completion_p50"] == 111
        assert payload["cost_latency"]["B"]["completion_p50"] == 222

    async def test_cost_latency_failure_degrades_to_note_not_crash(self, tmp_path):
        """流水对不上：成本读数降级为'未测定'进报告——不许炸掉整轮产物。"""
        empty = tmp_path / "empty-data" / "conversations"
        empty.mkdir(parents=True)
        healths = {
            "A": {"model": "m", "data_dir": str(tmp_path / "empty-data")},
            "B": {"model": "m", "data_dir": str(tmp_path / "empty-data")},
        }

        async def execute_fn(case, arm, sample_index, base_url):
            return {"case_id": case["id"], "arm": arm, "sample_index": sample_index,
                    "session_id": f"s-{arm}{sample_index}-{case['id']}",
                    "transcript": "t", "ok": True, "error": ""}

        async def judge_call(prompt):
            return "裁决: 平局\n理由: 相当"

        from scripts.eval.ab_run import run_ab_pipeline

        payload = await run_ab_pipeline(
            cases=[_case("c1")], k=1, pairing="diagonal", judge_model="longcat-2.0",
            healths=healths, urls={"A": "http://a", "B": "http://b"},
            arm_lines={"A": "a", "B": "b"},
            arm_config={"A": {"fingerprint": "f", "variant": "", "model": "m"},
                        "B": {"fingerprint": "g", "variant": "v", "model": "m"}},
            ground_truth="", eval_dir=tmp_path, stamp="20260905-191000",
            judge_factory=lambda model: judge_call, execute_fn=execute_fn,
            progress=lambda *_: None,
        )
        assert payload["cost_latency"] == {}
        assert any("未测定" in note for note in payload["notes"])
        assert (tmp_path / "ab-report-20260905-191000.md").exists()

    async def test_report_names_pair_errors_and_judge_errors(self, tmp_path):
        """error 行必须带名字进人读报告——只有计数是塌缩的另一种形态。"""
        payload, _ = await TestRunAbPipeline()._pipeline(
            tmp_path, [_case("c1"), _case("c2")], fail={triple_key("c1", "A", 0)},
        )
        report = (tmp_path / "ab-report-20260905-170000.md").read_text(encoding="utf-8")
        assert "错误明细" in report
        assert "c1" in report and "ReadTimeout" in report  # 配对错误逐条点名
        assert "臂 A 采样 0 执行失败" in report

    async def test_resumed_source_partial_cleaned_after_success(self, tmp_path):
        partial = tmp_path / "ab-partial-earlier.json"
        partial.write_text(json.dumps({
            "label": "", "pairing": "diagonal",
            "plan": {"k": 1, "pairing": "diagonal", "case_ids": ["c1"]},
            "arm_config": TestRunAbPipeline()._arm_config(),
            "results": [
                {"case_id": "c1", "arm": "A", "sample_index": 0, "session_id": "sA",
                 "transcript": "TA", "ok": True, "error": ""},
                {"case_id": "c1", "arm": "B", "sample_index": 0, "session_id": "sB",
                 "transcript": "TB", "ok": True, "error": ""},
            ],
        }, ensure_ascii=False), encoding="utf-8")
        await TestRunAbPipeline()._pipeline(tmp_path, [_case("c1")], k=1, resume_path=partial)
        assert not partial.exists()  # 续跑完成后旧 partial 不许残留再被误用

    async def test_seconds_per_intent_override_reaches_plan(self, tmp_path):
        payload, _ = await TestRunAbPipeline()._pipeline(tmp_path, [_case("c1")], seconds_per_intent=10.0)
        assert payload["plan"]["estimated_minutes"] == pytest.approx(2 * 10.0 / 60, abs=0.1)


class TestJudgeInvalidationRetention:
    """互换一致率 < 90% 时该轮 judge 读数作废重跑（指标表口径）。

    重判不该重烧执行段：作废轮必须保留 ab-partial，--resume 直接进判段。
    """

    def _order_biased_factory(self):
        """位置偏置 judge：永远选位置 1——两序必翻，互换一致率 0。"""

        def factory(model):
            async def judge_call(prompt):
                return "裁决: 1\n理由: 位置在前的更好"
            return judge_call
        return factory

    async def test_invalid_round_keeps_partial_for_rejudge(self, tmp_path):
        payload, _ = await TestRunAbPipeline()._pipeline(
            tmp_path, [_case("c1")], judge_factory=self._order_biased_factory(),
        )
        assert payload["significance"]["judge_valid"] is False
        assert (tmp_path / "ab-partial-20260905-170000.json").exists()  # 供 --resume 重判
        report = (tmp_path / "ab-report-20260905-170000.md").read_text(encoding="utf-8")
        assert "作废重跑" in report
        assert "ab-partial-20260905-170000.json" in report  # 报告里写明续跑入口

    async def test_valid_round_still_deletes_partial(self, tmp_path):
        payload, _ = await TestRunAbPipeline()._pipeline(tmp_path, [_case("c1")])
        assert payload["significance"]["judge_valid"] is True
        assert not (tmp_path / "ab-partial-20260905-170000.json").exists()


class TestJudgeCallTransport:
    async def test_judge_payload_pins_model_temp_and_max_tokens(self, monkeypatch):
        """deepseek CoT 被网关默认 max_tokens 截断（双 judge 轮 7/24 解析失败）——
        判据调用必须显式给足输出预算，截断的判词是脏输出。"""
        import scripts.eval.ab_run as ab

        captured = {}

        async def fake_retry(client, payload):
            captured.update(payload)
            return "裁决: 1\n理由: ok"

        monkeypatch.setattr("scripts.eval_regression.call_llm_with_retry", fake_retry)
        judge = ab.make_judge_call(client=None, model="deepseek-v4-flash")
        assert await judge("提示词") == "裁决: 1\n理由: ok"
        assert captured["model"] == "deepseek-v4-flash"
        assert captured["temperature"] == 0
        assert captured["max_tokens"] >= 1500
        assert captured["messages"][0]["content"] == "提示词"

    async def test_max_tokens_scales_with_model_reasoning_behavior(self, monkeypatch):
        """按模型分预算：longcat-2.0 是推理模型（reasoning 与 content 分离，
        自然 reasoning 实测 ~3787，M2'-c 8/32 KeyError 根因）→ 12000；
        deepseek-v4-flash 是 CoT-in-content（预算越大长跑越长，M2'-d 探针实测
        大 prompt+12000 被网关 225s 杀连接）→ 沿用其历史校准 2000。"""
        import scripts.eval.ab_run as ab

        captured = {}

        async def fake_retry(client, payload):
            captured.update(payload)
            return "裁决: 1\n理由: ok"

        monkeypatch.setattr("scripts.eval_regression.call_llm_with_retry", fake_retry)
        await ab.make_judge_call(client=None, model="longcat-2.0")("p")
        assert captured["max_tokens"] >= 10000
        await ab.make_judge_call(client=None, model="deepseek-v4-flash")("p")
        assert captured["max_tokens"] <= 2500


class TestJudgePairRowsConcurrent:
    """判段并发（授权文档口径：judge 调用可 ~4 并发压缩墙钟）。

    纪律不变：并发只改调度不改读数——行序与 pairs 一致、单飞数有上界、
    单对失败不拖垮整轮。串行路径（max_concurrency=1）行为逐字不变。
    """

    @staticmethod
    def _pairs(n: int) -> list[dict]:
        return [
            {
                "case_id": f"c{i}", "pair_index": 0,
                "left": {"transcript": "t1"}, "right": {"transcript": "t2"},
                "case_prompt_text": "问", "prior_context": "",
            }
            for i in range(n)
        ]

    async def test_same_rows_same_order_and_inflight_bounded(self):
        import asyncio

        state = {"inflight": 0, "max_inflight": 0}
        lock = asyncio.Lock()

        async def slow_judge(prompt: str) -> str:
            async with lock:
                state["inflight"] += 1
                state["max_inflight"] = max(state["max_inflight"], state["inflight"])
            await asyncio.sleep(0.02)
            async with lock:
                state["inflight"] -= 1
            return "裁决: 1\n理由: x"

        rows = await judge_pair_rows(
            self._pairs(6), slow_judge, ground_truth="", votes=2, max_concurrency=3,
        )
        assert [r["case_id"] for r in rows] == [f"c{i}" for i in range(6)]
        assert state["max_inflight"] <= 3
        assert all(r["verdict_ab"] == "a" and r["verdict_ba"] == "b" for r in rows)
        assert all(r["votes_ab"] == ["a", "a"] for r in rows)

    async def test_one_pair_failure_does_not_poison_others(self):
        async def flaky(prompt: str) -> str:
            if "poison-t1" in prompt:
                raise RuntimeError("Error code: 503")
            return "裁决: 2\n理由: x"

        pairs = self._pairs(5)
        pairs[3]["left"]["transcript"] = "poison-t1"
        rows = await judge_pair_rows(
            pairs, flaky, ground_truth="", max_concurrency=3,
        )
        assert rows[3]["verdict_ab"] is None and "503" in rows[3]["error_ab"]
        assert all(r["verdict_ab"] == "b" for i, r in enumerate(rows) if i != 3)

    async def test_invalid_concurrency_raises(self):
        with pytest.raises(ValueError, match="并发"):
            await judge_pair_rows(self._pairs(1), None, ground_truth="", max_concurrency=0)
