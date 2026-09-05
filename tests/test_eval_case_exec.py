# -*- coding: utf-8 -*-
"""eval_regression 为 A/B 真实路径拆出的共享件。

execute_case（多轮对话执行，不含 rubric judge）是 A/B 每次采样的执行单元：
  - k 次采样 = k 个独立会话：会话/买家 id 由调用方注入（样本索引进 id），
    否则 k 次采样串会话、记忆类用例互相污染（授权文档 M1）；
  - 基线臂与候选臂是两个服务实例：base_url 必须可以逐臂指定；
  - 故障注入失败/查询失败时必须清干净——漏清会让后面每条用例带着故障跑
    （run_case 原有注释的纪律，原来只在成功路径上成立，这是补上的洞）。

call_llm_with_retry / resolve_judge_model 是 judge 传输与模型解析的唯一一份
实现：rubric 判分与 A/B 成对判分共用，不许两处各回退各的。
"""
import pytest

import scripts.eval_regression as er


def _case(case_id: str = "demo", queries: int = 2, faults=None, buyer_id=None) -> dict:
    case = {
        "id": case_id,
        "queries": [f"问题{i}" for i in range(queries)],
        "description": case_id,
        "rubric": {"p0": ["数字必须可靠"], "p1": ["行为正确"], "p2": ["表达清楚"]},
    }
    if faults:
        case["faults"] = faults
    if buyer_id:
        case["buyer_id"] = buyer_id
    return case


class _IntentResponse:
    def __init__(self, text: str) -> None:
        self._text = text

    def json(self):
        return {"final_text": self._text}

    def raise_for_status(self):
        return None


class _FaultResponse:
    def __init__(self, components) -> None:
        self._components = components

    def json(self):
        return {}

    def raise_for_status(self):
        return None


class _FakeServerClient:
    """同时应答 /debug/faults 与 /commerce/intents 的假服务端。"""

    def __init__(self, replies: list[str] | None = None, fail_on_query: int | None = None) -> None:
        self.replies = replies or [f"答{i}" for i in range(10)]
        self.fail_on_query = fail_on_query
        self.intents: list[tuple[str, dict]] = []
        self.fault_posts: list[tuple[str, list]] = []
        self._query_count = 0

    async def post(self, url, json=None, timeout=None):
        if url.endswith("/debug/faults"):
            self.fault_posts.append((url, list((json or {}).get("components", []))))
            return _FaultResponse(json.get("components"))
        self._query_count += 1
        if self.fail_on_query is not None and self._query_count == self.fail_on_query:
            raise RuntimeError("boom: 第 2 轮炸了")
        self.intents.append((url, json or {}))
        return _IntentResponse(self.replies[len(self.intents) - 1])


class TestExecuteCase:
    async def test_runs_all_queries_and_builds_transcript(self):
        client = _FakeServerClient(["答一", "答二"])
        result = await er.execute_case(client, _case(queries=2))
        assert len(client.intents) == 2
        assert "[买家] 问题0" in result["transcript"]
        assert "[Agent] 答二" in result["transcript"]
        # 会话 id 派生逻辑与 run_case 同源：eval-<case id>-<随机尾>
        assert result["session_id"].startswith("eval-demo-")

    async def test_buyer_id_derivation_reused_from_run_case(self):
        client = _FakeServerClient()
        await er.execute_case(client, _case(buyer_id=None))
        assert client.intents[0][1]["buyer_id"] == "eval-buyer-demo"

    async def test_explicit_buyer_id_honored(self):
        client = _FakeServerClient()
        await er.execute_case(client, _case(buyer_id="eval-memory-buyer"))
        assert client.intents[0][1]["buyer_id"] == "eval-memory-buyer"

    async def test_explicit_session_id_used_not_regenerated(self):
        """A/B 的样本索引进会话 id：调用方注入什么就用什么。"""
        client = _FakeServerClient()
        result = await er.execute_case(client, _case(), session_id="ab-b-k1-demo-x")
        assert result["session_id"] == "ab-b-k1-demo-x"
        assert client.intents[0][1]["shopping_session_id"] == "ab-b-k1-demo-x"

    async def test_base_url_override_targets_that_instance(self):
        """两臂是两个服务实例：URL 必须逐臂可指定。"""
        client = _FakeServerClient()
        await er.execute_case(client, _case(), base_url="http://127.0.0.1:8012")
        assert client.intents[0][0] == "http://127.0.0.1:8012/commerce/intents"


class TestFaultLifecycle:
    async def test_faults_applied_and_cleared(self):
        client = _FakeServerClient(["答一"])
        await er.execute_case(client, _case(faults=["reranker"], queries=1))
        assert [payload for _, payload in client.fault_posts] == [["reranker"], []]
        assert all(url.endswith("/debug/faults") for url, _ in client.fault_posts)

    async def test_faults_cleared_when_query_fails(self):
        """查询炸了也必须清故障——漏清让后面每条用例带着故障跑。

        run_case 原实现只在成功路径上清理；A/B 一轮几百次执行，
        中途炸一条的概率不可忽略，这洞必须由 execute_case 兜住。
        """
        client = _FakeServerClient(fail_on_query=2)
        with pytest.raises(RuntimeError):
            await er.execute_case(client, _case(faults=["reranker"]))
        assert client.fault_posts[-1][1] == []
        assert len(client.fault_posts) == 2

    async def test_clear_failure_does_not_mask_original_error(self, monkeypatch):
        """清理自己失败时，原始异常才是主因——不许被清理异常顶掉。"""
        client = _FakeServerClient(["答一"], fail_on_query=1)
        original = er.apply_faults

        async def apply_faults_stub(client_, components, base_url=None):
            if components:
                return await original(client_, components, base_url=base_url)
            raise RuntimeError("清理也炸了")

        monkeypatch.setattr(er, "apply_faults", apply_faults_stub)
        with pytest.raises(RuntimeError, match="boom"):
            await er.execute_case(client, _case(faults=["reranker"], queries=1))

    async def test_run_case_clears_faults_too(self):
        """run_case 走 execute_case 后同样获得这个兜底。"""
        client = _FakeServerClient(["答一"], fail_on_query=1)
        with pytest.raises(RuntimeError):
            await er.run_case(client, _case(faults=["reranker"], queries=1), "事实表")
        assert client.fault_posts[-1][1] == []


class TestRunCaseSplit:
    async def test_run_case_still_scores_with_judge(self, monkeypatch):
        async def fake_judge(client, transcript, rubric, ground_truth, prior_context=""):
            return {"p0": [{"criterion": "c", "reason": "r", "pass": True}], "p1": [], "p2": []}

        monkeypatch.setattr(er, "call_judge", fake_judge)
        client = _FakeServerClient(["答一"])
        result = await er.run_case(client, _case(queries=1), "事实表")
        assert result["score"] == 1.0 and result["verdict"] == "PASS"
        assert result["p0_pass"] is True
        assert result["transcript"].startswith("[买家] 问题0")

    async def test_run_case_honors_base_url(self, monkeypatch):
        async def fake_judge(client, transcript, rubric, ground_truth, prior_context=""):
            return {"p0": [], "p1": [], "p2": []}

        monkeypatch.setattr(er, "call_judge", fake_judge)
        client = _FakeServerClient(["答一"])
        await er.run_case(client, _case(queries=1), "事实表", base_url="http://127.0.0.1:8011")
        assert client.intents[0][0] == "http://127.0.0.1:8011/commerce/intents"


class TestResolveJudgeModel:
    def test_eval_judge_model_wins(self, monkeypatch):
        monkeypatch.setenv("EVAL_JUDGE_MODEL", "longcat-2.0")
        monkeypatch.setenv("LLM_MODEL", "mimo-v2.5")
        assert er.resolve_judge_model() == "longcat-2.0"

    def test_falls_back_to_llm_model(self, monkeypatch):
        monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
        monkeypatch.setenv("LLM_MODEL", "mimo-v2.5")
        assert er.resolve_judge_model() == "mimo-v2.5"

    def test_default_is_longcat(self, monkeypatch):
        monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        assert er.resolve_judge_model() == "longcat-2.0"


class TestCallLlmWithRetry:
    class _FlakyClient:
        def __init__(self, failures: int) -> None:
            self.failures = failures
            self.calls = 0

        async def post(self, url, headers=None, json=None, timeout=None):
            self.calls += 1
            if self.calls <= self.failures:
                error = RuntimeError("Error code: 502")
                error.status_code = 502
                raise error

            class _Resp:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"choices": [{"message": {"content": "你好"}}]}

            return _Resp()

    async def test_retries_transient_then_returns_content(self, monkeypatch):
        monkeypatch.setattr(er, "_JUDGE_RETRY_BASE_SECONDS", 0.001)
        client = self._FlakyClient(failures=1)
        assert await er.call_llm_with_retry(client, {"model": "m"}) == "你好"
        assert client.calls == 2

    async def test_non_transient_raises_immediately(self, monkeypatch):
        monkeypatch.setattr(er, "_JUDGE_RETRY_BASE_SECONDS", 0.001)
        client = self._FlakyClient(failures=0)

        def raise_bad_request(*args, **kwargs):
            raise RuntimeError("Error code: 400")

        async def post(*args, **kwargs):
            client.calls += 1
            raise_bad_request()

        client.post = post
        with pytest.raises(RuntimeError, match="400"):
            await er.call_llm_with_retry(client, {"model": "m"})
        assert client.calls == 1


class TestFaultClearVisibility:
    """独立审查（M1 增量复核）：清理失败的可见性必须覆盖两条路径。"""

    async def test_success_with_failing_clear_records_error(self, monkeypatch):
        """执行成功 + 清理失败：结果字段必须带出清理失败，不许只有 stdout。"""
        client = _FakeServerClient()
        original = er.apply_faults

        async def apply_faults_stub(client_, components, base_url=None):
            if components:
                return await original(client_, components, base_url=base_url)
            raise RuntimeError("清理也炸了")

        monkeypatch.setattr(er, "apply_faults", apply_faults_stub)
        result = await er.execute_case(client, _case(faults=["reranker"], queries=1))
        assert result["fault_clear_error"] != ""
        assert "清理" in result["fault_clear_error"]

    async def test_failure_with_failing_clear_attaches_to_exception(self, monkeypatch):
        """执行失败 + 清理失败：原始异常为主因，但清理失败要挂在异常上留名。"""
        client = _FakeServerClient(fail_on_query=1)
        original = er.apply_faults

        async def apply_faults_stub(client_, components, base_url=None):
            if components:
                return await original(client_, components, base_url=base_url)
            raise RuntimeError("清理也炸了")

        monkeypatch.setattr(er, "apply_faults", apply_faults_stub)
        with pytest.raises(RuntimeError, match="boom") as err_info:
            await er.execute_case(client, _case(faults=["reranker"], queries=1))
        assert "清理" in getattr(err_info.value, "fault_clear_error", "")
