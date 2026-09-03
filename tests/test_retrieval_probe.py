# -*- coding: utf-8 -*-
"""检索依赖的深度探活，以及配置行如实渲染。

十四期小样本实测暴露的缺口：报告配置行写着"精排 开"，
而同一轮轨迹里 `recall_strategy` 是 `bm25_only`——两条隧道都是 502，
精排一次都没跑过。配置行不是写错了，是它**问错了问题**：
它读的是 `RERANKER_BASE_URL` 配没配，而不是那个地址通不通。

与踩坑 32（服务跑着旧代码）完全同构：分数标着一个并不成立的配置，
横向比较必然得出错的结论，甚至可能得出"精排没用"——
而真相是这一轮压根没有精排。
"""
import pytest

from app.application.harness.run_identity import describe_run
from app.infrastructure.retrieval_probe import probe_retrieval


class _Settings:
    def __init__(self, embedding_url="", reranker_url="") -> None:
        self.embedding_base_url = embedding_url
        self.embedding_api_key = "k"
        self.embedding_model = "m"
        self.reranker_base_url = reranker_url


class _Response:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Client:
    def __init__(self, post=None, get=None) -> None:
        self._post, self._get = post, get
        self.calls: list[str] = []

    async def post(self, url, **kwargs):
        self.calls.append(f"POST {url}")
        if isinstance(self._post, Exception):
            raise self._post
        return self._post or _Response()

    async def get(self, url, **kwargs):
        self.calls.append(f"GET {url}")
        if isinstance(self._get, Exception):
            raise self._get
        return self._get or _Response()


class TestProbe:
    async def test_unconfigured_components_report_disabled(self):
        """没配 ≠ 不可达。两者的含义相反，不能混成一个值。"""
        result = await probe_retrieval(_Settings(), client=_Client())
        assert result == {"embedding": "disabled", "reranker": "disabled"}

    async def test_reachable_components_report_ok(self):
        result = await probe_retrieval(
            _Settings("http://e/v1", "http://r"), client=_Client(),
        )
        assert result == {"embedding": "ok", "reranker": "ok"}

    async def test_unreachable_reports_the_error(self):
        """错误原文要留：502 与连接超时的排查方向完全不同。"""
        client = _Client(post=RuntimeError("HTTP 502"), get=RuntimeError("Connection refused"))
        result = await probe_retrieval(_Settings("http://e/v1", "http://r"), client=client)
        assert result["embedding"].startswith("error") and "502" in result["embedding"]
        assert result["reranker"].startswith("error") and "refused" in result["reranker"]

    async def test_probe_never_raises(self):
        """探活自己不能把 /health 打挂——它是给人看读数用的，不是关键路径。"""
        client = _Client(post=BaseException("boom"))  # type: ignore[arg-type]
        result = await probe_retrieval(_Settings("http://e/v1"), client=client)
        assert result["embedding"].startswith("error")

    async def test_hits_the_expected_endpoints(self):
        client = _Client()
        await probe_retrieval(_Settings("http://e/v1", "http://r/"), client=client)
        assert "POST http://e/v1/embeddings" in client.calls
        assert "GET http://r/health" in client.calls, "尾斜杠要削掉，否则打成 //health"


class TestRunLineTellsTheTruth:
    def _line(self, probe=None, reranker=True) -> str:
        retrieval = {"reranker": reranker, "lexical_index": True, "lexical_gate": 4.0}
        if probe is not None:
            retrieval["probe"] = probe
        return describe_run({"model": "m", "retrieval": retrieval})

    def test_configured_but_unreachable_is_called_out(self):
        line = self._line({"reranker": "error: HTTP 502", "embedding": "ok"})
        assert "精排 开(实测不可达)" in line

    def test_reachable_renders_as_before(self):
        assert "精排 开" in self._line({"reranker": "ok", "embedding": "ok"})
        assert "实测不可达" not in self._line({"reranker": "ok", "embedding": "ok"})

    def test_no_probe_adds_no_suffix(self):
        """没做深度探活时不加任何标记。

        多一个"未知"标记只会让每行都变长，而真正要跳出来的是
        "配了但没生效"那一种。
        """
        assert "精排 开" in self._line(None)
        assert "实测" not in self._line(None)

    def test_dead_vector_path_gets_its_own_segment(self):
        """向量路挂掉要单独说：它解释了为什么档位掉到 bm25_only。

        配置行里原本没有向量路这一格——它一直被当成"配了就有"，
        而十四期那一轮它整条不可用。
        """
        line = self._line({"embedding": "error: HTTP 502", "reranker": "ok"})
        assert "向量路 实测不可达" in line

    def test_healthy_vector_path_stays_quiet(self):
        assert "向量路" not in self._line({"embedding": "ok", "reranker": "ok"})

    @pytest.mark.parametrize("probe", [None, {}, "坏数据", {"reranker": None}])
    def test_malformed_probe_does_not_break_the_line(self, probe):
        """配置行是报告的第一行，任何情况下都得渲染得出来。"""
        assert "被测模型" in self._line(probe)


class TestWiring:
    """接线判据：探活写好了但没人调，等于没写（七期与十四期的同一条）。"""

    def test_health_accepts_deep_and_calls_the_probe(self):
        import inspect

        from app.presentation import server

        source = inspect.getsource(server)
        assert "async def health(deep: bool = False)" in source
        assert 'payload["retrieval"]["probe"] = await probe_retrieval(c.settings)' in source

    def test_default_health_does_no_probing(self):
        """默认 /health 必须零外部调用：它同时是容器存活探针，
        每 10 秒打一次外部服务会把探针本身变得不稳定。"""
        import inspect

        from app.presentation import server

        source = inspect.getsource(server)
        probe_line = source.index("probe_retrieval(c.settings)")
        guard_line = source.rindex("if deep:", 0, probe_line)
        assert guard_line < probe_line, "探活必须在 if deep 之内"

    def test_eval_regression_asks_for_the_deep_variant(self):
        """报告开头那行的可信度全靠这一处：不问 deep=1，配置行就还是在报"配了什么"。"""
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "scripts" / "eval_regression.py").read_text(
            encoding="utf-8",
        )
        assert "/health?deep=1" in source
