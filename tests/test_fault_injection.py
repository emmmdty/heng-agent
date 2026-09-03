# -*- coding: utf-8 -*-
"""故障注入（评测态基建，默认全关）。

要解决的问题：`tool-error-honesty` 这类用例目前只覆盖"目的国不支持"——
一个**参数错误**。而真正会在线上发生的是**服务不可达**：精排挂了、
向量库连不上、embedding 网关超时。这几条路径 `catalog_search` 里都有降级代码，
也都有单测，但**从没有一条端到端用例检验过"降级之后 Agent 怎么跟买家说"**。

做法上刻意选了装饰器：`catalog_search` 的降级链本来就由真实异常触发，
所以注入不需要在业务代码里加任何分支——在端口边界包一层、抛异常就行。
生产代码路径一行不改，风险收敛在组装根一处。
"""
import pytest

from app.domain.catalog.ports.retrieval_ports import VectorHit
from app.infrastructure.faults import (
    COMPONENTS,
    COMPONENT_EMBEDDING,
    COMPONENT_RERANKER,
    COMPONENT_VECTOR_INDEX,
    FaultRegistry,
    InjectedFault,
    install_fault_injection,
)


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [0.1, 0.2]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[0.1, 0.2] for _ in texts]


class _FakeReranker:
    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [1.0] * len(documents)


class _FakeIndex:
    def __init__(self) -> None:
        self.upserts = 0
        self.closed = False

    async def ensure_ready(self, vector_dim: int) -> None:
        return None

    async def upsert_products(self, products, embeddings) -> None:
        self.upserts += 1

    async def search(self, embedding, top_n: int):
        return [VectorHit(product_id="P1001", score=0.9)]

    async def close(self) -> None:
        self.closed = True


class _Settings:
    def __init__(self, enabled: bool) -> None:
        self.fault_injection_enabled = enabled


class TestRegistry:
    def test_disabled_registry_refuses_to_activate(self):
        """未启用的进程里激活故障必须**报错**，不能静默成功。

        评测那头据此判定：用例声明了 faults 而服务没开注入时，
        这条用例要判 ERROR 而不是在"一切正常"的情况下跑完然后 PASS——
        那样判据就成了绿色装饰（同 CI 里"没数据就当通过"的陷阱）。
        """
        registry = FaultRegistry(enabled=False)
        with pytest.raises(RuntimeError, match="FAULT_INJECTION_ENABLED"):
            registry.activate([COMPONENT_RERANKER])
        assert registry.active() == []

    def test_unknown_component_names_the_supported_ones(self):
        registry = FaultRegistry(enabled=True)
        with pytest.raises(ValueError) as err:
            registry.activate(["qdrant"])
        for component in COMPONENTS:
            assert component in str(err.value), "错误信息要给出可选值，调用方才好自纠"

    def test_activate_and_clear(self):
        registry = FaultRegistry(enabled=True)
        assert registry.activate([COMPONENT_RERANKER, COMPONENT_EMBEDDING]) == [
            COMPONENT_EMBEDDING, COMPONENT_RERANKER,
        ]
        registry.check(COMPONENT_VECTOR_INDEX)  # 未激活的组件不受影响
        with pytest.raises(InjectedFault):
            registry.check(COMPONENT_RERANKER)
        registry.clear()
        assert registry.active() == []
        registry.check(COMPONENT_RERANKER)

    def test_fault_message_is_unmistakable(self):
        """异常信息必须一眼看出是注入的。

        否则有人忘了关注入，会对着一个不存在的线上故障排查一下午——
        这类"自己制造的幻觉故障"比故障本身更贵。
        """
        registry = FaultRegistry(enabled=True)
        registry.activate([COMPONENT_EMBEDDING])
        with pytest.raises(InjectedFault, match="故障注入"):
            registry.check(COMPONENT_EMBEDDING)

    def test_describe_is_false_when_disabled(self):
        """`/health` 与报告配置行原样渲染它：未启用时是 false，启用时列出组件。"""
        assert FaultRegistry(enabled=False).describe() is False
        registry = FaultRegistry(enabled=True)
        assert registry.describe() == {"enabled": True, "active": []}
        registry.activate([COMPONENT_RERANKER])
        assert registry.describe() == {"enabled": True, "active": [COMPONENT_RERANKER]}


class TestDecorators:
    async def test_embedding_passes_through_when_inactive(self):
        registry = FaultRegistry(enabled=True)
        _, embedder, _, _ = install_fault_injection(
            _Settings(True), _FakeEmbedder(), _FakeIndex(), _FakeReranker(), registry=registry,
        )
        assert await embedder.embed("x") == [0.1, 0.2]
        assert await embedder.embed_batch(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]

    async def test_embedding_fails_when_active(self):
        registry = FaultRegistry(enabled=True)
        registry.activate([COMPONENT_EMBEDDING])
        _, embedder, _, _ = install_fault_injection(
            _Settings(True), _FakeEmbedder(), _FakeIndex(), _FakeReranker(), registry=registry,
        )
        with pytest.raises(InjectedFault):
            await embedder.embed("x")
        with pytest.raises(InjectedFault):
            await embedder.embed_batch(["x"])

    async def test_reranker_fails_when_active(self):
        registry = FaultRegistry(enabled=True)
        registry.activate([COMPONENT_RERANKER])
        _, _, _, reranker = install_fault_injection(
            _Settings(True), _FakeEmbedder(), _FakeIndex(), _FakeReranker(), registry=registry,
        )
        with pytest.raises(InjectedFault):
            await reranker.rerank("q", ["d"])

    async def test_vector_index_faults_only_on_search(self):
        """只注入检索，不注入建索引。

        `bootstrap_product_index` 在服务启动时跑，注入它等于让服务起不来，
        而要检验的是"运行时向量库不可达"，两件事。
        """
        registry = FaultRegistry(enabled=True)
        registry.activate([COMPONENT_VECTOR_INDEX])
        inner = _FakeIndex()
        _, _, index, _ = install_fault_injection(
            _Settings(True), _FakeEmbedder(), inner, _FakeReranker(), registry=registry,
        )
        await index.ensure_ready(2)
        await index.upsert_products([], [])
        assert inner.upserts == 1
        with pytest.raises(InjectedFault):
            await index.search([0.1], top_n=3)

    async def test_vector_index_decorator_keeps_close(self):
        """容器关停时要能 close：装饰器漏了它会在 shutdown 抛 AttributeError。"""
        registry = FaultRegistry(enabled=True)
        inner = _FakeIndex()
        _, _, index, _ = install_fault_injection(
            _Settings(True), _FakeEmbedder(), inner, _FakeReranker(), registry=registry,
        )
        await index.close()
        assert inner.closed is True

    async def test_none_reranker_stays_none(self):
        """未配置 RERANKER_BASE_URL 时组装根传进来的是 None，不能被包成一个对象——
        `catalog_search` 用 `is None` 判断精排是否可用。"""
        registry = FaultRegistry(enabled=True)
        _, _, _, reranker = install_fault_injection(
            _Settings(True), _FakeEmbedder(), _FakeIndex(), None, registry=registry,
        )
        assert reranker is None


class TestInstallIsOffByDefault:
    def test_disabled_settings_returns_the_originals_untouched(self):
        """默认关时**连装饰器都不构造**：生产进程里不该存在这条代码路径。"""
        embedder, index, reranker = _FakeEmbedder(), _FakeIndex(), _FakeReranker()
        registry, out_embedder, out_index, out_reranker = install_fault_injection(
            _Settings(False), embedder, index, reranker,
        )
        assert out_embedder is embedder
        assert out_index is index
        assert out_reranker is reranker
        assert registry.enabled is False
        assert registry.describe() is False


class TestServerWiring:
    """端点与上报的接线判据。

    七期教训：写完了不接线，外观与"故意不做"完全一样，没有任何告警。
    故障注入尤其如此——它平时是关的，接错了也不会有人发现，
    直到某天写故障用例时才发现注入根本没生效（而那时用例已经绿了一轮）。
    """

    def _build_app(self, monkeypatch, enabled: str):
        import importlib

        monkeypatch.setenv("FAULT_INJECTION_ENABLED", enabled)
        from app.presentation import server

        importlib.reload(server)
        return server.build_app(), server

    def _paths(self, app) -> set:
        return {getattr(route, "path", "") for route in app.routes}

    def test_debug_endpoint_absent_by_default(self, monkeypatch):
        """默认关时端点**不注册**：生产进程里不该存在这条路径。"""
        app, _ = self._build_app(monkeypatch, "0")
        assert "/debug/faults" not in self._paths(app)

    def test_debug_endpoint_present_when_enabled(self, monkeypatch):
        app, _ = self._build_app(monkeypatch, "1")
        assert "/debug/faults" in self._paths(app)

    def test_health_reports_fault_injection(self, monkeypatch):
        """`/health` 必须上报注入状态。

        踩坑 32 的同一条：分数变了要能归因到配置。开着精排故障跑出来的报告，
        配置行不写这件事的话，读的人只会看到"精排档分数崩了"然后去改检索参数。
        """
        import inspect

        _, server = self._build_app(monkeypatch, "0")
        source = inspect.getsource(server)
        assert '"fault_injection": c.faults.describe()' in source

    def test_run_identity_line_shows_injected_faults(self):
        from app.application.harness.run_identity import describe_run

        line = describe_run({
            "model": "m", "prompt_fingerprint": "abc", "semantic_cache": False,
            "retrieval": {"reranker": True, "lexical_index": True, "lexical_gate": 4.0},
            "fault_injection": {"enabled": True, "active": ["reranker"]},
        })
        assert "故障注入" in line and "reranker" in line

    def test_run_identity_line_stays_quiet_when_not_injected(self):
        """没注入时不该在配置行里占位——每多一个恒定字段，真正变的那个就更难被看见。"""
        from app.application.harness.run_identity import describe_run

        line = describe_run({"model": "m", "fault_injection": False})
        assert "故障注入" not in line


class TestDegradationChainActuallyMoves:
    """注入之后**降级链真的走到了预期档位**。

    这是这套基建唯一重要的判据：前面的测试证明了"装饰器会抛异常"，
    但抛异常不等于降级会按预期发生——`catalog_search` 里哪一档捕获它、
    捕获后 `recall_strategy` 标成什么，是另一回事。
    钉住这一条，故障用例才能拿降级档位当判据。
    """

    async def _usecase(self, registry, *, with_lexical=True):
        from app.application.usecases.catalog_search import CatalogSearchUseCase
        from app.infrastructure.persistence.in_memory_repositories import (
            InMemoryProductRepository,
        )
        from app.infrastructure.retrieval.bm25_index import Bm25LexicalIndex

        repo = InMemoryProductRepository()
        _, embedder, index, reranker = install_fault_injection(
            _Settings(True), _FakeEmbedder(), _FakeIndex(), _FakeReranker(), registry=registry,
        )
        lexical = None
        if with_lexical:
            lexical = Bm25LexicalIndex()
            lexical.index(await repo.list_all())
        return CatalogSearchUseCase(
            repo, embedder=embedder, vector_index=index,
            reranker=reranker, lexical_index=lexical,
        )

    async def _run(self, usecase):
        from app.domain.catalog.product_search_spec import ProductSearchSpec

        return await usecase.execute(ProductSearchSpec(normalized_query="露营灯 便携"))

    async def test_reranker_fault_drops_to_first_stage_ranking(self):
        registry = FaultRegistry(enabled=True)
        registry.activate([COMPONENT_RERANKER])
        result = await self._run(await self._usecase(registry))
        assert result["rerank_applied"] is False
        assert "rerank" not in result["recall_strategy"], (
            "recall_strategy 是给模型看的事实，精排没跑就不能标成精排档"
        )

    async def test_embedding_fault_falls_back_to_lexical_path(self):
        """向量路整条挂掉时退到字面路，而不是整轮失败。"""
        registry = FaultRegistry(enabled=True)
        registry.activate([COMPONENT_EMBEDDING])
        result = await self._run(await self._usecase(registry))
        assert result["hits"], "字面路是纯本地的，永远可用——不该返回空"
        assert result["recall_strategy"].startswith("bm25") or \
            result["recall_strategy"] == "keyword_2gram"

    async def test_vector_index_fault_also_degrades(self):
        registry = FaultRegistry(enabled=True)
        registry.activate([COMPONENT_VECTOR_INDEX])
        result = await self._run(await self._usecase(registry))
        assert result["hits"]
        assert "embedding" not in result["recall_strategy"]

    async def test_everything_down_still_answers_from_keyword_recall(self):
        """全挂时兜底到 keyword_2gram：这条是"所有召回基建都挂了"的最后一档。"""
        registry = FaultRegistry(enabled=True)
        registry.activate(list(COMPONENTS))
        result = await self._run(await self._usecase(registry, with_lexical=False))
        assert result["recall_strategy"] == "keyword_2gram"
        assert result["hits"]

    async def test_no_fault_keeps_the_top_tier(self):
        """对照组：不注入时仍走精排档——否则上面几条测的可能是别的原因。"""
        result = await self._run(await self._usecase(FaultRegistry(enabled=True)))
        assert result["rerank_applied"] is True
