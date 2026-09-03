# -*- coding: utf-8 -*-
"""faults —— 评测态的检索故障注入（默认全关）

要解决的问题：`catalog_search` 的降级链（hybrid_rerank → … → keyword_2gram）
每一档都有代码、有单测，但**没有一条端到端用例检验过"降级之后 Agent 怎么跟买家说"**。
现有的 `tool-error-honesty` 覆盖的是参数错误（目的国不支持），
而线上真正会发生的是服务不可达：精排挂了、向量库连不上、embedding 网关超时。

实现上刻意选了**装饰器**：降级链本来就由真实异常触发，所以注入不需要在业务代码里
加任何分支——在端口边界包一层、抛异常就行。三个好处：

    1. 生产代码路径一行不改，不多一次判断，不多一条 if；
    2. 风险收敛在组装根一处——没开开关时装饰器根本不会被构造；
    3. 注入的是"真实异常"而不是"模拟的降级状态"，走的就是线上那条路。

**双重开关**：
    `FAULT_INJECTION_ENABLED`（默认 0）决定装饰器与 `/debug/faults` 端点是否存在；
    运行时 `POST /debug/faults` 在已启用的进程里选择当前注入哪些组件。
只做环境变量的话，31 条用例里那一两条故障用例就要单独重启一次服务，
结果是这类用例永远不会被跑到——跑不动的判据等于没有判据。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from app.domain.catalog.ports.retrieval_ports import (
    EmbeddingClient,
    ProductVectorIndex,
    Reranker,
    VectorHit,
)
from app.domain.catalog.product import Product

COMPONENT_EMBEDDING = "embedding"
COMPONENT_RERANKER = "reranker"
COMPONENT_VECTOR_INDEX = "vector_index"

COMPONENTS = (COMPONENT_EMBEDDING, COMPONENT_RERANKER, COMPONENT_VECTOR_INDEX)


class InjectedFault(RuntimeError):
    """注入的故障。

    信息里必须带"故障注入"四个字：有人忘了关注入时，
    会对着一个不存在的线上故障排查一下午——自己制造的幻觉故障比故障本身贵。
    """


@dataclass
class FaultRegistry:
    """当前进程里激活了哪些故障。

    `enabled` 由环境变量定；未启用时 `activate` 直接报错而不是静默成功——
    评测据此判定"用例声明了 faults 但服务没开注入"，把那条用例判 ERROR。
    否则它会在一切正常的情况下跑完然后 PASS，判据成了绿色装饰。
    """

    enabled: bool = False
    _active: set[str] = field(default_factory=set)

    def activate(self, components: Iterable[str]) -> list[str]:
        if not self.enabled:
            raise RuntimeError(
                "本进程未启用故障注入，无法激活：请以 FAULT_INJECTION_ENABLED=1 重启服务。"
                "（默认关是有意的——生产进程里不该存在这条代码路径）",
            )
        requested = [str(item) for item in components]
        unknown = [item for item in requested if item not in COMPONENTS]
        if unknown:
            raise ValueError(
                f"未知的故障组件 {unknown}，可选：{list(COMPONENTS)}",
            )
        self._active = set(requested)
        return self.active()

    def clear(self) -> None:
        self._active = set()

    def active(self) -> list[str]:
        return sorted(self._active)

    def check(self, component: str) -> None:
        if component in self._active:
            raise InjectedFault(
                f"[故障注入] {component} 被人为置为不可用（FAULT_INJECTION_ENABLED=1 生效中）。"
                f"这不是真实故障；关闭方式：POST /debug/faults {{\"components\": []}}",
            )

    def describe(self) -> Any:
        """给 `/health` 与报告配置行用：未启用时是 false，启用时列出当前激活的组件。

        必须上报的理由同踩坑 32：**分数变了要能归因到配置**。
        一个开着精排故障的进程跑出来的报告，配置行不写这件事的话，
        读报告的人只会看到"精排档分数崩了"，然后去改检索参数。
        """
        if not self.enabled:
            return False
        return {"enabled": True, "active": self.active()}


@dataclass
class _FaultyEmbeddingClient(EmbeddingClient):
    inner: EmbeddingClient
    registry: FaultRegistry

    async def embed(self, text: str) -> list[float]:
        self.registry.check(COMPONENT_EMBEDDING)
        return await self.inner.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.registry.check(COMPONENT_EMBEDDING)
        return await self.inner.embed_batch(texts)


@dataclass
class _FaultyReranker(Reranker):
    inner: Reranker
    registry: FaultRegistry

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.registry.check(COMPONENT_RERANKER)
        return await self.inner.rerank(query, documents)


@dataclass
class _FaultyVectorIndex(ProductVectorIndex):
    """只注入 `search`。

    `bootstrap_product_index` 在服务启动时建索引，注入它等于让服务起不来；
    要检验的是"运行时向量库不可达"，两件事不能混。
    """

    inner: Any
    registry: FaultRegistry

    async def ensure_ready(self, vector_dim: int) -> None:
        await self.inner.ensure_ready(vector_dim)

    async def upsert_products(
        self, products: list[Product], embeddings: list[list[float]],
    ) -> None:
        await self.inner.upsert_products(products, embeddings)

    async def search(self, embedding: list[float], top_n: int) -> list[VectorHit]:
        self.registry.check(COMPONENT_VECTOR_INDEX)
        return await self.inner.search(embedding, top_n)

    async def close(self) -> None:
        # 容器关停要调它；漏了会在 shutdown 抛 AttributeError
        await self.inner.close()


def install_fault_injection(
    settings: Any,
    embedder: Any,
    vector_index: Any,
    reranker: Optional[Any],
    registry: Optional[FaultRegistry] = None,
) -> tuple[FaultRegistry, Any, Any, Optional[Any]]:
    """按开关决定是否给三个检索端口包上故障注入装饰器。

    关掉时**原样返回**传入的对象：生产进程里连装饰器都不构造。
    reranker 为 None（未配置 RERANKER_BASE_URL）时保持 None——
    `catalog_search` 用 `is None` 判断精排是否可用，包成对象会让它以为精排在线。
    """
    enabled = bool(getattr(settings, "fault_injection_enabled", False))
    registry = registry or FaultRegistry(enabled=enabled)
    if not enabled:
        return registry, embedder, vector_index, reranker
    return (
        registry,
        _FaultyEmbeddingClient(embedder, registry),
        _FaultyVectorIndex(vector_index, registry),
        _FaultyReranker(reranker, registry) if reranker is not None else None,
    )
