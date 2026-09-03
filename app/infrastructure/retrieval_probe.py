# -*- coding: utf-8 -*-
"""retrieval_probe —— 检索依赖的深度探活

回答的是配置行原先答不出的那个问题：**这个地址通不通**。

十四期小样本实测：配置行写着"精排 开"，同一轮轨迹里 `recall_strategy` 是
`bm25_only`——两条隧道都是 502，精排一次都没跑过。配置行不是写错了，
是它读的是 `RERANKER_BASE_URL` 配没配，而不是那个地址是否可达。
与踩坑 32（服务跑着旧代码）同构：**分数标着一个并不成立的配置**。

**只在有人明确要读数时探活**（`/health?deep=1`、`make health`、评测开跑前），
默认 `/health` 一次外部调用都不发——它同时是容器存活探针，
每 10 秒打一次外部服务既浪费，又会把探针本身变得不稳定
（外部服务抖一下，容器被判死）。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# 探活超时。短是刻意的：这是"给人看读数"用的，不是关键路径，
# 宁可报不可达也不要让 /health 卡住。
_PROBE_TIMEOUT_SECONDS = 3.0

STATUS_OK = "ok"
STATUS_DISABLED = "disabled"


def _error(err: BaseException) -> str:
    """错误原文要留：502 与连接超时的排查方向完全不同。"""
    return f"error: {type(err).__name__}: {str(err)[:120]}"


async def _probe_embedding(settings: Any, client: Any) -> str:
    base = (getattr(settings, "embedding_base_url", "") or "").rstrip("/")
    if not base:
        return STATUS_DISABLED
    try:
        response = await client.post(
            f"{base}/embeddings",
            headers={"Authorization": f"Bearer {getattr(settings, 'embedding_api_key', '')}"},
            json={"model": getattr(settings, "embedding_model", ""), "input": "ping"},
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except BaseException as err:  # noqa: BLE001 —— 探活自己绝不能把 /health 打挂
        return _error(err)
    return STATUS_OK


async def _probe_reranker(settings: Any, client: Any) -> str:
    base = (getattr(settings, "reranker_base_url", "") or "").rstrip("/")
    if not base:
        return STATUS_DISABLED
    try:
        response = await client.get(f"{base}/health", timeout=_PROBE_TIMEOUT_SECONDS)
        response.raise_for_status()
    except BaseException as err:  # noqa: BLE001
        return _error(err)
    return STATUS_OK


async def probe_retrieval(settings: Any, client: Optional[Any] = None) -> dict:
    """探活两个外部检索依赖，返回 {组件: "ok" | "disabled" | "error: ..."}。

    `disabled`（没配）与 `error`（配了但不可达）**必须是两个值**：
    含义相反，混成一个会让配置行同时失去两种信息。
    """
    if client is not None:
        return {
            "embedding": await _probe_embedding(settings, client),
            "reranker": await _probe_reranker(settings, client),
        }
    async with httpx.AsyncClient() as owned:
        return {
            "embedding": await _probe_embedding(settings, owned),
            "reranker": await _probe_reranker(settings, owned),
        }
