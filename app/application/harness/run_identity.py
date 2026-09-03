# -*- coding: utf-8 -*-
"""run_identity —— 把一次跑测的配置渲染成报告里的一行

要防的问题：**一个读数说不清它是哪套配置跑出来的。**

设计演进记录里写了"评测分数与所用模型绑定，换模型必须重跑并在报告里标注"，
但报告本身不记模型，全靠跑的人当时记得。同样不记的还有提示词版本
（改一句 prompt 分数就会动）、精排是否可用、字面门限取值。
过两周回头看一份报告，只剩一个数字和一堆无法归因的差异——
更糟的是"分数掉了"时，第一反应会去改 Agent，而真实原因可能是精排服务挂了。

所以配置由**被测服务自己报**（`GET /health`），评测脚本原样抄进报告。
脚本本来就为了拦语义缓存要读一次 /health，顺路留下整份配置，零额外成本。

缺字段一律写"未知"而不是省略：少一行会被读成"这项没启用"，比"未知"更误导。

**代码新鲜度（九期加）**：上面这些字段都答不了"这个服务跑的是不是我刚改的代码"。
九期实测踩到：uvicorn 16:43:06 起，修复 16:49:20 落地，进程再没重启，
之后两条定向回归打的都是装着旧代码的服务，而 /health 报的配置行一字不差、
408 单测也全绿（单测读磁盘、评测打进程，两者可以同时成立）。
判据取 **源码 mtime vs 进程启动时刻**而不是 git sha：sha 看不见未提交的改动，
而踩坑当时正是未提交状态。
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

_UNKNOWN = "未知"

# 模块在服务启动早期被导入，用导入时刻近似进程启动时刻。
# 晚于这一刻的源码改动，进程内存里一定没有。
_PROCESS_START = time.time()
_SOURCE_ROOT = Path(__file__).resolve().parents[2]  # 仓库里的 app/
_MAX_LISTED = 5


def _stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%m-%d %H:%M:%S")


def code_identity(root: Path | None = None, started_at: float | None = None) -> dict:
    """比对源码 mtime 与进程启动时刻，报出这个进程跑的代码是不是最新的。

    `__pycache__` 要排除：它在进程跑起来之后才写入是常态，据此判过期会天天误报。
    """
    root = Path(root) if root is not None else _SOURCE_ROOT
    started_at = _PROCESS_START if started_at is None else started_at

    newest = 0.0
    stale_files: list[str] = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        mtime = path.stat().st_mtime
        newest = max(newest, mtime)
        if mtime > started_at:
            stale_files.append(str(path.relative_to(root)))

    stale_files.sort()
    return {
        "started_at": _stamp(started_at),
        "source_mtime": _stamp(newest) if newest else _UNKNOWN,
        "stale": bool(stale_files),
        "stale_files": stale_files[:_MAX_LISTED],
        "stale_count": len(stale_files),
    }


def _text(value: Any) -> str:
    if value is None or value == "":
        return _UNKNOWN
    return str(value)


def _switch(value: Any) -> str:
    if value is None:
        return _UNKNOWN
    return "开" if value else "关"


def describe_run(health: dict, judge_model: str = "") -> str:
    """渲染成一行：被测模型 / 评审模型 / 提示词版本 / 检索配置。"""
    retrieval = health.get("retrieval") or {}
    return (
        f"被测模型 {_text(health.get('model'))}"
        f"｜评审模型 {_text(judge_model)}"
        f"｜提示词 {_text(health.get('prompt_fingerprint'))}"
        f"｜精排 {_probed_switch(retrieval.get('reranker'), _probe_of(retrieval, 'reranker'))}"
        f"{_dead_vector_path(retrieval)}"
        f"｜字面索引 {_switch(retrieval.get('lexical_index'))}"
        f"｜字面门限 {_text(retrieval.get('lexical_gate'))}"
        f"｜语义缓存 {_switch(health.get('semantic_cache'))}"
        f"｜代码 {_describe_code(health.get('code'))}"
        f"{_describe_faults(health.get('fault_injection'))}"
    )


def _describe_faults(faults: Any) -> str:
    """故障注入只在**真的注入了**的时候才占一格。

    每多一个恒定不变的字段，真正变了的那个就更难被看见；
    而一旦注入生效，它就是这一轮读数最该被归因到的东西。
    """
    if not isinstance(faults, dict) or not faults.get("active"):
        return ""
    return f"｜**故障注入 {'/'.join(faults['active'])}**"


def _probe_of(retrieval: Any, component: str) -> Any:
    """取深度探活结果；没探活或形状不对时返回 None（配置行任何情况下都得渲染得出来）。"""
    probe = retrieval.get("probe") if isinstance(retrieval, dict) else None
    return probe.get(component) if isinstance(probe, dict) else None


def _probed_switch(configured: Any, probe: Any) -> str:
    """配置 + 实测两截。

    十四期实测踩到的：配置行写着"精排 开"，而那一轮精排是 502、一次都没跑过。
    **没探活时不加任何后缀**——多一个"未知"标记只会让每行都变长，
    真正要跳出来的是"配了但没生效"那一种。
    """
    rendered = _switch(configured)
    if configured and isinstance(probe, str) and probe.startswith("error"):
        return f"{rendered}(实测不可达)"
    return rendered


def _dead_vector_path(retrieval: Any) -> str:
    """向量路挂掉要单独说：它解释了为什么召回档位掉到 bm25_only。

    配置行里原本没有这一格——向量路一直被当成"配了就有"，
    而十四期那一轮它整条不可用，报告上却看不出来。
    """
    probe = _probe_of(retrieval, "embedding")
    if isinstance(probe, str) and probe.startswith("error"):
        return "｜**向量路 实测不可达**"
    return ""


def _describe_code(code: Any) -> str:
    """代码新鲜度渲进同一行——报告开头是"分数变了先看它"的地方，过期要在这刺眼。"""
    if not isinstance(code, dict) or not code:
        return _UNKNOWN
    started = _text(code.get("started_at"))
    if not code.get("stale"):
        return f"新鲜(服务启动于 {started})"
    listed = "、".join(code.get("stale_files") or []) or "若干文件"
    count = code.get("stale_count") or len(code.get("stale_files") or [])
    more = f" 等 {count} 个" if count > len(code.get("stale_files") or []) else ""
    return (
        f"⚠️已过期(服务启动于 {started}，但 {listed}{more} 在那之后被改过"
        f"，最新 {_text(code.get('source_mtime'))})"
    )
