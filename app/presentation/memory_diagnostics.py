# -*- coding: utf-8 -*-
"""进程内存诊断（soak RSS 分析的观测端点后端）

soak 首轮给出负结果（拐点后 RSS 仍在增长）且 aclose 修复只削掉 20%，
对象计数恒定、线程/FD 正常——**增长源不在 Python 活对象层**。
要往下挖只有 tracemalloc：按"分配点（文件:行号）"对比两个时刻的分配，
直接指认增长代码。

约束：
    - tracemalloc 必须在进程早期启用（PYTHONTRACEMALLOC=<深度> 环境变量），
      本模块只负责 take/diff，不悄悄打开追踪（运行中打开丢历史）；
    - 输出为诊断线索不是判据（经验 9：诊断类输出宁可少说不可说错），
      端点挂 /debug 仅供单机诊断，无鉴权环境不该开。
"""
from __future__ import annotations

import gc
import tracemalloc
from collections import Counter
from typing import Any

# gc 对象计数的 top 上限：数太多没意义，增长大户集中在少数类型
_TOP_TYPES = 15


def tracemalloc_enabled() -> bool:
    return tracemalloc.is_tracing()


def take_snapshot() -> dict[str, Any]:
    """取一个内存快照（tracemalloc）+ gc 对象计数 top，存入模块级槽位。

    返回值里包含对象计数（始终可用）与 tracemalloc 可用性标记；
    diff 需要 take 两次。
    """
    global _LAST_SNAPSHOT
    gc.collect()
    counts = Counter(type(o).__name__ for o in gc.get_objects())
    payload: dict[str, Any] = {
        "tracemalloc_enabled": tracemalloc_enabled(),
        "top_types": [
            {"type": name, "count": n} for name, n in counts.most_common(_TOP_TYPES)
        ],
    }
    if tracemalloc_enabled():
        _LAST_SNAPSHOT = tracemalloc.take_snapshot()
        payload["snapshot_taken"] = True
    else:
        payload["snapshot_taken"] = False
        payload["hint"] = "以 PYTHONTRACEMALLOC=<深度> 启动进程后 tracemalloc 才有数据"
    return payload


def diff_snapshot(top: int = 15) -> dict[str, Any]:
    """与上一次 take 的快照对比，返回增长最大的分配点。

    没有上次快照时报错而不是给空 diff——空的 top 会被人读成"没有增长"。
    """
    global _LAST_SNAPSHOT
    if not tracemalloc_enabled():
        return {"tracemalloc_enabled": False, "hint": "需以 PYTHONTRACEMALLOC=<深度> 启动"}
    if _LAST_SNAPSHOT is None:
        raise ValueError("没有上次快照：先调 take_snapshot（GET /debug/memory?snapshot=1）")
    current = tracemalloc.take_snapshot()
    diff = current.compare_to(_LAST_SNAPSHOT, "lineno")
    _LAST_SNAPSHOT = current
    return {
        "tracemalloc_enabled": True,
        "top_growths": [
            {
                "where": str(item.traceback),
                "size_diff_kb": round(item.size_diff / 1024, 1),
                "count_diff": item.count_diff,
            }
            for item in diff[:top]
        ],
    }


_LAST_SNAPSHOT: Any = None
