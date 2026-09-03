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
"""
from __future__ import annotations

from typing import Any

_UNKNOWN = "未知"


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
        f"｜精排 {_switch(retrieval.get('reranker'))}"
        f"｜字面索引 {_switch(retrieval.get('lexical_index'))}"
        f"｜字面门限 {_text(retrieval.get('lexical_gate'))}"
        f"｜语义缓存 {_switch(health.get('semantic_cache'))}"
    )
