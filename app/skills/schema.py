# -*- coding: utf-8 -*-
"""Skill 打包单元 schema（#14 任务 C，C2）。

Skill = 提示词片段 + 工具子集 + 判据的打包单元，按任务阶段渐进注入
（交接文档「五之一」任务 C 范围）。schema 是纯数据 + 构造期校验：
一切会在运行期静默失效的形状（空壳 skill、空白片段、未知阶段）在
构造期就拒——静默存在的坏配置等于骗自己。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Stage(str, Enum):
    """任务阶段。COMMON 不参与路由——它对所有阶段生效。"""

    COMMON = "common"
    SEARCH = "search"
    TRADE = "trade"
    MEMORY = "memory"


def _ensure_tuple_str(value, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError(f"{field_name} 必须是字符串序列，收到单个字符串 {value!r}")
    try:
        items = tuple(value)
    except TypeError:
        raise ValueError(f"{field_name} 必须是字符串序列，收到 {type(value).__name__}")
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} 含空白或非字符串项：{item!r}")
    return items


@dataclass(frozen=True)
class SkillSpec:
    """一个 skill 的完整声明。字段不可变——注册后改字段等于绕过校验。"""

    skill_id: str
    stages: tuple[Stage, ...]
    tools: tuple[str, ...] = field(default=())
    prompt_fragments: tuple[str, ...] = field(default=())
    criteria: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not isinstance(self.skill_id, str) or not self.skill_id.strip():
            raise ValueError(f"skill_id 不能为空：{self.skill_id!r}")
        stages = tuple(self.stages)
        if not stages:
            raise ValueError(f"skill {self.skill_id} 未声明任何阶段")
        for stage in stages:
            if not isinstance(stage, Stage):
                raise ValueError(
                    f"skill {self.skill_id} 的阶段 {stage!r} 不是 Stage 枚举值"
                )
        object.__setattr__(self, "stages", stages)
        tools = _ensure_tuple_str(self.tools, f"skill {self.skill_id} 的 tools")
        fragments = _ensure_tuple_str(
            self.prompt_fragments, f"skill {self.skill_id} 的提示词片段"
        )
        criteria = _ensure_tuple_str(self.criteria, f"skill {self.skill_id} 的判据")
        if not (tools or fragments or criteria):
            raise ValueError(
                f"skill {self.skill_id} 是空壳（无片段/无工具/无判据）——"
                "注册了也不会有任何效果，构造期拒绝"
            )
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "prompt_fragments", fragments)
        object.__setattr__(self, "criteria", criteria)
