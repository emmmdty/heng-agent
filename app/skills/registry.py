# -*- coding: utf-8 -*-
"""Skill 注册表（#14 任务 C，C2）：按阶段查询 + 渐进注入的确定性渲染。

纯逻辑、零 I/O：工具全集由构造方注入（C3 接线时传运行时 Toolkit 的
真实工具名集合），注册期对未知工具名报错留名——工具名拼错 = 静默缺工具
= 行为回退，不能等 C4 护栏轮的"工具调用率下降"才炸。

渲染契约：同一注册状态下 render_prompt(stage) 逐字节一致（注册顺序即
渲染顺序）——产物会进 system prompt，输出抖动会破坏 prompt 缓存与指纹。
"""
from __future__ import annotations

from .schema import Stage, SkillSpec


class SkillRegistry:
    def __init__(self, tool_universe: set[str] | frozenset[str] = frozenset()) -> None:
        self._tool_universe = frozenset(tool_universe)
        self._skills: list[SkillSpec] = []
        self._by_id: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> None:
        if spec.skill_id in self._by_id:
            raise ValueError(f"skill_id 重复注册：{spec.skill_id}")
        unknown = [tool for tool in spec.tools if tool not in self._tool_universe]
        if unknown:
            raise ValueError(
                f"skill {spec.skill_id} 声明了工具全集里不存在的工具：{'、'.join(unknown)}"
                "——工具名拼错会在运行期静默缺工具，注册期拦下"
            )
        self._skills.append(spec)
        self._by_id[spec.skill_id] = spec

    def get(self, skill_id: str) -> SkillSpec | None:
        """按 id 读回（C3 接线时按名核对/单独渲染用）。"""
        return self._by_id.get(skill_id)

    def for_stage(self, stage: Stage) -> list[SkillSpec]:
        """该阶段生效的 skill（含 COMMON），按注册顺序。"""
        return [
            s for s in self._skills if stage in s.stages or Stage.COMMON in s.stages
        ]

    def render_prompt(self, stage: Stage) -> str:
        """该阶段全部提示词片段的确定性拼接；无内容返回空串（调用方整段省略）。"""
        parts: list[str] = []
        for spec in self.for_stage(stage):
            parts.extend(spec.prompt_fragments)
        return "\n".join(part for part in parts if part)

    def render_criteria(self, stage: Stage) -> list[str]:
        """该阶段生效的判据条目（去重、保序）。"""
        seen: set[str] = set()
        ordered: list[str] = []
        for spec in self.for_stage(stage):
            for item in spec.criteria:
                if item not in seen:
                    seen.add(item)
                    ordered.append(item)
        return ordered

    def tool_subset(self, stage: Stage) -> tuple[str, ...]:
        """该阶段允许的工具并集（去重、按首次出现排序）——渐进加载的另一半。"""
        subset: list[str] = []
        seen: set[str] = set()
        for spec in self.for_stage(stage):
            for tool in spec.tools:
                if tool not in seen:
                    seen.add(tool)
                    subset.append(tool)
        return tuple(subset)
