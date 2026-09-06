# -*- coding: utf-8 -*-
"""Skill 定义装载（#14 任务 C，C2）：app/skills/definitions.yml → SkillSpec 列表。

只做结构与字面校验（阶段枚举、字符串非空、id 唯一）；工具名对运行时
全集的核对在 SkillRegistry.register（C3 接线时传真集合）。
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .schema import Stage, SkillSpec

DEFINITIONS_PATH = Path(__file__).resolve().parent / "definitions.yml"


def load_skill_definitions(path: Path | None = None) -> list[SkillSpec]:
    source = path or DEFINITIONS_PATH
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    # 顶层必须是映射：list 等畸形结构不走 getattr 会好看些，但裸 traceback
    # 不合"降级路径必须留因"（硬纪律五.4）——转成带文件名的 ValueError。
    if not isinstance(raw, dict):
        raise ValueError(f"{source} 顶层必须是 skills 映射，收到 {type(raw).__name__}")
    entries = raw.get("skills") or []
    if not entries:
        raise ValueError(f"{source} 没有任何 skill 定义")
    skills: list[SkillSpec] = []
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{source} 的 skill 条目必须是映射：{entry!r}")
        skill_id = entry.get("id")
        if not skill_id:
            raise ValueError(f"{source} 存在没有 id 的 skill 条目：{entry!r}")
        skill_id = str(skill_id)
        if skill_id in seen_ids:
            raise ValueError(f"{source} 的 skill_id 重复：{skill_id}")
        seen_ids.add(skill_id)
        try:
            stages = tuple(Stage(stage) for stage in (entry.get("stages") or []))
        except ValueError as err:
            raise ValueError(
                f"skill {skill_id} 的 stages 含未知阶段（合法值："
                f"{[s.value for s in Stage]}）：{err}"
            )
        # 原始值直接交给 SkillSpec（只做 None→() 的缺省），不预转换 tuple：
        # schema 的字符串序列护栏（防裸字符串被拆成单字片段、防映射以 key
        # 静默通过）必须在构造期开火，loader 抢先转换等于把它拆了（烧前评审 A1）。
        skills.append(SkillSpec(
            skill_id=skill_id,
            stages=stages,
            tools=entry.get("tools") if entry.get("tools") is not None else (),
            prompt_fragments=entry.get("prompt_fragments")
            if entry.get("prompt_fragments") is not None else (),
            criteria=entry.get("criteria") if entry.get("criteria") is not None else (),
        ))
    return skills
