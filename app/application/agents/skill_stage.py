# -*- coding: utf-8 -*-
"""Skill 阶段注入中间件（#14 任务 C，C3）：agent 中间件协议的 on_system_prompt。

agentscope 每次模型调用都会经 `_get_system_prompt()` 重读 system prompt 并
顺序过中间件的 on_system_prompt 变换器——orchestrator 每轮经控制器刷新激活
阶段，这里做**替换式**变换（身份头 + common + 激活阶段片段）。

保底：注册表为空（skill 体系没装载成功）时原样返回原 prompt——
退回单块基线，绝不发一个空 system prompt 出去（脏输出不静默塌缩的同族纪律）。
"""
from __future__ import annotations

from agentscope.middleware import MiddlewareBase


class SkillStagePromptMiddleware(MiddlewareBase):
    def __init__(self, controller) -> None:
        self._controller = controller

    async def on_system_prompt(self, agent, current_prompt: str) -> str:
        if not self._controller.has_skills():
            return current_prompt
        return self._controller.render_system_prompt()
