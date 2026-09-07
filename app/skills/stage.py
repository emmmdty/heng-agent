# -*- coding: utf-8 -*-
"""Skill 阶段控制器（#14 任务 C，C3）：每轮的阶段状态 + system prompt 拼装。

替换式（设计笔记对账表）：flag 开时 system prompt = 身份头 + common +
激活阶段片段——heng.yml 单块不再发送，文件本身一字不动留作 skill-off 基线
（指纹 a0915fac = B2 resume 断点依赖的基线，C3 不许弄丢）。

保守默认：控制器初始态 = 全交付阶段（等价于"没有路由信息时什么都带上"）。
"""
from __future__ import annotations

import contextvars

from .registry import SkillRegistry
from .router import StageRouter
from .schema import Stage

SKILL_BASE_HEADER = (
    "你是「衡 · Heng」跨境电商超级框总管 (CommerceConcierge)，负责听懂买家自然语言诉求，"
    "自己动手完成购物任务；只有确有必要时才派发专家子代理。"
)

# 保守默认：没有任何路由信息时全交付阶段（等价于"什么都带上"）。
_ALL_STAGES: frozenset[Stage] = frozenset({Stage.SEARCH, Stage.TRADE, Stage.MEMORY})


class SkillStageController:
    def __init__(self, router: StageRouter | None = None,
                 registry: SkillRegistry | None = None) -> None:
        self._router = router or StageRouter()
        # 预构建的注册表仅供测试/手动装配；生产路径走 register_skills——
        # 它会用运行时实名重建注册表（universe = 实名），装载即校验。
        self._registry = registry or SkillRegistry(tool_universe=frozenset())
        self._registered = False
        # 阶段态放**当前异步任务的上下文**而不是实例字段：控制器是进程单例
        # （factory 与 orchestrator 同实例注入），worker 并发消费 / FastAPI
        # 并发请求下，会话 A set_stage 之后、模型调用读 prompt 之前，会话 B
        # 的 set_stage 会把 A 的阶段串掉（search 轮拿到 trade 纪律、trade 轮
        # 丢检索纪律）。ContextVar 按任务隔离——set_stage 与同一轮的模型调用
        # 在同一个任务上下文里，天然各持各的；未 set 的任务落到保守默认。
        self._stages_var: contextvars.ContextVar[frozenset[Stage] | None] = (
            contextvars.ContextVar(f"skill_active_stages_{id(self)}", default=None)
        )

    def set_stage(self, raw_query: str) -> None:
        """每轮意图进来时刷新激活阶段（orchestrator 注入点调用）。

        只影响当前异步任务——并发会话互不可见。"""
        self._stages_var.set(self._router.route(raw_query))

    def active_stages(self) -> frozenset[Stage]:
        return self._stages_var.get() or _ALL_STAGES

    def has_skills(self) -> bool:
        return bool(self._registry.for_stage(Stage.COMMON) or self._registry.for_stage(
            Stage.SEARCH
        ))

    def render_system_prompt(self) -> str:
        parts = [SKILL_BASE_HEADER, self._registry.render_stages(self.active_stages())]
        return "\n\n".join(part for part in parts if part.strip())

    def tool_subset(self) -> tuple[str, ...]:
        """当前阶段允许的工具并集（common 常驻工具恒在）。"""
        return self._registry.tool_subset_for_stages(self.active_stages())

    def register_skills(self, tool_names, definitions=None) -> None:
        """把仓内 definitions 装进注册表，并对**真实运行时工具名**校验。

        factory 在 build 路径上调用（此时工具已构造、名字是实名）；注册表
        以实名重建（universe = 实名），definitions 引用不在其中的工具 = 报错
        留名。幂等——Agent 每次构建都会走到这里，重复装载必须跳过。
        definitions 参数供测试注入坏定义，生产用仓内文件。
        """
        if self._registered:
            return
        specs = definitions if definitions is not None else self._load_definitions()
        unknown_by_spec: list[str] = []
        for spec in specs:
            unknown = [t for t in spec.tools if t not in set(tool_names)]
            if unknown:
                unknown_by_spec.append(f"{spec.skill_id}: {'、'.join(unknown)}")
        if unknown_by_spec:
            raise ValueError(
                "skill 定义引用了运行时不存在的工具（工具名拼错会在运行期静默缺工具）——"
                + "；".join(unknown_by_spec)
            )
        self._registry = SkillRegistry(tool_universe=set(tool_names))
        for spec in specs:
            self._registry.register(spec)
        self._registered = True

    @staticmethod
    def _load_definitions():
        from .loader import load_skill_definitions
        return load_skill_definitions()
