# -*- coding: utf-8 -*-
"""C3 接线（#14 任务 C）：skill 渐进加载进 app 运行时。

设计约束（二十七期任务书 C3 + 设计笔记）：
- **flag 门控**（SKILL_LOADING_ENABLED，默认关）：关 = 行为与指纹逐字节不变
  ——B2 resume 断点记的是 a0915fac，C3 不许把它弄丢；
- **替换式**：flag 开时 system prompt = 身份头 + common + 激活阶段片段
  （对账表全量搬运，heng.yml 单块不再发送、文件一字不动留作 skill-off 基线）；
- **保守路由**：关键词无命中 = 全阶段（宁可多带，C4 的 PASS 不回退一票否决）；
- **Task* 死重**（C1：全史 0 调用）只在 flag 开时移出 toolkit。
"""
import asyncio
import hashlib
from pathlib import Path

import pytest

from app.skills.registry import SkillRegistry
from app.skills.router import StageRouter
from app.skills.schema import Stage, SkillSpec
from app.skills.stage import SKILL_BASE_HEADER, SkillStageController

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_DELIVERY_STAGES = frozenset({Stage.SEARCH, Stage.TRADE, Stage.MEMORY})


def _registry() -> SkillRegistry:
    reg = SkillRegistry(tool_universe={
        "product_search_tool", "category_insight_tool", "task_dispatch",
        "quote_basket_tool", "optimize_basket_tool", "create_order_tool",
        "cancel_order_tool", "query_order_tool", "remember_preference_tool",
        "forget_preference_tool",
    })
    for spec in _load_definitions():
        reg.register(spec)
    return reg


def _load_definitions():
    from app.skills.loader import load_skill_definitions
    return load_skill_definitions()


class TestStageRouter:
    def test_search_query_hits_search_only(self):
        assert StageRouter().route("帮我推荐一款便携露营灯") == frozenset({Stage.SEARCH})

    def test_trade_query_hits_trade(self):
        assert StageRouter().route("帮我取消刚才那个订单") == frozenset({Stage.TRADE})

    def test_memory_query_hits_memory(self):
        assert StageRouter().route("记住我以后不要塑料材质") == frozenset({Stage.MEMORY})

    def test_no_hit_falls_back_to_all_stages(self):
        """保守铁律：认不出就全给——闲聊/歧义轮不缺任何纪律。"""
        assert StageRouter().route("你好呀") == _DELIVERY_STAGES

    def test_multi_signal_query_unions_stages(self):
        """"找一下我的订单"——search 与 trade 同时命中，取并集（宁可多带）。"""
        stages = StageRouter().route("帮我找一下我的订单")
        assert {Stage.SEARCH, Stage.TRADE} <= stages


class TestSkillStageController:
    def _controller(self):
        return SkillStageController(router=StageRouter(), registry=_registry())

    def test_default_is_all_stages_conservative(self):
        assert self._controller().active_stages() == _DELIVERY_STAGES

    def test_set_stage_narrows_prompt_to_hit_stages(self):
        c = self._controller()
        c.set_stage("帮我取消刚才那个订单")
        prompt = c.render_system_prompt()
        assert "确认卡" in prompt, "trade 片段必须在"
        assert "只提供判断口径" not in prompt, "search 专属片段不该在"

    def test_prompt_contains_common_and_header_once(self):
        c = self._controller()
        c.set_stage("帮我推荐一款露营灯")
        prompt = c.render_system_prompt()
        assert prompt.startswith(SKILL_BASE_HEADER)
        assert prompt.count("不涉及支付与物流") == 1, "common 只渲染一次"

    def test_render_is_deterministic(self):
        c = self._controller()
        c.set_stage("帮我推荐一款露营灯")
        assert c.render_system_prompt() == c.render_system_prompt()

    def test_off_stage_tools_not_in_subset(self):
        c = self._controller()
        c.set_stage("帮我取消刚才那个订单")
        tools = c.tool_subset()
        assert "create_order_tool" in tools and "remember_preference_tool" not in tools

    def test_tool_subset_keeps_resident_search_tool(self):
        """product_search_tool 是 common 常驻档（C1/B1 评审）——任何阶段都在。"""
        c = self._controller()
        c.set_stage("帮我取消刚才那个订单")
        assert "product_search_tool" in c.tool_subset()


class TestSkillStagePromptMiddleware:
    def _middleware(self):
        from app.application.agents.skill_stage import SkillStagePromptMiddleware
        return SkillStagePromptMiddleware(SkillStageController(router=StageRouter(), registry=_registry()))

    def test_replaces_monolith_under_flag(self):
        monolith = "计划工具（TaskCreate / TaskUpdate / TaskList / TaskGet）：任务需要 3 步以上"
        result = asyncio.run(self._middleware().on_system_prompt(object(), monolith))
        assert "计划工具" not in result, "Task* 段必须随替换消失"
        assert SKILL_BASE_HEADER in result

    def test_falls_back_to_original_when_registry_empty(self):
        from app.application.agents.skill_stage import SkillStagePromptMiddleware
        mw = SkillStagePromptMiddleware(SkillStageController(registry=SkillRegistry(tool_universe=set())))
        original = "原 prompt"
        assert asyncio.run(mw.on_system_prompt(object(), original)) == original


class TestSkillStageControllerRegistration:
    def test_register_skills_validates_against_real_tool_names(self):
        """definitions 引用了运行时不存在的工具 = 静默缺工具——装载时报错留名。"""
        c = SkillStageController(router=StageRouter(), registry=_registry())
        with pytest.raises(ValueError, match="no_such_tool"):
            c.register_skills(tool_names={"product_search_tool"}, definitions=[
                SkillSpec(skill_id="bad", stages=(Stage.SEARCH,), tools=("no_such_tool",),
                          prompt_fragments=("x",)),
            ])

    def test_register_skills_is_idempotent(self):
        c = SkillStageController(StageRouter())  # 生产路径：空控制器起步
        runtime_names = {t for s in _load_definitions() for t in s.tools}
        c.register_skills(tool_names=runtime_names)  # 首次：装载仓内定义
        c.register_skills(tool_names=runtime_names)  # 再次：幂等不炸
        assert c.render_system_prompt()


class TestOrchestratorStageInjection:
    async def test_set_stage_called_with_raw_query_each_intent(self):
        from app.application.agents.orchestrator import MainAgentOrchestrator, SubmitIntentInput

        class RecordingController:
            def __init__(self):
                self.queries = []

            def set_stage(self, raw_query):
                self.queries.append(raw_query)

        from tests.test_turn_usage_recording import FakeAgent, FakeRegistry, NullPreferenceStore
        from app.infrastructure.eventbus import TradeEventBus

        bus = TradeEventBus()
        controller = RecordingController()
        orchestrator = MainAgentOrchestrator(
            sessions=FakeRegistry(FakeAgent(bus=bus, session_id="s1", reply_text="好的")),
            bus=bus, preference_store=NullPreferenceStore(),
            skill_stage=controller,
        )
        await orchestrator.handle_intent(SubmitIntentInput(
            shopping_session_id="s1", buyer_id="b1",
            locale="zh-CN", currency="CNY", raw_query="找个露营灯",
        ))
        assert controller.queries == ["找个露营灯"]

    async def test_no_controller_is_noop(self):
        """flag 关（controller=None）= 现行为零变化。"""
        from app.application.agents.orchestrator import MainAgentOrchestrator, SubmitIntentInput

        from tests.test_turn_usage_recording import FakeAgent, FakeRegistry, NullPreferenceStore
        from app.infrastructure.eventbus import TradeEventBus

        bus = TradeEventBus()
        orchestrator = MainAgentOrchestrator(
            sessions=FakeRegistry(FakeAgent(bus=bus, session_id="s2", reply_text="好的")),
            bus=bus, preference_store=NullPreferenceStore(),
        )
        result = await orchestrator.handle_intent(SubmitIntentInput(
            shopping_session_id="s2", buyer_id="b1",
            locale="zh-CN", currency="CNY", raw_query="找个露营灯",
        ))
        assert not result.final_text.startswith("[error]")


class TestCompositionWiring:
    """容器级：flag 关 = 逐字节不变（指纹 a0915fac、Task* 在）；flag 开 = 换血。

    每个测试独立 VECTOR_STORE_DIR——容器对象持有的 qdrant 本地客户端是
    文件锁互斥的，同目录二次 build 会 AlreadyLocked。"""

    @staticmethod
    async def _toolkit_names_and_fingerprint(monkeypatch, flag: str, store_dir: str):
        # CI runner 没有 /tmp/opencode——qdrant 本地存储目录先建出来
        Path(store_dir).mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("VECTOR_STORE_DIR", store_dir)
        monkeypatch.setenv("SKILL_LOADING_ENABLED", flag)
        from app.composition import build_container
        container = await build_container()
        registry_obj = next(
            v for v in vars(container.orchestrator).values()
            if type(v).__name__ == "SessionRegistry"
        )
        factory = next(
            v for v in vars(registry_obj).values()
            if type(v).__name__ == "MainAgentFactory"
        )
        agent = factory.build()
        apis = await agent.toolkit.get_tool_schemas()
        names = {api.get("function", api).get("name", "?") for api in apis}
        return names, container.prompt_fingerprint

    async def test_flag_off_is_byte_identical(self, monkeypatch):
        names, fingerprint = await self._toolkit_names_and_fingerprint(
            monkeypatch, "0", "/tmp/opencode/qdrant-c3-off",
        )
        assert {"TaskCreate", "TaskUpdate", "TaskList", "TaskGet"} <= names, "flag 关必须保留 Task*"
        expected = hashlib.sha256(
            (PROJECT_ROOT / "app" / "application" / "prompts" / "heng.yml").read_bytes()
        ).hexdigest()[:8]
        assert fingerprint == expected, "flag 关指纹必须仍是 heng.yml 原值（B2 resume 依赖）"

    async def test_flag_on_drops_dead_weight_and_changes_fingerprint(self, monkeypatch):
        names, fingerprint = await self._toolkit_names_and_fingerprint(
            monkeypatch, "1", "/tmp/opencode/qdrant-c3-on",
        )
        assert not {"TaskCreate", "TaskUpdate", "TaskList", "TaskGet"} & names, "Task* 死重必须消失"
        assert {"product_search_tool", "quote_basket_tool", "remember_preference_tool"} <= names
        heng_hash = hashlib.sha256(
            (PROJECT_ROOT / "app" / "application" / "prompts" / "heng.yml").read_bytes()
        ).hexdigest()[:8]
        assert fingerprint != heng_hash, "flag 开 = 不同 prompt 配置，指纹必须不同"

    async def test_definitions_tools_all_exist_in_runtime(self, monkeypatch):
        """flag 开时 definitions 的工具名必须全部在真实 toolkit 里——
        registry 校验在 factory.build 路径上开火，能 build 成功即已通过。"""
        await self._toolkit_names_and_fingerprint(
            monkeypatch, "1", "/tmp/opencode/qdrant-c3-validate",
        )
