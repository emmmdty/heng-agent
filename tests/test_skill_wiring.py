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

    def test_stage_state_is_task_local_under_concurrent_intents(self):
        """并发隔离（C4 烧前 review 定位的竞态）：控制器是进程单例，阶段态
        原先是实例可变字段——worker 并发消费 / FastAPI 并发请求下，会话 A
        set_stage 之后、模型调用读 prompt 之前，会话 B 的 set_stage 会把 A
        的阶段串掉（search 轮拿到 trade 纪律、trade 轮丢检索纪律）。
        阶段态必须落在当前异步任务的上下文里，各会话各持各的。"""
        c = self._controller()
        results: dict = {}

        async def session(query: str, key: str, delay: float):
            await asyncio.sleep(delay)
            c.set_stage(query)
            await asyncio.sleep(0.01)
            results[key] = c.active_stages()

        async def main():
            await asyncio.gather(
                session("帮我推荐一款露营灯", "search", 0.0),
                session("帮我取消刚才那个订单", "trade", 0.005),
            )

        asyncio.run(main())
        assert results["search"] == frozenset({Stage.SEARCH}), (
            "会话 A 的阶段被并发会话 B 串掉了"
        )
        assert results["trade"] == frozenset({Stage.TRADE})

    def test_concurrent_renders_get_own_stage_prompt(self):
        """中间件读的是当前任务的阶段态：两个并发渲染各拿各的 system prompt。"""
        from app.application.agents.skill_stage import SkillStagePromptMiddleware

        mw = SkillStagePromptMiddleware(self._controller())
        prompts: dict = {}

        async def render(query: str, key: str, delay: float):
            await asyncio.sleep(delay)
            mw._controller.set_stage(query)
            await asyncio.sleep(0.01)
            prompts[key] = mw._controller.render_system_prompt()

        async def main():
            await asyncio.gather(
                render("帮我推荐一款露营灯", "search", 0.0),
                render("帮我取消刚才那个订单", "trade", 0.005),
            )

        asyncio.run(main())
        assert prompts["search"] != prompts["trade"]

    def test_fresh_task_without_set_stage_defaults_to_all_stages(self):
        """未经 set_stage 的任务（后台调用等）落保守默认 = 全交付阶段，
        与控制器初始态同口径——宁可多带，不缺纪律。"""
        c = self._controller()

        async def main():
            return c.active_stages()

        assert asyncio.run(main()) == _DELIVERY_STAGES


class TestFidelityLedger:
    """heng.yml ↔ definitions.yml 搬运对账的可执行版（设计笔记第五节的表）。

    保真度缺口的本质：flag 开时 heng.yml 单块不再发送，片段没搬到的纪律
    就静默消失——C4 烧前 review 抓出三处（landed_price/ship_to 内联纪律、
    派发语义只在 search 档、web_search_tool 零提及）。本测试把对账表钉成
    确定性断言：以后任何人改 heng.yml 或 definitions.yml，漏搬必红。
    断言用签名短语（工具名/字段名/关键词）而不是整句——卡的是
    "这条纪律涉及的工具与字段在片段里有没有被提及"这个保真度本质，
    搬运时的措辞差异不误报。
    """

    _LEDGER = [
        # (签名短语, heng.yml 纪律块)
        ("price_max_major", "工具说明：product_search 预算硬约束必传"),
        ("filtered_out", "工具说明/工作约定 7：被挡候选如实说明"),
        ("landed_price", "工具说明：product_search 传 ship_to 内联到手价，比价直接用"),
        ("ship_to", "工具说明：product_search 传 ship_to 内联到手价明细"),
        ("quote_basket_tool", "工具说明：组合到手价，禁止单品相加"),
        ("optimize_basket_tool", "工具说明：预算最优组合，三个数都由它返回"),
        ("taxable_base_major", "工作约定 2：计税基数是超出免税额度的部分"),
        ("de_minimis_threshold_native_major", "工作约定 2：免税额度原生币种口径"),
        ("separate_purchase_landed_major", "工作约定 2：分开买 vs 一起买由工具返回"),
        ("combining_saving_major", "工作约定 2：一起买省多少由工具返回"),
        ("category_insight_tool", "工具说明：品类洞察只给口径不代商品"),
        ("web_search_tool", "工具说明：web_search（若可用）查时效信息"),
        ("task_dispatch", "调度工具：派发语义 + demands 自包含"),
        ("确认卡", "工作约定 1：订单写操作先确认卡"),
        ("默认地址", "工作约定 1：系统不存在默认收货信息"),
        ("buyer-preferences", "记忆：<buyer-preferences> 注入与参考"),
        ("remember_preference_tool", "记忆工具：长期偏好写入"),
        ("forget_preference_tool", "记忆撤回：statement 照抄原文"),
        ("凭印象编造", "工作约定 2 总纲：数字事实必须来自工具返回"),
        ("退款路径", "工作约定 4：不承诺支付物流能力"),
        ("凭通用经验", "工作约定 6：知识库降级口径（不含具体金额）"),
        ("说成没有这个商品", "工作约定 7：filtered_out 不说成没有该商品"),
    ]

    def _render(self, stages):
        reg = _registry()
        return SKILL_BASE_HEADER + "\n\n" + reg.render_stages(frozenset(stages))

    def test_union_render_covers_all_heng_disciplines(self):
        rendered = self._render({Stage.SEARCH, Stage.TRADE, Stage.MEMORY})
        missing = [
            f"{sig!r}（{why}）" for sig, why in self._LEDGER if sig not in rendered
        ]
        assert not missing, f"flag 开时以下 heng.yml 纪律没有搬进任何片段：{'、'.join(missing)}"

    def test_dispatch_discipline_available_in_trade_and_memory_rounds(self):
        """task_dispatch 工具在 flag 开时全阶段可见（工具子集未接线），
        派发纪律却只挂在 search 档——trade/memory 轮模型看得见工具
        却没有任何用法指引。派发语义必须进 COMMON。"""
        for stage in (Stage.TRADE, Stage.MEMORY):
            rendered = self._render({stage})
            assert "task_dispatch" in rendered, (
                f"{stage.value} 轮可见 task_dispatch 工具却没有派发纪律"
            )


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
