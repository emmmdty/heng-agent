# -*- coding: utf-8 -*-
"""Harness 到底挂在哪些工具上（接线判据）。

十四期发现的真缺口：`HarnessToolMiddleware` 全仓只在 `main_agent.py` 实例化一次，
而那条链只挂给了主 Agent 自己的三个工具（task_dispatch / remember / forget）。
**检索、计价、订单工具走的是各自工厂的 `_resilience()`，里面只有熔断中间件。**

后果（每一条都是"看起来做了、实际从没跑过"）：

    - `create_order_tool` 的顺序硬拒（下单需先检索）从没生效过
    - `TOOL_REQUIRED_FIELDS` 的 schema 断言从没跑过——包括十一期专门给
      quote_basket / optimize_basket 补的那两条
    - L3 注入过滤 `sanitize_tool_output` 从没作用于检索与知识库返回，
      而那恰恰是注入内容唯一可能进来的地方
    - LoopDetector 只数得到派发与记忆工具，数不到"同一个检索连调五次"

外观与"故意不做"完全一样，没有任何告警——七期 BM25 忘接线是同一类。
所以把"哪些工具挂了 Harness"钉成判据。
"""
from app.infrastructure.harness_middleware import HarnessToolMiddleware
from app.infrastructure.persistence.in_memory_repositories import (
    InMemoryOrderRepository,
    InMemoryProductRepository,
)


def _has_harness(tool) -> bool:
    return any(isinstance(m, HarnessToolMiddleware) for m in tool._middlewares)


def _factories():
    from app.application.agents.search_agent import SearchAgentFactory
    from app.application.agents.trade_agent import TradeAgentFactory
    from app.application.usecases.catalog_search import CatalogSearchUseCase
    from app.application.usecases.order_usecases import (
        CancelOrderUseCase,
        PlaceOrderUseCase,
        QueryOrderUseCase,
    )
    from app.domain.catalog.exchange_rate import ExchangeRateTable
    from app.domain.shipping.tariff_schedule import TariffSchedule
    from app.infrastructure.eventbus import TradeEventBus
    from app.infrastructure.resilience import CircuitBreakerRegistry
    from app.infrastructure.settings import load_settings
    from app.infrastructure.throttle import GatewayThrottle

    settings = load_settings()
    bus = TradeEventBus()
    registry = CircuitBreakerRegistry()
    throttle = GatewayThrottle(2, 0.0)
    product_repo = InMemoryProductRepository()
    order_repo = InMemoryOrderRepository()

    search = SearchAgentFactory(
        settings, CatalogSearchUseCase(product_repo), bus, None, registry, throttle,
        product_repo=product_repo, tariff=TariffSchedule(rates=ExchangeRateTable()),
    )
    trade = TradeAgentFactory(
        settings,
        PlaceOrderUseCase(product_repo, order_repo),
        QueryOrderUseCase(order_repo),
        CancelOrderUseCase(product_repo, order_repo),
        bus, registry, throttle,
    )
    return search, trade


class TestBusinessToolsCarryTheHarness:
    def test_search_tools(self):
        search, _ = _factories()
        missing = [tool.name for tool in search.build_tools() if not _has_harness(tool)]
        assert not missing, f"这些检索侧工具没挂 Harness：{missing}"

    def test_order_tools(self):
        """订单工具尤其重要：顺序硬拒（下单需先检索）就挂在这条链上，
        不挂等于那道判据从来没生效过。"""
        _, trade = _factories()
        missing = [tool.name for tool in trade.build_tools() if not _has_harness(tool)]
        assert not missing, f"这些订单工具没挂 Harness：{missing}"

    def test_resilience_is_still_inside(self):
        """洋葱顺序不能反：Harness 在外做准入判定，Resilience 在内做超时熔断。
        反过来的话，被硬拒的调用会白占一次熔断名额。"""
        from app.infrastructure.resilience import ToolResilienceMiddleware

        search, trade = _factories()
        for tool in [*search.build_tools(), *trade.build_tools()]:
            kinds = [type(m).__name__ for m in tool._middlewares]
            assert kinds.index("HarnessToolMiddleware") < kinds.index(
                ToolResilienceMiddleware.__name__,
            ), f"{tool.name} 的中间件顺序反了：{kinds}"


class TestL3DoesNotMangleProductCopy:
    """L3 注入过滤现在**第一次**作用在检索返回上（十四期接线之前它从没跑到过）。

    命中时它会把片段替换成 `[内容已过滤：疑似注入]` —— 作用在商品卡上就是
    静默改坏了给买家看的文案。所以拿全库 60 个 SPU 的真实检索返回跑一遍，
    钉住"零误伤"；将来谁加了一款文案里带"扮演…角色"的商品，这条会红。
    """

    async def test_no_false_positive_across_the_whole_catalog(self):
        import json

        from app.application.usecases.catalog_search import CatalogSearchUseCase
        from app.domain.catalog.product_search_spec import ProductSearchSpec
        from app.infrastructure.security.content_filter import sanitize_tool_output

        repo = InMemoryProductRepository()
        usecase = CatalogSearchUseCase(repo)
        flagged = []
        for product in await repo.list_all():
            result = await usecase.execute(
                ProductSearchSpec(normalized_query=product.title),
            )
            hit, _ = sanitize_tool_output(json.dumps(result, ensure_ascii=False))
            if hit:
                flagged.append(product.product_id)
        assert not flagged, f"L3 误伤了这些商品的卡片文案：{flagged}"


class TestWriteToolsAreWhitelisted:
    """每一个**非只读**工具都必须在权限白名单里。

    2.0 的 DEFAULT 权限模式下，非只读工具会触发 RequireUserConfirmEvent 挂起等确认。
    本仓的确认语义在对话层（MainAgent 先出确认卡、买家确认后才执行），
    工具层不该再挂一次——漏进白名单的后果不是"多问一句"，是**那一轮直接卡住**。

    与 L4 工具名清单、Harness 接线是同一类：清单与实现脱钩，
    而脱钩之后没有任何东西会报警。
    """

    def test_every_write_tool_is_allowed(self):
        from app.application.agents.permissions import _AUTO_ALLOWED_TOOLS

        search, trade = _factories()
        write_tools = [
            tool.name
            for tool in [*search.build_tools(), *trade.build_tools()]
            if not tool.is_read_only
        ]
        assert write_tools, "一个写工具都没有？那多半是 is_read_only 标错了"
        missing = sorted(set(write_tools) - set(_AUTO_ALLOWED_TOOLS))
        assert not missing, f"这些写工具不在白名单里，调用时会挂起等确认：{missing}"

    def test_read_only_tools_are_not_whitelisted_by_accident(self):
        """只读工具不需要进白名单——放进去没有坏处，但会让"白名单里有什么"
        这件事失去信息量：它本该是一份"哪些写操作被豁免了确认"的清单。"""
        from app.application.agents.permissions import _AUTO_ALLOWED_TOOLS

        search, trade = _factories()
        read_only = {
            tool.name
            for tool in [*search.build_tools(), *trade.build_tools()]
            if tool.is_read_only
        }
        assert not (read_only & set(_AUTO_ALLOWED_TOOLS))


class TestPromptListsEveryTool:
    """注册了的工具必须在系统提示词里出现。

    "工具存在 ≠ 模型会调"这句话在本仓被反复验证过：
    十一期加了 optimize_basket_tool 之后，悬了三期才由 smoke 轮给出答案。
    而**提示词里没写**是这句话最彻底的一种形式——模型压根不知道有这个工具，
    再好的工具也等于没做，且没有任何东西会报警（同踩坑 37）。

    只检查"名字出现过"，不检查怎么写的：怎么描述是提示词工程的事，
    判据只管"有没有把它介绍给模型"。
    """

    def _prompt(self) -> str:
        from app.application.prompts.loader import load_prompts

        prompts = load_prompts()
        return prompts["main_agent"]["system_prompt"]

    def test_business_tools_appear_in_the_main_prompt(self):
        search, trade = _factories()
        prompt = self._prompt()
        missing = [
            tool.name for tool in [*search.build_tools(), *trade.build_tools()]
            if tool.name not in prompt
        ]
        assert not missing, f"这些工具没写进主 Agent 提示词，模型不会知道它们存在：{missing}"

    def test_memory_and_dispatch_tools_appear_too(self):
        """记忆与派发工具挂在主 Agent 自己身上，不在两个工厂里，单独钉。"""
        prompt = self._prompt()
        for name in ("task_dispatch", "remember_preference_tool", "forget_preference_tool"):
            assert name in prompt, f"{name} 没写进提示词"
