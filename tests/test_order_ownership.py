# -*- coding: utf-8 -*-
"""订单归属校验（红队用例挖出的真缺陷，二十三期清单 6 的产出）

红队首轮（eval/report-20260904-191241）静态分诊确认：`QueryOrderUseCase` 与
`CancelOrderUseCase` **不接收 buyer_id**——任何买家只要报对订单号，
就能查询乃至取消别人的订单。红队用例当时没探到洞底，是因为它用了
编造的订单号，"订单不存在"恰好掩护了归属校验的缺失。

修复原则与 create_order_tool 同一条：**买家身份从系统上下文取真实值，
不信任买家/模型的声明**。归属不符时对外一律报"订单不存在"——
不区分"不存在"与"非你所有"，防止订单号枚举探测（安全上正确的模糊）。

三个测试层：
    usecase 层   —— 归属规则本身（拒绝 + **不产生副作用**：取消被拒后订单必须原样）；
    工具接线层   —— buyer_id 必须来自 ShoppingContext，模型传不进来；
    DTO 契约层   —— REST 直调方必须显式声明身份，缺省拒收。
"""

import pytest

from app.application.usecases.order_usecases import (
    CancelOrderUseCase,
    OrderItemInput,
    PlaceOrderUseCase,
    QueryOrderUseCase,
)
from app.domain.order.address import Address
from app.domain.session.ports.conversation_store import ConversationEventRecord
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.persistence.in_memory_repositories import (
    InMemoryOrderRepository,
    InMemoryProductRepository,
)

ADDRESS = Address(
    recipient_name="张三", country="CN", state="浙江", city="杭州市",
    address_line="西湖区某路 1 号", postal_code="310000", phone="13800000000",
)


@pytest.fixture()
def repos():
    return InMemoryProductRepository(), InMemoryOrderRepository()


@pytest.fixture()
def owned_order(repos):
    """buyer-a 名下的真实订单（走真实下单用例创建，扣了库存的）。"""
    product_repo, order_repo = repos
    usecase = PlaceOrderUseCase(product_repo, order_repo)
    snapshot = usecase_sync(usecase, buyer_id="buyer-a")
    return snapshot


def usecase_sync(usecase, buyer_id: str) -> dict:
    import asyncio

    return asyncio.run(usecase.execute(
        buyer_id=buyer_id,
        items=[OrderItemInput(product_id="P1008", sku_id="P1008-S1", quantity=1)],
        shipping_address=ADDRESS,
    ))


class TestOwnershipAtUseCaseLayer:
    async def test_other_buyer_cannot_query(self, repos, owned_order):
        _, order_repo = repos
        usecase = QueryOrderUseCase(order_repo)
        with pytest.raises(ValueError, match="订单不存在"):
            await usecase.execute(owned_order["order_id"], "buyer-b")

    async def test_owner_can_query(self, repos, owned_order):
        _, order_repo = repos
        usecase = QueryOrderUseCase(order_repo)
        snapshot = await usecase.execute(owned_order["order_id"], "buyer-a")
        assert snapshot["order_id"] == owned_order["order_id"]

    async def test_other_buyer_cannot_cancel_and_nothing_happens(self, repos, owned_order):
        """不只是拒绝：**不能有任何副作用**——取消被拒后订单必须原样 CONFIRMED，
        库存不能被回补（否则等于用拒绝完成了攻击）。"""
        product_repo, order_repo = repos
        usecase = CancelOrderUseCase(product_repo, order_repo)
        with pytest.raises(ValueError, match="订单不存在"):
            await usecase.execute(owned_order["order_id"], "想白嫖", "buyer-b")
        stored = await order_repo.find_by_id(owned_order["order_id"])
        assert stored.snapshot()["status"] == "CONFIRMED", "取消必须没有发生"

    async def test_nonexistent_order_reads_the_same(self, repos):
        """不存在的订单与非你所有的订单，对外读数必须一致——可区分就是枚举探测的入口。"""
        _, order_repo = repos
        usecase = QueryOrderUseCase(order_repo)
        with pytest.raises(ValueError) as err:
            await usecase.execute("GBX-NOTEXIST", "buyer-b")
        assert "订单不存在" in str(err.value)


class TestOwnershipAtToolLayer:
    async def test_tool_injects_buyer_from_context_not_from_model(self, repos, owned_order):
        """查询工具没有 buyer_id 参数可传——身份只能来自 ShoppingContext。
        模型（或注入攻击）无法替买家声明身份。"""
        from app.application.tools.order_tools import build_query_order_tool
        from app.infrastructure.eventbus import TradeEventBus

        _, order_repo = repos
        bus = TradeEventBus()
        tool = build_query_order_tool(QueryOrderUseCase(order_repo), bus)
        token = ShoppingContext.set(ShoppingContextSnapshot(
            shopping_session_id="s-own", buyer_id="buyer-b",
            locale="zh-CN", currency="CNY",
        ))
        try:
            result = await tool(order_id=owned_order["order_id"])
        finally:
            ShoppingContext.reset(token)
        text = result.content[0].text
        assert text.startswith("[error]") and "订单不存在" in text, (
            "buyer-b 的会话里查 buyer-a 的订单必须被拒——哪怕模型很想查"
        )

    async def test_cancel_tool_same_boundary(self, repos, owned_order):
        from app.application.tools.order_tools import build_cancel_order_tool
        from app.infrastructure.eventbus import TradeEventBus

        product_repo, order_repo = repos
        bus = TradeEventBus()
        tool = build_cancel_order_tool(CancelOrderUseCase(product_repo, order_repo), bus)
        token = ShoppingContext.set(ShoppingContextSnapshot(
            shopping_session_id="s-own", buyer_id="buyer-b",
            locale="zh-CN", currency="CNY",
        ))
        try:
            result = await tool(order_id=owned_order["order_id"], reason="孩子乱点的")
        finally:
            ShoppingContext.reset(token)
        text = result.content[0].text
        assert text.startswith("[error]")
        stored = await order_repo.find_by_id(owned_order["order_id"])
        assert stored.snapshot()["status"] == "CONFIRMED"


class TestRestDtoContract:
    def test_cancel_request_requires_buyer_id(self):
        """REST 直调方没有会话上下文，必须显式声明身份——缺省拒收。"""
        from app.presentation.dto import CancelOrderRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CancelOrderRequest(reason="改主意了")
        request = CancelOrderRequest(reason="改主意了", buyer_id="buyer-a")
        assert request.buyer_id == "buyer-a"


class _NullPreferenceStore:
    async def append(self, preference):
        return None

    async def list_by_buyer(self, buyer_id):
        return []

    async def delete(self, buyer_id, statement):
        return False


class TestForgetSessionCleansLateAddedSources:
    """十七期 LRU 淘汰回调的清理清单漏了后加的两个按会话累积结构
    （contact：二十一期 / knowledge：二十二期）——soak 首轮 RSS 未持平，
    静态排查抓到的确定缺陷之一。淘汰后桶必须清空，否则只增不减。"""

    async def test_eviction_resets_contact_and_knowledge_sources(self):

        from app.application.agents.orchestrator import MainAgentOrchestrator, SubmitIntentInput

        class _Registry:
            async def get_or_create(self, session_id):  # pragma: no cover
                raise RuntimeError("本轮用不到 Agent")

            async def persist(self, session_id):  # pragma: no cover
                return None

        orchestrator = MainAgentOrchestrator(
            sessions=_Registry(), bus=TradeEventBus(), preference_store=_NullPreferenceStore(),
        )
        intent = SubmitIntentInput(
            shopping_session_id="s-leak", buyer_id="b", locale="zh-CN", currency="CNY",
            raw_query="上海市浦东新区世纪大道100号 收到请回复",
        )
        events = [
            ConversationEventRecord(session_id="s-leak", type="tool.result", payload={"hits": []}),
        ]
        orchestrator._number_sources.observe("s-leak", tool_results=events, buyer_texts=[intent.raw_query])
        orchestrator._contact_sources.observe("s-leak", tool_results=events, buyer_texts=[intent.raw_query])
        orchestrator._knowledge_sources.observe("s-leak", tool_results=events)

        orchestrator.forget_session("s-leak")  # 同步方法：LRU 淘汰回调是同步契约

        empty_number = orchestrator._number_sources.of("s-leak")
        assert not empty_number.numbers, "金额出处桶必须被清空"
        assert not orchestrator._contact_sources.of("s-leak").blob, "收货字段出处桶必须被清空（soak 抓到的泄漏）"
        assert orchestrator._knowledge_sources.of("s-leak").available is False, "知识库可用性标记必须被清空"


class TestRestContract:
    """归属校验的 REST 契约（审查 M4）：破坏性变更必须有回归保护。

    不起 TestClient 打路由：/commerce/* 的处理函数走 container()，而 container
    由 lifespan 构建整套外部依赖（app/presentation/server.py 的既有注释）。
    所以契约钉在 OpenAPI schema 上——buyer_id 必填是 FastAPI 422 行为的来源，
    schema 变了 422 行为就变了；"不存在与非你所有同读数"的语义在
    UseCase 层已测（TestOwnershipAtUseCaseLayer）。
    """

    def _openapi(self) -> dict:
        from app.presentation.server import build_app

        return build_app().openapi()

    def test_get_order_requires_buyer_id_query_param(self):
        spec = self._openapi()
        params = spec["paths"]["/commerce/orders/{order_id}"]["get"]["parameters"]
        buyer = [p for p in params if p["name"] == "buyer_id"]
        assert buyer and buyer[0]["required"] is True, "GET 缺 buyer_id 必须 422（fail-closed）"

    def test_cancel_body_requires_buyer_id(self):
        spec = self._openapi()
        schema_ref = spec["paths"]["/commerce/orders/{order_id}/cancel"]["post"]["requestBody"][
            "content"]["application/json"]["schema"]["$ref"]
        schema_name = schema_ref.rsplit("/", 1)[-1]
        required = spec["components"]["schemas"][schema_name]["required"]
        assert "buyer_id" in required, "取消请求体缺 buyer_id 必须 422（fail-closed）"
