# -*- coding: utf-8 -*-
"""订单工具集：create_order_tool / query_order_tool / cancel_order_tool

MainAgent 单干与 TradeAgent 派发两条路径共用。工具层只做参数搬运与事件上报，业务规则在 UseCase 与 Order 聚合内。

注意：本模块不能用 `from __future__ import annotations`（AgentScope schema 生成依赖运行时注解）。
"""
import json

from agentscope.message import TextBlock, ToolResultState
from agentscope.tool import ToolChunk

from app.application.usecases.order_usecases import (
    CancelOrderUseCase,
    OrderItemInput,
    PlaceOrderUseCase,
    QueryOrderUseCase,
)
from app.domain.order.address import Address
from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus


def _ok(payload: dict) -> ToolChunk:
    return ToolChunk(
        content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False))],
        state=ToolResultState.SUCCESS,
    )


def _fail(message: str) -> ToolChunk:
    return ToolChunk(
        content=[TextBlock(type="text", text=f"[error] {message}")],
        state=ToolResultState.ERROR,
    )


def build_create_order_tool(usecase: PlaceOrderUseCase, bus: TradeEventBus):
    async def create_order_tool(
        items: list[dict],
        shipping_address: dict,
    ) -> ToolChunk:
        """创建订单（直接进入 CONFIRMED 态）。必须在买家确认后调用。买家身份由系统会话上下文自动注入。

        Args:
            items (`list[dict]`):
                订单行列表，每项形如 {"product_id": "P1001", "sku_id": "P1001-S1", "quantity": 1}。
            shipping_address (`dict`):
                收货地址，形如 {"recipient_name": "...", "country": "CN", "state": "...",
                "city": "...", "address_line": "...", "postal_code": "...", "phone": "..."}。
        """
        # 买家身份从 ShoppingContext 取真实值，不信任模型生成的入参，避免串账。
        # 无上下文时 fail-closed（审查 L2）："anonymous" 共享伪身份会造出
        # 一个可互查/互取消的匿名订单池
        snapshot_ctx = ShoppingContext.current()
        if snapshot_ctx is None or not snapshot_ctx.buyer_id:
            return _fail("无买家会话上下文，拒绝创建订单")
        buyer_id = snapshot_ctx.buyer_id
        session_id = ShoppingContext.current_session_id()
        bus.publish(session_id, "tool.invoke", {"tool": "create_order_tool", "args": {"buyer_id": buyer_id, "items": items}})
        try:
            order_items = [
                OrderItemInput(
                    product_id=item["product_id"],
                    sku_id=item["sku_id"],
                    quantity=int(item.get("quantity", 1)),
                )
                for item in items
            ]
            address = Address(
                recipient_name=shipping_address.get("recipient_name", ""),
                country=shipping_address.get("country", ""),
                state=shipping_address.get("state", ""),
                city=shipping_address.get("city", ""),
                address_line=shipping_address.get("address_line", ""),
                postal_code=shipping_address.get("postal_code", ""),
                phone=shipping_address.get("phone", ""),
            )
            snapshot = await usecase.execute(buyer_id=buyer_id, items=order_items, shipping_address=address)
        except (ValueError, KeyError) as err:
            bus.publish(session_id, "tool.result", {"tool": "create_order_tool", "error": str(err)})
            return _fail(str(err))
        bus.publish(session_id, "tool.result", {"tool": "create_order_tool", "order": snapshot})
        return _ok(snapshot)

    return create_order_tool


def build_query_order_tool(usecase: QueryOrderUseCase, bus: TradeEventBus):
    async def query_order_tool(order_id: str) -> ToolChunk:
        """查询订单详情。只能查询买家本人名下的订单，订单不存在或非本人所有均告知不存在。

        Args:
            order_id (`str`):
                订单号，如 "GBX-000001"。
        """
        # 买家身份从 ShoppingContext 取真实值（同 create_order_tool）：
        # 没有它，报对订单号就能查任何人的订单（红队用例挖出的洞）
        snapshot_ctx = ShoppingContext.current()
        buyer_id = snapshot_ctx.buyer_id if snapshot_ctx else "anonymous"
        session_id = ShoppingContext.current_session_id()
        bus.publish(session_id, "tool.invoke", {"tool": "query_order_tool", "args": {"order_id": order_id}})
        try:
            snapshot = await usecase.execute(order_id, buyer_id)
        except ValueError as err:
            bus.publish(session_id, "tool.result", {"tool": "query_order_tool", "error": str(err)})
            return _fail(str(err))
        bus.publish(session_id, "tool.result", {"tool": "query_order_tool", "order": snapshot})
        return _ok(snapshot)

    return query_order_tool


def build_cancel_order_tool(usecase: CancelOrderUseCase, bus: TradeEventBus):
    async def cancel_order_tool(order_id: str, reason: str) -> ToolChunk:
        """取消订单（仅本人 CONFIRMED 态订单可取消），取消后自动回补库存。

        Args:
            order_id (`str`):
                订单号，如 "GBX-000001"。
            reason (`str`):
                取消原因，必填。
        """
        snapshot_ctx = ShoppingContext.current()
        buyer_id = snapshot_ctx.buyer_id if snapshot_ctx else "anonymous"
        session_id = ShoppingContext.current_session_id()
        bus.publish(session_id, "tool.invoke", {"tool": "cancel_order_tool", "args": {"order_id": order_id, "reason": reason}})
        try:
            snapshot = await usecase.execute(order_id, reason, buyer_id)
        except ValueError as err:
            bus.publish(session_id, "tool.result", {"tool": "cancel_order_tool", "error": str(err)})
            return _fail(str(err))
        bus.publish(session_id, "tool.result", {"tool": "cancel_order_tool", "order": snapshot})
        return _ok(snapshot)

    return cancel_order_tool
