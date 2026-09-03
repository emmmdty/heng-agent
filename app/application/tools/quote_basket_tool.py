# -*- coding: utf-8 -*-
"""quote_basket_tool

组合到手价工具：多个商品一起买的总价（小计 + 运费 + 关税）由领域层算，不交给模型。

为什么补这个工具：本仓原有的取舍是「计价收敛进检索链路，不给 Agent 单独的比价/运费
工具」，理由是每多一个工具就多一轮模型调用。这个取舍对**单品**是对的——到手价随商品卡
内联返回。但它留了个缺口：买家问"这两个一起多少钱"时，组合总价没有任何工具提供，
模型只能自己相加。评测 compare-two 实测到了后果：单品到手价全对（工具算的），
组合价 $51.26 + $21.69 被模型算成 $62.95（正确 $72.95）——一个纯加法错误，
却直接违反了"数字事实必须来自工具返回"的 P0 底线。

所以这不是推翻原取舍，是把它补全：**凡是要出现在回复里的金额，都必须有工具出处。**
组合报价还有两处口径模型不可能自己推对（运费按一次履约计、免税额度按整批判定），
更加不该让它算。

注意：本模块不能用 `from __future__ import annotations`——
AgentScope 用 pydantic 从函数签名动态生成 JSON schema，字符串化注解会解析失败。
"""
import json

from agentscope.message import TextBlock, ToolResultState
from agentscope.tool import ToolChunk

from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.shipping.tariff_schedule import BasketLine, TariffSchedule
from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus


def build_quote_basket_tool(
    product_repo: ProductRepository, tariff: TariffSchedule, bus: TradeEventBus,
):
    async def quote_basket_tool(
        items: list,
        ship_to: str,
        target_currency: str = "CNY",
    ) -> ToolChunk:
        """计算多个商品一起购买的组合到手价（小计+运费+关税）。

        买家问"这几个一起多少钱""两个加起来到手价"时**必须**调用本工具，
        不要自己把单品到手价相加：运费按一次履约计（不是各单品运费之和），
        免税额度按整批小计判定（不是逐件判定），自行相加会得到错误金额。

        Args:
            items (`list`):
                商品清单，每项形如 {"product_id": "P1004", "quantity": 1}；
                quantity 缺省为 1。
            ship_to (`str`):
                收货国家二位码，如 "US"、"CN"。
            target_currency (`str`):
                金额口径币种，默认 "CNY"。
        """
        session_id = ShoppingContext.current_session_id()
        args = {"items": items, "ship_to": ship_to, "target_currency": target_currency}
        bus.publish(session_id, "tool.invoke", {"tool": "quote_basket_tool", "args": args})

        try:
            if not items:
                raise ValueError("items 不能为空")

            lines = []
            for raw in items:
                if not isinstance(raw, dict) or "product_id" not in raw:
                    raise ValueError(f"items 元素需含 product_id：{raw}")
                product_id = str(raw["product_id"])
                # 模型偶尔把数量当字符串传，与 product_search_tool 同样宽松处理
                quantity = int(raw.get("quantity", 1) or 1)
                found = await product_repo.find_by_ids([product_id])
                if not found:
                    raise ValueError(f"商品不存在：{product_id}")
                product = found[0]
                if ship_to not in product.ships_to:
                    raise ValueError(f"{product_id}（{product.title}）不可寄往 {ship_to}")
                sku = product.primary_sku()
                lines.append(
                    BasketLine(
                        product_id=product.product_id,
                        title=product.title,
                        category=product.category,
                        unit_price=sku.price,
                        quantity=quantity,
                    ),
                )

            quote = tariff.quote_basket(lines, ship_to=ship_to, target_currency=target_currency)
            payload = quote.to_dict()
        except Exception as err:  # noqa: BLE001 —— 报价失败要让模型如实告知，不得编数字
            message = f"[error] 组合报价失败：{err}"
            bus.publish(session_id, "tool.result", {"tool": "quote_basket_tool", "error": str(err)})
            return ToolChunk(
                content=[TextBlock(type="text", text=message)],
                state=ToolResultState.ERROR,
            )

        bus.publish(session_id, "tool.result", {"tool": "quote_basket_tool", **payload})
        return ToolChunk(
            content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False))],
            state=ToolResultState.SUCCESS,
        )

    return quote_basket_tool
