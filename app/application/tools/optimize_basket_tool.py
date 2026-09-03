# -*- coding: utf-8 -*-
"""optimize_basket_tool

预算感知的组合优化：给定预算与若干需求，在候选商品里枚举出最优组合。

为什么补这个工具：`quote_basket_tool` 解决了"这几个一起多少钱"，但买家说
"预算 300，想要个降噪耳机和登机箱"时，**选哪几件**仍然由模型自己拼——
一件件查、自己加、自己减。代价在金额出处校验里长期可见："预算还剩 $22"
"分开买贵 $3.65"这类数字没有任何工具出处，被归成 `suspected_difference`。

这条能力值得做深的另一个原因是**可验证性**：60 SPU 的候选空间小到可以暴力枚举出
真最优解，ground truth 是确定性算出来的，不需要 LLM judge 打分——
与金额出处校验、与给 judge 补规则表是同一条方法论主线。

被硬约束挡掉的候选（不可寄往目的国）**回传而不是静默丢弃**，同 `filtered_out`
的思路：让模型能区分"库里没有"与"有但不满足约束"。

注意：本模块不能用 `from __future__ import annotations`——
AgentScope 用 pydantic 从函数签名动态生成 JSON schema，字符串化注解会解析失败。
"""
import json

from agentscope.message import TextBlock, ToolResultState
from agentscope.tool import ToolChunk

from app.domain.catalog.money import Money
from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.shipping.basket_optimizer import NeedCandidate, NeedGroup, optimize_basket
from app.domain.shipping.tariff_schedule import TariffSchedule
from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus


def build_optimize_basket_tool(
    product_repo: ProductRepository, tariff: TariffSchedule, bus: TradeEventBus,
):
    async def optimize_basket_tool(
        needs: list,
        ship_to: str,
        budget_major: float = 0.0,
        target_currency: str = "CNY",
    ) -> ToolChunk:
        """在预算内挑出最优商品组合，并给出预算余额与"一起买省多少"。

        买家给了预算又提了多个需求（"预算 300 想要耳机和登机箱"）时**必须**调用本工具，
        不要自己挑完再把价格相加相减：预算余额、缺口差额、分开买与合并买的差额
        都由本工具返回，自行计算会得到没有出处的数字。

        先用 product_search_tool 为每个需求检索候选，再把候选的 product_id 传进来。
        组合内的运费按一次履约计、免税额度按整批小计判定，与 quote_basket_tool 同口径。

        Args:
            needs (`list`):
                需求清单，每项形如
                {"need": "降噪耳机", "product_ids": ["P1004", "P1005"], "quantity": 1}；
                每个需求最多选中一件（quantity 是选中后买几件，缺省 1）。
                最多 4 个需求、每个需求最多 12 个候选。
            ship_to (`str`):
                目的国代码，**只能取** "CN" / "US" / "EU" / "JP" / "SG"
                （EU 是整个欧盟，不要填 DE、FR 这类成员国代码）。
            budget_major (`float`):
                预算上限，与 target_currency 同币种，按**到手价**（含运费关税）判定。
                传 0 或不传表示不设预算，此时返回覆盖全部需求的最便宜组合。
            target_currency (`str`):
                金额口径币种，默认 "CNY"。
        """
        session_id = ShoppingContext.current_session_id()
        args = {
            "needs": needs, "ship_to": ship_to,
            "budget_major": budget_major, "target_currency": target_currency,
        }
        bus.publish(session_id, "tool.invoke", {"tool": "optimize_basket_tool", "args": args})

        try:
            if not needs:
                raise ValueError("needs 不能为空：至少要有一个需求及其候选商品")

            # 规则表支持性**先于**商品可达性判断（十期教训）：顺序反了的话，
            # 传 DE 会先撞上"某商品不可寄往 DE"，模型据此告诉买家"这些商品不发欧盟"，
            # 而真相是规则表根本没有 DE 这个目的国，模型无从自纠。
            if ship_to not in tariff.supported_destinations():
                raise ValueError(
                    f"计价规则表不支持目的国 {ship_to}"
                    f"（支持 {tariff.supported_destinations()}）。"
                    f"若买家说的是欧盟/日本/新加坡等，请改用括号里的代码重试；"
                    f"确实不在支持范围内时，如实告知买家无法计算到手价，"
                    f"不要自行估算运费或关税。",
                )

            groups: list[NeedGroup] = []
            excluded: list[dict] = []
            for raw in needs:
                if not isinstance(raw, dict) or "need" not in raw or "product_ids" not in raw:
                    raise ValueError(f"needs 元素需含 need 与 product_ids：{raw}")
                need = str(raw["need"])
                # 模型偶尔把数量当字符串传，与 product_search_tool 同样宽松处理
                quantity = int(raw.get("quantity", 1) or 1)
                product_ids = [str(pid) for pid in (raw.get("product_ids") or [])]

                found = await product_repo.find_by_ids(product_ids)
                by_id = {product.product_id: product for product in found}
                missing = [pid for pid in product_ids if pid not in by_id]
                if missing:
                    raise ValueError(f"商品不存在：{'、'.join(missing)}")

                candidates: list[NeedCandidate] = []
                for pid in product_ids:
                    product = by_id[pid]
                    if ship_to not in product.ships_to:
                        # 排除而不是报错：优化器本来就是在候选里做选择，
                        # 一个候选不可用不等于问题无解。但必须回传，否则模型
                        # 会把"这件不发美国"答成"没有这个商品"。
                        excluded.append({
                            "need": need,
                            "product_id": pid,
                            "title": product.title,
                            "reason": "not_shippable",
                            "ships_to": list(product.ships_to),
                        })
                        continue
                    sku = product.primary_sku()
                    candidates.append(NeedCandidate(
                        product_id=product.product_id,
                        title=product.title,
                        category=product.category,
                        unit_price=sku.price,
                    ))
                groups.append(NeedGroup(need=need, candidates=candidates, quantity=quantity))

            budget_value = float(budget_major or 0)
            budget = (
                Money.from_major_units(budget_value, target_currency)
                if budget_value > 0 else None
            )
            plan = optimize_basket(
                tariff, groups, ship_to=ship_to,
                target_currency=target_currency, budget=budget,
            )
            payload = {**plan.to_dict(), "excluded_candidates": excluded}
        except Exception as err:  # noqa: BLE001 —— 优化失败要让模型如实告知，不得编数字
            message = f"[error] 组合优化失败：{err}"
            bus.publish(
                session_id, "tool.result",
                {"tool": "optimize_basket_tool", "error": str(err)},
            )
            return ToolChunk(
                content=[TextBlock(type="text", text=message)],
                state=ToolResultState.ERROR,
            )

        # 事件发的就是喂给模型的那一份：少发一部分，金额出处校验扫轨迹时
        # 会把有出处的数字判成无出处（八期实测的失真源头）
        bus.publish(session_id, "tool.result", {"tool": "optimize_basket_tool", **payload})
        return ToolChunk(
            content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False))],
            state=ToolResultState.SUCCESS,
        )

    return optimize_basket_tool
