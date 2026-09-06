# -*- coding: utf-8 -*-
"""阶段路由（#14 任务 C，C3）：买家 query → 生效阶段集合。

保守铁律（宁可多带，C4 的 PASS 不回退是一票否决）：
- 关键词命中取**并集**（一句话可以同时是检索+交易）；
- 无命中 = 全阶段（闲聊/歧义轮不缺任何纪律）；
- 关键词表是预登记的确定性规则，不是模型分类——零成本、可测试、可回放。
"""
from __future__ import annotations

from .schema import Stage

_DELIVERY_STAGES = frozenset({Stage.SEARCH, Stage.TRADE, Stage.MEMORY})

# 交付三档的关键词（COMMON 不参与路由，永远生效）。
# 词表来源：cases.yaml 用例问句形态 + heng.yml 工具职责描述，人工预登记。
_STAGE_KEYWORDS: dict[Stage, tuple[str, ...]] = {
    Stage.SEARCH: (
        "推荐", "找", "搜", "有哪些", "看看", "挑", "选购", "怎么挑", "怎样",
        "一套", "一款", "哪个好", "适合", "多少钱", "价位",
    ),
    Stage.TRADE: (
        "订单", "下单", "买", "取消", "预算", "组合", "一起", "到手价", "运费",
        "关税", "免税", "确认", "库存", "地址", "包邮",
    ),
    Stage.MEMORY: (
        "记住", "记一下", "偏好", "删掉", "不用再", "忘记", "撤回", "别管",
        "以后不要", "过敏",
    ),
}


class StageRouter:
    def route(self, raw_query: str) -> frozenset[Stage]:
        query = raw_query or ""
        hits = {
            stage
            for stage, keywords in _STAGE_KEYWORDS.items()
            if any(keyword in query for keyword in keywords)
        }
        return frozenset(hits) if hits else _DELIVERY_STAGES
