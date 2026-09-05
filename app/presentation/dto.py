# -*- coding: utf-8 -*-
"""presentation DTO

REST 请求 / 响应模型。shopping_session_id 缺省时由服务端生成。
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SubmitIntentRequest(BaseModel):
    shopping_session_id: Optional[str] = Field(default=None, description="会话 ID，缺省则新建会话")
    buyer_id: str = Field(min_length=1, description="买家 ID")
    locale: str = Field(default="zh-CN")
    currency: str = Field(default="CNY")
    raw_query: str = Field(min_length=1, description="买家自然语言购物意图")


class SubmitIntentResponse(BaseModel):
    shopping_session_id: str
    final_text: str


class CancelOrderRequest(BaseModel):
    # REST 直调方没有会话上下文，必须显式声明买家身份做归属校验
    # （红队用例挖出的洞：此前任何人报对订单号就能取消别人的订单）
    buyer_id: str = Field(min_length=1, description="买家 ID（订单归属校验）")
    reason: str = Field(min_length=1, description="取消原因")


class FaultInjectionRequest(BaseModel):
    """评测态故障注入的运行时开关（仅在 FAULT_INJECTION_ENABLED=1 的进程里存在）。

    空列表表示清空——不设单独的"清空"端点，是因为两个端点会让调用方
    多一次分支判断，而 `components: []` 已经把意图表达清楚了。
    """

    components: list[str] = []
