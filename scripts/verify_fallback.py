# -*- coding: utf-8 -*-
"""模型回退链的真实验证 —— 把「只有单测证据」变成真上游证据。

设计演进记录四期曾如实记录一条边界：

    回退链（`model.fallback`）**只有单测证据**：真实网关限流间歇发生，本期未能主动复现。

限流不可主动复现（配额池是共享的，见踩坑档案第 8 条），但**回退链本身**可以：
把主模型指向一个不可达的 base_url，`connection error` 命中瞬时故障判据，
重试耗尽后应当回退到备用模型——而备用模型是**真实网关上的真实模型**，
会返回一条真实补全。整条链路（闸门 → 重试退避 → 回退 → 事件发布）跑的都是生产代码，
唯一被替换的是"故障从哪来"。

这比单测强在：单测里 fallback 是桩，这里 fallback 真的产出了内容；
比"等一次真限流"强在：可复现、可进 CI。

用法（需 .env 里配好 LLM_* 与 LLM_FALLBACK_MODEL）：
    uv run python scripts/verify_fallback.py
退出码：0 = 回退链成立；1 = 未回退或未发事件。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentscope.credential import OpenAICredential  # noqa: E402
from agentscope.message import TextBlock, UserMsg  # noqa: E402
from agentscope.model import OpenAIChatModel  # noqa: E402

from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot  # noqa: E402
from app.infrastructure.eventbus import TradeEventBus  # noqa: E402
from app.infrastructure.llm import ThrottledChatModel  # noqa: E402
from app.infrastructure.settings import load_settings  # noqa: E402
from app.infrastructure.throttle import GatewayThrottle  # noqa: E402

_UNREACHABLE = "http://127.0.0.1:1/v1"  # 保留端口，必定 connection error


async def main() -> int:
    settings = load_settings()
    if not settings.llm_api_key or not settings.llm_fallback_model:
        print("需要 LLM_API_KEY 与 LLM_FALLBACK_MODEL，跳过")
        return 1

    bus = TradeEventBus()
    session_id = "verify-fallback"
    queue = bus.subscribe(session_id)
    # model.fallback 按 ShoppingContext 的会话 id 路由，不设上下文事件就没有去处。
    # 这本身值得记一笔：模型层降级事件绑定在会话上下文上，
    # 若回退发生在没有会话上下文的路径（如后台预热），前端将收不到任何提示。
    ctx_token = ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id=session_id, buyer_id="verifier", locale="zh-CN", currency="CNY",
        ),
    )

    common = {"stream": False, "context_size": settings.context_size}
    # 备用模型：真实网关、真实模型
    fallback = OpenAIChatModel(
        model=settings.llm_fallback_model,
        credential=OpenAICredential(
            api_key=settings.llm_api_key, base_url=settings.llm_base_url,
        ),
        **common,
    )
    # 主模型：真实模型名，但 base_url 不可达 —— 制造可复现的瞬时故障
    model = ThrottledChatModel(
        model=settings.llm_model,
        credential=OpenAICredential(api_key=settings.llm_api_key, base_url=_UNREACHABLE),
        throttle=GatewayThrottle(settings.llm_max_concurrency, 0.0),
        fallback=fallback,
        max_transient_retries=1,
        retry_base_seconds=0.5,   # 验证脚本不需要真实退避时长
        bus=bus,
        **common,
    )

    print(f"主模型   {settings.llm_model} @ {_UNREACHABLE}（不可达）")
    print(f"备用模型 {settings.llm_fallback_model} @ {settings.llm_base_url}（真实）")
    print("发起一次真实补全…\n")

    msg = UserMsg(name="user", content=[TextBlock(type="text", text="只回复两个字：可用")])
    response = await model([msg])

    # ChatResponse 继承 DictMixin→dict，content 里的 block 可能是 dict 也可能是
    # pydantic 对象；两种都取一遍，别假设其中一种。
    text = ""
    for block in (response.get("content") if isinstance(response, dict) else None) or []:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype == "text":
            text += (block.get("text") if isinstance(block, dict) else getattr(block, "text", "")) or ""

    # 事件总线是 fire-and-forget：publish 起一个 task，不让出事件循环就读不到
    await asyncio.sleep(0.2)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    fallback_events = [e for e in events if e.type == "model.fallback"]

    print(f"备用模型返回内容：{text.strip()[:80]!r}")
    print(f"model.fallback 事件：{len(fallback_events)} 条")
    for e in fallback_events:
        print(f"  from={e.payload.get('from')} → to={e.payload.get('to')}")
        print(f"  reason={str(e.payload.get('reason'))[:110]}")

    ShoppingContext.reset(ctx_token)
    ok = bool(text.strip()) and len(fallback_events) == 1
    print("\n结论：" + ("回退链成立（备用模型产出真实内容 + 事件如实发布）" if ok
                       else "未通过——回退未发生或事件缺失"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
