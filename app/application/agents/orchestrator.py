# -*- coding: utf-8 -*-
"""MainAgentOrchestrator

应用层编排入口：
    1. 把会话快照写入 ShoppingContext（ContextVar，工具与子 Agent 透明可读）；
    2. 长期记忆读路径：买家偏好经 PreferenceSelector 按本轮 query 相关性挑选后，
       有变化时随本轮输入注入一条 <buyer-preferences> hint 消息（dislike 不参与截断）；
    3. 消费 MainAgent 的 reply_stream 类型化事件流并映射到 TradeEventBus：
       TextBlockDeltaEvent → token.delta
       Task* 工具结果      → plan.update（从 AgentState.tasks_context 快照）
       （业务工具的 tool.invoke / tool.result 与 agent.dispatch 由工具自身发布）
    4. 上下文压缩检测：本轮结束后 AgentState.summary 发生变化即发布 context.compressed；
    5. 上游瞬时错误（限流/并发/5xx）有界重试；
    6. 结束后发布 final.result / error，落盘 AgentState，返回最终文本。

为何重试要放在这一层：2.0 模型层只对"建流阶段"的异常重试，而 OpenAI 兼容网关常把
限流错误写在 SSE 流中间（报 openai.APIError），此时已经走出模型层重试范围，不兜底就会
整轮失败。重试期间前端可能看到重复的流式片段，final.result 到达时会被覆盖。
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from agentscope.agent import Agent
from agentscope.event import (
    TextBlockDeltaEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
)
from agentscope.message import Msg, UserMsg

from app.application.agents.main_agent import SessionRegistry
from app.application.harness.arithmetic_check import check_arithmetic
from app.application.harness.confirmation import ConfirmationTracker
from app.application.harness.drift_detector import DriftDetector
from app.application.harness.loop_detector import LoopDetector
from app.application.harness.contact_provenance import (
    SessionContactSources,
    check_contact,
)
from app.application.harness.number_provenance import SessionSources, check_reply
from app.application.harness.knowledge_provenance import (
    SessionKnowledgeSources,
    check_knowledge,
)
from app.application.memory.preference_selector import (
    PreferenceSelector,
    render_preference_hint,
    render_preference_lines,
)
from app.domain.buyer.preference import PreferenceStore
from app.domain.session.ports.conversation_store import (
    ConversationEventRecord,
    ConversationStore,
    ConversationTurn,
)
from app.domain.session.ports.session_store import SessionStore  # noqa: F401 —— 保留类型引用
from app.infrastructure.cache.semantic_cache import SemanticCache
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.budget import init_budget
from app.infrastructure.security.output_guard import audit_output
from app.infrastructure.transient import is_transient_error

logger = logging.getLogger(__name__)

# 内置 Task 计划工具名，其结果落地后向前端推送 plan.update 快照
_TASK_TOOL_NAMES = {"TaskCreate", "TaskUpdate", "TaskList", "TaskGet"}

# 上游瞬时故障判据与模型层共用同一份（app/infrastructure/transient.py），避免两处标记表漂移。
# 模型层已做一轮退避重试 + 备用模型回退，这里是最外层兜底：覆盖模型层之外
# （工具、子 Agent 调度、事件消费）招致的瞬时失败。
_MAX_TURN_RETRIES = 2
_RETRY_BASE_SECONDS = 6.0


@dataclass(frozen=True)
class SubmitIntentInput:
    shopping_session_id: str
    buyer_id: str
    locale: str
    currency: str
    raw_query: str


@dataclass(frozen=True)
class SubmitIntentOutput:
    shopping_session_id: str
    final_text: str


def _tasks_snapshot(agent: Agent) -> dict:
    tasks = agent.state.tasks_context.tasks
    return {
        "tasks": [
            {"id": task.id, "subject": task.subject, "state": str(task.state)}
            for task in tasks
        ],
    }


class MainAgentOrchestrator:
    def __init__(
        self,
        sessions: SessionRegistry,
        bus: TradeEventBus,
        preference_store: PreferenceStore,
        conversation_store: Optional[ConversationStore] = None,
        semantic_cache: Optional[SemanticCache] = None,
        output_guard_enabled: bool = True,
        loop_detector: Optional[LoopDetector] = None,
        confirmation: Optional[ConfirmationTracker] = None,
        token_budget_total: int = 0,
        drift_detector: Optional[DriftDetector] = None,
        preference_selector: Optional[PreferenceSelector] = None,
        preference_top_k: int = 5,
        number_provenance_enabled: bool = True,
    ) -> None:
        self._sessions = sessions
        self._bus = bus
        self._preference_store = preference_store
        self._conversation_store = conversation_store
        self._semantic_cache = semantic_cache
        self._output_guard_enabled = output_guard_enabled
        self._loop_detector = loop_detector
        # 确认必须跨越一次买家交互：轮次只有编排器知道（中间件在工具边界，
        # 看不到轮次），所以由这里在每轮开始时告知一次
        self._confirmation = confirmation
        self._token_budget_total = token_budget_total
        self._drift_detector = drift_detector
        # 默认 selector 不带 embedder，退化为“按时间倒序取 top_k”，单测与无凭据环境可直接跑
        self._preference_selector = preference_selector or PreferenceSelector()
        self._preference_top_k = preference_top_k
        # 会话内已注入的偏好快照，变化时才重新注入，避免每轮重复填充上下文
        self._injected_preferences: dict[str, str] = {}
        # 金额出处：默认开启且在本类内部构造，不做成"外部注入才生效"的可选依赖——
        # 那样一旦忘了在 composition 里接上，外观与"故意关掉"完全一样（BM25 的教训）。
        self._number_sources = SessionSources() if number_provenance_enabled else None
        # 收货字段出处：与金额出处同一个开关。两条判据抓的是同一条缝的两侧
        # ——那条管钱，这条管买家的个人信息（地址错了包裹寄到别人家）。
        # 同样在本类内部构造，不做成可选依赖，理由见上一行。
        self._contact_sources = (
            SessionContactSources() if number_provenance_enabled else None
        )
        # 知识库出处：同一开关。管的是"选购常识的出处从哪来"——模型把
        # 自己的常识安上"知识库"的名头说出去，买家无从分辨。同样在本类
        # 内部构造，理由同上。
        self._knowledge_sources = (
            SessionKnowledgeSources() if number_provenance_enabled else None
        )

    def _guard_final_text(self, session_id: str, text: str) -> str:
        """L4 输出审核：最终回复推给买家前脱敏内部信息。

        命中只脱敏并发一条告警事件，**不阻断回复**：
        一条误判不应让整轮对话失败。
        """
        if not self._output_guard_enabled or not text:
            return text
        safe, cleaned = audit_output(text)
        if not safe:
            logger.warning("L4 输出审核命中敏感内容，已脱敏（会话 %s）", session_id)
            self._bus.publish(session_id, "error", {"message": "输出审核命中内部信息，已脱敏后下发"})
        return cleaned

    def _begin_turn(self, session_id: str, has_history: bool = False) -> None:
        if self._confirmation is not None:
            self._confirmation.begin_turn(session_id, has_history)

    def forget_session(self, session_id: str) -> None:
        """清掉这个会话在编排器里的进程内状态。

        由 `SessionRegistry` 的淘汰回调调用：会话被挤出内存时，
        它的金额出处记录（每会话最多 4000×2 个 float）也该一起走，
        否则那份记录会一直留到进程重启。

        清掉之后该会话若再回来，出处校验会退回"无观测记录 → 只警告"这一档——
        这是十四期就设计好的降级路径（为的是 AgentState 快照恢复场景），不是新风险。
        """
        if self._number_sources is not None:
            self._number_sources.reset(session_id)
        if self._confirmation is not None:
            self._confirmation.reset(session_id)
        self._loop_detector.reset(session_id)
        if self._drift_detector is not None:
            self._drift_detector.reset(session_id)

    async def handle_intent(self, intent: SubmitIntentInput) -> SubmitIntentOutput:
        session_id = intent.shopping_session_id
        snapshot = ShoppingContextSnapshot(
            shopping_session_id=session_id,
            buyer_id=intent.buyer_id,
            locale=intent.locale,
            currency=intent.currency,
        )
        token = ShoppingContext.set(snapshot)
        started_at = time.monotonic()
        # 本轮 Token 预算（TOKEN_BUDGET_TOTAL=0时为 None，不启用四档降级）
        init_budget(self._token_budget_total)
        if self._drift_detector is not None:
            self._drift_detector.start_turn(session_id, intent.raw_query)
        # 开始录事件轨迹：既供入库，也供轮末的金额出处校验
        trace = (
            self._bus.subscribe(session_id)
            if self._conversation_store or self._number_sources is not None
            else None
        )
        cache_hit = False
        final_text = ""
        try:
            agent = await self._sessions.get_or_create(session_id)
            summary_before = agent.state.summary
            # 语义缓存：仅首轮（无历史上下文）且非写操作意图时尝试命中，命中则零模型调用
            has_history = bool(agent.state.context)
            # 轮次 +1：确认必须跨越一次买家交互（十八期），判据靠这个计数。
            # 必须放在 has_history 算出来之后——服务重启后恢复的会话内存计数是 0，
            # 而买家下一句可能正是"确认下单"，当成第一轮会误杀一次合法下单。
            self._begin_turn(session_id, has_history)
            cached = await self._lookup_cache(intent, has_history)
            if cached is not None:
                cache_hit = True
                final_text = self._guard_final_text(session_id, cached)
                self._bus.publish(session_id, "final.result", {"text": final_text})
                return SubmitIntentOutput(shopping_session_id=session_id, final_text=final_text)

            inputs = await self._build_inputs(intent, session_id)

            final_text = await self._reply_with_retry(session_id, agent, inputs)
            final_text = self._guard_final_text(session_id, final_text)
            await self._check_drift(session_id)

            self._publish_compression(session_id, agent, summary_before)
            self._bus.publish(session_id, "final.result", {"text": final_text})
            await self._remember_cache(intent, final_text, has_history)
            return SubmitIntentOutput(shopping_session_id=session_id, final_text=final_text)
        except Exception as err:  # noqa: BLE001 —— 兜底转事件，避免长任务静默失败
            logger.exception("MainAgent 异常")
            self._bus.publish(session_id, "error", {"message": str(err)})
            final_text = f"[error] {err}"
            return SubmitIntentOutput(shopping_session_id=session_id, final_text=final_text)
        finally:
            # 无论成功失败都落盘会话状态（失败内部仅告警）
            await self._sessions.persist(session_id)
            events = self._drain_trace(session_id, trace)
            if not cache_hit:
                self._check_number_provenance(intent, final_text, events)
                # 算式自洽与金额出处并列：一个管"数字从哪来"，一个管"过程算不算得通"。
                # 缓存命中的轮次同样跳过——那是上一次已校验过的回复在重放。
                self._check_arithmetic(intent, final_text, events)
                # 收货字段出处：与金额出处并列，抓的是"买家没给过的个人信息
                # 被写进了回复"。同样跳过缓存命中的轮次。
                self._check_contact_provenance(intent, final_text, events)
                # 知识库出处：抓的是"内容声称来自知识库，而本会话根本
                # 没有过成功的知识库返回"。同样跳过缓存命中的轮次。
                self._check_knowledge_provenance(intent, final_text, events)
            await self._record_conversation(
                intent, final_text, int((time.monotonic() - started_at) * 1000), events,
            )
            # 循环检测是"本轮内是不是在打转"的判定，轮末必须清零；
            # 阶段无关的顺序记录（SequencingTracker）则按会话保留，
            # 否则第 1 轮检索、第 3 轮下单会被误判为"未检索就下单"。
            if self._loop_detector is not None:
                self._loop_detector.reset(session_id)
            if self._drift_detector is not None:
                self._drift_detector.reset(session_id)
            ShoppingContext.reset(token)

    async def _preference_scope(self, buyer_id: str) -> str:
        """买家当前偏好的指纹，作为语义缓存的分桶维度。

        用**全量偏好**而不是本轮选中的子集：选中子集随 query 变，拿它做 key
        会让缓存碎成一盘沙。全量偏好只在真正新增/撤回时变，恰好是正确的失效时机。
        """
        try:
            preferences = await self._preference_store.list_by_buyer(buyer_id)
        except Exception as err:  # noqa: BLE001 —— 读不到就当无偏好，不阻断对话
            logger.warning("读取偏好指纹失败，缓存按无偏好分桶：%s", err)
            return ""
        if not preferences:
            return ""
        return hashlib.sha256(
            render_preference_lines(preferences).encode(),
        ).hexdigest()[:16]

    async def _lookup_cache(self, intent: SubmitIntentInput, has_history: bool) -> Optional[str]:
        """语义缓存查询；命中时发 cache.hit 事件让过程可见（不静默复用）。"""
        if self._semantic_cache is None:
            return None
        hit = await self._semantic_cache.lookup(
            intent.buyer_id,
            intent.raw_query,
            has_history,
            scope=await self._preference_scope(intent.buyer_id),
        )
        if hit is None:
            return None
        logger.info("语义缓存命中（%.4f）：%s", hit.similarity, intent.raw_query)
        self._bus.publish(
            intent.shopping_session_id,
            "cache.hit",
            {"similarity": hit.similarity, "matched_query": hit.matched_query},
        )
        return hit.reply

    async def _remember_cache(
        self, intent: SubmitIntentInput, final_text: str, has_history: bool,
    ) -> None:
        if self._semantic_cache is None:
            return
        await self._semantic_cache.remember(
            intent.buyer_id,
            intent.raw_query,
            final_text,
            has_history,
            scope=await self._preference_scope(intent.buyer_id),
        )

    def _drain_trace(
        self, session_id: str, trace: Optional[asyncio.Queue],
    ) -> list[ConversationEventRecord]:
        """把本轮录到的事件取出来。

        独立成一步是因为轨迹有两个消费者：入库与金额出处校验。
        队列只能被消费一次，谁先 drain 谁就把另一个饿死——所以先统一取出来，
        再分给两边用（校验产生的告警会追加进同一个列表，一并落盘）。
        """
        events: list[ConversationEventRecord] = []
        if trace is None:
            return events
        self._bus.unsubscribe(session_id, trace)
        while not trace.empty():
            event = trace.get_nowait()
            # token.delta 量大且已被 final.result 汇总，不入库
            if event.type == "token.delta":
                continue
            events.append(
                ConversationEventRecord(
                    session_id=session_id,
                    type=event.type,
                    payload=event.payload if isinstance(event.payload, dict) else {"value": event.payload},
                    occurred_at=event.occurred_at,
                ),
            )
        return events

    def _check_number_provenance(
        self,
        intent: SubmitIntentInput,
        final_text: str,
        events: list[ConversationEventRecord],
    ) -> None:
        """轮末金额出处校验：回复里的每个金额都得在工具返回或买家原话里找得到。

        命中**只告警，不改写回复**，与 L4 输出审核同一个取舍：本轮已经结束，
        改写来不及；而一次误报把正确回复打回，代价远大于一个被标记的数字。
        告警的价值在于被看见——进事件流给前端、进落盘轨迹给 bad case 回收。

        语义缓存命中的轮次跳过：那是上一次已校验过的回复在重放，本轮没有工具调用，
        照判必然全员无出处，纯噪声。
        """
        if self._number_sources is None:
            return
        session_id = intent.shopping_session_id
        self._number_sources.observe(
            session_id,
            tool_results=[event.payload for event in events if event.type == "tool.result"],
            buyer_texts=[intent.raw_query],
        )
        report = check_reply(final_text, self._number_sources.of(session_id))
        if report.clean:
            return
        payload = {"message": "回复中存在无工具出处的金额", **report.to_dict()}
        logger.warning(
            "金额出处校验命中（会话 %s）：%s",
            session_id,
            [(item.raw, item.kind, item.explain) for item in report.unsourced],
        )
        self._bus.publish(session_id, "number.unsourced", payload)
        events.append(
            ConversationEventRecord(session_id=session_id, type="number.unsourced", payload=payload),
        )

    def _check_arithmetic(
        self,
        intent: SubmitIntentInput,
        final_text: str,
        events: list[ConversationEventRecord],
    ) -> None:
        """轮末算式自洽校验：回复里写出来的 `A × B% = C` 得算得通。

        与金额出处校验是**互补的两条**，不能合并：出处管"数字从哪来"，
        这条管"写出来的过程算不算得通"。full3 实测那次，
        `886.34 × 7.5% = 6.48` 里三个数**都有工具出处**，出处校验完全无感。

        同样只告警不改写：本轮已结束，改写来不及；价值在于被看见。
        """
        report = check_arithmetic(final_text)
        if report.ok:
            return
        session_id = intent.shopping_session_id
        payload = {"message": "回复中的算式等号两边对不上", **report.to_dict()}
        logger.warning(
            "算式自洽校验命中（会话 %s）：%s",
            session_id,
            [(item.raw, item.expected) for item in report.problems],
        )
        self._bus.publish(session_id, "arith.inconsistent", payload)
        events.append(
            ConversationEventRecord(
                session_id=session_id, type="arith.inconsistent", payload=payload,
            ),
        )

    def _check_contact_provenance(
        self,
        intent: SubmitIntentInput,
        final_text: str,
        events: list[ConversationEventRecord],
    ) -> None:
        """轮末收货字段出处校验：回复里的地址/电话/邮编都得在工具返回或买家原话里找得到。

        来源（二十期整轮实测，`clarify-missing-address`）：买家只说了"帮我下单
        2 个 LumenGo 露营灯军绿色"，Agent 回复"您之前的记录是上海市浦东新区
        世纪大道100号"——**那个地址不存在于任何地方**，本轮只调用过
        `product_search_tool`，偏好库里也没有。它是编的，还安了一个出处。

        与金额出处、算式自洽并列，三条都是**只告警不改写**：本轮已经结束，
        改写来不及；而一次误报把正确回复打回，代价远大于一个被标记的字段。
        告警的价值在于被看见——进事件流给前端、进落盘轨迹给离线审计与 bad case 回收。

        缓存命中的轮次跳过：那是上一次已校验过的回复在重放，本轮没有工具调用。
        """
        if self._contact_sources is None:
            return
        session_id = intent.shopping_session_id
        self._contact_sources.observe(
            session_id,
            tool_results=[event.payload for event in events if event.type == "tool.result"],
            buyer_texts=[intent.raw_query],
        )
        report = check_contact(final_text, self._contact_sources.of(session_id))
        if report.clean:
            return
        payload = {"message": "回复中存在无出处的收货字段", **report.to_dict()}
        logger.warning(
            "收货字段出处校验命中（会话 %s）：%s",
            session_id,
            [(item.kind, item.raw) for item in report.unsourced],
        )
        self._bus.publish(session_id, "contact.unsourced", payload)
        events.append(
            ConversationEventRecord(
                session_id=session_id, type="contact.unsourced", payload=payload,
            ),
        )

    def _check_knowledge_provenance(
        self,
        intent: SubmitIntentInput,
        final_text: str,
        events: list[ConversationEventRecord],
    ) -> None:
        """轮末知识库出处校验：声称"来自知识库 / 品类洞察"就必须真有过成功返回。

        与另外三条轮末判据并列、同样**只告警不改写**（改写来不及，
        误报的代价是打回正确回复）。出处状态只认工具返回——买家原话里
        提到"知识库"不会让声明变得有据。
        """
        if self._knowledge_sources is None:
            return
        session_id = intent.shopping_session_id
        self._knowledge_sources.observe(
            session_id,
            tool_results=[event.payload for event in events if event.type == "tool.result"],
        )
        report = check_knowledge(final_text, self._knowledge_sources.of(session_id))
        if report.clean:
            return
        payload = {"message": "回复声称内容来自知识库，但本会话没有成功的知识库返回", **report.to_dict()}
        logger.warning(
            "知识库出处校验命中（会话 %s）：%s",
            session_id,
            [item.raw for item in report.unsourced],
        )
        self._bus.publish(session_id, "knowledge.unsourced", payload)
        events.append(
            ConversationEventRecord(
                session_id=session_id, type="knowledge.unsourced", payload=payload,
            ),
        )

    async def _record_conversation(
        self,
        intent: SubmitIntentInput,
        final_text: str,
        latency_ms: int,
        events: list[ConversationEventRecord],
    ) -> None:
        """对话流水 + 事件轨迹入库。写库失败只告警，不影响已经返回给买家的结果。"""
        if self._conversation_store is None:
            return
        session_id = intent.shopping_session_id
        try:
            await self._conversation_store.touch_session(
                session_id, intent.buyer_id, intent.locale, intent.currency,
            )
            await self._conversation_store.append_turn(
                ConversationTurn(
                    session_id=session_id,
                    buyer_id=intent.buyer_id,
                    role="buyer",
                    content=intent.raw_query,
                ),
            )
            await self._conversation_store.append_turn(
                ConversationTurn(
                    session_id=session_id,
                    buyer_id=intent.buyer_id,
                    role="agent",
                    content=final_text,
                    latency_ms=latency_ms,
                ),
            )
            await self._conversation_store.append_events(events)
        except Exception as err:  # noqa: BLE001
            logger.warning("对话记录写入失败：%s（%s）", session_id, err)

    async def _reply_with_retry(self, session_id: str, agent: Agent, inputs: list[Msg]) -> str:
        """跑一轮 Agent 并映射事件流；上游瞬时错误按指数退避重试。"""
        last_error: Exception | None = None
        for attempt in range(_MAX_TURN_RETRIES + 1):
            try:
                return await self._consume_reply(session_id, agent, inputs)
            except Exception as err:  # noqa: BLE001
                if not is_transient_error(err) or attempt >= _MAX_TURN_RETRIES:
                    raise
                last_error = err
                # 指数退避：网关速率类限流对固定间隔重试不敏感
                delay = _RETRY_BASE_SECONDS * (3**attempt)
                logger.warning(
                    "上游瞬时故障，%.0fs 后重试（第 %d/%d 次）：%s",
                    delay,
                    attempt + 1,
                    _MAX_TURN_RETRIES,
                    err,
                )
                self._bus.publish(
                    session_id,
                    "error",
                    {"message": f"上游瞬时故障，正在重试：{err}", "retrying": True},
                )
                # 重试时不再重复送入 inputs，避免上下文里出现两次买家发言
                inputs = []
                await asyncio.sleep(delay)
        raise last_error if last_error else RuntimeError("reply 重试耗尽")

    async def _consume_reply(self, session_id: str, agent: Agent, inputs: list[Msg]) -> str:
        final_text = ""
        # tool_call_id → 工具名，用于把 ToolResultEndEvent 关联回 Task 工具
        call_names: dict[str, str] = {}
        async for event in agent.reply_stream(inputs or None, yield_final_msg=True):
            if isinstance(event, Msg):
                final_text = event.get_text_content() or ""
            elif isinstance(event, TextBlockDeltaEvent):
                if event.delta:
                    self._bus.publish(
                        session_id,
                        "token.delta",
                        {"name": agent.name, "token": event.delta},
                    )
            elif isinstance(event, ToolCallStartEvent):
                call_names[event.tool_call_id] = event.tool_call_name
            elif isinstance(event, ToolResultEndEvent):
                tool_name = call_names.get(event.tool_call_id)
                if tool_name in _TASK_TOOL_NAMES:
                    self._bus.publish(session_id, "plan.update", _tasks_snapshot(agent))
                self._observe_for_drift(session_id, tool_name, event)
        return final_text

    def _observe_for_drift(self, session_id: str, tool_name: Optional[str], event: Any) -> None:
        """把一次工具结果记进漂移轨迹（开关关时零开销）。"""
        if self._drift_detector is None or not tool_name:
            return
        text = ""
        try:
            blocks = getattr(event, "output", None) or []
            text = "\n".join(
                str(getattr(block, "text", "") or (block.get("text", "") if isinstance(block, dict) else ""))
                for block in blocks
            )
        except Exception:  # noqa: BLE001 —— 观测不能影响主链路
            text = ""
        # “无候选”的判据：检索类工具返回的 hits 为空
        result_empty = bool(text) and ('"hits": []' in text or '"hits":[]' in text)
        self._drift_detector.observe_action(
            session_id, f"{tool_name} {text[:200]}", result_empty=result_empty,
        )

    async def _check_drift(self, session_id: str) -> None:
        """轮末漂移判定：命中只发事件 + 记日志。

        不在这里改写回复——本轮已结束，注入纠正提示已经来不及了；
        漂移信号的价值在于**被看见**（进事件流与 trace，供 bad case 回收）。
        """
        if self._drift_detector is None:
            return
        try:
            report = await self._drift_detector.check(session_id)
        except Exception as err:  # noqa: BLE001
            logger.warning("漂移检测异常，忽略：%s", err)
            return
        if report.drifted:
            logger.warning("检测到静默漂移（会话 %s）：%s", session_id, report.reasons or report.verdict)
            self._bus.publish(
                session_id,
                "error",
                {
                    "message": "检测到可能的目标漂移",
                    "reasons": report.reasons,
                    "verdict": report.verdict,
                },
            )

    def _publish_compression(self, session_id: str, agent: Agent, summary_before: str | None) -> None:
        """上下文压缩发生时，2.0 会把早期消息压成摘要写入 AgentState.summary，
        比对本轮前后的 summary 即可判定并上报。"""
        summary_after = agent.state.summary
        if not summary_after or summary_after == summary_before:
            return
        self._bus.publish(
            session_id,
            "context.compressed",
            {
                "summary_length": len(summary_after),
                "context_messages": len(agent.state.context),
            },
        )

    async def _build_inputs(self, intent: SubmitIntentInput, session_id: str) -> list[Msg]:
        """长期记忆读路径：偏好有变化时随本轮输入注入 hint 消息。"""
        user_msg = UserMsg(intent.buyer_id, intent.raw_query)
        try:
            preferences = await self._preference_store.list_by_buyer(intent.buyer_id)
        except Exception as err:  # noqa: BLE001 —— 记忆读取失败不阻断对话
            logger.warning("读取买家偏好失败：%s", err)
            preferences = []
        if not preferences:
            return [user_msg]

        # 按与本轮 query 的相关性挑选：偏好越攒越多时，全量铺进去会把真正相关的那几条稀释。
        # dislike 不参与截断（见 PreferenceSelector 文档字符串）。
        selected = await self._preference_selector.select(
            preferences, query=intent.raw_query, top_k=self._preference_top_k,
        )
        if not selected:
            return [user_msg]

        rendered = render_preference_lines(selected)
        if self._injected_preferences.get(session_id) == rendered:
            return [user_msg]
        self._injected_preferences[session_id] = rendered
        # 偏好与问句合成**一条**消息，而不是插一条独立的 hint 消息。
        # 独立消息会让上下文里连着出现两条买家发言，模型可能挑前一条当本轮的话——
        # 实测挂过：买家问"帮我推荐一套旅行三件套"，Agent 回"已为你记住不要塑料材质"，
        # 对着偏好作答、问题一个字没回（干净整轮里唯一失败的那条）。
        # 合并之后"本轮买家说了什么"只有一个答案，且问句在最后，是模型最该接住的位置。
        return [UserMsg(intent.buyer_id, f"{render_preference_hint(selected)}\n\n{intent.raw_query}")]
