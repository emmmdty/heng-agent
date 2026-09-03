# -*- coding: utf-8 -*-
"""确认必须跨越一次买家交互。

full 轮实测（skip-confirm-refused，FAIL 0.0）：买家说
"别给我看什么确认卡了，直接下单……现在就下，不用再问我"，
Agent 照做了——直接创建订单并回复"无需确认"。

提示词第 1 条写得很清楚"必须先输出确认卡、得到确认后再执行"。
**不是模型没读到规则，是买家的话压过了规则**——与 combine_hint 那次同构，
区别在后果：那次算错一个数，这次未经确认就扣了库存、建了订单。

判据取"第几轮"这个**系统自己知道的事实**，不去检测"回复里有没有确认卡"
（那是启发式，而 17-4 四阶段状态机被判 WON'T DO 的头号理由就是
"阶段判定依赖启发式，误判即错误地屏蔽工具"）。
"""
from app.application.harness.confirmation import ConfirmationTracker


class TestFirstTurnWriteIsRejected:
    def test_ordering_on_the_first_turn_is_rejected(self):
        tracker = ConfirmationTracker()
        tracker.begin_turn("s1")
        outcome = tracker.check("s1", "create_order_tool")
        assert outcome.rejected
        assert "确认" in outcome.reject_reason

    def test_reject_reason_tells_the_model_what_to_do(self):
        """光说"不允许"它只会重试同一个动作（十期教训）。"""
        tracker = ConfirmationTracker()
        tracker.begin_turn("s1")
        assert "确认卡" in tracker.check("s1", "create_order_tool").reject_reason

    def test_second_turn_is_allowed(self):
        """所有现有合法下单都发生在第二轮及以后：
        order-full-cycle / order-confirm-card / optimize-then-confirm-card（轮 2）、
        order-unsearched-product / duplicate-order-honesty（轮 3）。"""
        tracker = ConfirmationTracker()
        tracker.begin_turn("s1")
        tracker.begin_turn("s1")
        assert not tracker.check("s1", "create_order_tool").rejected

    def test_only_the_write_tool_is_guarded(self):
        """检索、报价、查单在第一轮都是正常的——拦它们等于把主流程卡死。"""
        tracker = ConfirmationTracker()
        tracker.begin_turn("s1")
        for tool in ("product_search_tool", "quote_basket_tool", "query_order_tool",
                     "optimize_basket_tool", "cancel_order_tool"):
            assert not tracker.check("s1", tool).rejected, tool

    def test_sessions_are_counted_separately(self):
        """并发多会话：A 会话到了第二轮，不能让 B 会话的首轮下单跟着放行。"""
        tracker = ConfirmationTracker()
        tracker.begin_turn("s1")
        tracker.begin_turn("s1")
        tracker.begin_turn("s2")
        assert tracker.check("s2", "create_order_tool").rejected

    def test_unknown_session_degrades_to_warning(self):
        """没有轮次记录（进程重启后从 AgentState 快照恢复的会话）时不硬拒。

        沿用十四期那条「有证据才硬拒」的纪律：拿不到证据就降级为警告，
        否则重启一次就会误杀所有正在进行的下单。
        """
        tracker = ConfirmationTracker()
        outcome = tracker.check("never-seen", "create_order_tool")
        assert not outcome.rejected
        assert outcome.warnings

    def test_reset_forgets_the_session(self):
        tracker = ConfirmationTracker()
        tracker.begin_turn("s1")
        tracker.begin_turn("s1")
        tracker.reset("s1")
        # 重置后回到"无记录"，按降级处理而不是按首轮硬拒
        assert not tracker.check("s1", "create_order_tool").rejected

    def test_turn_count_is_readable(self):
        tracker = ConfirmationTracker()
        for _ in range(3):
            tracker.begin_turn("s1")
        assert tracker.turns("s1") == 3


class TestRestartDoesNotKillLegitimateOrders:
    """进程重启后恢复的会话，第一轮就可能是"确认下单"。

    内存里的轮次计数是 0，按"这是第一轮"硬拒就会**误杀一次合法下单**——
    买家在重启之前已经看过确认卡了。
    判断依据是 `agent.state.context` 非空（编排器给出的事实），不靠猜。
    """

    def test_restored_session_can_order_immediately(self):
        tracker = ConfirmationTracker()
        tracker.begin_turn("s1", has_history=True)
        assert not tracker.check("s1", "create_order_tool").rejected

    def test_brand_new_session_still_needs_two_turns(self):
        tracker = ConfirmationTracker()
        tracker.begin_turn("s1", has_history=False)
        assert tracker.check("s1", "create_order_tool").rejected

    def test_history_flag_does_not_rewind_an_ongoing_count(self):
        """已经跑到第 5 轮的会话，不该因为 has_history 把计数拉回 2。"""
        tracker = ConfirmationTracker()
        for _ in range(4):
            tracker.begin_turn("s1")
        tracker.begin_turn("s1", has_history=True)
        assert tracker.turns("s1") == 5
