# -*- coding: utf-8 -*-
"""C1：prompt token 构成分解（#14 Skill 渐进加载的设计依据）。

任务书口径（二十七期 C1）：下钻 prompt token 构成——system prompt 固定段 /
工具 schema / 上下文历史 / 注入块各占多少，P50/P95 分布按用例形态分组；
产出 skill 化候选片段清单。没有构成数据不写 loader（C2 的前置）。

本模块只钉**纯函数**：会话轮解析、用例形态归组、构成分解算式。
容器探针（真工具 schema）与报告渲染在脚本侧，不在这里重复。
"""
from scripts.eval.skill_token_breakdown import (
    FixedComponents,
    case_form,
    dead_weight_tools,
    decompose_turn,
    floor_tokens,
    turn_tool_chars,
)


def _turn(role, content, prompt_tokens=None):
    return {"role": role, "content": content, "prompt_tokens": prompt_tokens}


class TestCaseForm:
    def test_eval_session_strips_prefix_and_suffix(self):
        assert case_form("eval-memory-recall-5a0934") == "memory-recall"

    def test_ab_session_keeps_arm_tag(self):
        assert case_form("ab-a-k0-memory-write-06e1dc") == "ab(memory-write)"

    def test_unknown_shape_returns_as_is(self):
        assert case_form("soak-137-4fffb5") == "soak-137-4fffb5"


class TestDecomposeTurn:
    def test_first_turn_is_system_plus_tools_plus_query(self):
        fixed = FixedComponents(system_chars=1000, tool_schema_chars=2000)
        turns = [_turn("buyer", "hello"), _turn("agent", "hi", prompt_tokens=600)]
        d = decompose_turn(turns, turn_index=1, fixed=fixed, ratio=0.5)
        # history = 当前买家问句 5 chars；无前文
        assert d["history_chars"] == 5
        assert d["total_chars"] == 1000 + 2000 + 5
        assert d["prompt_tokens"] == 600
        assert d["ratio"] == 600 / 3005

    def test_history_accumulates_prior_turns(self):
        fixed = FixedComponents(system_chars=100, tool_schema_chars=100)
        turns = [
            _turn("buyer", "12345"),            # 5
            _turn("agent", "abcdef", 100),      # 6
            _turn("buyer", "xy"),               # 2
            _turn("agent", "z", 200),           # 1
        ]
        d = decompose_turn(turns, turn_index=3, fixed=fixed, ratio=0.5)
        # 第 4 轮（agent）的请求历史 = 前 3 条 + 本轮买家问句 = 5+6+2
        assert d["history_chars"] == 13

    def test_injection_counted_when_estimated(self):
        fixed = FixedComponents(system_chars=10, tool_schema_chars=10)
        turns = [_turn("buyer", "q"), _turn("agent", "a", 100)]
        d = decompose_turn(turns, turn_index=1, fixed=fixed, ratio=0.5, injection_chars=40)
        assert d["injection_chars"] == 40
        assert d["total_chars"] == 10 + 10 + 1 + 40


class TestDeadWeightTools:
    def test_zero_use_tools_sorted_by_schema_size(self):
        usage = {"a": 10, "b": 0, "c": 0, "d": 3}
        sizes = {"a": 500, "b": 900, "c": 100, "d": 50}
        dead = dead_weight_tools(usage, sizes)
        assert dead == [("b", 900), ("c", 100)]

    def test_no_dead_weight_when_all_used(self):
        assert dead_weight_tools({"a": 1}, {"a": 100}) == []


class TestFloorAnchored:
    def test_floor_is_min_single_turn_zero_tool_prompt(self):
        """固定段真值 = 无工具、单轮、prompt_tokens 最小的那轮
        （零工具轮的 prompt ≈ 固定段 + 一句问句）。"""
        rows = [
            {"case": "a", "turns_in_session": 1, "tool_payload_chars": 0, "prompt_tokens": 9000},
            {"case": "b", "turns_in_session": 1, "tool_payload_chars": 0, "prompt_tokens": 7094},
            {"case": "c", "turns_in_session": 3, "tool_payload_chars": 4000, "prompt_tokens": 20000},
            {"case": "d", "turns_in_session": 1, "tool_payload_chars": 800, "prompt_tokens": 6500},
        ]
        assert floor_tokens(rows) == 7094

    def test_floor_returns_none_without_zero_tool_single_turn(self):
        rows = [{"case": "c", "turns_in_session": 3, "tool_payload_chars": 4000, "prompt_tokens": 20000}]
        assert floor_tokens(rows) is None

    def test_turn_tool_chars_splits_events_between_agent_turns(self):
        """agent 轮 i 的工具载荷 = 它之前、上一轮之后的 event 载荷之和
        （ReAct 循环里的检索结果在本轮回复前进入上下文）。"""
        stream = [
            {"kind": "turn", "role": "buyer", "content": "q1"},
            {"kind": "event", "type": "tool.invoke", "payload": {"tool": "t", "args": {"x": "12345"}}},
            {"kind": "event", "type": "tool.result", "payload": {"tool": "t", "hit_count": 3}},
            {"kind": "turn", "role": "agent", "content": "a1", "prompt_tokens": 100},
            {"kind": "turn", "role": "buyer", "content": "q2"},
            {"kind": "turn", "role": "agent", "content": "a2", "prompt_tokens": 200},
        ]
        per_turn = turn_tool_chars(stream)
        assert per_turn[0] > 0        # a1 的轮里有一次工具载荷
        assert per_turn[1] == 0       # a2 之前没有任何 event
