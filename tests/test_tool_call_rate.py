# -*- coding: utf-8 -*-
"""C4 护栏"工具调用率不降"的纯函数守卫。

口径（交接文档任务 C 指标表）：skill 渐进加载动的是"工具存在"这层，
护栏防的是"schema 瘦身把该调的工具也瘦没了"——聚合 calls/intent 候选
不得低于基线；逐用例下降必须点名（聚合不许洗掉局部退化）。
"""
import json

from pathlib import Path

from scripts.eval.tool_call_rate import (
    case_of,
    collect,
    compare,
    count_agent_turns,
    count_tool_invokes,
)


def test_count_tool_invokes_only_counts_invoke_events():
    records = [
        {"kind": "event", "type": "tool.invoke", "payload": {}},
        {"kind": "event", "type": "tool.result", "payload": {}},
        {"kind": "event", "type": "final.result", "payload": {}},
        {"kind": "turn", "role": "agent"},
    ]
    assert count_tool_invokes(records) == 1


def test_count_agent_turns():
    records = [
        {"kind": "turn", "role": "buyer"},
        {"kind": "turn", "role": "agent"},
        {"kind": "turn", "role": "agent"},
    ]
    assert count_agent_turns(records) == 2


def test_case_of_strips_eval_prefix_and_suffix():
    assert case_of("eval-search-budget-abc123") == "search-budget"
    assert case_of("weird-session") == "weird-session"


def test_compare_gate_fails_when_candidate_rate_lower():
    baseline = {"a": {"calls": 10, "intents": 10}}   # 1.0
    candidate = {"a": {"calls": 4, "intents": 10}}   # 0.4
    result = compare(baseline, candidate)
    assert result["gate_ok"] is False
    assert result["dropped_cases"] == ["a"]


def test_compare_gate_passes_and_still_flags_local_drops():
    """聚合达标但个别用例下降必须点名——聚合数字不许洗掉局部退化。"""
    baseline = {
        "a": {"calls": 10, "intents": 10},
        "b": {"calls": 2, "intents": 2},
    }
    candidate = {
        "a": {"calls": 14, "intents": 10},  # 1.4 拉高聚合
        "b": {"calls": 0, "intents": 2},    # 掉零
    }
    result = compare(baseline, candidate)
    assert result["gate_ok"] is True
    assert result["dropped_cases"] == ["b"]


def test_compare_equal_rate_passes():
    baseline = {"a": {"calls": 5, "intents": 5}}
    candidate = {"a": {"calls": 5, "intents": 5}}
    assert compare(baseline, candidate)["gate_ok"] is True


def test_collect_maps_sessions_to_cases(tmp_path):
    from scripts.eval.tool_call_rate import collect as _collect
    conv = tmp_path / "conversations"
    conv.mkdir(parents=True)
    (conv / "eval-search-budget-aaa111.jsonl").write_text(
        "\n".join([
            json.dumps({"kind": "event", "type": "tool.invoke", "payload": {}}),
            json.dumps({"kind": "turn", "role": "agent"}),
        ]),
        encoding="utf-8",
    )
    report = {"results": [{"session_id": "eval-search-budget-aaa111"}]}
    out = _collect(report, conv)
    assert out == {"search-budget": {"calls": 1, "intents": 1}}


def test_collect_missing_session_fails_loud(tmp_path):
    from scripts.eval.tool_call_rate import collect as _collect
    report = {"results": [{"session_id": "eval-ghost-aaa111"}]}
    try:
        _collect(report, tmp_path)
    except SystemExit as err:
        assert "ghost" in str(err)
    else:
        raise AssertionError("流水缺失必须报错，不许按 0 计洗成假绿")
