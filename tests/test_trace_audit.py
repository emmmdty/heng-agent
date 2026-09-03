# -*- coding: utf-8 -*-
"""对话流水审计单测

被测对象是离线侧的两件事：
    1. 把落盘的 JSONL 流水还原成"每一轮的买家问句 / Agent 回复 / 本轮工具返回"
    2. 在还原结果上跑金额出处校验，聚合成可回归的指标

第 1 步单独测是因为它有一个不显眼的坑：JsonFileConversationStore 是先写两条
turn、再批量写 events，**落盘顺序与真实发生顺序不同**。按行顺序天真地
"边读边判"会让第一轮的工具返回全部晚于回复出现，出处集合恒为空、
无出处率恒为 100%——指标看起来还挺像样，其实全是假的。
"""
from scripts.eval.trace_audit import audit_session, load_session


def _write(tmp_path, records):
    path = tmp_path / "s1.jsonl"
    import json

    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


TRACE_WITH_SUM_ERROR = [
    {"kind": "session", "buyer_id": "b1", "locale": "zh-CN", "currency": "CNY"},
    {"kind": "turn", "role": "buyer", "content": "两个一起多少钱"},
    {"kind": "turn", "role": "agent", "content": "分别 ¥364 和 ¥154，一起买 ¥518。"},
    {"kind": "event", "type": "tool.result", "payload": {
        "tool": "product_search_tool",
        "hits": [
            {"landed_price": {"landed_total_major": 364.0}},
            {"landed_price": {"landed_total_major": 154.0}},
        ],
    }},
    {"kind": "event", "type": "final.result", "payload": {"text": "..."}},
]


class TestLoadSession:
    def test_tool_results_are_available_despite_append_order(self, tmp_path):
        """事件写在 turn 之后，还原时必须归到同一会话，而不是被判成"回复时还没有出处"。"""
        session = load_session(_write(tmp_path, TRACE_WITH_SUM_ERROR))
        assert len(session.tool_results) == 1
        assert session.buyer_texts == ["两个一起多少钱"]
        assert len(session.agent_replies) == 1

    def test_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "s1.jsonl"
        path.write_text('{"kind": "turn", "role": "buyer", "content": "hi"}\n{ 坏行\n', encoding="utf-8")
        assert load_session(path).buyer_texts == ["hi"]

    def test_missing_file_yields_empty_session(self, tmp_path):
        assert load_session(tmp_path / "nope.jsonl").agent_replies == []


class TestAuditSession:
    def test_reports_unsourced_sum(self, tmp_path):
        result = audit_session(load_session(_write(tmp_path, TRACE_WITH_SUM_ERROR)))
        assert result.total_amounts == 3
        assert [item.value for item in result.unsourced] == [518.0]
        assert result.unsourced[0].kind == "suspected_sum"

    def test_clean_session_has_no_findings(self, tmp_path):
        records = list(TRACE_WITH_SUM_ERROR)
        records[2] = {"kind": "turn", "role": "agent", "content": "分别 ¥364 和 ¥154。"}
        result = audit_session(load_session(_write(tmp_path, records)))
        assert result.unsourced == []
        assert result.total_amounts == 2

    def test_runtime_warning_is_picked_up(self, tmp_path):
        """运行时护栏已经发过 number.unsourced 的会话，离线侧要能认出来。"""
        records = TRACE_WITH_SUM_ERROR + [
            {"kind": "event", "type": "number.unsourced", "payload": {"total_amounts": 3, "unsourced": []}},
        ]
        assert audit_session(load_session(_write(tmp_path, records))).runtime_flagged is True

    def test_session_without_runtime_warning_is_marked_false(self, tmp_path):
        assert audit_session(load_session(_write(tmp_path, TRACE_WITH_SUM_ERROR))).runtime_flagged is False
