# -*- coding: utf-8 -*-
"""basket_misadd 审计与门禁

判据从落地第一天就带下游（二十期的教训：只做"检测"不做"暴露"的判据
在真实评测里等价于不存在）。本模块钉住三件事：

1. 违规判定的四个条件在审计路径上同样成立（与运行时同一份纯函数）；
2. "分开买合计"这类合法用法在真实流水的原文措辞下不被误报；
3. 门禁口径：命中一处即红，且"没东西可判"不许冒充"全对"（踩坑 33）。
"""
import json

from scripts.eval.audit_basket_sum import (
    audit_session_basket,
    gate_verdict,
    summarize,
)
from scripts.eval.trace_audit import load_session

_LANDED_HITS = {"tool": "product_search_tool", "hits": [
    {"landed_price": {"landed_total_major": 364.0}},
    {"landed_price": {"landed_total_major": 154.0}},
]}
_BASKET_QUOTE = {
    "tool": "quote_basket_tool",
    "landed_total_major": 492.0,
    "separate_purchase_landed_major": None,
}


def _write_trace(directory, session_id, buyer, replies, tool_results=(), events=()):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session_id}.jsonl"
    lines = [json.dumps({"kind": "turn", "role": "buyer", "content": buyer}, ensure_ascii=False)]
    lines += [
        json.dumps({"kind": "turn", "role": "agent", "content": r}, ensure_ascii=False)
        for r in replies
    ]
    lines += [
        json.dumps({"kind": "event", "type": "tool.result", "payload": p}, ensure_ascii=False)
        for p in tool_results
    ]
    lines += [json.dumps(e, ensure_ascii=False) for e in events]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestSessionAudit:
    def test_combined_wording_against_a_basket_quote_is_a_violation(self, tmp_path):
        path = _write_trace(
            tmp_path, "eval-compare-two-aaa111", "两个一起多少钱",
            ["两个一起下单的组合到手价 ¥518。"],
            tool_results=[_LANDED_HITS, _BASKET_QUOTE],
        )
        audit = audit_session_basket(load_session(path))
        assert [v.kind for v in audit.violations] == ["basket_misadd"]
        assert audit.has_basket_quote

    def test_separate_wording_is_not_a_violation(self, tmp_path):
        """真实流水措辞（eval-compare-two-1b9144）：分开买合计是合法用法。"""
        path = _write_trace(
            tmp_path, "eval-compare-two-bbb222", "两个一起多少钱",
            ["两件分开买合计：¥518。"],
            tool_results=[_LANDED_HITS, _BASKET_QUOTE],
        )
        assert audit_session_basket(load_session(path)).clean

    def test_no_basket_quote_means_no_verdict(self, tmp_path):
        path = _write_trace(
            tmp_path, "eval-compare-two-ccc333", "两个一起多少钱",
            ["两个一起下单合计 ¥518。"],
            tool_results=[_LANDED_HITS],
        )
        audit = audit_session_basket(load_session(path))
        assert audit.clean, "没有组合报价就没有 ground truth，只作线索不定罪"
        assert not audit.has_basket_quote

    def test_runtime_warning_is_detected(self, tmp_path):
        path = _write_trace(
            tmp_path, "eval-compare-two-ddd444", "两个一起多少钱", ["随便"],
            tool_results=[_LANDED_HITS, _BASKET_QUOTE],
            events=[{"kind": "event", "type": "number.unsourced", "payload": {
                "unsourced": [{"value": 518.0, "kind": "basket_misadd"}],
            }}],
        )
        assert audit_session_basket(load_session(path)).runtime_flagged


class TestGate:
    def test_any_violation_fails(self):
        summary = {"sessions": 3, "sessions_with_basket_quote": 2,
                   "amounts": 40, "violations": 1}
        assert not gate_verdict(summary).passed

    def test_no_basket_quotes_is_not_a_pass(self):
        """0 个组合报价的"0 违规"没有信息量，判词必须把它说破（踩坑 33）。"""
        summary = {"sessions": 5, "sessions_with_basket_quote": 0,
                   "amounts": 12, "violations": 0}
        verdict = gate_verdict(summary)
        assert verdict.passed
        assert "无从判定" in verdict.reason

    def test_clean_with_basket_quotes_passes(self):
        summary = {"sessions": 3, "sessions_with_basket_quote": 2,
                   "amounts": 40, "violations": 0}
        assert gate_verdict(summary).passed

    def test_summarize_counts_are_consistent(self, tmp_path):
        _write_trace(
            tmp_path, "eval-a-111111", "q", ["两个一起下单的组合到手价 ¥518。"],
            tool_results=[_LANDED_HITS, _BASKET_QUOTE],
        )
        _write_trace(tmp_path, "eval-b-222222", "q", ["聊点别的"], tool_results=[_LANDED_HITS])
        audits = [audit_session_basket(load_session(p)) for p in sorted(tmp_path.glob("*.jsonl"))]
        summary = summarize(audits)
        assert summary["sessions"] == 2
        assert summary["sessions_with_basket_quote"] == 1
        assert summary["violations"] == 1
