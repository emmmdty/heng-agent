# -*- coding: utf-8 -*-
"""知识库出处审计与门禁

判据从落地第一天就带下游（二十期的教训：只做"检测"不做"暴露"的判据
在真实评测里等价于不存在）。本模块钉住审计层的三件事：

1. 审计路径与运行时用同一份纯函数、同一套出处口径；
2. 门禁口径：命中一处即红，且"没东西可判"不许冒充"全对"（踩坑 33）；
3. 报错与零命中的知识库返回都不算出处——它们恰恰是张冠李戴的高发时刻。
"""
import json

from scripts.eval.audit_knowledge_provenance import (
    audit_session_knowledge,
    gate_verdict,
    summarize,
)
from scripts.eval.trace_audit import load_session

_KB_SUCCESS = {
    "tool": "category_insight_tool", "hit_count": 1,
    "insights": [{"content": "先看材质", "source": "travel-gear.md"}],
}
_KB_ERROR = {"tool": "category_insight_tool", "error": "索引未就绪"}


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
    def test_claim_without_kb_return_is_unsourced(self, tmp_path):
        path = _write_trace(
            tmp_path, "eval-kb-aaa111", "这个品类怎么挑",
            ["根据知识库，这个品类先看材质。"],
            tool_results=[{"tool": "product_search_tool", "hits": []}],
        )
        audit = audit_session_knowledge(load_session(path))
        assert len(audit.unsourced) == 1
        assert audit.kb_called is False

    def test_error_result_does_not_license_a_claim(self, tmp_path):
        path = _write_trace(
            tmp_path, "eval-kb-bbb222", "这个品类怎么挑",
            ["知识库里说这个品类先看材质。"],
            tool_results=[_KB_ERROR],
        )
        audit = audit_session_knowledge(load_session(path))
        assert not audit.clean
        assert audit.kb_called and not audit.kb_available

    def test_successful_return_covers_later_rounds(self, tmp_path):
        path = _write_trace(
            tmp_path, "eval-kb-ccc333", "这个品类怎么挑",
            ["先看材质。", "知识库里也提醒注意面料。"],
            tool_results=[_KB_SUCCESS],
        )
        audit = audit_session_knowledge(load_session(path))
        assert audit.clean
        assert audit.kb_available

    def test_runtime_warning_is_detected(self, tmp_path):
        path = _write_trace(
            tmp_path, "eval-kb-ddd444", "q", ["随便"],
            tool_results=[_KB_ERROR],
            events=[{"kind": "event", "type": "knowledge.unsourced", "payload": {
                "claims": 1, "unsourced": [{"raw": "知识库里说…"}],
            }}],
        )
        assert audit_session_knowledge(load_session(path)).runtime_flagged


class TestGate:
    def test_any_unsourced_fails(self):
        summary = {"sessions": 3, "claims": 2, "unsourced": 1,
                   "sessions_kb_called": 2, "sessions_kb_available": 1}
        assert not gate_verdict(summary).passed

    def test_no_claims_is_not_a_pass(self):
        summary = {"sessions": 5, "claims": 0, "unsourced": 0,
                   "sessions_kb_called": 0, "sessions_kb_available": 0}
        verdict = gate_verdict(summary)
        assert verdict.passed
        assert "无从判定" in verdict.reason

    def test_all_sourced_passes(self):
        summary = {"sessions": 3, "claims": 4, "unsourced": 0,
                   "sessions_kb_called": 2, "sessions_kb_available": 2}
        assert gate_verdict(summary).passed

    def test_summarize_counts_are_consistent(self, tmp_path):
        _write_trace(
            tmp_path, "eval-a-111111", "q", ["根据知识库，先看材质。"],
            tool_results=[_KB_ERROR],
        )
        _write_trace(tmp_path, "eval-b-222222", "q", ["聊点别的"])
        audits = [audit_session_knowledge(load_session(p)) for p in sorted(tmp_path.glob("*.jsonl"))]
        summary = summarize(audits)
        assert summary["sessions"] == 2
        assert summary["sessions_kb_called"] == 1
        assert summary["unsourced"] == 1
