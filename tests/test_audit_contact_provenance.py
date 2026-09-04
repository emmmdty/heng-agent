# -*- coding: utf-8 -*-
"""收货字段出处的离线审计与门禁

二十期的教训写得很直白：**判据只做"检测"、没做"暴露"，在真实评测里等价于不存在**。
`arith.inconsistent` 当时在 app 之外零消费方——不进门禁、不进报告、没有审计脚本，
而它写进去的流水正好被另一条缺陷删掉了。

所以 `contact.unsourced` 这条判据从落地的第一天就带着它的下游：
本模块扫 `--report latest` 那一轮的流水，命中即非零退出。

**门禁口径与算式自洽相同，与金额出处不同：不设阈值、不设样本量下限，命中一处即红。**
无出处金额率是比率指标（对已有出处数字的修辞取整本来就占几个点），小样本不判定是对的；
而"编造了一个收货地址"是能指着原文说"这个地址不存在"的事实错误，
"发生了没有"不是"高了低了"（踩坑 45）。

**扫的是回复，不是订单**：地址被写进回复就已经把错误信息给了买家，
等它进到下单入参才拦就晚了（而入参那一层由 `order_provenance` 管，
它的 docstring 明确说明了为什么入参里的地址判不了）。
"""
import json


from scripts.eval.audit_contact_provenance import (
    CONTACT_WARNING_EVENT,
    audit_directory_contact,
    audit_session_contact,
    gate_verdict,
    summarize,
)
from scripts.eval.trace_audit import load_session

FABRICATED = "收货地址：您之前的记录是上海市浦东新区世纪大道100号，这次还是这个地址吗？"


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


class TestTraceCarriesContactWarnings:
    def test_runtime_event_is_loaded(self, tmp_path):
        """要能区分"运行时就告警了"与"这次补判才发现"——补判发现说明这份流水
        跑在判据落地之前，它是改动前的对照数据（同 trace_audit 的理由）。"""
        path = _write_trace(
            tmp_path, "eval-a-111111", "下单", ["随便"],
            events=[{"kind": "event", "type": CONTACT_WARNING_EVENT, "payload": {"x": 1}}],
        )
        assert load_session(path).contact_warnings == [{"x": 1}]

    def test_other_warnings_are_not_confused(self, tmp_path):
        path = _write_trace(
            tmp_path, "eval-b-222222", "下单", ["随便"],
            events=[{"kind": "event", "type": "number.unsourced", "payload": {"y": 2}}],
        )
        trace = load_session(path)
        assert trace.runtime_warnings == [{"y": 2}]
        assert trace.contact_warnings == []


class TestSessionAudit:
    def test_fabricated_address_is_caught(self, tmp_path):
        """二十期实测那一轮的复现：只检索过商品，地址是编的。"""
        path = _write_trace(
            tmp_path, "eval-clarify-000001",
            "帮我下单 2 个 LumenGo 露营灯军绿色。", [FABRICATED],
            tool_results=[{"tool": "product_search_tool", "hits": [{"product_id": "P1008"}]}],
        )
        audit = audit_session_contact(load_session(path))
        assert not audit.clean
        assert audit.unsourced[0].raw == "上海市浦东新区世纪大道100号"
        assert audit.runtime_flagged is False, "这份流水跑在判据落地之前，属补判发现"

    def test_address_the_buyer_gave_is_clean(self, tmp_path):
        path = _write_trace(
            tmp_path, "eval-ok-000002",
            "寄到上海市浦东新区世纪大道100号", ["好的，寄往上海市浦东新区世纪大道100号。"],
        )
        assert audit_session_contact(load_session(path)).clean

    def test_asking_for_the_address_is_clean(self, tmp_path):
        path = _write_trace(
            tmp_path, "eval-ok-000003",
            "帮我下单", ["还需要您提供收货地址、收件人和联系电话。"],
        )
        assert audit_session_contact(load_session(path)).clean

    def test_sources_accumulate_across_turns_in_one_session(self, tmp_path):
        """买家第 1 轮给的地址，第 3 轮复述不算编造——同运行时的会话级口径。"""
        path = _write_trace(
            tmp_path, "eval-ok-000004",
            "寄到上海市浦东新区世纪大道100号",
            ["收到。", "确认一下，寄往上海市浦东新区世纪大道100号。"],
        )
        assert audit_session_contact(load_session(path)).clean


class TestGate:
    def test_one_hit_is_enough_to_fail(self):
        """不设阈值：编造一个地址是事实错误，没有摊薄它的口径。"""
        verdict = gate_verdict({"sessions": 40, "claims": 3, "unsourced": 1})
        assert not verdict.passed

    def test_no_claims_is_not_reported_as_all_clean(self, ):
        """0 个断言算出的"0 处问题"不能冒充满分（踩坑 33 的同一条）。"""
        verdict = gate_verdict({"sessions": 40, "claims": 0, "unsourced": 0})
        assert verdict.passed and "无从判定" in verdict.reason

    def test_all_sourced_passes(self):
        verdict = gate_verdict({"sessions": 40, "claims": 5, "unsourced": 0})
        assert verdict.passed and "5" in verdict.reason


class TestDirectoryScan:
    def test_summarize_counts_across_sessions(self, tmp_path):
        _write_trace(tmp_path, "eval-x-000001", "下单", [FABRICATED])
        _write_trace(tmp_path, "eval-y-000002", "下单", ["请提供收货地址。"])
        summary = summarize(audit_directory_contact(tmp_path))
        assert summary["sessions"] == 2
        assert summary["unsourced"] == 1
        assert summary["sessions_with_findings"] == 1
