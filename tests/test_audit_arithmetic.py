# -*- coding: utf-8 -*-
"""算式自洽的离线审计与门禁

十九期把 `check_arithmetic` 接进了编排器轮末，命中就发 `arith.inconsistent` 事件。
**但事件发出去之后没有任何人读它**——不进 `make check`，不进报告，
grep 全仓只有 app 自己和单测。那份"346 条回复、命中 2 处、零误报"的读数
是一次性手工扫描，结果被硬编码成了 `test_arithmetic_check.py` 里的两个字符串，
没有可重跑的脚本。

于是这道判据在真实评测里等价于不存在：它写进流水，而流水会被清理
（见 [test_evidence_durability.py]）。护栏的验收标准在交接文档里写得很清楚——
"只有一个证据算数：它拒绝过一次，而且那次拒绝被记了下来"。
记下来还不够，得有人去读。

本模块补的就是"读"这一半，形状照抄 `audit_number_provenance.py`：
扫 `--report latest` 那一轮的流水，命中即非零退出。

**与金额出处门禁的关键差别：算式不自洽不设阈值、不设样本量下限。**
无出处金额率是比率指标（对已有出处数字的修辞取整本来就会占掉几个点），
所以要样本量才能下结论；而 `886.34 × 7.5% = 6.48` 是一次具体的事实错误，
"发生了没有"不是"高了低了"——踩坑 45 那条纪律的同一面。命中一处即红。
"""
import json

import pytest

from scripts.eval.audit_arithmetic import (
    audit_directory_arithmetic,
    audit_session_arithmetic,
    gate_verdict,
    summarize,
)
from scripts.eval.trace_audit import ARITHMETIC_WARNING_EVENT, load_session


def _write_trace(directory, session_id, replies, events=()):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session_id}.jsonl"
    lines = [json.dumps({"kind": "turn", "role": "buyer", "content": "报个价"}, ensure_ascii=False)]
    lines += [
        json.dumps({"kind": "turn", "role": "agent", "content": r}, ensure_ascii=False)
        for r in replies
    ]
    lines += [json.dumps(e, ensure_ascii=False) for e in events]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestTraceCarriesArithmeticWarnings:
    def test_runtime_event_is_loaded(self, tmp_path):
        """流水里要能区分"当时就告警了"与"这次补判才发现"——同金额出处那条。"""
        path = _write_trace(
            tmp_path, "eval-a-111111", ["随便"],
            events=[{"kind": "event", "type": ARITHMETIC_WARNING_EVENT, "payload": {"x": 1}}],
        )
        assert load_session(path).arith_warnings == [{"x": 1}]

    def test_absent_event_means_empty(self, tmp_path):
        path = _write_trace(tmp_path, "eval-b-222222", ["随便"])
        assert load_session(path).arith_warnings == []

    def test_number_provenance_warnings_are_not_confused_with_arithmetic(self, tmp_path):
        path = _write_trace(
            tmp_path, "eval-c-333333", ["随便"],
            events=[{"kind": "event", "type": "number.unsourced", "payload": {"y": 2}}],
        )
        trace = load_session(path)
        assert trace.runtime_warnings == [{"y": 2}]
        assert trace.arith_warnings == []


class TestAuditSession:
    def test_catches_the_real_defect(self, tmp_path):
        """十九期那条原文：结果对（来自工具），过程错。"""
        path = _write_trace(
            tmp_path, "eval-taxable-1", ["关税 = $886.34 × 7.5% = $6.48"],
        )
        audit = audit_session_arithmetic(load_session(path))
        assert not audit.clean
        assert audit.equations == 1
        assert audit.problems[0].expected == pytest.approx(66.4755, abs=0.01)

    def test_correct_working_is_clean(self, tmp_path):
        path = _write_trace(tmp_path, "eval-ok-1", ["应税基数 $86.34 × 7.5% ≈ $6.48"])
        audit = audit_session_arithmetic(load_session(path))
        assert audit.clean
        assert audit.equations == 1

    def test_prose_without_equations_is_clean(self, tmp_path):
        path = _write_trace(tmp_path, "eval-ok-2", ["到手价约为原价的九成"])
        audit = audit_session_arithmetic(load_session(path))
        assert audit.clean
        assert audit.equations == 0

    def test_directory_scan_sorts_by_session(self, tmp_path):
        _write_trace(tmp_path, "eval-b-2", ["关税 = $886.34 × 7.5% = $6.48"])
        _write_trace(tmp_path, "eval-a-1", ["随便"])
        audits = audit_directory_arithmetic(tmp_path)
        assert [a.session_id for a in audits] == ["eval-a-1", "eval-b-2"]


class TestGate:
    def test_any_inconsistency_fails_the_gate(self):
        """一处即红：算式错是事实错误，不是可以摊薄的比率。"""
        verdict = gate_verdict({"sessions": 1, "equations": 1, "problems": 1})
        assert not verdict.passed
        assert "1" in verdict.reason

    def test_clean_run_passes(self):
        verdict = gate_verdict({"sessions": 12, "equations": 30, "problems": 0})
        assert verdict.passed

    def test_no_equations_at_all_still_passes_but_says_so(self):
        """判据刻意窄，整轮抽不出算式是常态（346 条里只有 4 个）。

        但读数必须说清楚"这轮没东西可判"，否则 0 处问题看起来
        和"判过了、全对"一模一样——那正是金额出处门禁踩过的坑。
        """
        verdict = gate_verdict({"sessions": 12, "equations": 0, "problems": 0})
        assert verdict.passed
        assert "未抽出" in verdict.reason


class TestSummarize:
    def test_counts_across_sessions(self, tmp_path):
        _write_trace(tmp_path, "eval-a-1", ["关税 = $886.34 × 7.5% = $6.48"])
        _write_trace(tmp_path, "eval-b-2", ["$86.34 × 7.5% ≈ $6.48"])
        summary = summarize(audit_directory_arithmetic(tmp_path))
        assert summary["sessions"] == 2
        assert summary["equations"] == 2
        assert summary["problems"] == 1
        assert summary["sessions_with_findings"] == 1
