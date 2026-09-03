# -*- coding: utf-8 -*-
"""金额出处门禁的扫描范围

要防的问题：**门禁扫的目录只增不减，历史流水会把读数永久推高。**

`data/conversations/` 是累积目录：每跑一条用例就多一份流水，旧的不会消失。
交接文档已经写明"无出处金额率只在同一轮内横向比较，不要拿两轮的绝对值当趋势"——
但门禁恰恰是拿绝对值比阈值的，扫全目录等于把这条纪律作废。

九期实测撞上：修复前那两轮的流水（含模型凭知识说的 "$800"）留在目录里，
即便新一轮已经把 $800 修没了，全量比率仍是 10.2%，`make check` 照样红。
再往后每跑一轮，分母分子一起涨，阈值只能不断往上调——门禁就废了。

所以门禁要扫"最近一轮"，范围由那一轮的报告界定，而不是靠目录里剩下什么。
"""
import json

import pytest

from scripts.eval.audit_number_provenance import (
    latest_report,
    select_audits,
    sessions_from_report,
)


class _Audit:
    def __init__(self, session_id):
        self.session_id = session_id


class TestSessionsFromReport:
    def test_reads_session_ids_of_that_run(self):
        report = {"results": [
            {"id": "a", "session_id": "eval-a-111111"},
            {"id": "b", "session_id": "eval-b-222222"},
        ]}
        assert sessions_from_report(report) == {"eval-a-111111", "eval-b-222222"}

    def test_report_without_session_ids_fails_loudly(self):
        """九期之前的报告不记 session_id。

        静默回退到全量扫描是最坏选项：门禁会拿一个被历史污染的数当本轮读数，
        而这个错读数看上去和真读数一模一样（踩坑档案第 10 条同型）。
        """
        with pytest.raises(SystemExit, match="session_id"):
            sessions_from_report({"results": [{"id": "a", "score": 1.0}]})


class TestSelectAudits:
    def test_keeps_only_that_runs_sessions(self):
        audits = [_Audit("eval-a-111111"), _Audit("eval-b-222222"), _Audit("eval-old-999999")]
        kept = select_audits(audits, {"eval-a-111111", "eval-b-222222"})
        assert [a.session_id for a in kept] == ["eval-a-111111", "eval-b-222222"]

    def test_none_scope_means_everything(self):
        audits = [_Audit("x"), _Audit("y")]
        assert select_audits(audits, None) == audits

    def test_report_naming_sessions_with_no_trace_fails_loudly(self):
        """报告说跑了这些会话，目录里却一份流水都找不到——多半是流水被清了，
        此时"0 处金额 / 0 处无出处 = 0%"会当成满分通过，比红灯更危险。"""
        with pytest.raises(SystemExit, match="找不到"):
            select_audits([_Audit("eval-old-999999")], {"eval-a-111111"})


class TestLatestReport:
    def test_picks_the_newest_report(self, tmp_path):
        for name, payload in [("report-20260903-100000.json", {"n": 1}),
                              ("report-20260903-200000.json", {"n": 2})]:
            (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
        assert latest_report(tmp_path)["n"] == 2

    def test_no_report_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="没有找到"):
            latest_report(tmp_path)


class TestMinimumSampleSize:
    """样本太小时不要用比率下判断。

    实测：`--report latest` 指向一条单用例的报告，金额总数 17 处。
    此时 1 处无出处 = 5.9%、2 处 = 11.8%，阈值 8% 恰好落在两个可能取值之间——
    **门禁的结论完全取决于模型这一轮多写了一句还是少写了一句**，等于抛硬币。

    这与踩坑 30 记下的读数纪律是同一条（n=105 时 1 条 query ≈ 0.95pt，
    单次跑出的 1pt 差异不构成结论）：那条写在文档里，这条要写进判据。

    小样本时的正确行为是**不判定**（照常打印发现，退出码 0），
    而不是放宽阈值——放宽会让整轮的真劣化也一起漏过去。
    """

    def test_small_sample_is_not_judged(self):
        from scripts.eval.audit_number_provenance import gate_verdict

        verdict = gate_verdict(
            summary={"total_amounts": 17, "unsourced_ratio": 0.118},
            max_ratio=0.08, min_amounts=30,
        )
        assert verdict.passed is True
        assert "样本量不足" in verdict.reason
        assert "17" in verdict.reason and "30" in verdict.reason

    def test_large_sample_over_threshold_fails(self):
        from scripts.eval.audit_number_provenance import gate_verdict

        verdict = gate_verdict(
            summary={"total_amounts": 150, "unsourced_ratio": 0.118},
            max_ratio=0.08, min_amounts=30,
        )
        assert verdict.passed is False
        assert "11.8%" in verdict.reason

    def test_large_sample_under_threshold_passes(self):
        from scripts.eval.audit_number_provenance import gate_verdict

        verdict = gate_verdict(
            summary={"total_amounts": 150, "unsourced_ratio": 0.05},
            max_ratio=0.08, min_amounts=30,
        )
        assert verdict.passed is True

    def test_sample_exactly_at_the_minimum_is_judged(self):
        """边界取"够了就判"，否则 min_amounts 的语义会变成"严格大于"。"""
        from scripts.eval.audit_number_provenance import gate_verdict

        verdict = gate_verdict(
            summary={"total_amounts": 30, "unsourced_ratio": 0.118},
            max_ratio=0.08, min_amounts=30,
        )
        assert verdict.passed is False


class TestAuditFollowsTheReportDataDir:
    """审计要跟着报告走，而不是死认仓库默认目录。

    对着非默认 DATA_DIR 的实例跑评测时（比如另起一个不抢 Qdrant 文件锁的实例），
    报告落在仓库 eval/、流水落在别处，门禁两头对不上——
    而原来的报错指向"流水可能被清过"，把人引向完全错误的方向。
    """

    def test_reads_data_dir_from_the_report_health_block(self, tmp_path):
        from scripts.eval.audit_number_provenance import conversations_dir_from_report

        (tmp_path / "conversations").mkdir()
        report = {"health": {"data_dir": str(tmp_path)}}
        assert conversations_dir_from_report(report) == tmp_path / "conversations"

    def test_returns_none_when_the_directory_is_gone(self, tmp_path):
        """记着但目录不在了：回落到默认，而不是拿一个不存在的路径去扫。"""
        from scripts.eval.audit_number_provenance import conversations_dir_from_report

        assert conversations_dir_from_report({"health": {"data_dir": str(tmp_path / "nope")}}) is None

    def test_returns_none_for_old_reports(self):
        """九期到十五期之间的报告不记 data_dir，不能因此报错。"""
        from scripts.eval.audit_number_provenance import conversations_dir_from_report

        assert conversations_dir_from_report({"health": {}}) is None
        assert conversations_dir_from_report({}) is None

    def test_health_reports_data_dir(self):
        import inspect

        from app.presentation import server

        assert '"data_dir": str(c.settings.data_dir)' in inspect.getsource(server)


class TestMinSessionsGuard:
    """用例数太少同样不判定——金额数与用例数是两个独立的样本量维度。

    实测：定向重跑 3 条用例的切片算出 10.2%（金额 49 处，过了金额下限），
    而同期整轮是 4.9%。那 5 处无出处全部来自其中**一条**用例的
    "超出预算多少"差额（bad-case 池里早已人工定为 wontfix）。
    比率由个别用例主导时，它不构成结论。
    """

    def _summary(self, sessions: int, total: int, unsourced: int) -> dict:
        return {
            "sessions": sessions,
            "total_amounts": total,
            "unsourced_ratio": unsourced / total if total else 0.0,
        }

    def test_small_slice_is_not_judged(self):
        from scripts.eval.audit_number_provenance import gate_verdict

        verdict = gate_verdict(self._summary(3, 49, 5), max_ratio=0.08, min_amounts=30)
        assert verdict.passed and "用例数不足" in verdict.reason

    def test_full_round_is_judged(self):
        from scripts.eval.audit_number_provenance import gate_verdict

        verdict = gate_verdict(self._summary(12, 200, 30), max_ratio=0.08, min_amounts=30)
        assert not verdict.passed, "用例数够了就该按比率判定"

    def test_amount_guard_still_applies(self):
        """两道守卫互不替代：用例数够、金额太少时仍然不判定。"""
        from scripts.eval.audit_number_provenance import gate_verdict

        verdict = gate_verdict(self._summary(12, 10, 5), max_ratio=0.08, min_amounts=30)
        assert verdict.passed and "样本量不足" in verdict.reason
