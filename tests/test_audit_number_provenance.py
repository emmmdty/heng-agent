# -*- coding: utf-8 -*-
"""金额出处审计的暴露面双指标（二十三期清单 1）

**为什么要把一个数拆成两个**：无出处金额率是比率指标，它有一个 Goodhart 口子
——模型少写解释性算术（"分开买合计 $72.95"这类自己算的数），
分子分母一起缩，比率照样好看，但那不是修复，是少干活。

所以报告必须把两个数**分开呈现**：
    无出处金额数          —— 判据抓到的真问题（门禁用的还是它）
    疑似自行算术数        —— 其中 classified 成 suspected_sum / difference /
                             basket_misadd 的部分，即"模型自己算且算错出处"的暴露面
两者的差（纯无出处）与各自的占比一起读，才能分辨
"暴露面下降"究竟是缺陷变少，还是解释变少。
"""
from app.application.harness.number_provenance import UnsourcedAmount
from scripts.eval.audit_number_provenance import render, summarize
from scripts.eval.trace_audit import SessionAudit


def _finding(value: float, kind: str) -> UnsourcedAmount:
    return UnsourcedAmount(value=value, raw=f"¥{value:g}", kind=kind)


def _audit(session_id: str, total: int, findings: list[UnsourcedAmount]) -> SessionAudit:
    return SessionAudit(
        session_id=session_id, total_amounts=total, unsourced=findings,
        runtime_flagged=False,
    )


class TestExposureDualMetric:
    def test_unsourced_and_explanatory_arithmetic_are_reported_separately(self):
        """两个数必须同时出现：只报无出处数，Goodhart 口子还在。"""
        audits = [_audit("s1", 20, [
            _finding(518.0, "suspected_sum"),
            _finding(72.95, "suspected_sum"),
            _finding(21.85, "suspected_difference"),
            _finding(999.0, "unsourced"),
            _finding(123.0, "unsourced"),
        ])]
        summary = summarize(audits)
        assert summary["unsourced_amounts"] == 5
        assert summary["explanatory_arithmetic"] == 3
        assert summary["pure_unsourced"] == 2
        assert summary["arithmetic_ratio"] == 0.15  # 3 / 20

    def test_basket_misadd_counts_as_explanatory_arithmetic(self):
        """basket_misadd 是 suspected_sum 的升级形态，同为自行算术暴露面。"""
        audits = [_audit("s1", 10, [
            _finding(518.0, "basket_misadd"),
            _finding(3.0, "unsourced"),
        ])]
        summary = summarize(audits)
        assert summary["explanatory_arithmetic"] == 1
        assert summary["pure_unsourced"] == 1

    def test_clean_sessions_zero_everything(self):
        summary = summarize([_audit("s1", 8, [])])
        assert summary["unsourced_amounts"] == 0
        assert summary["explanatory_arithmetic"] == 0
        assert summary["pure_unsourced"] == 0
        assert summary["arithmetic_ratio"] == 0.0

    def test_zero_amounts_zero_ratio(self):
        """0 处金额算 0% 会被当满分放行（踩坑 33），比率必须安全归零。"""
        summary = summarize([_audit("s1", 0, [])])
        assert summary["arithmetic_ratio"] == 0.0


class TestRenderShowsBothNumbers:
    def test_render_contains_both_metrics(self):
        audits = [_audit("s1", 20, [
            _finding(518.0, "suspected_sum"),
            _finding(999.0, "unsourced"),
        ])]
        text = render(audits, summarize(audits))
        assert "疑似自行算术 1 处" in text
        assert "纯无出处 1 处" in text
