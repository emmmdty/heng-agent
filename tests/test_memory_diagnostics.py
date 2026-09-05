# -*- coding: utf-8 -*-
"""内存诊断模块（二十三期 soak 分析的观测后端）

soak 首轮负结果（拐点后 RSS 仍在增长）在对象计数、线程、FD 三层都看不到
异常——只剩 tracemalloc 能指认到行号。这里测的是诊断逻辑本身：
快照槽位的时序约束（diff 前必须 take）、tracemalloc 未启用时的明确提示。
"""
import tracemalloc

from app.presentation import memory_diagnostics as diag


class TestTakeSnapshot:
    def test_object_counts_always_available(self):
        payload = diag.take_snapshot()
        assert payload["top_types"], "gc 对象计数必须始终可用"
        assert all(item["count"] > 0 for item in payload["top_types"][:3])

    def test_snapshot_flag_reflects_tracing_state(self, monkeypatch):
        monkeypatch.setattr(tracemalloc, "is_tracing", lambda: False)
        payload = diag.take_snapshot()
        assert payload["snapshot_taken"] is False
        assert "PYTHONTRACEMALLOC" in payload["hint"]


class TestDiffSnapshot:
    def test_diff_without_take_is_rejected_not_empty(self, monkeypatch):
        """空 diff 会被读成"没有增长"——比报错危险得多。"""
        monkeypatch.setattr(tracemalloc, "is_tracing", lambda: True)
        monkeypatch.setattr(diag, "_LAST_SNAPSHOT", None)
        try:
            diag.diff_snapshot()
            raise AssertionError("必须拒绝")
        except ValueError as err:
            assert "take_snapshot" in str(err)

    def test_diff_reports_growth_with_tracing(self, monkeypatch):
        """真实 tracemalloc 的端到端：take → 分配一批 → diff 必须能看到增长。"""
        monkeypatch.setattr(diag, "_LAST_SNAPSHOT", None)
        tracemalloc.start(1)
        try:
            baseline = diag.take_snapshot()
            assert baseline["snapshot_taken"] is True
            leaked = [[0] * 128 for _ in range(64)]  # 增长大户
            payload = diag.diff_snapshot()
            assert payload["top_growths"], "必须报出增长分配点"
            assert payload["top_growths"][0]["size_diff_kb"] > 0
            assert leaked  # 防优化
        finally:
            tracemalloc.stop()


class TestDebugEndpointRegistered:
    def test_memory_endpoint_is_in_openapi(self):
        from app.presentation.server import build_app

        spec = build_app().openapi()
        assert "/debug/memory" in spec["paths"]
        params = spec["paths"]["/debug/memory"]["get"]["parameters"]
        assert {p["name"] for p in params} == {"snapshot", "compare"}
