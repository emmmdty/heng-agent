# -*- coding: utf-8 -*-
"""soak 内存验证工具（二十三期清单 7：十七期 LRU 的验收欠账）

十七期给四个按会话累积的东西加了 LRU 上限（SESSION_CACHE_MAX，默认 200），
但"加了上限"和"上限真的生效"是两回事——验收欠账至今：
数百会话打服务，**RSS 应当在会话数越过上限后持平**（拐点 ≈ 上限），
而不是随会话数线性上涨直到 OOM。

本工具把这件事变成可回归的读数：打 N 个独立会话，定期采样服务进程 RSS，
按"会话数 < 上限"与">= 上限"两段分别拟合每会话的 RSS 增长斜率，输出判读。

**判读是观测不是门禁**：RSS 还受索引、碎片、其他租户影响，硬性断言会把
一次环境的抖动固化成红灯。工具给出两段斜率与比值，结论由人下——
但"样本不足不判读"必须硬编码（两头各至少几个采样点才谈得上斜率）。
"""
from __future__ import annotations

import pytest

from scripts.soak_memory import (
    LRU_ASSUMED_MAX,
    parse_rss_kb,
    pick_query,
    plateau_verdict,
    slope_per_session,
)


class TestParseRss:
    def test_parses_vm_rss_from_proc_status(self):
        text = "Name:\tuvicorn\nVmRSS:\t 123456 kB\nVmHWM:\t 200000 kB\n"
        assert parse_rss_kb(text) == 123456

    def test_missing_vm_rss_returns_none(self):
        assert parse_rss_kb("Name:\tuvicorn\n") is None

    def test_garbage_returns_none(self):
        assert parse_rss_kb("VmRSS:\tnot-a-number kB") is None


class TestPickQuery:
    def test_queries_rotate_and_are_deterministic(self):
        assert pick_query(0) == pick_query(0)
        # 轮换保证不是同一句话打三百遍：不同 query 至少两种
        assert len({pick_query(i) for i in range(6)}) >= 2

    def test_queries_stay_off_the_write_path(self):
        """soak 只用读路径/闲聊 query：下单会扣库存、写订单，
        三百个 soak 会话会把商品库状态搅乱——那是对用例集的污染。"""
        for i in range(10):
            assert "下单" not in pick_query(i)
            assert "取消" not in pick_query(i)


class TestSlope:
    def test_perfect_linear_series_gives_exact_slope(self):
        # 每 10 个会话涨 1000 kB → 每 1 个会话 100 kB
        points = [(10, 10000), (20, 11000), (30, 12000), (40, 13000)]
        assert slope_per_session(points) == pytest.approx(100.0)

    def test_flat_series_gives_zero_slope(self):
        points = [(10, 10000), (20, 10000), (30, 10000)]
        assert slope_per_session(points) == pytest.approx(0.0)

    def test_fewer_than_two_points_return_zero(self):
        assert slope_per_session([(10, 10000)]) == 0.0
        assert slope_per_session([]) == 0.0


class TestPlateauVerdict:
    def test_flat_after_cap_reads_as_plateau(self):
        assert plateau_verdict(before_slope=800.0, after_slope=40.0) == "持平"

    def test_still_growing_reads_as_growth(self):
        assert plateau_verdict(before_slope=800.0, after_slope=600.0) == "仍在增长"

    def test_tiny_before_slope_is_inconclusive(self):
        """前段本身就平（或为 0），比值没有意义——不判读比硬判更诚实。"""
        assert plateau_verdict(before_slope=0.0, after_slope=0.0) == "前段无增长，无法判读"
        assert plateau_verdict(before_slope=1.0, after_slope=1.0) == "前段无增长，无法判读"


class TestLruAssumption:
    def test_assumed_max_matches_settings_default(self):
        """脚本标注的 LRU 拐点必须与服务配置同源——漂了拐点就找错地方。"""
        import dataclasses

        from app.infrastructure.settings import Settings

        field = next(f for f in dataclasses.fields(Settings) if f.name == "session_cache_max")
        assert field.default == LRU_ASSUMED_MAX


class TestProjectRoot:
    def test_eval_dir_is_the_repo_eval(self, tmp_path, monkeypatch):
        """本脚本在 scripts/ 下（不是 scripts/eval/），root 少取一级——
        首跑 300 会话的读数就栽在落盘路径指到仓库外，数据全丢。"""
        from scripts import soak_memory

        assert soak_memory.PROJECT_ROOT.name == "globex-agent"
        assert soak_memory.EVAL_DIR == soak_memory.PROJECT_ROOT / "eval"


class TestTargetProcessMatching:
    """subagent 审查 M3：多 uvicorn 实例并存时必须按端口核对目标进程——
    静默采到别的实例，斜率结论失真且无告警。"""

    def test_serves_port_matches_explicit_flag(self):
        from scripts.soak_memory import _serves_port

        args = ["/venv/bin/uvicorn", "app.presentation.server:app", "--port", "8011"]
        assert _serves_port(args, "8011") is True
        assert _serves_port(args, "8000") is False

    def test_serves_port_matches_app_string(self):
        from scripts.soak_memory import _serves_port

        assert _serves_port(["uvicorn", "app.presentation.server:app"], "8000") is False
        # 端口只认显式 --port；app 字符串里的 "server:app" 不含端口语义

    def test_port_of_parses_base_url(self):
        from scripts.soak_memory import _port_of

        assert _port_of("http://127.0.0.1:8011") == "8011"
        assert _port_of("http://127.0.0.1") == "80"
        assert _port_of("https://example.com") == "443"
