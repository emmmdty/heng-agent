# -*- coding: utf-8 -*-
"""token 成本 / 轮延迟指标脚本（二十三期清单 2）

读 data/conversations/ 流水的 usage 与 latency_ms，聚合
「每意图 completion token 分布 / 轮延迟 P50-P95」，落进贡献证明。

测试钉三类读数行为：
    1. 新流水（有 usage 字段）出完整分布；
    2. 旧流水（无 usage 字段）不算进 token 分布，但**要被点名**——
       覆盖面缩水不能静默，"没记账"和"零成本"必须分得开；
    3. --report 收敛范围与出处审计同一套语义（跟着报告记的 DATA_DIR 走）。
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.eval.audit_cost_latency import audit_directory, summarize


def _session_line(buyer: str = "b1") -> str:
    return json.dumps({"kind": "session", "buyer_id": buyer, "locale": "zh-CN", "currency": "CNY"},
                      ensure_ascii=False)


def _turn(role: str, content: str, *, latency_ms: int = 0, model: str = "",
          prompt: int | None = None, completion: int | None = None) -> str:
    record: dict = {
        "kind": "turn", "buyer_id": "b1", "role": role, "content": content,
        "model": model, "latency_ms": latency_ms, "created_at": "2026-09-04T00:00:00+00:00",
    }
    if prompt is not None:
        record["prompt_tokens"] = prompt
    if completion is not None:
        record["completion_tokens"] = completion
    return json.dumps(record, ensure_ascii=False)


def _make_dir(tmp_path: Path) -> Path:
    conv = tmp_path / "conversations"
    conv.mkdir()
    # 新流水：一次调用，usage 记账齐全
    (conv / "eval-new-1.jsonl").write_text("\n".join([
        _session_line(),
        _turn("buyer", "找个露营灯"),
        _turn("agent", "找到了。", latency_ms=8000, model="mimo-v2.5", prompt=1000, completion=500),
        json.dumps({"kind": "event", "type": "llm.usage",
                    "payload": {"model": "mimo-v2.5", "prompt_tokens": 1000,
                                "completion_tokens": 500, "total_tokens": 1500},
                    "occurred_at": ""}),
    ]), encoding="utf-8")
    # 多轮流水：一轮缓存命中（字段在、值为 0）+ 一轮回退到备用模型
    (conv / "eval-multi-2.jsonl").write_text("\n".join([
        _session_line(),
        _turn("buyer", "再找个充电器"),
        _turn("agent", "缓存命中轮。", latency_ms=2000, model="", prompt=0, completion=0),
        _turn("buyer", "下单"),
        _turn("agent", "好的。", latency_ms=15000, model="longcat-2.0", prompt=2000, completion=900),
    ]), encoding="utf-8")
    # 旧流水：没有 usage 字段（二十三期之前），latency 一直有
    (conv / "eval-old-3.jsonl").write_text("\n".join([
        _session_line(),
        _turn("buyer", "老流水"),
        _turn("agent", "老回复。", latency_ms=5000),
    ]), encoding="utf-8")
    return conv


class TestAggregation:
    def test_intent_count_and_latency_percentiles(self, tmp_path):
        audits = audit_directory(_make_dir(tmp_path))
        summary = summarize(audits)
        assert summary["sessions"] == 3
        assert summary["intents"] == 4  # agent 轮 = 意图
        # 最近邻排名法：[2000, 5000, 8000, 15000]，P90 rank=ceil(3.6)=4 → 15000
        latency = summary["latency_ms"]
        assert [latency["p50"], latency["p90"], latency["p95"], latency["max"]] == [5000, 15000, 15000, 15000]

    def test_token_distribution_excludes_unrecorded_turns(self, tmp_path):
        """旧流水的轮不计入 token 分布——把 None 当 0 会把覆盖缺口藏进分母。"""
        audits = audit_directory(_make_dir(tmp_path))
        summary = summarize(audits)
        assert summary["usage_recorded_intents"] == 3
        assert summary["intents_without_usage"] == 1
        completion = summary["completion_tokens"]
        # 记账轮的 completion：[500, 0, 900]，最近邻排名法 P50 = 500
        assert [completion["p50"], completion["p90"], completion["p95"]] == [500, 900, 900]
        assert completion["total"] == 1400
        assert summary["prompt_tokens"]["total"] == 3000

    def test_zero_token_turn_is_counted_not_hidden(self, tmp_path):
        """缓存命中轮（0/0）是"没调模型"，必须单独计数——它不是记账缺口。"""
        audits = audit_directory(_make_dir(tmp_path))
        summary = summarize(audits)
        assert summary["zero_token_intents"] == 1

    def test_model_distribution_by_turn(self, tmp_path):
        audits = audit_directory(_make_dir(tmp_path))
        summary = summarize(audits)
        # 缓存命中轮与旧流水轮都记不到模型，归入"未记账/缓存"一档
        assert summary["models"] == {"未记账/缓存": 2, "mimo-v2.5": 1, "longcat-2.0": 1}

    def test_sessions_without_usage_are_named(self, tmp_path):
        audits = audit_directory(_make_dir(tmp_path))
        summary = summarize(audits)
        assert summary["sessions_without_usage"] == ["eval-old-3"]


class TestEmptyInputs:
    def test_empty_directory_gives_zeros_not_crash(self, tmp_path):
        conv = tmp_path / "conversations"
        conv.mkdir()
        summary = summarize(audit_directory(conv))
        assert summary["intents"] == 0
        assert summary["completion_tokens"]["p50"] == 0.0

    def test_session_line_only_counts_nothing(self, tmp_path):
        conv = tmp_path / "conversations"
        conv.mkdir()
        (conv / "s1.jsonl").write_text(_session_line() + "\n", encoding="utf-8")
        summary = summarize(audit_directory(conv))
        assert summary["sessions"] == 1
        assert summary["intents"] == 0


class TestReportScoping:
    def test_scopes_to_the_reported_run(self, tmp_path):
        conv = _make_dir(tmp_path)
        report = tmp_path / "report-x.json"
        report.write_text(json.dumps({
            "health": {"data_dir": str(tmp_path)},
            "results": [{"session_id": "eval-new-1"}, {"session_id": "eval-old-3"}],
        }, ensure_ascii=False), encoding="utf-8")
        audits = audit_directory(conv, sessions={"eval-new-1", "eval-old-3"})
        summary = summarize(audits)
        assert summary["sessions"] == 2
        assert summary["intents"] == 2
        assert summary["sessions_without_usage"] == ["eval-old-3"]


class TestRender:
    def test_render_names_coverage_and_percentiles(self, tmp_path):
        from scripts.eval.audit_cost_latency import render

        audits = audit_directory(_make_dir(tmp_path))
        text = render(audits, summarize(audits))
        assert "记账覆盖 3/4" in text
        assert "eval-old-3" in text
        assert "零 token" in text
