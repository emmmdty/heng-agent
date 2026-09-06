# -*- coding: utf-8 -*-
"""分层判读（二十七期 B2 预登记口径）的纯函数守卫。

判读人群冻结在交接文档「五之一」任务 B 指标表【口径回写 · 2026-09-06】：
敏感层主判读（decisive≥30 + 显著性只作用本层）、其余为预期平局对照层
（照常报告、不作废整轮）。统计函数零新造（ab_stats 全套复用）。
"""
from scripts.eval.mem_layer_readout import LAYER_SENSITIVE, layer_of, split_rows


def test_sensitive_layer_matches_frozen_population():
    """敏感层 = 2 条现有 + 4 条 B1 新增（二十七期任务书 B2 节）。
    预登记人群再扩时这里同步——口径冻结在指标表，不在脚本里悄悄改。"""
    assert LAYER_SENSITIVE == {
        "memory-recall", "memory-forget",
        "preference-inject-multi", "preference-cross-category",
        "preference-like-drives-choice", "preference-round-override",
    }


def test_layer_of_known_cases():
    assert layer_of("memory-recall") == "sensitive"
    assert layer_of("memory-write") == "neutral"
    assert layer_of("preference-round-override") == "sensitive"
    assert layer_of("preference-conflict-cheapest-vs-dislike") == "neutral"


def test_layer_of_unknown_case_fails_loudly():
    """不在预登记人群的 case = 人群漂移——报错而不是静默归层。"""
    try:
        layer_of("no-fabrication")
    except ValueError as err:
        assert "no-fabrication" in str(err)
    else:
        raise AssertionError("未知用例应报错")


def test_split_rows_routes_by_case_id():
    rows = [
        {"case_id": "memory-recall", "verdict_ab": "b", "verdict_ba": "b"},
        {"case_id": "memory-write", "verdict_ab": "tie", "verdict_ba": "tie"},
    ]
    sensitive, neutral = split_rows(rows)
    assert [r["case_id"] for r in sensitive] == ["memory-recall"]
    assert [r["case_id"] for r in neutral] == ["memory-write"]
