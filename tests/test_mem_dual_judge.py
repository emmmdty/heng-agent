# -*- coding: utf-8 -*-
"""B2 双 judge 证据段补跑的纯函数守卫（零网络）。

覆盖三块新胶水：配对身份核对（对不上必须拒出证据）、行索引、
补判对齐（存储 None 跳过 / 重判失败留名 / 顺序映射与 ab_run 双 judge 段
逐字节同构）。选取与一致率复用 select_dual_judge_pairs / judge_agreement
（各自已有测试），这里只钉胶水语义。
"""
import pytest

from scripts.eval.mem_dual_judge import (
    build_agreement_rows,
    index_rows,
    pair_identity,
    verify_pair_identity,
)


def _exec(case_id="c0", arm="A", idx=0, session="s-l", transcript="A词"):
    """生产形状的执行记录（build_pairs_from_executions 的输入/配对 left/right 形状）。"""
    return {
        "case_id": case_id, "arm": arm, "sample_index": idx,
        "session_id": session, "transcript": transcript,
        "ok": True, "error": "",
    }


def _pair(case_id="c0", idx=0, left="s-l", right="s-r", prompt="问句", lt="A词", rt="B词"):
    """生产形状的重建配对（ab_run.build_pairs_from_executions 输出）：
    left/right 是完整执行记录，没有 left_session/right_session——那两个键
    只在落盘序列化时才装配。夹具曾同时塞两套键，掩盖过一次 KeyError。"""
    return {
        "case_id": case_id, "pair_index": idx,
        "left": _exec(case_id, "A", idx, left, lt),
        "right": _exec(case_id, "B", idx, right, rt),
        "case_prompt_text": prompt, "prior_context": "",
    }


def _stored(case_id="c0", idx=0, ab="a", ba="b", left="s-l", right="s-r"):
    return {"case_id": case_id, "pair_index": idx, "verdict_ab": ab, "verdict_ba": ba,
            "left_session": left, "right_session": right}


def test_verify_pair_identity_passes_on_identical_sequence():
    rebuilt = [_pair("c0", 0), _pair("c0", 1, left="s-l2", right="s-r2")]
    stored = [_stored("c0", 0), _stored("c0", 1, left="s-l2", right="s-r2")]
    verify_pair_identity(rebuilt, stored)


def test_verify_pair_identity_rejects_on_divergence():
    rebuilt = [_pair("c0", 0), _pair("c1", 0)]
    stored = [_stored("c0", 0), _stored("c2", 0, left="x", right="y")]
    with pytest.raises(SystemExit, match="不一致"):
        verify_pair_identity(rebuilt, stored)


def test_verify_pair_identity_rejects_on_length():
    with pytest.raises(SystemExit, match="不一致"):
        verify_pair_identity([_pair("c0", 0)], [])


def test_index_rows_keys_on_case_and_pair():
    rows = [_stored("c0", 0), _stored("c0", 1), _stored("c1", 0)]
    idx = index_rows(rows)
    assert idx[("c0", 1)]["verdict_ab"] == "a"
    assert len(idx) == 3


async def _stub_judge(verdict_by_transcript):
    async def judge_call(prompt: str) -> str:
        for marker, verdict in verdict_by_transcript.items():
            if marker in prompt:
                return f"裁决: {verdict}\n理由: 测试"
        raise AssertionError(f"prompt 未命中任何桩标记：{prompt[:80]}")

    return judge_call


def test_build_agreement_rows_compares_both_orders():
    """ab 序 = 左A右B，ba 序 = 左B右A——顺序映射必须与 ab_run 双 judge 段同构。"""
    import asyncio

    pairs = [_pair(lt="词A", rt="词B")]
    rows_by_key = index_rows([_stored(ab="a", ba="b")])
    judge_call = asyncio.run(_stub_judge({}))

    async def fake_judge_pair(judge_call_, prompt, lt, rt, order, ground_truth, prior_context):
        # 桩：按收到的 transcript 与 order 返回可预测裁决
        if order == ("a", "b"):
            winner = "a" if lt == "词A" else "b"
        else:
            winner = "b" if lt == "词B" else "a"
        return {"winner": winner, "rationale": "", "raw": ""}

    import scripts.eval.mem_dual_judge as m
    orig = m.judge_pair
    m.judge_pair = fake_judge_pair
    try:
        rows = asyncio.run(
            build_agreement_rows(pairs, rows_by_key, judge_call, ground_truth="gt")
        )
    finally:
        m.judge_pair = orig

    assert [r["order"] for r in rows] == ["ab", "ba"]
    assert rows[0]["verdict_first"] == "a" and rows[0]["verdict_second"] == "a"
    assert rows[1]["verdict_first"] == "b" and rows[1]["verdict_second"] == "b"


def test_build_agreement_rows_skips_stored_error_cells():
    """主判段 error 格（verdict None）跳过重判、进 n_error 不烧调用。"""
    import asyncio

    pairs = [_pair("c0", 0), _pair("c1", 0)]
    rows_by_key = index_rows([
        _stored("c0", 0, ab=None, ba="b"),
        _stored("c1", 0, ab="a", ba=None),
    ])

    calls = []

    async def fake_judge_pair(judge_call_, prompt, lt, rt, order, ground_truth, prior_context):
        calls.append(order)
        return {"winner": "a", "rationale": "", "raw": ""}

    import scripts.eval.mem_dual_judge as m
    orig = m.judge_pair
    m.judge_pair = fake_judge_pair
    try:
        rows = asyncio.run(
            build_agreement_rows(pairs, rows_by_key, None, ground_truth="gt")
        )
    finally:
        m.judge_pair = orig

    assert len(rows) == 4
    skipped = [r for r in rows if r["verdict_first"] is None]
    assert len(skipped) == 2
    assert all("跳过" in r["note"] for r in skipped)
    assert len(calls) == 2  # 只补判两个有存储裁决的序


def test_build_agreement_rows_transport_failure_stays_named():
    """重判侧传输失败记 None + 留名，不塌缩成"不一致"也不炸整段。"""
    import asyncio

    pairs = [_pair()]
    rows_by_key = index_rows([_stored(ab="a", ba="b")])

    async def fake_judge_pair(*args, **kwargs):
        raise RuntimeError("网关炸了")

    import scripts.eval.mem_dual_judge as m
    orig = m.judge_pair
    m.judge_pair = fake_judge_pair
    try:
        rows = asyncio.run(
            build_agreement_rows(pairs, rows_by_key, None, ground_truth="gt")
        )
    finally:
        m.judge_pair = orig

    assert all(r["verdict_second"] is None for r in rows)
    assert all("RuntimeError" in r["note"] or "网关" in r["note"] for r in rows)


def test_pair_identity_uses_session_refs():
    assert pair_identity(_pair("c9", 3, left="L", right="R")) == ("c9", 3, "L", "R")


def test_pair_identity_accepts_stored_shape():
    stored = _stored("c9", 3, left="L", right="R")
    assert pair_identity(stored) == ("c9", 3, "L", "R")


def test_rebuild_then_verify_end_to_end():
    """端到端守卫：真实 build_pairs_from_executions 产出 → verify 对存储形状。
    回归起因：夹具曾同时带两套键，放过 pair_identity 对重建形状的 KeyError。"""
    from scripts.eval.ab_run import build_pairs_from_executions

    cases = [{"id": "c0", "queries": ["问句"]}, {"id": "c1", "queries": ["问句2"]}]
    executions = []
    for cid in ("c0", "c1"):
        for i in range(2):
            executions.append(_exec(cid, "A", i, f"{cid}-s-a{i}"))
            executions.append(_exec(cid, "B", i, f"{cid}-s-b{i}"))
    pairs, errors = build_pairs_from_executions(executions, cases, 2, "diagonal")
    assert not errors
    stored = [
        {"case_id": p["case_id"], "pair_index": p["pair_index"],
         "left_session": p["left"]["session_id"], "right_session": p["right"]["session_id"]}
        for p in pairs
    ]
    verify_pair_identity(pairs, stored)
