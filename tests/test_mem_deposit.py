# -*- coding: utf-8 -*-
"""沉淀提取与验证（#12 任务 B，M1 接线）。

提取是确定性的：只认会话流水里 tool.result 有 saved 的成功写入；验证
对照同 sample 两臂的下游 transcript（on=B 注入开 / off=A 注入关）。
假件全部按真实流水/产物 schema 造，schema 变了这里先红。
"""
import json
from pathlib import Path

import pytest

from scripts.eval.mem_deposit import (
    build_deposit,
    extract_remember_calls,
    verify_against_run,
)
from app.application.memory.deposit_store import DepositStore


def _session_file(data_dir: Path, session_id: str, buyer_id: str, lines: list[dict]) -> None:
    records = [{"kind": "session", "buyer_id": buyer_id}] + lines
    path = data_dir / "conversations"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{session_id}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8",
    )


def _write_events(statement: str = "不要塑料材质", *, kind: str = "dislike", saved: bool = True) -> list[dict]:
    return [
        {"kind": "turn", "role": "buyer", "content": "记住：我以后买东西都不要塑料材质的，我对塑料过敏。"},
        {"kind": "event", "type": "tool.invoke",
         "payload": {"tool": "remember_preference_tool", "args": {"kind": kind, "statement": statement}}},
        {"kind": "event", "type": "tool.result",
         "payload": {"tool": "remember_preference_tool", "saved": statement} if saved
         else {"tool": "remember_preference_tool", "error": "statement required"}},
        {"kind": "turn", "role": "agent", "content": "已记住"},
    ]


class TestExtractRememberCalls:
    def test_successful_write_extracted_with_trigger_query(self, tmp_path: Path):
        session_id = "ab-b-k0-memory-write-x1"
        _session_file(tmp_path, session_id, "b1", _write_events())
        writes = extract_remember_calls(session_id, tmp_path)
        assert writes == [{
            "kind": "dislike", "statement": "不要塑料材质",
            "trigger_query": "记住：我以后买东西都不要塑料材质的，我对塑料过敏。",
        }]

    def test_failed_write_is_not_a_deposit(self, tmp_path: Path):
        """tool.result 报 error 的写入没发生——编一条沉淀出来反而污染对账。"""
        session_id = "ab-b-k0-memory-write-x2"
        _session_file(tmp_path, session_id, "b1", _write_events(saved=False))
        assert extract_remember_calls(session_id, tmp_path) == []

    def test_other_tools_ignored(self, tmp_path: Path):
        session_id = "ab-b-k0-memory-write-x3"
        _session_file(tmp_path, session_id, "b1", [
            {"kind": "turn", "role": "buyer", "content": "推荐个杯子"},
            {"kind": "event", "type": "tool.invoke",
             "payload": {"tool": "product_search_tool", "args": {"query": "杯子"}}},
        ])
        assert extract_remember_calls(session_id, tmp_path) == []

    def test_missing_transcript_fails_loudly(self, tmp_path: Path):
        """执行记录声明了 session 而流水不存在 = 对账链断了，静默返回空会让
        沉淀凭空消失。必须报错。"""
        with pytest.raises(SystemExit):
            extract_remember_calls("no-such-session", tmp_path)


class TestBuildDeposit:
    def test_dislike_write_gets_result_verifier(self):
        deposit = build_deposit("memory-write", "b1", "s1", {
            "kind": "dislike", "statement": "不要塑料材质", "trigger_query": "q",
        })
        assert deposit.verifier_spec["kind"] == "product_presence"
        assert deposit.verifier_spec["expect_on"] is False
        assert deposit.verifier_spec["require_contrast"] is True
        assert "Voyager" in deposit.assertion
        assert deposit.deposit_id  # session 在构造前补齐 → id 按真实会话派生

    def test_like_write_gets_mention_verifier(self):
        deposit = build_deposit("memory-write", "b1", "s1", {
            "kind": "like", "statement": "喜欢军绿色", "trigger_query": "q",
        })
        assert deposit.verifier_spec == {"kind": "preference_mention", "keywords": ["喜欢军绿色"]}

    def test_unknown_kind_rejected(self):
        with pytest.raises(SystemExit):
            build_deposit("memory-write", "b1", "s1", {
                "kind": "habit", "statement": "x", "trigger_query": "q",
            })


def _run_json(tmp_path: Path, *, on_transcript: str, off_transcript: str, data_dir: Path) -> dict:
    """按 run_ab_pipeline 落盘 schema 造最小 run json：2 用例 × 2 臂 × k=1。"""
    write_session = "ab-b-k0-memory-write-x9"
    _session_file(data_dir, write_session, "eval-memory-buyer-abBk0", _write_events())
    return {
        "healths": {"A": {"data_dir": str(data_dir)}, "B": {"data_dir": str(data_dir)}},
        "results": [
            {"case_id": "memory-write", "arm": "B", "sample_index": 0,
             "session_id": write_session, "transcript": "已记住", "ok": True, "error": ""},
            {"case_id": "memory-write", "arm": "A", "sample_index": 0,
             "session_id": "ab-a-k0-memory-write-y9", "transcript": "已记住", "ok": True, "error": ""},
            {"case_id": "memory-recall", "arm": "B", "sample_index": 0,
             "session_id": "ab-b-k0-memory-recall-z1", "transcript": on_transcript, "ok": True, "error": ""},
            {"case_id": "memory-recall", "arm": "A", "sample_index": 0,
             "session_id": "ab-a-k0-memory-recall-z2", "transcript": off_transcript, "ok": True, "error": ""},
        ],
    }


class TestVerifyAgainstRun:
    def test_behavior_change_confirmed_and_stored(self, tmp_path: Path):
        """注入开避开 Voyager 且提到偏好、注入关推了 Voyager——行为差异确认，
        沉淀入库。这是 M1'读数符合预期'的确定性部分。"""
        run = _run_json(
            tmp_path,
            on_transcript="[Agent] 考虑到您不要塑料材质，推荐 Nomadica 帆布三件套 189 元",
            off_transcript="[Agent] 推荐 Voyager 旅行三件套 记忆棉款，139 元",
            data_dir=tmp_path,
        )
        store = DepositStore(data_dir=str(tmp_path))
        report = verify_against_run(run, tmp_path, store=store)
        assert report["n_deposits"] == 1
        assert report["verifiable_rate"] == 1.0
        assert report["n_behavior_confirmed"] == 1
        assert report["deposits"][0]["ok"] is True
        stored = store.list_by_buyer("eval-memory-buyer-abBk0")
        assert len(stored) == 1
        assert stored[0].statement == "不要塑料材质"

    def test_no_contrast_is_not_confirmed(self, tmp_path: Path):
        """两臂都没出现 Voyager = 注入没有改变该行为面——沉淀不确认、不入库
        （留名）。期望本身满足（注入开不含），缺的是与注入关的对比。"""
        run = _run_json(
            tmp_path,
            on_transcript="[Agent] 推荐 Nomadica 帆布三件套（考虑到您不要塑料）",
            off_transcript="[Agent] 推荐 Nomadica 帆布三件套",
            data_dir=tmp_path,
        )
        store = DepositStore(data_dir=str(tmp_path))
        report = verify_against_run(run, tmp_path, store=store)
        assert report["n_behavior_confirmed"] == 0
        assert "都不包含" in report["deposits"][0]["detail"]
        assert store.list_by_buyer("eval-memory-buyer-abBk0") == []

    def test_missing_downstream_product_recorded_not_crashed(self, tmp_path: Path):
        run = _run_json(tmp_path, on_transcript="x", off_transcript="y", data_dir=tmp_path)
        run["results"] = [r for r in run["results"] if r["case_id"] != "memory-recall"]
        report = verify_against_run(run, tmp_path)
        assert report["deposits"][0]["ok"] is False
        assert "缺失" in report["deposits"][0]["detail"]

    def test_unverifiable_write_aborts(self, tmp_path: Path):
        """没有预登记验证器形态的写入 = 写入门被绕过，管线缺陷直接炸。"""
        session_id = "ab-b-k0-memory-write-u1"
        _session_file(tmp_path, session_id, "b1", _write_events(kind="habit", statement="常买露营装备"))
        run = _run_json(tmp_path, on_transcript="x", off_transcript="y", data_dir=tmp_path)
        run["results"][0]["session_id"] = session_id
        with pytest.raises(SystemExit):
            verify_against_run(run, tmp_path)

    def test_arm_a_writes_not_extracted(self, tmp_path: Path):
        """沉淀只从臂 B（注入开）提取——臂 A 是反事实基线，它的写入不参与
        对账（对账语义是'注入改变了什么'，臂 A 的记忆闭环本来就不在跑）。"""
        session_a = "ab-a-k0-memory-write-w1"
        _session_file(tmp_path, session_a, "eval-memory-buyer-abAk0", _write_events())
        run = _run_json(tmp_path, on_transcript="on", off_transcript="off", data_dir=tmp_path)
        run["results"] = [r for r in run["results"] if r != {
            "case_id": "memory-write", "arm": "A", "sample_index": 0,
            "session_id": "ab-a-k0-memory-write-y9", "transcript": "已记住", "ok": True, "error": "",
        }]
        run["results"].append({
            "case_id": "memory-write", "arm": "A", "sample_index": 0,
            "session_id": session_a, "transcript": "已记住", "ok": True, "error": "",
        })
        report = verify_against_run(run, tmp_path)
        assert report["n_deposits"] == 1  # 只有臂 B 的写入进了对账
