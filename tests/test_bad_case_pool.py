# -*- coding: utf-8 -*-
"""Bad-case 标注池单测

飞轮的形状：**运行时/离线发现的失败 → 去重入池 → 人工定级 → 升级成回归用例**。

池子只有两条硬性质，两条都在这里钉住：
    1. **幂等**：同一个失败反复扫到不能反复入池，否则池子会被最容易复现的
       那几条淹掉，人根本看不到新问题。
    2. **不覆盖人工状态**：条目被人标成 promoted/wontfix 之后，再扫到同一条
       只更新"最近一次出现时间"，不能把状态打回 new——否则每扫一次，
       分诊工作就白做一次。
"""
import json

from scripts.eval.bad_case_pool import BadCase, load_pool, merge, write_pool


def _case(fingerprint="fp-1", reason="金额 ¥518 无工具出处", status="new"):
    return BadCase(
        fingerprint=fingerprint,
        source="provenance",
        session_id="eval-compare-two-6d0690",
        case_id="compare-two",
        buyer_query="两个一起多少钱",
        agent_excerpt="一起买 ¥518。",
        reason=reason,
        status=status,
    )


class TestPoolMerge:
    def test_new_case_is_added(self, tmp_path):
        path = tmp_path / "bad_cases.jsonl"
        added, updated = merge(path, [_case()])
        assert (added, updated) == (1, 0)
        assert len(load_pool(path)) == 1

    def test_same_fingerprint_is_not_duplicated(self, tmp_path):
        path = tmp_path / "bad_cases.jsonl"
        merge(path, [_case()])
        added, updated = merge(path, [_case()])
        assert (added, updated) == (0, 1)
        assert len(load_pool(path)) == 1

    def test_rescan_keeps_triaged_status(self, tmp_path):
        path = tmp_path / "bad_cases.jsonl"
        merge(path, [_case()])
        pool = load_pool(path)
        pool["fp-1"].status = "wontfix"
        write_pool(path, pool)

        merge(path, [_case(status="new")])
        assert load_pool(path)["fp-1"].status == "wontfix", "重扫不得把人工分诊结果打回"

    def test_rescan_refreshes_last_seen(self, tmp_path):
        path = tmp_path / "bad_cases.jsonl"
        merge(path, [_case()])
        first = load_pool(path)["fp-1"].last_seen_at
        merge(path, [_case()])
        assert load_pool(path)["fp-1"].last_seen_at >= first

    def test_pool_file_is_valid_jsonl(self, tmp_path):
        path = tmp_path / "bad_cases.jsonl"
        merge(path, [_case("fp-1"), _case("fp-2")])
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert {row["fingerprint"] for row in rows} == {"fp-1", "fp-2"}


class TestHarvesters:
    def test_provenance_findings_become_cases(self, tmp_path):
        from scripts.eval.bad_case_pool import from_provenance
        from scripts.eval.trace_audit import audit_session, load_session

        path = tmp_path / "eval-compare-two-x.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
            {"kind": "turn", "role": "buyer", "content": "两个一起多少钱"},
            {"kind": "turn", "role": "agent", "content": "一起买 ¥518。"},
            {"kind": "event", "type": "tool.result", "payload": {"hits": [
                {"landed_price": {"landed_total_major": 364.0}},
                {"landed_price": {"landed_total_major": 154.0}},
            ]}},
        ]) + "\n", encoding="utf-8")

        cases = from_provenance([audit_session(load_session(path))])
        assert len(cases) == 1
        assert cases[0].case_id == "compare-two", "会话 id 里带的 case 名要还原出来，便于定位"
        assert "518" in cases[0].reason
        # 买家问句是"能复现这个失败"的唯一线索，不带上的话 --promote 只能吐 TODO，
        # 飞轮的最后一步（升级成回归用例）就得靠人回去翻流水
        assert cases[0].buyer_query == "两个一起多少钱"

    def test_clean_session_yields_no_case(self, tmp_path):
        from scripts.eval.bad_case_pool import from_provenance
        from scripts.eval.trace_audit import SessionAudit

        assert from_provenance([SessionAudit("s1", 3, [], False, [])]) == []

    def test_failed_rubric_cases_become_cases(self):
        from scripts.eval.bad_case_pool import from_report

        report = {"results": [
            {"id": "compare-two", "verdict": "FAIL", "score": 0.5,
             "transcript": "[买家] 两个一起多少钱\n[Agent] 一起买 ¥518。",
             "judged": {"p0": [{"criterion": "组合总价不得是简单相加", "pass": False, "reason": "相加了"}]}},
            {"id": "search-budget", "verdict": "PASS", "score": 1.0, "transcript": "", "judged": {}},
        ]}
        cases = from_report(report)
        assert [case.case_id for case in cases] == ["compare-two"]
        assert "组合总价" in cases[0].reason

    def test_same_failure_across_runs_shares_fingerprint(self):
        from scripts.eval.bad_case_pool import from_report

        def report(score):
            return {"results": [{
                "id": "compare-two", "verdict": "FAIL", "score": score, "transcript": "x",
                "judged": {"p0": [{"criterion": "组合总价不得是简单相加", "pass": False}]},
            }]}

        assert from_report(report(0.5))[0].fingerprint == from_report(report(0.6))[0].fingerprint


class TestTriage:
    """分诊必须有命令可用，否则"人工定级"这一环只能手改 JSONL——
    手改的东西没人会去改，飞轮就停在"发现"这一步。"""

    def test_triage_by_case_id_updates_matching_entries(self, tmp_path):
        from scripts.eval.bad_case_pool import triage

        path = tmp_path / "bad_cases.jsonl"
        merge(path, [_case("fp-1"), _case("fp-2")])
        changed = triage(path, "compare-two", "wontfix", note="轨迹漏发已修，非模型问题")

        assert changed == 2
        entries = list(load_pool(path).values())
        assert {entry.status for entry in entries} == {"wontfix"}
        assert all("轨迹漏发" in entry.triage_note for entry in entries)

    def test_triage_by_fingerprint_is_precise(self, tmp_path):
        from scripts.eval.bad_case_pool import triage

        path = tmp_path / "bad_cases.jsonl"
        merge(path, [_case("fp-1"), _case("fp-2")])
        assert triage(path, "fp-1", "promoted") == 1
        assert load_pool(path)["fp-2"].status == "new"

    def test_unknown_status_is_rejected(self, tmp_path):
        import pytest

        from scripts.eval.bad_case_pool import triage

        path = tmp_path / "bad_cases.jsonl"
        merge(path, [_case()])
        with pytest.raises(ValueError, match="状态"):
            triage(path, "fp-1", "随便写的")

    def test_triage_survives_a_rescan(self, tmp_path):
        from scripts.eval.bad_case_pool import triage

        path = tmp_path / "bad_cases.jsonl"
        merge(path, [_case()])
        triage(path, "fp-1", "wontfix", note="已确认非缺陷")
        merge(path, [_case()])

        entry = load_pool(path)["fp-1"]
        assert entry.status == "wontfix"
        assert entry.triage_note == "已确认非缺陷"
