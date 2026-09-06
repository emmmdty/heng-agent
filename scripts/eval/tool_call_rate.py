# -*- coding: utf-8 -*-
"""工具调用率对比（#14 任务 C，C4 护栏的确定性验收工具）。

护栏口径（交接文档「五之一」任务 C 指标表）："'工具存在 ≠ 模型会调'回归：
同一批用例的工具调用轨迹统计（trace 审计，确定性）——不得低于全量加载基线"。

skill 渐进加载恰恰动的是"工具存在"这层（Task* 移出、阶段子集），这条护栏
就是防"schema 瘦身把模型该调的工具也瘦身没了"。纯函数 + CLI，零 LLM。

用法：
    uv run python scripts/eval/tool_call_rate.py \
        --baseline eval/report-20260905-142017.json --candidate eval/report-<C4 stamp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def count_tool_invokes(records: list[dict]) -> int:
    """一份会话流水里的 tool.invoke 事件数（别的 event 类型不计）。"""
    return sum(
        1
        for r in records
        if r.get("kind") == "event" and str(r.get("type", "")) == "tool.invoke"
    )


def count_agent_turns(records: list[dict]) -> int:
    return sum(
        1 for r in records if r.get("kind") == "turn" and r.get("role") == "agent"
    )


def case_of(session_id: str) -> str:
    """eval-<case>-<hex> → case；其余形态原样返回（ab- 会话不该出现在
    eval_regression 报告里，出现即原样展示便于人工发现）。"""
    if session_id.startswith("eval-"):
        stem = session_id[len("eval-"):]
        return stem.rsplit("-", 1)[0] if "-" in stem else stem
    return session_id


def collect(report: dict, conversations_dir: Path) -> dict[str, dict]:
    """报告 → {case: {"calls": n, "intents": n}}。

    会话选取与出处审计同一语义（sessions_from_report）；流水缺失的会话
    报错而不是静默按 0 计——把"没跑到"洗成"没调用"就是假绿。
    """
    from scripts.eval.audit_number_provenance import sessions_from_report

    sessions = sessions_from_report(report)
    files = {p.stem: p for p in Path(conversations_dir).glob("*.jsonl")}
    missing = sessions - set(files)
    if missing:
        raise SystemExit(
            f"报告里的 {len(missing)} 个会话在流水目录里找不到（另机跑测？）："
            f"{'、'.join(sorted(missing)[:5])}…"
        )
    out: dict[str, dict] = {}
    for sid in sorted(sessions):
        records = [
            json.loads(line)
            for line in files[sid].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        case = case_of(sid)
        slot = out.setdefault(case, {"calls": 0, "intents": 0})
        slot["calls"] += count_tool_invokes(records)
        slot["intents"] += count_agent_turns(records)
    return out


def compare(baseline: dict[str, dict], candidate: dict[str, dict]) -> dict:
    """护栏判定：聚合工具调用率（calls/intent）候选 ≥ 基线才算过。

    per_case 同时给出逐用例对照（候选比基线低的用例点名）——聚合达标但
    个别用例掉零也要在结论里显形，不许聚合数字把局部退化洗掉。
    """
    base_rate = (
        sum(v["calls"] for v in baseline.values()) / sum(v["intents"] for v in baseline.values())
        if sum(v["intents"] for v in baseline.values()) else None
    )
    cand_rate = (
        sum(v["calls"] for v in candidate.values()) / sum(v["intents"] for v in candidate.values())
        if sum(v["intents"] for v in candidate.values()) else None
    )
    per_case = {}
    for case in sorted(set(baseline) | set(candidate)):
        b = baseline.get(case, {"calls": 0, "intents": 0})
        c = candidate.get(case, {"calls": 0, "intents": 0})
        per_case[case] = {
            "baseline_calls": b["calls"], "candidate_calls": c["calls"],
            "dropped": c["calls"] < b["calls"],
        }
    dropped_cases = [c for c, v in per_case.items() if v["dropped"]]
    gate_ok = (
        base_rate is not None and cand_rate is not None and cand_rate >= base_rate
    )
    return {
        "baseline_rate": base_rate,
        "candidate_rate": cand_rate,
        "gate_ok": gate_ok,
        "dropped_cases": dropped_cases,
        "per_case": per_case,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="工具调用率对比（C4 护栏，零 LLM）")
    parser.add_argument("--baseline", required=True, help="基线报告 JSON（钉死的 R8）")
    parser.add_argument("--candidate", required=True, help="候选轮报告 JSON（flag on）")
    parser.add_argument("--dir", default=str(PROJECT_ROOT / "data" / "conversations"))
    args = parser.parse_args(argv)

    conversations_dir = Path(args.dir)
    base = collect(json.loads(Path(args.baseline).read_text(encoding="utf-8")), conversations_dir)
    cand = collect(json.loads(Path(args.candidate).read_text(encoding="utf-8")), conversations_dir)
    result = compare(base, cand)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if not result["gate_ok"]:
        print("⚠️ 工具调用率低于基线——护栏破，回滚（flag off）。", file=sys.stderr)
        return 1
    if result["dropped_cases"]:
        print(f"⚠️ 聚合达标但以下用例调用数下降（结论里必须点名）：{'、'.join(result['dropped_cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
