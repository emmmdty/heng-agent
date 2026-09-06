# -*- coding: utf-8 -*-
"""C1：prompt token 构成分解（#14 Skill 渐进加载，零 LLM）。

复用 audit_cost_latency 的记账口径（同一种会话选取、同一个 usage 字段），
把每意图 prompt token 下钻成四块：system prompt 固定段 / 工具 schema /
上下文历史 / 注入块。P50/P95 按用例形态分组；工具 schema 按真实 Toolkit
探针逐个计字符，并结合同一批会话里的真实调用次数给出"死重"清单。

这是 C2（skill 单元与 loader）的设计依据——没有构成数据不写 loader。

用法（零模型调用，只读流水与代码）：
    uv run python scripts/eval/skill_token_breakdown.py \
        --report eval/report-20260905-142017.json
产物：eval/skill-breakdown-<stamp>.json / .md（skill- 前缀分桶）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"

# 工具 schema 探针要建容器；Qdrant 本地存储是文件锁互斥的（运行中的服务
# 持有默认目录），探针必须落在独立目录——只读 schema，不碰向量。
PROBE_VECTOR_STORE_DIR = "/tmp/opencode/qdrant-skill-probe"

# 注入块在流水里不可见（transcript 记原始问句，不记合并后的 UserMsg）。
# 按用例形态做**有界估算**：R8 里注入只发生在跨会话记忆用例上，
# 注入条数由用例前提决定（recall=1 条 dislike；forget 撤回后剩 1 条 like）。
INJECTION_PREFS_BY_CASE = {
    "memory-recall": ("不要塑料材质",),
    "memory-forget": ("偏好军绿色",),
}


@dataclass(frozen=True)
class FixedComponents:
    system_chars: int
    tool_schema_chars: int


def case_form(session_id: str) -> str:
    """会话 id → 用例形态。eval-<case>-<hex> 归组到 case；
    ab 臂会话保留 ab() 前缀以免与 eval 会话混桶；认不出就原样返回。"""
    sid = session_id
    if sid.startswith("eval-"):
        stem = sid[len("eval-"):]
        return stem.rsplit("-", 1)[0] if "-" in stem else stem
    if sid.startswith("ab-"):
        stem = sid.rsplit("-", 1)[0]
        case = stem.split("-", 3)[-1] if stem.count("-") >= 3 else stem
        return f"ab({case})"
    return sid


def decompose_turn(
    turns: list[dict], turn_index: int, fixed: FixedComponents,
    ratio: float, injection_chars: int = 0,
) -> dict:
    """单个 agent 轮的构成分解（字符口径；token 用 ratio 折算）。

    agent 轮 i 的请求 = 固定段（system + 工具 schema）+ 历史（turns[0..i-1]
    的正文）+ 注入块（估算）。prompt_tokens 来自流水 usage，是唯一真值；
    ratio = prompt_tokens / total_chars，按轮记录、按全体取中位数折算分量。
    """
    history_chars = sum(len(t.get("content") or "") for t in turns[:turn_index])
    total = fixed.system_chars + fixed.tool_schema_chars + history_chars + injection_chars
    prompt_tokens = turns[turn_index].get("prompt_tokens")
    return {
        "system_chars": fixed.system_chars,
        "tool_schema_chars": fixed.tool_schema_chars,
        "history_chars": history_chars,
        "injection_chars": injection_chars,
        "total_chars": total,
        "prompt_tokens": prompt_tokens,
        "ratio": (prompt_tokens / total) if (prompt_tokens and total) else None,
    }


def dead_weight_tools(usage: dict[str, int], sizes: dict[str, int]) -> list[tuple[str, int]]:
    """零调用工具按 schema 大小降序——skill 化/裁剪的第一批候选。"""
    return sorted(
        ((name, size) for name, size in sizes.items() if not usage.get(name)),
        key=lambda item: -item[1],
    )


def turn_tool_chars(stream: list[dict]) -> list[int]:
    """按 agent 轮归因工具载荷字符（tool.invoke args + tool.result 全载荷）。

    流水是顺序追加的：两条 agent turn 之间的 event 属于**后一条** agent 轮
    （ReAct 循环里本轮的检索结果在本轮回复生成前进入上下文）。
    返回列表与 agent 轮一一对应（按出现顺序）。
    """
    per_agent_turn: list[int] = []
    pending = 0
    for record in stream:
        kind = record.get("kind")
        if kind == "event":
            payload = record.get("payload") or {}
            if str(record.get("type", "")).startswith("tool."):
                pending += len(json.dumps(payload, ensure_ascii=False))
        elif kind == "turn" and record.get("role") == "agent":
            per_agent_turn.append(pending)
            pending = 0
    return per_agent_turn


def floor_tokens(turn_rows: list[dict]) -> float | None:
    """固定段 token 真值：零工具、单轮 agent 轮的 prompt_tokens 取最小。

    零工具轮的 prompt = 固定段（system + 工具 schema）+ 一句问句，
    是固定段成本的直接测量——不依赖任何 chars/token 折算假设。
    没有这样的轮（全是有工具的多轮）时返回 None（不能硬估）。
    """
    candidates = [
        row["prompt_tokens"]
        for row in turn_rows
        if row.get("turns_in_session") == 1
        and not row.get("tool_payload_chars")
        and row.get("prompt_tokens")
    ]
    return min(candidates) if candidates else None


def load_session_stream(path: Path) -> list[dict]:
    """整份会话流水（turn + event，保序）——turn_tool_chars 的输入。"""
    records: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def load_session_turns(path: Path) -> list[dict]:
    """按顺序抽 turn 记录（buyer=问句正文；agent=回复正文 + usage）。"""
    turns: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("kind") != "turn":
            continue
        turns.append({
            "role": record.get("role"),
            "content": record.get("content") or "",
            "prompt_tokens": record.get("prompt_tokens"),
        })
    return turns


def tool_usage_from_session(path: Path) -> dict[str, int]:
    """一份会话流水里的工具调用计数（event 记录，载荷键宽匹配）。"""
    usage: dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("kind") != "event":
            continue
        payload = record.get("payload") or {}
        name = payload.get("tool") or payload.get("tool_name") or payload.get("name")
        if name and "tool" in str(record.get("type", "")).lower():
            usage[str(name)] = usage.get(str(name), 0) + 1
    return usage


def _p50_p95(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    ordered = sorted(values)
    p50 = ordered[len(ordered) // 2] if len(ordered) % 2 else (
        (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    )
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return p50, p95


def probe_tool_schema_chars() -> tuple[dict[str, int], int, int]:
    """真 Toolkit 探针：逐工具 schema 字符数。需要建容器（Qdrant 文件锁
    互斥——强制独立 VECTOR_STORE_DIR，见模块 docstring）。"""
    os.environ.setdefault("VECTOR_STORE_DIR", PROBE_VECTOR_STORE_DIR)
    from app.application.prompts.loader import load_prompts
    from app.composition import build_container

    async def _probe() -> tuple[dict[str, int], int, int]:
        container = await build_container()
        registry = next(
            v for v in vars(container.orchestrator).values()
            if type(v).__name__ == "SessionRegistry"
        )
        factory = next(
            v for v in vars(registry).values() if type(v).__name__ == "MainAgentFactory"
        )
        agent = factory.build()
        apis = await agent.toolkit.get_tool_schemas()
        sizes = {
            (api.get("function", api)).get("name", "?"): len(json.dumps(api, ensure_ascii=False))
            for api in apis
        }
        system = load_prompts()["main_agent"]["system_prompt"]
        return sizes, len(system), sum(sizes.values())

    return asyncio.run(_probe())


def injection_estimate(case: str) -> int:
    """按用例形态估算注入块字符数（有界：真实 hint 由已知偏好渲染）。"""
    prefs = INJECTION_PREFS_BY_CASE.get(case)
    if not prefs:
        return 0
    from app.application.memory.preference_selector import render_preference_hint
    from app.domain.buyer.preference import BuyerPreference
    return len(render_preference_hint([
        BuyerPreference(buyer_id="c1", kind="dislike", statement=p, created_at="") for p in prefs
    ]))


def analyze(report_path: Path, conversations_dir: Path) -> dict:
    """主分析：地板锚定构成分解 + 形态分组 + 死重清单。

    固定段 token 用**真值**（零工具单轮的 prompt_tokens 最小值，不靠折算）；
    历史段 = prompt_tokens − 固定段（含工具返回载荷与注入块，不再拆猜）。
    工具载荷字符按轮归因，给出历史内部"prose vs 工具载荷"的构成。
    """
    from scripts.eval.audit_number_provenance import sessions_from_report

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    sessions = sessions_from_report(report)
    conv_files = {p.stem: p for p in Path(conversations_dir).glob("*.jsonl")}
    missing = sessions - set(conv_files)
    if missing:
        raise SystemExit(
            f"报告里的 {len(missing)} 个会话在流水目录里一份都找不到（另机跑测？）"
        )

    sizes, system_chars, tool_chars = probe_tool_schema_chars()
    fixed = FixedComponents(system_chars=system_chars, tool_schema_chars=tool_chars)

    usage: dict[str, int] = {}
    rows: list[dict] = []
    for sid in sorted(sessions):
        stream = load_session_stream(conv_files[sid])
        for name, count in tool_usage_from_session(conv_files[sid]).items():
            usage[name] = usage.get(name, 0) + count
        turns = [r for r in stream if r.get("kind") == "turn"]
        tool_chars_per_turn = turn_tool_chars(stream)
        case = case_form(sid)
        inj = injection_estimate(case)
        agent_seen = 0
        for idx, turn in enumerate(turns):
            if turn["role"] != "agent":
                continue
            d = decompose_turn(turns, idx, fixed, ratio=0.0, injection_chars=inj)
            rows.append({
                "session_id": sid,
                "case": case,
                "turn_index": agent_seen,
                "turns_in_session": sum(1 for t in turns if t["role"] == "agent"),
                "prompt_tokens": turn.get("prompt_tokens"),
                "history_prose_chars": d["history_chars"],
                "injection_chars": inj,
                "tool_payload_chars": tool_chars_per_turn[agent_seen]
                if agent_seen < len(tool_chars_per_turn) else 0,
            })
            agent_seen += 1

    floor = floor_tokens(rows)
    for row in rows:
        row["fixed_tokens"] = floor
        row["history_tokens_est"] = (
            row["prompt_tokens"] - floor
            if (floor and row["prompt_tokens"] and row["prompt_tokens"] >= floor) else None
        )

    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)
    case_stats = {}
    for case, case_rows in sorted(by_case.items()):
        pts = [float(r["prompt_tokens"]) for r in case_rows if r["prompt_tokens"]]
        tools_c = [float(r["tool_payload_chars"]) for r in case_rows]
        p50, p95 = _p50_p95(pts)
        case_stats[case] = {
            "turns": len(case_rows),
            "prompt_p50": p50,
            "prompt_p95": p95,
            "tool_payload_chars_p50": statistics.median(tools_c) if tools_c else None,
            "injection_chars": case_rows[0]["injection_chars"] if case_rows else 0,
        }

    all_prompt = [float(r["prompt_tokens"]) for r in rows if r["prompt_tokens"]]
    overall_p50, overall_p95 = _p50_p95(all_prompt)
    dead = dead_weight_tools(usage, sizes)
    summary = {
        "report": str(report_path),
        "sessions": len(sessions),
        "agent_turns": len(rows),
        "fixed": {"system_chars": system_chars, "tool_schema_chars": tool_chars,
                  "tools": len(sizes)},
        "floor_tokens": floor,
        "floor_basis": "零工具、单轮 agent 轮的 prompt_tokens 最小值"
                       "（= system + 工具 schema + 一句问句的直接测量）",
        "prompt_tokens_p50": overall_p50,
        "prompt_tokens_p95": overall_p95,
        "fixed_share_of_p50": (floor / overall_p50) if (floor and overall_p50) else None,
        "dead_weight_tools": [{"tool": n, "schema_chars": s} for n, s in dead],
        "tool_usage": dict(sorted(usage.items(), key=lambda kv: -kv[1])),
        "tool_schema_sizes": dict(sorted(sizes.items(), key=lambda kv: -kv[1])),
        "by_case": case_stats,
        "injection_estimate_note": "流水不记合并后的 UserMsg；注入块按用例形态"
                                   "以已知偏好渲染估算（memory-recall=1 条 /"
                                   " memory-forget=1 条），其余形态计 0",
    }
    return {"summary": summary, "rows": rows}


def render_markdown(summary: dict) -> str:
    fixed = summary["fixed"]
    floor = summary["floor_tokens"]
    p50 = summary["prompt_tokens_p50"]
    lines = [
        "# Skill 化候选：prompt token 构成分解（C1）",
        "",
        f"- 基线：`{summary['report']}`（{summary['sessions']} 会话 / "
        f"{summary['agent_turns']} agent 轮）",
        f"- prompt token P50 **{p50:.0f}** / P95 **{summary['prompt_tokens_p95']:.0f}**"
        f"（与 `make cost --report` 同口径）",
        "",
        "## 固定段（每请求都发，与轮数无关）",
        "",
        f"- **真值 {floor:.0f} token**（{summary['floor_basis']}）"
        f"= P50 的 **{summary['fixed_share_of_p50'] * 100:.0f}%**",
        f"- 字符构成：system prompt（heng.yml 单块）{fixed['system_chars']} chars"
        f" + 工具 schema（{fixed['tools']} 个）{fixed['tool_schema_chars']} chars",
        "- schema 与 system 的 token 拆分不做均匀折算假设（中文与 JSON 密度不同）；"
        "C4 护栏轮的 bootstrap CI 才是最终口径，本表只用于排候选优先级",
        "",
        "## 历史段（随轮数增长）= prompt − 固定段",
        "",
        "P50 轮的历史约占 {:.0f}%；工具返回载荷（商品卡 JSON 等）进上下文后"
        "随轮累积——单轮检索型用例（no-fabrication 50,880）可以比无工具轮"
        "（chitchat-boundary {:.0f}）贵 7 倍，差距几乎全在工具载荷。".format(
            (1 - summary["fixed_share_of_p50"]) * 100, floor),
        "",
        "## 工具死重（同一批会话零调用的 schema）",
        "",
        "| 工具 | schema 字符 | 历史调用 |",
        "|---|---|---|",
    ]
    for item in summary["dead_weight_tools"]:
        lines.append(f"| {item['tool']} | {item['schema_chars']} | 0 |")
    lines += [
        "",
        "## 按用例形态分组（prompt token P50 / P95）",
        "",
        "| 形态 | 轮数 | P50 | P95 | 工具载荷 chars P50 | 注入估算 chars |",
        "|---|---|---|---|---|---|",
    ]
    for case, s in summary["by_case"].items():
        p50c = f"{s['prompt_p50']:.0f}" if s["prompt_p50"] else "—"
        p95c = f"{s['prompt_p95']:.0f}" if s["prompt_p95"] else "—"
        t = f"{s['tool_payload_chars_p50']:.0f}" if s["tool_payload_chars_p50"] is not None else "—"
        lines.append(f"| {case} | {s['turns']} | {p50c} | {p95c} | {t} | {s['injection_chars']} |")
    lines += [
        "",
        f"> 注入块估算口径：{summary['injection_estimate_note']}。",
        "",
        "## 工具调用频次（同一批会话）",
        "",
        "| 工具 | 调用次数 | schema 字符 |",
        "|---|---|---|",
    ]
    for name, count in summary["tool_usage"].items():
        lines.append(f"| {name} | {count} | {summary['tool_schema_sizes'].get(name, '—')} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C1：prompt token 构成分解（零 LLM）")
    parser.add_argument("--report", default=str(EVAL_DIR / "report-20260905-142017.json"),
                        help="基线报告（决定扫哪些会话；默认钉死的 R8）")
    parser.add_argument("--dir", default=str(PROJECT_ROOT / "data" / "conversations"))
    parser.add_argument("--out", default=str(EVAL_DIR), help="产物目录（skill- 前缀）")
    args = parser.parse_args(argv)

    result = analyze(Path(args.report), Path(args.dir))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out)
    json_path = out_dir / f"skill-breakdown-{stamp}.json"
    md_path = out_dir / f"skill-breakdown-{stamp}.md"
    payload = {"generated_at": stamp, **result["summary"]}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    md_path.write_text(render_markdown(result["summary"]), encoding="utf-8")

    s = result["summary"]
    print(f"# Token 成本构成（C1）\n")
    print(f"会话 {s['sessions']}｜agent 轮 {s['agent_turns']}｜"
          f"prompt P50 {s['prompt_tokens_p50']:.0f} / P95 {s['prompt_tokens_p95']:.0f}")
    print(f"固定段真值 {s['floor_tokens']:.0f} token（P50 的 "
          f"{s['fixed_share_of_p50'] * 100:.0f}%）｜"
          f"system {s['fixed']['system_chars']} chars + "
          f"工具 schema {s['fixed']['tool_schema_chars']} chars"
          f"（{s['fixed']['tools']} 个）")
    print(f"死重工具（零调用）：{', '.join(i['tool'] for i in s['dead_weight_tools']) or '无'}")
    print(f"\n报告：{md_path}\n结构化：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
