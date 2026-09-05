# -*- coding: utf-8 -*-
"""token 成本 / 轮延迟指标（二十三期清单 2）

用法：
    uv run python scripts/eval/audit_cost_latency.py
    uv run python scripts/eval/audit_cost_latency.py --report latest
    uv run python scripts/eval/audit_cost_latency.py --json

读 data/conversations/ 流水，聚合两个生产声明里的硬缺口：
    每意图 completion token 分布（P50-P95）
    轮延迟分布（P50-P95）

口径：
    意图 = agent 轮。一次买家发言到最终回复算一个意图，
    多模型调用（工具循环、续调用）求和进同一个意图——轮次就是自然边界，
    编排器已把 llm.usage 求和写上 turn，这里不做事件-轮次的相关
    （落盘顺序不等于发生顺序，地雷 10）。

    记账覆盖必须单独报告：二十三期之前的流水没有 usage 字段，
    把"没记账"混进 token 分布，等于把覆盖缺口藏进分母——
    所以旧流水只进延迟分布，token 分布只统计记账轮，并把缺口点名。

    零 token 轮 = 缓存命中（字段在、值为 0），它不是记账缺口，
    单独计数。"零成本"和"没记账"是两种读数。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.eval.audit_number_provenance import (  # noqa: E402
    conversations_dir_from_report,
    latest_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "conversations"


@dataclass
class IntentAudit:
    """一个意图（agent 轮）的读数。usage 缺席与零值是两种读数，分开记。"""

    session_id: str
    latency_ms: int = 0
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class SessionCostAudit:
    session_id: str
    intents: list[IntentAudit] = field(default_factory=list)


def load_session(path: Path) -> SessionCostAudit:
    """读一份会话流水，抽 agent 轮的 latency 与 usage。

    与 trace_audit 同一条纪律：先整份读完再判——不过这里只读 turn，
    不做跨行相关，顺序问题不构成误判源。
    """
    audit = SessionCostAudit(session_id=Path(path).stem)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("kind") != "turn" or record.get("role") != "agent":
            continue
        audit.intents.append(IntentAudit(
            session_id=audit.session_id,
            latency_ms=int(record.get("latency_ms") or 0),
            model=str(record.get("model") or ""),
            # 脏数据强转：字段存在但不是数（手改的流水）不该让排序语义漂移
            prompt_tokens=_int_or_none(record.get("prompt_tokens")),
            completion_tokens=_int_or_none(record.get("completion_tokens")),
        ))
    return audit


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def audit_directory(
    conversations_dir: Path, sessions: set[str] | None = None,
) -> list[SessionCostAudit]:
    """扫一个 conversations 目录，按会话 id 排序返回。

    sessions 给定时收敛到该范围；范围与目录对不上时报错而不是给空读数
    （0 意图算出的 0 延迟会被当成满分放行，比红灯更危险）。
    """
    paths = sorted(Path(conversations_dir).glob("*.jsonl"))
    if sessions is not None:
        by_id = {path.stem: path for path in paths}
        missing = sessions - set(by_id)
        if missing:
            raise SystemExit(
                f"报告里的 {len(sessions)} 个会话在流水目录里一份都找不到"
                f"（如 {sorted(missing)[0]}）。可能的原因按概率排：\n"
                f"  1. 这份报告是对着**另一个 DATA_DIR** 的实例跑的"
                f"——用 --dir <那个实例的 data/conversations> 重扫；\n"
                f"  2. 流水被清过——重跑一轮再扫",
            )
        paths = [by_id[sid] for sid in sorted(sessions)]
    return [load_session(path) for path in paths]


def _percentile(values: list[int], q: float) -> float:
    """最近邻排名法：取排序后第 ceil(q/100·n) 个（1-based）。

    不用插值：单轮样本本来就小，插值算出的中间数在对账时对不上任何一条
    真实流水——诊断类输出宁可少说，不可说错（经验 9 的同一面）。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return float(ordered[rank - 1])


def _dist(values: list[int], keys: tuple[str, ...] = ("p50", "p90", "p95")) -> dict:
    if not values:
        return {key: 0.0 for key in keys} | {"mean": 0.0, "max": 0, "total": 0}
    return {
        **{key: _percentile(values, {"p50": 50, "p90": 90, "p95": 95}[key]) for key in keys},
        "mean": round(sum(values) / len(values), 1),
        "max": max(values),
        "total": sum(values),
    }


def summarize(audits: list[SessionCostAudit]) -> dict:
    intents = [intent for audit in audits for intent in audit.intents]
    recorded = [i for i in intents if i.prompt_tokens is not None and i.completion_tokens is not None]
    # 显式判据，不做对象相等性反查：dataclass 相等性去重既脆又 O(n²)，
    # 流水目录是累积的，规模只涨不跌
    unrecorded = [i for i in intents if i.prompt_tokens is None or i.completion_tokens is None]
    zero_token = [i for i in recorded if i.prompt_tokens == 0 and i.completion_tokens == 0]
    models: dict[str, int] = {}
    for intent in intents:
        key = intent.model if intent.model else "未记账/缓存"
        models[key] = models.get(key, 0) + 1
    return {
        "sessions": len(audits),
        "intents": len(intents),
        "usage_recorded_intents": len(recorded),
        "intents_without_usage": len(unrecorded),
        "zero_token_intents": len(zero_token),
        "completion_tokens": _dist([i.completion_tokens for i in recorded if i.completion_tokens is not None]),
        "prompt_tokens": _dist(
            [i.prompt_tokens for i in recorded if i.prompt_tokens is not None],
            keys=("p50", "p95"),
        ),
        "latency_ms": _dist([i.latency_ms for i in intents]),
        "models": dict(sorted(models.items(), key=lambda kv: -kv[1])),
        "sessions_without_usage": sorted({
            audit.session_id for audit in audits
            if any(i.prompt_tokens is None or i.completion_tokens is None for i in audit.intents)
        }),
    }


def render(audits: list[SessionCostAudit], summary: dict) -> str:
    lines = ["# Token 成本与轮延迟", ""]
    lines.append(
        f"会话 {summary['sessions']} 个｜意图（agent 轮）{summary['intents']} 个｜"
        f"记账覆盖 {summary['usage_recorded_intents']}/{summary['intents']}"
        f"（旧流水无 usage 字段，不计入 token 分布）",
    )
    completion = summary["completion_tokens"]
    prompt = summary["prompt_tokens"]
    latency = summary["latency_ms"]
    lines.append(
        f"每意图 completion tokens: P50 {completion['p50']:.0f} / P90 {completion['p90']:.0f}"
        f" / P95 {completion['p95']:.0f}｜均值 {completion['mean']:.0f}｜最大 {completion['max']}"
        f"｜合计 {completion['total']}",
    )
    lines.append(
        f"每意图 prompt tokens:     P50 {prompt['p50']:.0f} / P95 {prompt['p95']:.0f}"
        f"｜合计 {prompt['total']}",
    )
    lines.append(
        f"每意图轮延迟:             P50 {latency['p50'] / 1000:.1f}s"
        f" / P90 {latency['p90'] / 1000:.1f}s / P95 {latency['p95'] / 1000:.1f}s"
        f"｜最大 {latency['max'] / 1000:.1f}s",
    )
    if summary["models"]:
        lines.append("模型分布（按 turn）：" + "，".join(
            [f"{name} ×{count}" for name, count in summary["models"].items()],
        ))
    if summary["zero_token_intents"]:
        lines.append(
            f"零 token 轮（缓存命中，非记账缺口）：{summary['zero_token_intents']} 个",
        )
    if summary["sessions_without_usage"]:
        lines.append("")
        lines.append(
            f"⚠️ {len(summary['sessions_without_usage'])} 个会话无 usage 记账"
            f"（早于本功能的流水）：{'、'.join(summary['sessions_without_usage'][:10])}"
            + ("…" if len(summary["sessions_without_usage"]) > 10 else ""),
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None, help="conversations 目录（默认仓库 data/conversations）")
    parser.add_argument("--json", action="store_true", help="只输出汇总 JSON")
    parser.add_argument(
        "--report", default=None, metavar="PATH|latest",
        help="只扫该报告那一轮的会话（与出处审计同一语义；不传则扫全目录）",
    )
    args = parser.parse_args()

    sessions: set[str] | None = None
    report = None
    if args.report is not None:
        report = (latest_report() if args.report == "latest"
                  else json.loads(Path(args.report).read_text(encoding="utf-8")))

    directory = Path(args.dir) if args.dir else None
    if directory is None and report is not None:
        directory = conversations_dir_from_report(report)
    if directory is None:
        directory = DEFAULT_DIR
    if args.dir is None and directory != DEFAULT_DIR:
        print(f"（按报告记下的 DATA_DIR 扫描：{directory}）\n")

    if report is not None:
        results = report.get("results") or []
        sessions = {r["session_id"] for r in results if r.get("session_id")}
        if not sessions:
            raise SystemExit(
                "报告里没有 session_id，范围收敛不了（旧报告）。\n"
                "  重跑一轮评测，或去掉 --report 扫全目录。",
            )

    audits = audit_directory(directory, sessions=sessions)
    summary = summarize(audits)
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render(audits, summary))


if __name__ == "__main__":
    main()
