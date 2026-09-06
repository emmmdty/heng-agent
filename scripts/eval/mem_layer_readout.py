# -*- coding: utf-8 -*-
"""分层判读（#12 任务 B，B2 认证轮的预登记口径）。

判读人群冻结在交接文档「五之一」任务 B 指标表【口径回写 · 2026-09-06】：
- 记忆敏感层（主判读层）：decisive ≥ 30 与显著性判读只作用于本层；
- 预期平局对照层：平局率照常报告、不作废整轮，不参与主判读。

统计零新造：ab_stats 全套（sign_test_p / bootstrap_ci_win_rate /
position_swap_consistency / win_rate_summary / significance）原样复用，
本模块只负责**按冻结口径分层**与产层读数——过滤口径本身是指标定义的一部分。

用法（零 LLM，跑完 B2 后对 mem-run-*.json 出分层判读）：
    uv run python scripts/eval/mem_layer_readout.py eval/mem-run-<stamp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.eval.ab_stats import (  # noqa: E402
    bootstrap_ci_win_rate,
    decisive_indicators,
    decisive_pairs_gate,
    position_swap_consistency,
    significance,
    sign_test_p,
    win_rate_summary,
)

# 敏感层（主判读层）= 2 条现有 + 4 条 B1 新增。冻结在指标表回写块，
# 增删走回写通道。其余预登记用例 = 预期平局对照层。
LAYER_SENSITIVE = {
    "memory-recall", "memory-forget",
    "preference-inject-multi", "preference-cross-category",
    "preference-like-drives-choice", "preference-round-override",
}

# 全部预登记用例（mem_replay.PREFERENCE_PRESET_IDS 的同源清单——import 会把
# eval 脚本目录拉进 app 依赖方向，这里按值冻结并在测试钉住一致性）。
PRESET_IDS = LAYER_SENSITIVE | {
    "memory-write", "preference-conflict-cheapest-vs-dislike",
    "memory-forget-setup", "preference-inject-setup",
}


def layer_of(case_id: str) -> str:
    if case_id in LAYER_SENSITIVE:
        return "sensitive"
    if case_id in PRESET_IDS:
        return "neutral"
    raise ValueError(
        f"用例 {case_id} 不在预登记人群（mem_replay.PREFERENCE_PRESET_IDS）——"
        "人群漂移，报错而不是静默归层"
    )


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """rows → (敏感层 rows, 对照层 rows)，未知用例报错。"""
    sensitive: list[dict] = []
    neutral: list[dict] = []
    for row in rows:
        (sensitive if layer_of(row["case_id"]) == "sensitive" else neutral).append(row)
    return sensitive, neutral


def layer_readout(rows: list[dict], layer: str) -> dict:
    """层读数——与 run_ab_pipeline 的统计语义逐位一致（ab_stats 原样复用）：
    win/CI 只取位置互换一致对（ab_stats.decisive_indicators 的冻结口径），
    indicator 1 = A 臂胜（mem_replay 语义里 A=注入关）；n_flip 必须点名。"""
    swap = position_swap_consistency(rows)
    feed, ci_pairs, n_flip = decisive_indicators(rows)
    summary = win_rate_summary(feed)
    n_decisive = summary.get("n_decisive", 0)
    p = (
        sign_test_p(summary.get("wins", 0), summary.get("losses", 0))
        if n_decisive else None
    )
    ci = bootstrap_ci_win_rate(ci_pairs) if ci_pairs else None
    readout = {
        "layer": layer,
        "n_pairs": len(rows),
        "win_rate": summary,
        "swap": swap,
        "n_flip": n_flip,
    }
    if layer == "sensitive":
        readout["ci"] = ci
        readout["p_value"] = p
        readout["significance"] = significance(summary, swap, p, ci, min_decisive=30)
        readout["decisive_gate"] = decisive_pairs_gate(summary, min_pairs=30)
    return readout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B2 分层判读（零 LLM）")
    parser.add_argument("run_json", help="eval/mem-run-*.json")
    args = parser.parse_args(argv)

    data = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
    sensitive, neutral = split_rows(data.get("rows") or [])
    out = {
        "run": args.run_json,
        "sensitive": layer_readout(sensitive, "sensitive"),
        "neutral": layer_readout(neutral, "neutral"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
