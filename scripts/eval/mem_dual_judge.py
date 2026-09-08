# -*- coding: utf-8 -*-
"""B2 双 judge 证据段补跑（三重认证第 ③ 条，v25 修订口径照抄）。

背景：B2 认证轮主判段统计门槛全过（decisive/p/CI/方向），但三重认证第
③ 条"双 judge ≤20 对证据段"没随轮烧出——ab_run 的双 judge 段只能跟在
主判段同进程里跑，对已落盘的 run JSON 没有单独补跑入口。本模块离线重建
配对（与存储 pair 身份逐位核对，不一致即报错——用例集或配对逻辑漂移时
拒绝出证据），按 select_dual_judge_pairs 同一选取逻辑取 ≤20 对，用
deepseek-v4-flash 重判两个顺序，与已存 longcat 裁决比对一致率。

口径（v25 M3 证据段同款）：deepseek max_tokens 2000（make_judge_call
按模型分预算）；一致率只作 judge 可信度证据，不参与胜负判定；任一侧
裁决缺失的对不进分母（judge_agreement 既有规矩）。

用法（零执行重烧，≤40 次 deepseek 调用）：
    uv run python scripts/eval/mem_dual_judge.py eval/mem-run-<stamp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 导入 settings 是为了它的副作用：模块顶层会 load_dotenv(.env)（地雷 8）。
import app.infrastructure.settings  # noqa: F401, E402
from app.infrastructure.transient import describe_error  # noqa: E402
from scripts.eval.ab_pairwise import judge_pair  # noqa: E402
from scripts.eval.ab_run import (  # noqa: E402
    build_pairs_from_executions,
    make_judge_call,
    select_dual_judge_pairs,
)
from scripts.eval.ab_stats import judge_agreement  # noqa: E402
from scripts.eval.mem_replay import build_ground_truth  # noqa: E402

SECOND_JUDGE_MODEL = "deepseek-v4-flash"
DUAL_PAIRS = 20

_ORDER_SPECS = (
    ("ab", "verdict_ab", ("a", "b")),
    ("ba", "verdict_ba", ("b", "a")),
)


def pair_identity(pair: dict) -> tuple:
    """配对身份四元组。兼容两种形状：重建侧 left/right 是完整执行记录
    （session_id 在内层），落盘侧才是扁平的 left_session/right_session。"""
    left = pair.get("left_session") or pair["left"]["session_id"]
    right = pair.get("right_session") or pair["right"]["session_id"]
    return (pair["case_id"], pair["pair_index"], left, right)


def verify_pair_identity(rebuilt: list[dict], stored: list[dict]) -> None:
    """重建配对必须与落盘 pair 序列逐位一致——对不上 = 用例集或配对逻辑
    相对出证据的那一轮漂移过，证据段不许出。"""
    a = [pair_identity(p) for p in rebuilt]
    b = [pair_identity(p) for p in stored]
    if a != b:
        first_diff = next(
            (i for i, (x, y) in enumerate(zip(a, b)) if x != y),
            min(len(a), len(b)),
        )
        raise SystemExit(
            f"重建配对与 run JSON 存储的 pair 序列不一致（第 {first_diff} 个起分歧："
            f"重建 {a[first_diff:first_diff + 1]} vs 存储 {b[first_diff:first_diff + 1]}；"
            f"重建 {len(a)} 对 / 存储 {len(b)} 对）——cases.yaml 或配对逻辑已漂移，拒绝出证据。"
        )


def index_rows(rows: list[dict]) -> dict[tuple, dict]:
    return {(r["case_id"], r["pair_index"]): r for r in rows}


async def build_agreement_rows(
    selected: list[dict],
    rows_by_key: dict[tuple, dict],
    judge_call,
    ground_truth: str,
    progress=None,
) -> list[dict]:
    """对选中的对按两个顺序补判，与已存 longcat 裁决对齐成一致率行。

    存储侧裁决为 None（主判段 error 格）的对跳过重判——一边缺失比对
    无从谈起，进 n_error 不进分母（judge_agreement 口径），也不烧这 40 次
    里用不上的调用。重判侧传输失败记 None 留名，不塌缩成"不一致"。
    """
    out: list[dict] = []
    for pair in selected:
        row = rows_by_key[(pair["case_id"], pair["pair_index"])]
        for order, first_key, order_names in _ORDER_SPECS:
            stored_verdict = row.get(first_key)
            if stored_verdict is None:
                out.append({
                    "case_id": pair["case_id"], "pair_index": pair["pair_index"],
                    "order": order, "verdict_first": None, "verdict_second": None,
                    "note": "主判段该序为 error 格，跳过重判",
                })
                if progress:
                    progress(f"   [dual] {pair['case_id']} 对 {pair['pair_index']} {order}: 主判段 error 格，跳过")
                continue
            pos_left, pos_right = (
                (pair["left"], pair["right"]) if order == "ab" else (pair["right"], pair["left"])
            )
            try:
                judged = await judge_pair(
                    judge_call, pair["case_prompt_text"],
                    pos_left["transcript"], pos_right["transcript"],
                    order=order_names, ground_truth=ground_truth,
                    prior_context=pair.get("prior_context", ""),
                )
                second_verdict: str | None = judged["winner"]
            except Exception as err:  # noqa: BLE001 —— 传输失败留名进 n_error，不塌缩
                second_verdict = None
                note = describe_error(err)
                if progress:
                    progress(f"   [dual] {pair['case_id']} 对 {pair['pair_index']} {order} 重判失败：{note}")
                out.append({
                    "case_id": pair["case_id"], "pair_index": pair["pair_index"],
                    "order": order, "verdict_first": stored_verdict,
                    "verdict_second": None, "note": note,
                })
                continue
            agree_mark = "一致" if second_verdict == stored_verdict else "不一致"
            if progress:
                progress(
                    f"   [dual] {pair['case_id']} 对 {pair['pair_index']} {order}: "
                    f"longcat={stored_verdict} deepseek={second_verdict}（{agree_mark}）"
                )
            out.append({
                "case_id": pair["case_id"], "pair_index": pair["pair_index"],
                "order": order, "verdict_first": stored_verdict,
                "verdict_second": second_verdict,
            })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B2 双 judge 证据段补跑（三重认证第 ③ 条）")
    parser.add_argument("run_json", help="已落盘的 B2 run JSON（mem-run-*.json）")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "eval" / "cases.yaml"))
    parser.add_argument("--pairs", type=int, default=DUAL_PAIRS, help="证据段对数上限（预登记 ≤20）")
    parser.add_argument("--second-judge-model", default=SECOND_JUDGE_MODEL)
    parser.add_argument("--eval-dir", default=str(PROJECT_ROOT / "eval"))
    args = parser.parse_args(argv)
    if args.pairs > DUAL_PAIRS:
        raise SystemExit(f"证据段对数上限预登记为 {DUAL_PAIRS}，收到 {args.pairs}——加样走回写通道")

    import yaml

    run = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
    with open(args.cases, encoding="utf-8") as handle:
        cases = yaml.safe_load(handle)["cases"]

    rebuilt, _pair_errors = build_pairs_from_executions(
        run["results"], cases, run["plan"]["k"], run["pairing"],
    )
    verify_pair_identity(rebuilt, run["pairs"])

    selected = select_dual_judge_pairs(rebuilt, args.pairs)
    rows_by_key = index_rows(run["rows"])
    ground_truth = build_ground_truth()

    import httpx

    async def _run() -> tuple[list[dict], dict]:
        async with httpx.AsyncClient() as client:
            judge_call = make_judge_call(client, args.second_judge_model)
            return await build_agreement_rows(
                selected, rows_by_key, judge_call, ground_truth, progress=print,
            )

    import asyncio

    agreement_rows = asyncio.run(_run())
    agreement = judge_agreement(agreement_rows)
    payload = {
        "run": args.run_json,
        "second_judge_model": args.second_judge_model,
        "selected_pairs": [pair_identity(p) for p in selected],
        **agreement,
        "detail": agreement_rows,
    }
    stamp = run.get("stamp") or "nodate"
    out_path = Path(args.eval_dir) / (
        f"mem-dualjudge-{stamp}-{datetime.now().strftime('%H%M%S')}.json"
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    n_comparable = agreement.get("n_pairs", 0)
    print(
        f"\n双 judge 一致率：{agreement.get('n_agree', 0)}/{n_comparable}"
        f"（n_error {agreement.get('n_error', 0)}）"
        f"｜第二评审 {args.second_judge_model}｜产物 {out_path}"
    )
    if n_comparable == 0:
        print("⚠️ 可比对数为 0——证据段无效，查主判段 error 格与第二评审连通性。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
