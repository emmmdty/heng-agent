# -*- coding: utf-8 -*-
"""judge 运行间一致性（二十三期清单 3）

用法：
    uv run python scripts/eval/rejudge_transcripts.py --only compare-two   # 小样本试判词格式
    uv run python scripts/eval/rejudge_transcripts.py --report latest --sample 20
    uv run python scripts/eval/rejudge_transcripts.py --json

**没有人工时，"judge 可信吗"唯一能自动回答的办法**：把同一批已判过的
transcripts 让同一个 judge 重判一遍，量它自己的判分波动。读数回答两个问题：
    1. 同一输入判两次，分数能差多少（= 判词波动带）——
       "单条分数变化小于 X 不构成结论"从此有 judge 侧的量化依据；
    2. PASS/FAIL 会不会翻转（翻转的判词要逐条人读，那是判据最脆的地方）。

口径与纪律：
    - **只重判**，不重跑会话——被测 Agent 不参与，烧的只有 judge 的 token；
    - 指纹不匹配的用例不进样本：判据改过的用例，"judge 不一致"和
      "判据变了"会混在一起，量出来的就不是 judge 波动；
    - 判分波动带如实报告，不设阈值：这是在量一把尺子，不是在考它。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.eval_regression import (  # noqa: E402
    build_ground_truth,
    call_judge,
    rubric_fingerprint,
    score_case,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"

_JUDGE_MODEL = "longcat-2.0"

# 判分等级与权重（score_case 同源，这里只引用名称）
_LEVELS = ("p0", "p1", "p2")


def pick_cases(
    results: list[dict],
    cases: dict[str, dict],
    sample: int,
    only: str | None = None,
) -> list[dict]:
    """挑出可重判的用例：verdict 非 ERROR、cases.yaml 里有判据、指纹一致。

    指纹核对是这道闸门的核心：结果里存的是**当轮**判据的指纹，cases.yaml 是
    **现在**的。两者不一致（**含报告压根没记指纹**——无法验证尺子的极端
    情形）说明这轮读数和现在的判据对不上——重判它量出的
    不是 judge 波动，是两把尺子的差。宁可少样本，不混尺子。
    """
    if sample < 1:
        raise SystemExit(f"--sample 至少为 1，收到 {sample}")
    eligible: list[dict] = []
    skipped: list[str] = []
    for result in results:
        case_id = result.get("id")
        if only is not None and case_id != only:
            continue
        if result.get("verdict") == "ERROR":
            skipped.append(f"{case_id}（ERROR，无判分）")
            continue
        case = cases.get(case_id)
        if case is None:
            skipped.append(f"{case_id}（cases.yaml 里已不存在）")
            continue
        current = rubric_fingerprint(case)
        stored = result.get("rubric_fingerprint")
        if stored != current:
            skipped.append(f"{case_id}（判据指纹不符或缺记：报告 {stored} ≠ 当前 {current}）")
            continue
        eligible.append(result)

    if only is not None:
        if not eligible:
            raise SystemExit(f"--only {only} 没有可重判的用例。跳过原因：{('；'.join(skipped)) or '报告里没有这条用例'}")
        return eligible

    if not eligible:
        return []
    if len(eligible) <= sample:
        return eligible
    if sample == 1:
        return eligible[:1]
    # 均匀铺开取确定性子集（不用随机：重跑要能对上同一批）：
    # 取首尾与等距中点，样本覆盖用例集的各个区段，不偏向开头的类别
    stride = (len(eligible) - 1) / (sample - 1)
    indexes = sorted({round(i * stride) for i in range(sample)})
    return [eligible[i] for i in indexes]


def _validate_judged(judged: dict, rubric: dict, case_id: str, side: str) -> None:
    """判词结构校验：脏判词要么报错留名，要么被静默记成假读数，二选一。

    两条硬规则（都在真实网关上出现过前兆）：
        1. rubric 声明了某档（非空 list），judge 就必须返回该档——
           score_case 的"空档位按满分"语义在重判语境下会把
           judge 的脏返回变成满分 PASS（假波动读数）；
        2. criterion 必须是非空字符串——缺失/None/空串会在比较层
           塌缩成一个键（None 与 None 互撞）静默少计。
    """
    for level in _LEVELS:
        declared = rubric.get(level) or []
        returned = judged.get(level) if isinstance(judged, dict) else None
        if declared and (not isinstance(returned, list) or not returned):
            raise ValueError(
                f"{case_id}[{side}]：rubric 声明了 {level} 档 {len(declared)} 条，"
                f"judge 返回缺该档（判词脏数据，不计入波动）",
            )
        for item in (returned or []):
            criterion = item.get("criterion") if isinstance(item, dict) else None
            if not isinstance(criterion, str) or not criterion.strip():
                raise ValueError(
                    f"{case_id}[{side}]：{level} 档出现 criterion 缺失/为空的判词条目"
                    f"（判词脏数据，不计入波动）",
                )


def compare_result(row: dict) -> dict:
    """比较两次判分：分数差、verdict 翻转、逐条 criterion 的一致性。

    criterion 按**文本**配对，不按位置——judge 返回的列表顺序没有契约。
    单边出现的 criterion 记 disagree（保守口径：说不清的不算一致）。
    两边的判词都先过结构校验：脏判词当场抛 ValueError（由调用方记 error 行），
    而不是塌缩成一个看着正常的读数。
    """
    first, second = row["first"], row["second"]
    case_id = row["id"]
    rubric = row["rubric"]
    _validate_judged(first, rubric, case_id, side="首次")
    _validate_judged(second, rubric, case_id, side="重判")
    first_score, first_p0 = score_case(first)
    second_score, second_p0 = score_case(second)
    agree = 0
    disagree = 0
    disagreed_criteria: list[str] = []
    for level in _LEVELS:
        first_items = {item.get("criterion"): bool(item.get("pass")) for item in (first.get(level) or [])}
        second_items = {item.get("criterion"): bool(item.get("pass")) for item in (second.get(level) or [])}
        for criterion in first_items.keys() | second_items.keys():
            if criterion not in first_items or criterion not in second_items:
                disagree += 1
                disagreed_criteria.append(f"{level}:{criterion[:24]}")
            elif first_items[criterion] == second_items[criterion]:
                agree += 1
            else:
                disagree += 1
                disagreed_criteria.append(f"{level}:{criterion[:24]}")
    return {
        "id": row["id"],
        "first_score": first_score,
        "second_score": second_score,
        "delta_score": round(second_score - first_score, 3),
        "first_verdict": "PASS" if first_p0 and first_score >= 0.7 else "FAIL",
        "second_verdict": "PASS" if second_p0 and second_score >= 0.7 else "FAIL",
        "flip": ("PASS" if first_p0 and first_score >= 0.7 else "FAIL")
                != ("PASS" if second_p0 and second_score >= 0.7 else "FAIL"),
        "agree": agree,
        "disagree": disagree,
        "disagreed_criteria": disagreed_criteria,
        "error": "",
    }



def summarize_comparisons(rows: list[dict]) -> dict:
    """聚合：波动带（最大 |Δscore|）、翻转清单、逐条一致率。"""
    scored = [row for row in rows if not row.get("error")]
    deltas = [abs(row["delta_score"]) for row in scored]
    flips = [row["id"] for row in scored if row.get("flip")]
    agree = sum(row["agree"] for row in scored)
    disagree = sum(row["disagree"] for row in scored)
    return {
        "n": len(rows),
        "n_scored": len(scored),
        "n_error": len(rows) - len(scored),
        "mean_abs_delta": round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
        "max_abs_delta": max(deltas) if deltas else 0.0,
        "exact_score_rate": (
            round(sum(1 for d in deltas if d == 0) / len(deltas), 4) if deltas else 0.0
        ),
        "flips": flips,
        "item_agreement": round(agree / (agree + disagree), 4) if (agree + disagree) else 0.0,
    }


async def rejudge_all(
    results: list[dict],
    cases: dict[str, dict],
    judge: Callable[[str, dict, str, str], Awaitable[dict]],
    ground_truth: str,
) -> list[dict]:
    """逐条重判并比较。单条失败标记该行、不报废整批——但必须留名。"""
    rows: list[dict] = []
    for result in results:
        case = cases[result["id"]]
        row_id = result["id"]
        try:
            second = await judge(
                result["transcript"], case["rubric"], ground_truth,
                case.get("prior_context", ""),
            )
            row = compare_result({
                "id": row_id, "first": result["judged"], "second": second,
                "rubric": case["rubric"],
            })
        except Exception as err:  # noqa: BLE001 —— 一条挂了整批还得跑完
            # first_score 与成功行同口径重算（存储值只在重算也失败时兜底），
            # 否则错误行与成功行在一张表里用两把尺子
            try:
                first_score, first_p0 = score_case(result["judged"])
                first_verdict = "PASS" if first_p0 and first_score >= 0.7 else "FAIL"
            except Exception:  # noqa: BLE001
                first_score, first_verdict = result.get("score"), result.get("verdict")
            row = {
                "id": row_id, "first_score": first_score,
                "second_score": None, "delta_score": None,
                "first_verdict": first_verdict, "second_verdict": None,
                "flip": None, "agree": 0, "disagree": 0,
                "disagreed_criteria": [], "error": f"{type(err).__name__}: {err}",
            }
        rows.append(row)
        status = row["error"] or f"{row['first_score']} → {row['second_score']}"
        print(f"   {row_id}: {status}", flush=True)
    return rows


def render(summary: dict, rows: list[dict]) -> str:
    lines = ["# judge 运行间一致性（同 judge 重判）", ""]
    lines.append(
        f"重判 {summary['n']} 条（成功 {summary['n_scored']}，失败 {summary['n_error']}）｜"
        f"判词波动带（最大 |Δscore|）**{summary['max_abs_delta']}**｜"
        f"均值 |Δscore| {summary['mean_abs_delta']}｜同分率 {summary['exact_score_rate']:.0%}｜"
        f"逐条一致率 {summary['item_agreement']:.1%}",
    )
    if summary["flips"]:
        lines.append(f"⚠️ verdict 翻转 {len(summary['flips'])} 条：{'、'.join(summary['flips'])}")
    else:
        lines.append("verdict 翻转 0 条")
    failed = [row for row in rows if row.get("error")]
    if failed:
        lines.append(f"失败（未计入波动）：{'、'.join(row['id'] for row in failed)}")
    flipped_rows = [row for row in rows if row.get("flip")]
    if flipped_rows:
        lines.append("")
        for row in flipped_rows:
            lines.append(
                f"- `{row['id']}`：{row['first_verdict']} → {row['second_verdict']}"
                f"（{row['first_score']} → {row['second_score']}），"
                f"分歧条目：{'；'.join(row['disagreed_criteria']) or '仅分数浮动'}",
            )
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    # 报告只解析一次：先 glob 出路径再读内容。latest_report() 内部也是
    # sorted(glob)[-1]，调两次的话两次之间若恰好落了新报告，
    # 内容与文件名会指向不同的轮次——读数对不上号且无症状
    if args.report == "latest":
        reports = sorted(EVAL_DIR.glob("report-*.json"))
        if not reports:
            raise SystemExit(f"没有找到报告（{EVAL_DIR}/report-*.json）。先跑一轮评测")
        report_path = reports[-1]
    else:
        report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # yaml → json 归一化（日期等非 JSON 类型会当场 TypeError，红灯可接受）；
    # rubric_fingerprint 是对 json 序列化后哈希的，先归一化才能与报告指纹同构
    cases_list = json.loads(json.dumps(_load_cases(args.cases)))
    cases = {case["id"]: case for case in cases_list}
    results = report.get("results") or []
    selected = pick_cases(results, cases, args.sample, only=args.only)
    if not selected:
        raise SystemExit(
            "没有可重判的用例（全 ERROR / 指纹不符 / cases.yaml 对不上）。\n"
            "  产出一份 n=0 的'成功'报告比报错更危险——那是假绿。",
        )
    print(f"重判 {len(selected)} 条（源报告 {report_path.name}，judge {_JUDGE_MODEL}）\n", flush=True)

    ground_truth = build_ground_truth()

    async with _make_client() as client:
        async def judge(transcript: str, rubric: dict, ground_truth_text: str, prior_context: str) -> dict:
            import os

            saved = os.environ.get("EVAL_JUDGE_MODEL")
            os.environ["EVAL_JUDGE_MODEL"] = _JUDGE_MODEL
            try:
                return await call_judge(client, transcript, rubric, ground_truth_text, prior_context)
            finally:
                if saved is None:
                    os.environ.pop("EVAL_JUDGE_MODEL", None)
                else:
                    os.environ["EVAL_JUDGE_MODEL"] = saved

        rows = await rejudge_all(selected, cases, judge, ground_truth)

    summary = summarize_comparisons(rows)
    stamp = report_path.stem.replace("report-", "")
    # 文件名带时刻：同一报告跑两次（不同 sample / 修了判词后重跑），
    # 各自的波动带读数都留档，不静默覆盖
    stamp = f"{stamp}-{datetime.now().strftime('%H%M%S')}"
    out_path = EVAL_DIR / f"rejudge-{stamp}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render(summary, rows))
    print(f"\n（明细已写入 {out_path}）")
    # 全批失败（网关缺配置、判词全脏）不能 exit 0——接进脚本链就是假绿
    if rows and summary["n_scored"] == 0:
        print("\n[全部失败] 没有一条成功重判，波动带无从谈起", file=sys.stderr)
        return 1
    return 0


def _load_cases(path_str: str) -> list[dict]:
    import yaml

    path = Path(path_str)
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["cases"]


def _make_client() -> "httpx.AsyncClient":  # noqa: F821
    import httpx

    return httpx.AsyncClient()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="latest", metavar="PATH|latest", help="源报告（默认最近一轮）")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "eval" / "cases.yaml"))
    parser.add_argument("--sample", type=int, default=20, help="重判条数上限（默认 20）")
    parser.add_argument("--only", default=None, help="只重判这一条（小样本试判词格式用）")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
