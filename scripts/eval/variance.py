# -*- coding: utf-8 -*-
"""跑测方差：同一条用例在多轮报告里的分数散布

用法：
    uv run python scripts/eval/variance.py             # 扫 eval/report-*.json
    uv run python scripts/eval/variance.py --min-runs 3

为什么需要它：**没有方差就没有显著性**。
"这次 0.973、上次 0.95，是改好了还是抖了一下"——不量方差就只能靠感觉回答，
而靠感觉回答的结果是：真退化被当成抖动放过，抖动被当成退化去"修"。

Prompt A/B 分流（主线四）的前置条件也是它：两个分支的分数差要大于同配置的
自然波动才有意义，否则建出来的只是两个不能比较的数。

**口径**：只把**配置行相同、且判据指纹相同**的轮次放在一起比。
模型、提示词指纹、精排开关、代码新鲜度任一不同，分数就不可比
（十期换 ground_truth 那次的教训）；判据自己改了同样不可比——
**rubric 本身就是配置的一部分**，这一条是本工具第一版栽过之后补上的：
它把我改判据前后的读数混在一起，算出 0.325 的"波动"，
而那其实是判据被改对了。
配置行与判据指纹都由报告自己记着，不用人回忆。
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"


def load_reports(directory: Path) -> list[dict]:
    reports = []
    for path in sorted(directory.glob("report-*.json")):
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return reports


def config_key(report: dict) -> str:
    """把配置行里"会影响分数"的部分抽出来当分组键。

    评审模型不进键：它写在同一行里，但换 judge 等于换尺子——
    真要比较不同 judge 的读数，那是另一个问题，不该混进方差里。
    所以这里把整行原样当键，最保守：配置行只要有一个字不同就分开算。
    """
    return str(report.get("run") or "未知配置")


def collect_scores(
    reports: list[dict], require_fingerprint: bool = False,
) -> dict[tuple[str, str], list[float]]:
    """(配置行, case_id) → 分数列表。ERROR 轮次不计入——它没有判分，
    把 0.0 混进方差会把"跑挂了"算成"分数低"。"""
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for report in reports:
        key_config = config_key(report)
        for result in report.get("results") or []:
            if result.get("verdict") == "ERROR":
                continue
            # 判据指纹进键：**rubric 本身就是配置的一部分**。
            # 少了它，"改判据之前"和"改判据之后"的分数会被混在一起算方差——
            # 本工具第一版就栽在这里，算出 0.325 的"波动"，
            # 而那其实是我自己把判据改对了。
            rubric = str(result.get("rubric_fingerprint") or "未记录")
            if require_fingerprint and rubric == "未记录":
                # 十八期之前的报告不记判据指纹，混进来会把"改判据前后"当成波动
                continue
            scores[(f"{key_config}｜判据 {rubric}", result["id"])].append(
                float(result.get("score", 0.0)),
            )
    return scores


def run_level_means(
    reports: list[dict], require_fingerprint: bool = False,
) -> dict[str, list[float]]:
    """每一轮的均分，按（配置行 + 用例集规模）分组。

    人们引用的是均分（"这轮 0.973"），而单条散布回答不了"均分能抖多少"。
    带上用例数是因为 smoke 轮与 full 轮的均分本来就不可比。
    """
    means: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        valid = [
            r for r in report.get("results") or []
            if r.get("verdict") != "ERROR"
            and (not require_fingerprint or r.get("rubric_fingerprint"))
        ]
        if not valid:
            continue
        key = f"{config_key(report)}｜{len(valid)} 条"
        means[key].append(statistics.fmean(float(r.get("score", 0.0)) for r in valid))
    return means


def summarize(scores: dict[tuple[str, str], list[float]], min_runs: int) -> list[dict]:
    rows = []
    for (config, case_id), values in scores.items():
        if len(values) < min_runs:
            continue
        rows.append({
            "case_id": case_id,
            "runs": len(values),
            "min": min(values),
            "max": max(values),
            "spread": round(max(values) - min(values), 3),
            "mean": round(statistics.fmean(values), 3),
            "stdev": round(statistics.pstdev(values), 3),
            "config": config,
        })
    # 散布大的排在前面：那是"同配置下也不稳定"的用例，拿它们判回归最不可靠
    return sorted(rows, key=lambda row: (-row["spread"], row["case_id"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(EVAL_DIR), help="报告目录")
    parser.add_argument("--min-runs", type=int, default=2, help="至少跑过几轮才统计")
    parser.add_argument(
        "--require-fingerprint",
        action="store_true",
        help="只统计带判据指纹的报告（十八期之后的）。要拿到干净的自然波动读数就加上它",
    )
    args = parser.parse_args()

    reports = load_reports(Path(args.dir))
    if not reports:
        raise SystemExit(f"没有找到报告（{args.dir}/report-*.json）")

    rows = summarize(
        collect_scores(reports, require_fingerprint=args.require_fingerprint),
        args.min_runs,
    )
    print(f"# 跑测方差（{len(reports)} 份报告，至少 {args.min_runs} 轮的用例）\n")
    if not rows:
        print(f"没有用例在同一配置下跑过 {args.min_runs} 轮以上——"
              f"方差量不出来。同配置连跑两轮即可。")
        return 0

    print("| case | 轮次 | 最低 | 最高 | 散布 | 均值 | 总体标准差 |")
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['case_id']} | {row['runs']} | {row['min']} | {row['max']} "
              f"| **{row['spread']}** | {row['mean']} | {row['stdev']} |")

    unrecorded = [row for row in rows if "判据 未记录" in row["config"]]
    if unrecorded:
        print(
            f"\n⚠️ 其中 {len(unrecorded)} 行来自**没有判据指纹**的旧报告"
            f"（十八期之后的报告才记这个字段）。这些行可能把改判据前后的读数混在一起，"
            f"算出来的散布**不能当成模型的自然波动**——重跑两轮新的再看。",
        )

    means = run_level_means(reports, require_fingerprint=args.require_fingerprint)
    repeated = {k: v for k, v in means.items() if len(v) >= args.min_runs}
    if repeated:
        print("\n## 整轮均分的波动（人们引用的就是这个数）\n")
        print("| 配置｜规模 | 轮次 | 最低 | 最高 | 散布 |")
        print("|---|---|---|---|---|")
        for key, values in sorted(repeated.items()):
            print(f"| {key[-40:]} | {len(values)} | {min(values):.3f} | {max(values):.3f} "
                  f"| **{max(values) - min(values):.3f}** |")

    unstable = [row for row in rows if row["spread"] > 0]
    print(f"\n同配置下分数有波动的用例：{len(unstable)} / {len(rows)}")
    if unstable:
        worst = unstable[0]
        print(f"波动最大的是 `{worst['case_id']}`（{worst['min']} ~ {worst['max']}）。")
        if "判据 未记录" in worst["config"]:
            print("（这一条正来自没有判据指纹的旧报告，先别拿它当波动上界。）")
        else:
            print("**小于这个散布的分数变化不构成结论**——它落在同配置的自然波动里。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
