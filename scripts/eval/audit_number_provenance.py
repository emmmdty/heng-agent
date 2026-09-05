# -*- coding: utf-8 -*-
"""金额出处扫描：把"回复里的数字有没有工具出处"变成一个可回归的数

用法：
    uv run python scripts/eval/audit_number_provenance.py
    uv run python scripts/eval/audit_number_provenance.py --report latest --max-ratio 0.08   # 当门禁用
    uv run python scripts/eval/audit_number_provenance.py --json

指标：**无出处金额率 = 无出处金额数 / 回复中金额总数**。

这个数是**下界**，不是全貌：只统计带货币标记的数字（¥ $ 元 USD ...），
表格里裸写的 "| 65 |" 不算。放宽到所有数字会把"续航 40 小时""库存 150 件"
一并扫进来，判据立刻不可用。宁可漏报，不要一个没人敢信的数。

分类只是诊断线索：
    suspected_sum          约等于若干工具金额之和 —— 典型是把单品到手价相加
    suspected_difference   约等于两数之差       —— 典型是"预算还剩多少"
    unsourced              找不到成因

**暴露面双指标**（二十三期清单 1）：无出处金额数与"疑似自行算术数"
（suspected_sum / difference / basket_misadd 的候选数）**分开呈现**。
比率指标有一个 Goodhart 口子——模型少写解释性算术，分子分母一起缩，
比率照样好看，但那不是修复，是少干活。两个数一起读才能分辨
"暴露面下降"究竟是缺陷变少，还是解释变少：
门禁判的仍是"无出处金额率"，本拆分不改判定、只改呈现。

注意分类会有巧合命中（三个金额凑出的和恰好等于另一个金额），所以它不参与
"通过与否"的判定，只写在报告里供人复核。

**扫描范围**：`data/conversations/` 是累积目录，旧流水不会消失。不加 `--report`
时扫的是全部历史，而这个数只在同一轮内可比——把它拿去和固定阈值比，
每跑一轮分子分母一起涨，阈值只能不断上调，门禁很快就废了。
当门禁用时一律加 `--report latest`，把范围收敛到最近一轮。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.eval.trace_audit import SessionAudit, audit_directory  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "conversations"

# "疑似自行算术"的构成：模型把工具数/买家数自己加减出来的三类成因。
# basket_misadd 是 suspected_sum 的升级形态（确定性违规），同属算术暴露面。
_ARITHMETIC_KINDS = frozenset({"suspected_sum", "suspected_difference", "basket_misadd"})


EVAL_DIR = PROJECT_ROOT / "eval"


def latest_report(directory: Path | None = None) -> dict:
    """最近一份回归报告。门禁默认扫这一轮，而不是目录里累积下来的全部流水。"""
    directory = EVAL_DIR if directory is None else Path(directory)
    reports = sorted(directory.glob("report-*.json"))
    if not reports:
        raise SystemExit(
            f"没有找到回归报告（{directory}/report-*.json）。\n"
            f"  先跑一轮评测：make eval-smoke",
        )
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def conversations_dir_from_report(report: dict) -> Path | None:
    """报告记下的那一轮，流水落在哪。

    对着非默认 `DATA_DIR` 的实例跑评测时（比如另起一个不抢 Qdrant 文件锁的实例），
    报告在仓库 `eval/`、流水在别处，门禁两头对不上——而报错会指向
    "流水可能被清过"，把人引向完全错误的方向。
    报告里既然抄了 `/health`，就让审计跟着它走。
    """
    data_dir = ((report.get("health") or {}).get("data_dir") or "").strip()
    if not data_dir:
        return None
    candidate = Path(data_dir) / "conversations"
    return candidate if candidate.is_dir() else None


def sessions_from_report(report: dict) -> set[str]:
    """报告里这一轮跑了哪些会话。

    缺 session_id 时**报错而不是回退到全量**：静默回退会拿一个被历史流水
    污染的数当本轮读数，而错读数和真读数在报告里长得一模一样。
    """
    results = report.get("results") or []
    sessions = {r["session_id"] for r in results if r.get("session_id")}
    if not sessions:
        raise SystemExit(
            "报告里没有 session_id（九期之前的报告不记这个字段）。\n"
            "  重跑一轮即可：make eval-smoke\n"
            "  或去掉 --report 扫全目录（读数会被历史流水污染，不要当门禁用）。",
        )
    return sessions


def select_audits(audits: list[SessionAudit], sessions: set[str] | None) -> list[SessionAudit]:
    """把扫描范围收敛到指定会话。

    一份都没匹配上时报错：0 处金额算出来的比率是 0%，会被门禁当满分放行——
    比红灯更危险。
    """
    if sessions is None:
        return audits
    kept = [item for item in audits if item.session_id in sessions]
    if not kept:
        raise SystemExit(
            f"报告里的 {len(sessions)} 个会话在流水目录里一份都找不到"
            f"（如 {sorted(sessions)[0]}）。可能的原因按概率排：\n"
            f"  1. 这份报告是对着**另一个 DATA_DIR** 的实例跑的（报告在仓库 eval/、"
            f"流水在别处）——用 --dir <那个实例的 data/conversations> 重扫；\n"
            f"  2. 流水被清过——重跑一轮再扫：make eval-smoke",
        )
    return kept


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    reason: str


def gate_verdict(
    summary: dict, max_ratio: float, min_amounts: int, min_sessions: int = 8,
) -> GateVerdict:
    """门禁判定：样本量够了才用比率下结论。

    单条用例的报告只有十几处金额，1 处无出处 = 5.9%、2 处 = 11.8%——
    阈值 8% 落在两个可能取值之间，结论完全取决于模型这轮多写还是少写了一句话。
    小样本时**不判定**（照常打印发现），而不是放宽阈值：
    放宽会让整轮的真劣化一起漏过去。

    同一条读数纪律在踩坑 30 里写过（n=105 时 1 条 query ≈ 0.95pt），
    那次写在文档里，这次写进判据。
    """
    total = summary["total_amounts"]
    ratio = summary["unsourced_ratio"]
    sessions = summary.get("sessions", 0)
    if sessions and sessions < min_sessions:
        # 用例数太少同样不能下结论，哪怕金额数够了：定向重跑三条用例时，
        # 比率完全由其中一条决定（实测 3 条切片 10.2%，而同期整轮是 4.9%）。
        # 金额数与用例数是两个独立的样本量维度，都得够。
        return GateVerdict(True, (
            f"用例数不足，门禁未判定：{sessions} 条 < 下限 {min_sessions} 条"
            f"（金额 {total} 处，当前 {ratio:.1%}）。"
            f"定向重跑的切片里比率由个别用例主导，不构成结论；"
            f"要门禁真正生效，跑 make eval-smoke 或 make eval"
        ))
    if total < min_amounts:
        return GateVerdict(True, (
            f"样本量不足，门禁未判定：金额 {total} 处 < 下限 {min_amounts} 处。"
            f"（当前 {ratio:.1%}，这个比率在小样本上跳变太大，不构成结论；"
            f"要门禁真正生效，跑一轮 make eval 或 make eval-smoke 再扫）"
        ))
    if ratio > max_ratio:
        return GateVerdict(False, f"无出处金额率 {ratio:.1%} > 上限 {max_ratio:.1%}（金额 {total} 处）")
    return GateVerdict(True, f"无出处金额率 {ratio:.1%} ≤ 上限 {max_ratio:.1%}（金额 {total} 处）")


def summarize(audits: list[SessionAudit]) -> dict:
    total = sum(item.total_amounts for item in audits)
    unsourced = sum(len(item.unsourced) for item in audits)
    kinds = Counter(finding.kind for item in audits for finding in item.unsourced)
    # 暴露面双指标（二十三期清单 1）：自行算术的暴露面 = 判成 sum/difference/
    # basket_misadd 的部分；两者之差是"连成因都找不到"的纯无出处。
    arithmetic = sum(count for kind, count in kinds.items() if kind in _ARITHMETIC_KINDS)
    return {
        "sessions": len(audits),
        "sessions_with_findings": sum(1 for item in audits if not item.clean),
        "sessions_flagged_at_runtime": sum(1 for item in audits if item.runtime_flagged),
        "total_amounts": total,
        "unsourced_amounts": unsourced,
        "unsourced_ratio": round(unsourced / total, 4) if total else 0.0,
        "explanatory_arithmetic": arithmetic,
        "arithmetic_ratio": round(arithmetic / total, 4) if total else 0.0,
        "pure_unsourced": unsourced - arithmetic,
        "kinds": dict(kinds),
    }


def render(audits: list[SessionAudit], summary: dict) -> str:
    lines = ["# 金额出处扫描", ""]
    lines.append(
        f"会话 {summary['sessions']} 个（{summary['sessions_with_findings']} 个有发现，"
        f"{summary['sessions_flagged_at_runtime']} 个在运行时就已告警）｜"
        f"金额 {summary['total_amounts']} 处｜无出处 {summary['unsourced_amounts']} 处"
        f"（{summary['unsourced_ratio']:.1%}）",
    )
    # 暴露面双指标：这两个数一起读才能分辨"缺陷变少"还是"解释变少"（Goodhart 口）
    lines.append(
        f"其中疑似自行算术 {summary['explanatory_arithmetic']} 处"
        f"（{summary['arithmetic_ratio']:.1%}）｜"
        f"纯无出处 {summary['pure_unsourced']} 处",
    )
    if summary["kinds"]:
        lines.append("成因分布：" + "，".join(f"{k} {v}" for k, v in sorted(summary["kinds"].items())))
    lines.append("")
    for item in audits:
        if item.clean:
            continue
        mark = "运行时已告警" if item.runtime_flagged else "本次补判发现"
        lines.append(f"## {item.session_id}（{len(item.unsourced)}/{item.total_amounts}，{mark}）")
        seen: set[float] = set()
        for finding in item.unsourced:
            if finding.value in seen:
                continue
            seen.add(finding.value)
            explain = f"  ← {finding.explain}" if finding.explain else ""
            lines.append(f"  - {finding.raw}  [{finding.kind}]{explain}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default=None,
        help="conversations 目录（不传时：有 --report 就跟着报告记下的 DATA_DIR 走，"
             "否则用仓库默认的 data/conversations）",
    )
    parser.add_argument("--json", action="store_true", help="只输出汇总 JSON")
    parser.add_argument(
        "--report",
        default=None,
        metavar="PATH|latest",
        help="只扫该报告那一轮的会话（门禁用这个；不传则扫全目录，读数会被历史流水污染）",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=None,
        help="无出处金额率上限，超过则退出码 1（当回归门禁用）",
    )
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=8,
        help="判定所需的最小用例数（定向重跑的切片里比率由个别用例主导）",
    )
    parser.add_argument(
        "--min-amounts",
        type=int,
        default=30,
        help="判定所需的最小金额总数，低于它则不判定（比率在小样本上跳变太大）",
    )
    args = parser.parse_args()

    scope = None
    report = None
    if args.report is not None:
        report = (latest_report() if args.report == "latest"
                  else json.loads(Path(args.report).read_text(encoding="utf-8")))

    # 目录解析顺序：显式 --dir > 报告记下的 DATA_DIR > 仓库默认。
    # 中间那一档是为了让审计**跟着报告走**：对着非默认 DATA_DIR 的实例跑评测时，
    # 报告在仓库 eval/、流水在别处，两头对不上。
    directory = Path(args.dir) if args.dir else None
    if directory is None and report is not None:
        directory = conversations_dir_from_report(report)
    if directory is None:
        directory = DEFAULT_DIR
    if args.dir is None and directory != DEFAULT_DIR:
        print(f"（按报告记下的 DATA_DIR 扫描：{directory}）\n")

    audits = audit_directory(directory)
    if report is not None:
        scope = sessions_from_report(report)
        audits = select_audits(audits, scope)
    summary = summarize(audits)
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render(audits, summary))

    if args.max_ratio is not None:
        verdict = gate_verdict(
            summary, args.max_ratio, args.min_amounts, args.min_sessions,
        )
        if not verdict.passed:
            print(f"\n[门禁不通过] {verdict.reason}", file=sys.stderr)
            raise SystemExit(1)
        print(f"\n[门禁通过] {verdict.reason}")


if __name__ == "__main__":
    main()
