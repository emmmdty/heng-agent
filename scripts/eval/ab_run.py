# -*- coding: utf-8 -*-
"""A/B 分流跑测器（任务 A 第 4 项）——本版本只有 dry-run 路径。

用法：
    uv run python scripts/eval/ab_run.py --dry-run                       # 主线 44 条，k=2
    uv run python scripts/eval/ab_run.py --dry-run --only compare-two    # 单条试前置
    uv run python scripts/eval/ab_run.py --dry-run --arm-b-url http://127.0.0.1:8011

机制：A/B 两臂各是一个独立服务实例（基线臂不设 PROMPT_VARIANT，候选臂
PROMPT_VARIANT=<变体名>），评测侧按臂发流量、按臂归因读数。在线按比例
分流不做（评测读数站住之前不做，YAGNI + 配额约束——任务书口径）。

**真实跑测路径本期刻意不开**：候选提示词（臂 B 的内容）还没立项产出，
预算方案（前置 P1-4 的三方案表）待拍板。真实档现在的行为是打印说明并退出，
不留 `--yes-i-know` 之类的逃生阀——烧钱闸门上不留顺手旁路（同
「不许给门禁加'没数据就当通过'的旁路」的纪律）。

dry-run 复用 eval_regression 的拦截链（语义缓存 / 旧代码 / 临时目录 /
故障注入支持），外加两道 A/B 特有的闸：
  - 降级探活拒绝：一臂降级，两臂读数一起作废，而 A/B 报告不像主线报告
    那样有逐条召回档位可查——必须在开跑前拦（前置 P0-2 空索引的同族）；
  - 两臂同址拒绝：配错 URL 时 A/B 变 A/A，烧完才发现是最贵的失败形态。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.application.harness.run_identity import describe_run  # noqa: E402
from scripts.eval_regression import (  # noqa: E402
    _guard_ephemeral_data_dir,
    _guard_stale_service,
    guard_fault_support,
    select_cases,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"

# 墙钟估算系数：R7 实测 64 意图 ≈ 55 分钟（含 judge 调用）→ ~51.6s/意图。
# 用例集或模型换了这个数就过时——报告里打印假设，允许 --seconds-per-intent 覆盖。
_DEFAULT_SECONDS_PER_INTENT = 55 * 60 / 64

_ARMS = (("A", "arm_a_url"), ("B", "arm_b_url"))


def plan_ab_run(cases: list[dict], k: int, pairing: str = "diagonal") -> dict:
    """A/B 跑测的账本：用例执行数 / 意图数 / judge 调用数 / 墙钟估算。

    这是前置 P1-4 重算的代码化（原文"2 整轮 full/220 次意图"与 k=2 双臂
    对不上）。意图按 queries 数计（R7 口径：44 条主线声明 65 轮、usage 实测
    64 意图，差 1 属多轮合并计数）；墙钟按 R7 的实测秒/意图折算并打印假设。
    决定性对是跑出来才知道的量：这里只报上限（=pairs）与门槛 30 的占比，
    **不许许诺达标**。
    """
    if not cases:
        raise ValueError("用例集为空——0 条用例的 A/B 是假绿")
    if k < 1:
        raise ValueError(f"k 至少为 1，收到 {k}")
    if pairing not in ("diagonal", "cross"):
        raise ValueError(f"未知 pairing：{pairing!r}（应为 diagonal/cross）")

    intents_per_pass = sum(len(case.get("queries") or []) for case in cases)
    executions = 2 * len(cases) * k
    intents = intents_per_pass * 2 * k
    pairs_per_case = k if pairing == "diagonal" else k * k
    pairs = len(cases) * pairs_per_case
    judge_calls = pairs * 2  # 位置互换：每对正反两个顺序各判一次（MT-Bench 口径）

    return {
        "n_cases": len(cases),
        "k": k,
        "pairing": pairing,
        "executions": executions,
        "intents": intents,
        "pairs": pairs,
        "judge_calls": judge_calls,
        "decisive_ceiling": pairs,
        "decisive_gate": 30,
        "decisive_needed_ratio": (30 / pairs) if pairs else None,
        "estimated_minutes": round(intents * _DEFAULT_SECONDS_PER_INTENT / 60, 1),
        "case_ids": [case["id"] for case in cases],
    }


def _guard_semantic_cache_state(health: dict, arm: str, allow: bool) -> None:
    """语义缓存开着时拒绝——评的会是缓存而不是两臂的真实行为。"""
    if health.get("semantic_cache") and not allow:
        raise SystemExit(
            f"拒绝开跑：臂 {arm} 的语义缓存处于开启状态，评分会变成评缓存。\n"
            "请用 SEMANTIC_CACHE_ENABLED=0 重启该臂后重试"
            "（或 --allow-semantic-cache 确认要带缓存跑）。"
        )


def _guard_probe(health: dict, arm: str, allow: bool) -> str:
    """深度探活不过的臂不开跑：降级态的 A/B 读数整体作废。

    disabled 与 error 含义相反（retrieval_probe 的契约）：未配精排/嵌入是
    本仓的合法配置（零外部依赖模式），不拦但要点名——A/B 的归因需要知道
    两臂检索档位；error 才是真故障，必须拦。探活结果整体缺失时无从判定，
    同样拦（不拦的话空索引陷阱会从这道闸的缝里过去）。
    """
    retrieval = health.get("retrieval") or {}
    probe = retrieval.get("probe")
    if not isinstance(probe, dict):
        raise SystemExit(
            f"拒绝开跑：臂 {arm} 的 /health 没有 deep 探活结果（probe 缺失）——\n"
            "无从判定该臂是否降级。确认服务是当前代码且支持 ?deep=1 后重试。"
        )
    bad = {name: state for name, state in probe.items() if str(state).startswith("error")}
    if bad and not allow:
        raise SystemExit(
            f"拒绝开跑：臂 {arm} 处于降级态（{bad}）。\n"
            "A/B 比的是两臂差异，一臂降级则两臂读数一起作废，\n"
            "且 A/B 报告没有主线报告那样的逐条召回档位可供事后排查。\n"
            "先修好外部依赖（隧道/服务）再跑；确认要在降级态跑则加 "
            "--allow-degraded-probe（读数将如实标注降级）。"
        )
    disabled = {name: state for name, state in probe.items() if str(state) == "disabled"}
    if disabled:
        return f"⚠️ 臂 {arm} 有未启用的检索组件（{disabled}）——合法配置，但两臂必须一致，归因时注意"
    return ""


def _variant_note(health: dict) -> str:
    variant = health.get("prompt_variant")
    if variant is None:
        return "⚠️ 未上报 prompt_variant（服务可能是旧代码）——两臂归因缺主键"
    return f"提示词变体 {variant or '(基线)'}"


def run_dry_run(
    health_a: dict,
    health_b: dict,
    cases: list[dict],
    k: int,
    judge_model: str,
    arm_a_url: str = "http://127.0.0.1:8000",
    arm_b_url: str = "http://127.0.0.1:8011",
    allow_semantic_cache: bool = False,
    allow_stale_service: bool = False,
    allow_ephemeral_data_dir: bool = False,
    allow_degraded_probe: bool = False,
) -> str:
    """两臂前置检查 + 账本渲染，返回报告文本。只读 /health，不发模型调用。"""
    # 结构化比较而不是字符串比较；localhost 与 127.0.0.1 是同一台机的别名，
    # 不归一的话"两臂同址"会从字符串比较的缝里漏过去
    def _normalized(url_str: str) -> tuple[str, str, int]:
        url = httpx.URL(url_str)
        host = {"localhost": "127.0.0.1"}.get(str(url.host), str(url.host))
        return (url.scheme, host, url.port)

    if _normalized(arm_a_url) == _normalized(arm_b_url):
        raise SystemExit(
            f"拒绝开跑：两臂指向同一个服务（{arm_a_url} 与 {arm_b_url}）——A/B 会变成 A/A。\n"
            "两臂各是一个服务实例：基线臂不设 PROMPT_VARIANT，候选臂设之。"
        )

    healths = {"A": health_a, "B": health_b}
    urls = {"A": arm_a_url, "B": arm_b_url}
    notes: list[str] = []
    for arm, health in healths.items():
        _guard_semantic_cache_state(health, arm, allow_semantic_cache)
        _guard_stale_service(health, allow_stale_service)
        _guard_ephemeral_data_dir(health, allow_ephemeral_data_dir)
        note = _guard_probe(health, arm, allow_degraded_probe)
        if note:
            notes.append(note)

    # 故障注入支持：任一臂不支持都拦（故障用例必须两臂都真的注入才可比）
    guard_fault_support(cases, health_a)
    guard_fault_support(cases, health_b)

    plan = plan_ab_run(cases, k)
    lines: list[str] = []
    for arm in ("A", "B"):
        health = healths[arm]
        lines.append(f"臂 {arm}（{urls[arm]}）：{_variant_note(health)}")
        lines.append(f"  跑测配置：{describe_run(health, judge_model)}")
    fingerprints = [h.get("prompt_fingerprint") for h in healths.values()]
    if any(not fp for fp in fingerprints):
        lines.append("⚠️ 至少一臂未上报提示词指纹——两臂内容是否同稿无从判定，先确认服务是当前代码。")
    elif len(set(fingerprints)) == 1:
        lines.append(
            "⚠️ 两臂提示词指纹相同：这是 A/A 同稿对照，量的是 judge 与采样噪声，"
            "不是提示词差异——除非你就是有意做 A/A 校验。"
        )
    lines.extend(notes)
    lines.append(
        f"\n[ab dry-run] 计划：{plan['n_cases']} 条用例 × 2 臂 × k={plan['k']}"
        f"（{plan['pairing']} 配对）"
    )
    lines.append(f"  用例执行 {plan['executions']} 次｜意图 {plan['intents']} 次")
    lines.append(
        f"  成对比较 {plan['pairs']} 对｜judge 调用 {plan['judge_calls']} 次"
        f"（含位置互换）｜decisive 上限 {plan['decisive_ceiling']}"
        f"（门槛 {plan['decisive_gate']}，需 ≥{plan['decisive_needed_ratio']:.0%} 不平局）"
    )
    lines.append(
        f"  墙钟估算 ≈ {plan['estimated_minutes']} 分钟（按 R7 实测 "
        f"{_DEFAULT_SECONDS_PER_INTENT:.1f}s/意图，两臂串行；judge 调用另计，可并发压缩）"
    )
    lines.append(f"  用例：{', '.join(plan['case_ids'])}")
    lines.append("\n未发起任何模型调用。真实跑测未开放：候选提示词与预算拍板前不开闸。")
    return "\n".join(lines)


async def _fetch_health(url: str) -> dict:
    # trust_env=False：探活显式旁路本机代理。地雷 12——全局 http_proxy 会把
    # "连不上 127.0.0.1"伪装成代理的 502，看着像远端服务坏了
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        response = await client.get(f"{url.rstrip('/')}/health?deep=1")
        response.raise_for_status()
        return response.json()


async def main_async(args: argparse.Namespace) -> int:
    urls = {"A": args.arm_a_url, "B": args.arm_b_url}
    healths: dict[str, dict[str, Any]] = {}
    for arm in ("A", "B"):
        try:
            healths[arm] = await _fetch_health(urls[arm])
        except Exception as err:  # noqa: BLE001
            raise SystemExit(f"臂 {arm}（{urls[arm]}）的 /health 不可达：{err}") from err

    with open(args.cases, encoding="utf-8") as handle:
        cases = yaml.safe_load(handle)["cases"]
    cases = select_cases(cases, only=args.only, tag=args.tag, exclude_tag=args.exclude_tag)

    # judge 模型归因与 eval_regression.call_judge 同一解析式：
    # 配置行写的必须就是将来真判分的那个，不能两处各回退各的
    judge_model = (
        os.environ.get("EVAL_JUDGE_MODEL") or os.environ.get("LLM_MODEL", "longcat-2.0")
    )
    print(
        run_dry_run(
            health_a=healths["A"],
            health_b=healths["B"],
            cases=cases,
            k=args.k,
            judge_model=judge_model,
            arm_a_url=args.arm_a_url,
            arm_b_url=args.arm_b_url,
            allow_semantic_cache=args.allow_semantic_cache,
            allow_stale_service=args.allow_stale_service,
            allow_ephemeral_data_dir=args.allow_ephemeral_data_dir,
            allow_degraded_probe=args.allow_degraded_probe,
        )
    )
    if not args.dry_run:
        print(
            "\n[真实跑测未开放] 二十五期任务 A 第 4 项本期只交付 dry-run 路径：\n"
            "  1. 候选提示词（臂 B 的内容）还没立项产出；\n"
            "  2. 预算方案（前置 P1-4 的三方案表）待拍板。\n"
            "两项齐了再实现 real-run（--only 小样本 → smoke → 全量，沿用 partial 续跑）。",
            file=sys.stderr,
        )
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B 分流跑测器（当前仅 dry-run）")
    parser.add_argument("--arm-a-url", default=os.environ.get("EVAL_BASE_URL_A", "http://127.0.0.1:8000"))
    parser.add_argument("--arm-b-url", default=os.environ.get("EVAL_BASE_URL_B", "http://127.0.0.1:8011"))
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "eval" / "cases.yaml"))
    parser.add_argument("--only", default=None, help="只跑指定 case id（逗号分隔）")
    parser.add_argument("--tag", default=None, help="只跑带该标签的用例")
    parser.add_argument("--exclude-tag", default=None, help="剔除带该标签的用例（主线 = --exclude-tag redteam）")
    parser.add_argument("--k", type=int, default=2, help="每臂每用例采样次数（默认 2）")
    parser.add_argument("--dry-run", action="store_true", help="只跑前置检查与算式，不发模型调用")
    parser.add_argument("--allow-semantic-cache", action="store_true")
    parser.add_argument("--allow-stale-service", action="store_true")
    parser.add_argument("--allow-ephemeral-data-dir", action="store_true")
    parser.add_argument("--allow-degraded-probe", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
