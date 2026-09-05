# -*- coding: utf-8 -*-
"""A/B 分流跑测器（任务 A 第 4 项）——dry-run 前置 + 真实跑测路径。

用法：
    uv run python scripts/eval/ab_run.py --dry-run                       # 主线 44 条，k=2
    uv run python scripts/eval/ab_run.py --dry-run --only compare-two    # 单条试前置
    uv run python scripts/eval/ab_run.py --only compare-two --k 1        # 真实跑 2 条 ×2 臂
    uv run python scripts/eval/ab_run.py --tag smoke                     # 先导档（12 条 ×k=2 两臂）
    uv run python scripts/eval/ab_run.py --resume eval/ab-partial-*.json # 中断续跑

机制：A/B 两臂各是一个独立服务实例（基线臂不设 PROMPT_VARIANT，候选臂
PROMPT_VARIANT=<变体名> + 独立 VECTOR_STORE_DIR 躲 Qdrant 单进程文件锁），
评测侧按臂发流量、按臂归因读数。在线按比例分流不做（评测读数站住之前不做，
YAGNI + 配额约束——任务书口径）。

真实路径的三段结构（授权文档 M1）：
  1. 执行段：每 (case, arm, sample) 一次独立会话打该臂服务，产物逐次落
     eval/ab-partial-*.json（整份重写，同 eval_regression 的 partial 纪律），
     跑完才进下一段——**没跑完不许配对**；
  2. 判段：按 pairing（diagonal/cross）配对，每对正反两个顺序各判一次
     （MT-Bench 位置互换口径），脏判词记 error 行留名不塌缩；
  3. 统计与报告：ab_stats 口径（互换一致率 / 胜率 / 符号检验 / bootstrap CI）
     渲染 eval/ab-report-*.md，机器可读 eval/ab-run-*.json。

烧钱闸门上的既有闸全部有效：降级探活拒绝、两臂同址拒绝、语义缓存拒绝、
旧代码拒绝、评审模型=被测模型拒绝。dry-run 与真实路径共享同一份账本
（plan_ab_run）与同一道前置闸（preflight）——不许两套算式。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.application.harness.run_identity import describe_run  # noqa: E402
from app.infrastructure.transient import describe_error  # noqa: E402
from scripts.eval.ab_pairwise import judge_pair  # noqa: E402
from scripts.eval.ab_report import _fmt_rate  # noqa: E402  # 渲染与收尾摘要共用同一格式化
from scripts.eval.ab_stats import (  # noqa: E402
    bootstrap_ci_win_rate,
    decisive_pairs_gate,
    decisive_indicators,
    judge_agreement,
    position_swap_consistency,
    significance,
    sign_test_p,
    win_rate_summary,
)
from scripts.eval_regression import (  # noqa: E402
    _guard_ephemeral_data_dir,
    _guard_stale_service,
    build_ground_truth,
    execute_case,
    guard_fault_support,
    resolve_judge_model,
    select_cases,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"

# 墙钟估算系数：R7 实测 64 意图 ≈ 55 分钟（含 judge 调用）→ ~51.6s/意图。
# 用例集或模型换了这个数就过时——报告里打印假设，允许 --seconds-per-intent 覆盖。
_DEFAULT_SECONDS_PER_INTENT = 55 * 60 / 64

# 双 judge 一致率的授权上限（授权文档第一节：≤20 对 × 2 顺序）。
_DUAL_JUDGE_MAX_PAIRS = 20

_ARMS = (("A", "arm_a_url"), ("B", "arm_b_url"))


def plan_ab_run(cases: list[dict], k: int, pairing: str = "diagonal", seconds_per_intent: float | None = None) -> dict:
    """A/B 跑测的账本：用例执行数 / 意图数 / judge 调用数 / 墙钟估算。

    这是前置 P1-4 重算的代码化（原文"2 整轮 full/220 次意图"与 k=2 双臂
    对不上）。意图按 queries 数计（R7 口径：44 条主线声明 65 轮、usage 实测
    64 意图，差 1 属多轮合并计数）；墙钟按 R7 的实测秒/意图折算并打印假设，
    `seconds_per_intent` 可覆盖（模型/用例集换了实测系数就过时）。
    决定性对是跑出来才知道的量：这里只报上限（=pairs）与门槛 30 的占比，
    **不许许诺达标**。
    """
    if not cases:
        raise ValueError("用例集为空——0 条用例的 A/B 是假绿")
    if k < 1:
        raise ValueError(f"k 至少为 1，收到 {k}")
    if pairing not in ("diagonal", "cross"):
        raise ValueError(f"未知 pairing：{pairing!r}（应为 diagonal/cross）")

    rate = _DEFAULT_SECONDS_PER_INTENT
    if seconds_per_intent is not None:
        if seconds_per_intent <= 0:
            raise ValueError(f"seconds-per-intent 必须为正数，收到 {seconds_per_intent}")
        rate = float(seconds_per_intent)
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
        "estimated_minutes": round(intents * rate / 60, 1),
        "seconds_per_intent": rate,
        "case_ids": [case["id"] for case in cases],
    }


def triple_key(case_id: str, arm: str, sample_index: int) -> str:
    return f"{case_id}|{arm}|{sample_index}"


def ab_participant_ids(case: dict, arm: str, sample_index: int) -> tuple[str, str]:
    """(session_id, buyer_id)——run_case 的派生逻辑 + 臂与样本索引进 id。

    k 次采样 = k 个独立会话（授权文档 M1）：样本索引进 id 后，同一用例的
    不同样本、不同臂互不见对方的会话与记忆写入。记忆链（memory-write →
    memory-recall 声明同一 buyer_id）在同 (arm, sample) 内派生出同一买家，
    跨样本/跨臂自然隔离——偏好谁写的、哪一次写的，可归因。
    """
    arm_tag = arm.lower()
    session_id = f"ab-{arm_tag}-k{sample_index}-{case['id']}-{uuid.uuid4().hex[:6]}"
    base_buyer = case.get("buyer_id") or f"eval-buyer-{case['id']}"
    buyer_id = f"{base_buyer}-ab{arm_tag}k{sample_index}"
    return session_id, buyer_id


def build_execution_plan(cases: list[dict], k: int) -> list[dict]:
    """执行计划：arm-major → case-major → sample-inner 的三元组序列。

    臂整体串行（墙钟估算按两臂串行）；臂内按 cases.yaml 顺序（记忆链靠
    顺序成立），同用例的 k 个样本连着跑。k>=1：k=0 的 A/B 是假跑。
    """
    if not cases:
        raise ValueError("用例集为空")
    if k < 1:
        raise ValueError(f"k 至少为 1，收到 {k}")
    plan: list[dict] = []
    for arm in ("A", "B"):
        for case in cases:
            for sample_index in range(k):
                plan.append({
                    "case_id": case["id"], "arm": arm, "sample_index": sample_index,
                    "requires": list(case.get("requires") or []),
                })
    return plan


def partition_pending(plan: list[dict], completed: set[str]) -> list[dict]:
    """从执行计划里取待跑三元组；待跑项的前置若已完成要一并补跑。

    与 eval_regression.plan_resume 同一条纪律的 A/B 版：memory-recall 依赖
    memory-write 先写偏好，续跑时跳过前置直接跑后继，评的是一个不成立的
    前提，而外观完全正常。A/B 的前置补跑限定**同 (arm, sample)**——样本
    隔离语义下，(B,1) 的前置是 (B,1)，不是 (B,0)。
    """
    pending = [item for item in plan if triple_key(item["case_id"], item["arm"], item["sample_index"]) not in completed]
    rerun: set[str] = set()
    for item in pending:
        for req in item.get("requires") or []:
            key = triple_key(req, item["arm"], item["sample_index"])
            if key in completed:
                rerun.add(key)
    ordered = [
        item for item in plan
        if triple_key(item["case_id"], item["arm"], item["sample_index"]) in rerun
    ] + pending
    return ordered


async def execute_ab_case(
    case: dict, arm: str, sample_index: int, client: httpx.AsyncClient | None, base_url: str,
) -> dict:
    """执行一次采样，返回 execution 记录；失败记名不炸轮、不静默。"""
    session_id, buyer_id = ab_participant_ids(case, arm, sample_index)
    record = {
        "case_id": case["id"], "arm": arm, "sample_index": sample_index,
        "session_id": session_id, "transcript": "", "ok": False, "error": "",
        "fault_clear_error": "",
    }
    try:
        executed = await execute_case(
            client, case, base_url=base_url, session_id=session_id, buyer_id=buyer_id,
        )
        record.update(
            session_id=executed["session_id"],
            transcript=executed["transcript"],
            ok=True,
            fault_clear_error=executed.get("fault_clear_error", ""),
        )
    except Exception as err:  # noqa: BLE001 —— 单次失败降级为 error 行，整轮继续
        record["error"] = describe_error(err)
        # 执行炸了且清理也炸了：清理失败挂在原始异常上，一并留名
        record["fault_clear_error"] = getattr(err, "fault_clear_error", "")
    return record


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


def guard_judge_models(judge_model: str, healths: dict, second_judge_model: str = "") -> None:
    """评审模型的闸：judge 必须与被测不同模型（纪律 6），双 judge 两模型必须不同。

    同模型自评是系统性偏高的读数（eval_regression 头注释同一条纪律）；
    两个"双 judge"相同则一致率恒 100%，量的是噪声不是可信度——
    这两种配置烧的都是真配额，拦在开跑前。
    """
    tested = {str((healths.get(arm) or {}).get("model") or "") for arm in ("A", "B")}
    if judge_model and judge_model in tested:
        raise SystemExit(
            f"拒绝开跑：评审模型（{judge_model}）与被测模型相同——同模型自评存在"
            "自我偏好偏差，胜率读数无效。EVAL_JUDGE_MODEL 必须与被测不同"
            "（judge 固定 longcat-2.0）。"
        )
    if second_judge_model:
        if judge_model and second_judge_model == judge_model:
            raise SystemExit(
                "拒绝开跑：双 judge 的两个模型相同——一致率恒 100%，"
                "量的是采样噪声不是 judge 可信度。"
            )
        if second_judge_model in tested:
            raise SystemExit(
                f"拒绝开跑：第二评审模型（{second_judge_model}）与被测模型相同"
                "——自我偏好偏差同样成立。"
            )


def check_resume_config(partial: dict, healths: dict, expected_plan: dict | None = None) -> None:
    """续跑前核对两臂配置与跑测计划：指纹/变体/模型/k/pairing/用例选择。

    半批样本跑在旧提示词上、另外半批跑在新提示词上，是 A/B 里最贵的
    失败形态（读数混装且外观正常）。partial 里记下了当初两臂各自报的
    配置与本轮计划，续跑时对不上就拒绝——不提供覆盖开关（配置对不上
    唯一的正确动作是查清楚，不是绕过去）。
    """
    saved = partial.get("arm_config") or {}
    if not saved:
        raise SystemExit("续跑文件缺 arm_config——两臂配置无法核对，拒绝混装。")
    field_map = {"fingerprint": "prompt_fingerprint", "variant": "prompt_variant", "model": "model"}
    for arm in ("A", "B"):
        current = healths.get(arm) or {}
        was = saved.get(arm) or {}
        for field, health_field in field_map.items():
            now_value = str(current.get(health_field) if current.get(health_field) is not None else "")
            was_value = str(was.get(field, ""))
            if now_value != was_value:
                raise SystemExit(
                    f"拒绝续跑：臂 {arm} 的 {field} 与续跑文件不一致"
                    f"（文件里 {was_value!r}，当前服务 {now_value!r}）。\n"
                    "半批样本跑在新配置上会让两臂读数混装——先查清楚原因，"
                    "确认两臂配置后再续跑。"
                )
    if expected_plan is not None:
        saved_plan = partial.get("plan") or {}
        for field in ("k", "pairing"):
            if saved_plan.get(field) != expected_plan.get(field):
                raise SystemExit(
                    f"拒绝续跑：续跑文件的 {field}={saved_plan.get(field)!r} 与本次实参 "
                    f"{expected_plan.get(field)!r} 不一致——同一轮的样本必须同一种跑法，"
                    "换档请开新 stamp 重跑。"
                )
        if sorted(saved_plan.get("case_ids") or []) != sorted(expected_plan.get("case_ids") or []):
            raise SystemExit(
                "拒绝续跑：续跑文件的用例选择与本次不一致——用例集变了就不该接着旧 partial 跑。"
            )


def _latest_by_triple(results: list[dict], order: dict[str, int]) -> list[dict]:
    """同一 (case_id, arm, sample_index) 只留最新一条，按执行计划排序。

    续跑的前置补跑会重写已完成的样本——旧记录若留在 results 里，执行
    总数虚高、ab-run json 出现重复键，统计分母被污染（独立审查抓出）。
    配对本身用 by_key 字典不受影响，但落盘与报告的分母必须干净。
    """
    latest: dict[str, dict] = {}
    for record in results:
        latest[triple_key(record["case_id"], record["arm"], record["sample_index"])] = record
    return sorted(
        latest.values(),
        key=lambda r: order.get(triple_key(r["case_id"], r["arm"], r["sample_index"]), 10**9),
    )


def build_pairs_from_executions(
    executions: list[dict], cases: list[dict], k: int, pairing: str = "diagonal",
) -> tuple[list[dict], list[dict]]:
    """执行产物 → 成对列表与 error 行。缺样本/失败样本都记名，不静默丢。

    diagonal：样本 i↔i 配对（k 对/用例）；cross：全组合（k² 对/用例，
    judge 成本翻倍、被测成本不变——decisive 不够时的第一档扩容）。
    """
    if pairing not in ("diagonal", "cross"):
        raise ValueError(f"未知 pairing：{pairing!r}（应为 diagonal/cross）")
    cases_by_id = {case["id"]: case for case in cases}
    by_key = {(e["case_id"], e["arm"], e["sample_index"]): e for e in executions}
    combos = [(i, i) for i in range(k)] if pairing == "diagonal" else [
        (i, j) for i in range(k) for j in range(k)
    ]
    pairs: list[dict] = []
    errors: list[dict] = []
    for case in cases:
        case_id = case["id"]
        case_prompt_text = "\n".join(case.get("queries") or [])
        for i, j in combos:
            pair_index = combos.index((i, j))
            left = by_key.get((case_id, "A", i))
            right = by_key.get((case_id, "B", j))
            problem = ""
            if left is None and right is None:
                problem = f"用例 {case_id} 臂 A 采样 {i} 与臂 B 采样 {j} 的产物缺失（未跑完）"
            elif left is None:
                problem = f"用例 {case_id} 臂 A 采样 {i} 的产物缺失（未跑完）"
            elif right is None:
                problem = f"用例 {case_id} 臂 B 采样 {j} 的产物缺失（未跑完）"
            elif not left["ok"]:
                problem = f"用例 {case_id} 臂 A 采样 {i} 执行失败：{left['error']}"
            elif not right["ok"]:
                problem = f"用例 {case_id} 臂 B 采样 {j} 执行失败：{right['error']}"
            if problem:
                errors.append({"case_id": case_id, "pair_index": pair_index, "reason": problem})
                continue
            pairs.append({
                "case_id": case_id,
                "pair_index": pair_index,
                "left": left,
                "right": right,
                "case_prompt_text": case_prompt_text,
                "prior_context": cases_by_id[case_id].get("prior_context", ""),
            })
    return pairs, errors


async def judge_pair_rows(
    pairs: list[dict],
    judge_call,
    ground_truth: str,
    progress=None,
) -> list[dict]:
    """成对跑判：每对正反两个顺序各判一次，脏判词/网络失败记 error 行。

    两行都 None 的对由 ab_stats 计入 n_error；只坏一边的对同样如实在
    error_ab/error_ba 里留名——宁可少一对，不进一条假读数。
    """
    rows: list[dict] = []
    for pair in pairs:
        row = {
            "case_id": pair["case_id"], "pair_index": pair["pair_index"],
            "verdict_ab": None, "verdict_ba": None,
            "rationale_ab": "", "rationale_ba": "",
            "raw_ab": "", "raw_ba": "",
            "error_ab": "", "error_ba": "",
        }
        try:
            result = await judge_pair(
                judge_call, pair["case_prompt_text"],
                pair["left"]["transcript"], pair["right"]["transcript"],
                order=("a", "b"), ground_truth=ground_truth,
                prior_context=pair.get("prior_context", ""),
            )
            row.update(verdict_ab=result["winner"], rationale_ab=result["rationale"], raw_ab=result["raw"])
        except Exception as err:  # noqa: BLE001
            row["error_ab"] = f"用例 {pair['case_id']} 对 {pair['pair_index']} 正序判词失败：{describe_error(err)}"
        try:
            result = await judge_pair(
                judge_call, pair["case_prompt_text"],
                pair["right"]["transcript"], pair["left"]["transcript"],
                order=("b", "a"), ground_truth=ground_truth,
                prior_context=pair.get("prior_context", ""),
            )
            row.update(verdict_ba=result["winner"], rationale_ba=result["rationale"], raw_ba=result["raw"])
        except Exception as err:  # noqa: BLE001
            row["error_ba"] = f"用例 {pair['case_id']} 对 {pair['pair_index']} 反序判词失败：{describe_error(err)}"
        rows.append(row)
        if progress is not None:
            progress(f"   [judge] {pair['case_id']} 对 {pair['pair_index']}: ab={row['verdict_ab']} ba={row['verdict_ba']}")
    return rows


def select_dual_judge_pairs(rows: list[dict], n: int) -> list[dict]:
    """双 judge 子集：每用例只取第一对、按用例顺序取满 n 对。

    摊到更多用例而不是同一用例的相邻对——一致率要量的是跨请求的稳定性，
    不是同一个判两次的稳定性。
    """
    if n <= 0:
        return []
    selected: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if row["case_id"] in seen:
            continue
        seen.add(row["case_id"])
        selected.append(row)
        if len(selected) >= n:
            break
    return selected


def make_judge_call(client: httpx.AsyncClient | None, model: str):
    """评审适配器：纯文本裁决（不强制 JSON），传输与重试共用 call_llm_with_retry。

    与 eval_regression.call_judge 同一条网关直连路径（EVAL_JUDGE_MODEL 解析
    同源），但成对判词是"裁决: 1|2|平局"的文本格式，不能带 response_format
    的 JSON 约束——格式约束交给提示词，脏输出由 parse_verdict 拦。
    """
    from scripts.eval_regression import call_llm_with_retry

    async def judge_call(prompt: str) -> str:
        return await call_llm_with_retry(client, {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        })

    return judge_call


async def run_ab_pipeline(
    *,
    cases: list[dict],
    k: int,
    pairing: str,
    judge_model: str,
    healths: dict,
    urls: dict,
    arm_lines: dict,
    arm_config: dict,
    ground_truth: str,
    eval_dir: Path,
    stamp: str,
    label: str = "",
    execute_fn=None,
    judge_factory=None,
    client: httpx.AsyncClient | None = None,
    second_judge_model: str = "",
    dual_judge_pairs: int = 0,
    resume_path: Path | None = None,
    conversations_dir: Path | None = None,
    positive_control: bool = False,
    progress=None,
    seconds_per_intent: float | None = None,
    judge_client: httpx.AsyncClient | None = None,
) -> dict:
    """A/B 真实跑测三段管线：执行 → 配对判 → 统计与报告。

    产物三件套（都在 eval_dir，ab- 前缀分桶）：
      - ab-partial-{stamp}.json：逐样本落盘（整份重写），跑完并出报告后删除；
      - ab-run-{stamp}.json：机器可读（执行产物 + 判行 + 统计 + 两臂配置）；
      - ab-report-{stamp}.md：人读报告（render_ab_report 渲染）。

    execute_fn / judge_factory 是注入点：测试用假件零 LLM 走通全链，
    真实路径用 execute_ab_case / make_judge_call——接线只有这一处。
    client 是执行段（两臂本机服务，trust_env=False 绕代理——地雷 12）；
    judge_client 是评审段（LLM 网关，保持默认代理语义）。
    """
    from scripts.eval.ab_report import render_ab_report

    if progress is None:
        progress = (lambda *_: None)
    if not cases:
        raise ValueError("用例集为空——0 条用例的 A/B 是假绿")
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    if dual_judge_pairs > 0 and not second_judge_model:
        raise SystemExit("双 judge 需要第二评审模型（--second-judge-model，授权的是 deepseek-v4-flash）")
    if dual_judge_pairs > _DUAL_JUDGE_MAX_PAIRS:
        raise SystemExit(
            f"双 judge 子集 {dual_judge_pairs} 对超过授权上限 {_DUAL_JUDGE_MAX_PAIRS} 对"
            "（授权文档第一节：≤20 对 × 2 顺序）——超限是预算边界，不是调个参数就行。"
        )

    plan = plan_ab_run(cases, k, pairing, seconds_per_intent=seconds_per_intent)  # 与 dry-run 同一份账本
    exec_plan = build_execution_plan(cases, k)
    order = {
        triple_key(item["case_id"], item["arm"], item["sample_index"]): index
        for index, item in enumerate(exec_plan)
    }
    cases_by_id = {case["id"]: case for case in cases}
    results: list[dict] = []

    if resume_path is not None:
        if not resume_path.exists():
            raise SystemExit(f"续跑文件不存在：{resume_path}")
        saved = json.loads(resume_path.read_text(encoding="utf-8"))
        check_resume_config(saved, healths, expected_plan=plan)
        results = list(saved.get("results") or [])
        progress(f"续跑：已完成 {len(results)} 个样本，继续补齐余下部分")

    completed = {triple_key(r["case_id"], r["arm"], r["sample_index"]) for r in results}
    pending = partition_pending(build_execution_plan(cases, k), completed)

    partial_path = eval_dir / f"ab-partial-{stamp}.json"

    def _write_partial() -> None:
        partial_path.write_text(json.dumps({
            "label": label, "pairing": pairing, "plan": plan,
            "arm_config": arm_config, "urls": urls,
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_partial()

    execute = execute_fn if execute_fn is not None else (
        lambda case, arm, sample_index, base_url: execute_ab_case(case, arm, sample_index, client, base_url)
    )
    for item in pending:
        case = cases_by_id[item["case_id"]]
        try:
            record = await execute(case, item["arm"], item["sample_index"], urls[item["arm"]])
        except Exception as err:  # noqa: BLE001 —— 执行层抛错也降级为记名 error 行，整轮继续
            record = {
                "case_id": item["case_id"], "arm": item["arm"],
                "sample_index": item["sample_index"], "session_id": "",
                "transcript": "", "ok": False, "error": describe_error(err),
            }
        results.append(record)
        _write_partial()  # 每样本落一次盘（整份重写，同 eval_regression 纪律）
        mark = "✓" if record["ok"] else f"✗ {record['error']}"
        progress(f"   [exec] {record['case_id']} 臂 {record['arm']} k{record['sample_index']} {mark}")

    done_keys = {triple_key(r["case_id"], r["arm"], r["sample_index"]) for r in results}
    missing = [
        item for item in exec_plan
        if triple_key(item["case_id"], item["arm"], item["sample_index"]) not in done_keys
    ]
    if missing:
        raise SystemExit(
            f"执行产物不完整：{len(missing)} 个样本没有产物（如 {missing[0]['case_id']}"
            f"/臂 {missing[0]['arm']}/k{missing[0]['sample_index']}）——没跑完不许配对。"
        )
    # 前置补跑会重写已完成样本：按三元组去重（留最新），分母不许被续跑污染
    results = _latest_by_triple(results, order)

    pairs, pair_errors = build_pairs_from_executions(results, cases, k, pairing)
    judge_call = judge_factory(judge_model) if judge_factory else make_judge_call(judge_client or client, judge_model)
    progress(f"[judge] {len(pairs)} 对 × 2 顺序 = {len(pairs) * 2} 次评审调用（{judge_model}）")
    rows = await judge_pair_rows(pairs, judge_call, ground_truth, progress=progress)

    dual_payload = None
    if dual_judge_pairs > 0:
        if not second_judge_model:
            raise SystemExit("双 judge 需要第二评审模型（--second-judge-model，授权的是 deepseek-v4-flash）")
        second_call = judge_factory(second_judge_model) if judge_factory else make_judge_call(judge_client or client, second_judge_model)
        selected_pairs = select_dual_judge_pairs(pairs, dual_judge_pairs)
        rows_by_key = {(r["case_id"], r["pair_index"]): r for r in rows}
        agreement_rows: list[dict] = []
        for pair in selected_pairs:
            row = rows_by_key[(pair["case_id"], pair["pair_index"])]
            for order, (pos_left, pos_right, order_names, first_key) in (
                ("ab", (pair["left"], pair["right"], ("a", "b"), "verdict_ab")),
                ("ba", (pair["right"], pair["left"], ("b", "a"), "verdict_ba")),
            ):
                try:
                    judged = await judge_pair(
                        second_call, pair["case_prompt_text"],
                        pos_left["transcript"], pos_right["transcript"],
                        order=order_names, ground_truth=ground_truth,
                        prior_context=pair.get("prior_context", ""),
                    )
                    second_verdict: str | None = judged["winner"]
                except Exception as err:  # noqa: BLE001 —— 第二评审失败进 n_error，不塌缩
                    second_verdict = None
                    progress(f"   [dual] {pair['case_id']} {order} 第二评审失败：{describe_error(err)}")
                agreement_rows.append({
                    "case_id": pair["case_id"], "pair_index": pair["pair_index"], "order": order,
                    "verdict_first": row[first_key], "verdict_second": second_verdict,
                })
        agreement = judge_agreement(agreement_rows)
        dual_payload = {"model": second_judge_model, **agreement, "detail": agreement_rows}
        progress(f"[dual] 双 judge 一致率：{agreement['n_agree']}/{agreement['n_pairs']}")

    swap = position_swap_consistency(rows)
    feed, ci_pairs, n_flip = decisive_indicators(rows)
    summary = win_rate_summary(feed)
    n_decisive = summary["n_decisive"]
    p_value = sign_test_p(summary["wins"], summary["losses"]) if n_decisive else None
    ci = bootstrap_ci_win_rate(ci_pairs) if ci_pairs else None
    sig = significance(summary, swap, p_value, ci, min_decisive=plan["decisive_gate"])
    gate = decisive_pairs_gate(summary, plan["decisive_gate"])

    from scripts.eval.audit_cost_latency import audit_directory, summarize

    # 成本/延迟按臂各扫各的 data_dir（两臂独立 DATA_DIR 是合理配置）。
    # 流水对不上时降级为"未测定"进报告——烧完的执行与判段产物不许陪葬。
    cost_latency: dict = {}
    cost_notes: list[str] = []
    for arm in ("A", "B"):
        data_dir = str((healths.get(arm) or {}).get("data_dir") or "").strip()
        sessions = {
            r["session_id"] for r in results
            if r["arm"] == arm and r["ok"] and r.get("session_id")
        }
        if not data_dir or not sessions:
            cost_notes.append(f"臂 {arm} 成本/延迟读数未测定（data_dir 或流水会话缺失）")
            continue
        try:
            summary_cost = summarize(audit_directory(Path(data_dir) / "conversations", sessions=sessions))
        except (SystemExit, Exception) as err:  # noqa: BLE001 —— 诊断读数不许炸掉烧完的执行+判段产物
            cost_notes.append(f"臂 {arm} 成本/延迟读数未测定：{describe_error(err)}")
            continue
        cost_latency[arm] = {
            "completion_p50": summary_cost["completion_tokens"]["p50"],
            "latency_p50_s": round(summary_cost["latency_ms"]["p50"] / 1000, 1),
            "intents": summary_cost["intents"],
        }

    failed = [
        {"case_id": r["case_id"], "arm": r["arm"], "sample_index": r["sample_index"], "error": r["error"]}
        for r in results if not r["ok"]
    ]
    fault_clear_failures = [
        {"case_id": r["case_id"], "arm": r["arm"], "sample_index": r["sample_index"], "error": r["fault_clear_error"]}
        for r in results if r.get("fault_clear_error")
    ]
    report_payload = {
        "label": label,
        "stamp": stamp,
        "pairing": pairing,
        "plan": plan,
        "wall_clock_assumption": (
            f"{plan['seconds_per_intent']:.1f}s/意图（R7 实测，可 --seconds-per-intent 覆盖），"
            "两臂串行；judge 调用另计"
        ),
        "arm_lines": arm_lines,
        "arm_config": arm_config,
        "executions": {
            "total": len(results),
            "ok": sum(1 for r in results if r["ok"]),
            "failed": failed,
            "fault_clear_failures": fault_clear_failures,
        },
        "swap": swap,
        "win_rate": summary,
        "n_flip": n_flip,
        "p_value": p_value,
        "ci": ci,
        "significance": sig,
        "decisive_gate": gate,
        "dual_judge": dual_payload,
        "positive_control": positive_control,
        "guardrails": [],
        "cost_latency": cost_latency,
        "rows": rows,
        "pair_errors": pair_errors,
        "notes": [f"pair error 行 {len(pair_errors)} 条" if pair_errors else ""],
    }
    report_payload["notes"] = [n for n in report_payload["notes"] if n] + cost_notes
    report_text = render_ab_report(report_payload)

    report_path = eval_dir / f"ab-report-{stamp}.md"
    report_path.write_text(report_text, encoding="utf-8")
    run_json_path = eval_dir / f"ab-run-{stamp}.json"
    run_json_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "label": label, "pairing": pairing, "stamp": stamp,
        "judge_model": judge_model,
        "plan": plan,
        "arm_config": arm_config, "arm_lines": arm_lines, "urls": urls, "healths": healths,
        "results": results,
        "pairs": [{"case_id": p["case_id"], "pair_index": p["pair_index"],
                   "left_session": p["left"]["session_id"], "right_session": p["right"]["session_id"]}
                  for p in pairs],
        "pair_errors": pair_errors,
        "rows": rows,
        "dual_judge": dual_payload,
        "stats": {"swap": swap, "win_rate": summary, "n_flip": n_flip,
                  "p_value": p_value, "ci": ci, "significance": sig, "decisive_gate": gate},
        "cost_latency": cost_latency,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    # 正式报告落盘之后才删增量文件（同 eval_regression：顺序反了两份都没了）；
    # 被续跑的旧 partial 一并清掉——留着会被再次误 resume（配置没变时核对能通过）
    if resume_path is None or resume_path.resolve() != partial_path.resolve():
        partial_path.unlink(missing_ok=True)
    if resume_path is not None:
        resume_path.unlink(missing_ok=True)

    report_payload["report_path"] = str(report_path)
    report_payload["run_json_path"] = str(run_json_path)
    return report_payload


def _normalized_url(url_str: str) -> tuple[str, str, int]:
    """结构化比较而不是字符串比较；localhost 与 127.0.0.1 是同一台机的别名，
    不归一的话"两臂同址"会从字符串比较的缝里漏过去。"""
    url = httpx.URL(url_str)
    host = {"localhost": "127.0.0.1"}.get(str(url.host), str(url.host))
    return (url.scheme, host, url.port)


def preflight(
    health_a: dict,
    health_b: dict,
    cases: list[dict],
    *,
    arm_a_url: str,
    arm_b_url: str,
    allow_semantic_cache: bool = False,
    allow_stale_service: bool = False,
    allow_ephemeral_data_dir: bool = False,
    allow_degraded_probe: bool = False,
) -> list[str]:
    """两臂全部开跑闸：致命项直接 SystemExit，非致命告警以行返回。

    dry-run 与真实路径共用这同一道闸（不许两套前置）；真实路径烧的是
    真配额，更没有理由绕过其中任何一道。
    """
    if _normalized_url(arm_a_url) == _normalized_url(arm_b_url):
        raise SystemExit(
            f"拒绝开跑：两臂指向同一个服务（{arm_a_url} 与 {arm_b_url}）——A/B 会变成 A/A。\n"
            "两臂各是一个服务实例：基线臂不设 PROMPT_VARIANT，候选臂设之。"
        )

    healths = {"A": health_a, "B": health_b}
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
    return notes


def _arm_header_lines(healths: dict, urls: dict, judge_model: str) -> list[str]:
    """两臂配置行 + 指纹/变体告警（dry-run 与真实路径共用）。"""
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
            "不是提示词差异——A/A 校验是有意的验证手段（授权文档第一节第 3 条），"
            "其平局率结构性偏高，只作管线验证，不能当平局率先验。"
        )
    return lines


def _plan_lines(plan: dict) -> list[str]:
    """账本渲染（dry-run 与真实路径共用同一份 plan_ab_run 输出）。"""
    return [
        f"[ab] 计划：{plan['n_cases']} 条用例 × 2 臂 × k={plan['k']}"
        f"（{plan['pairing']} 配对）",
        f"  用例执行 {plan['executions']} 次｜意图 {plan['intents']} 次",
        f"  成对比较 {plan['pairs']} 对｜judge 调用 {plan['judge_calls']} 次"
        f"（含位置互换）｜decisive 上限 {plan['decisive_ceiling']}"
        f"（门槛 {plan['decisive_gate']}，需 ≥{plan['decisive_needed_ratio']:.0%} 不平局）",
        f"  墙钟估算 ≈ {plan['estimated_minutes']} 分钟（按 R7 实测 "
        f"{_DEFAULT_SECONDS_PER_INTENT:.1f}s/意图，两臂串行；judge 调用另计，可并发压缩）",
        f"  用例：{', '.join(plan['case_ids'])}",
    ]


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
    notes = preflight(
        health_a, health_b, cases,
        arm_a_url=arm_a_url, arm_b_url=arm_b_url,
        allow_semantic_cache=allow_semantic_cache,
        allow_stale_service=allow_stale_service,
        allow_ephemeral_data_dir=allow_ephemeral_data_dir,
        allow_degraded_probe=allow_degraded_probe,
    )
    healths = {"A": health_a, "B": health_b}
    urls = {"A": arm_a_url, "B": arm_b_url}
    plan = plan_ab_run(cases, k)
    lines = _arm_header_lines(healths, urls, judge_model)
    lines.extend(notes)
    lines.append("")
    lines.extend(_plan_lines(plan))
    lines.append("\n未发起任何模型调用。")
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

    # judge 模型与被测模型都必须过闸：配置行写的必须就是将来真判分的那个
    judge_model = resolve_judge_model()
    guard_judge_models(judge_model, healths, second_judge_model=args.second_judge_model)
    # 双 judge 的配置完整性在开跑前核对：烧完执行段+判段才发现缺第二评审
    # 模型，等于全部重烧（独立审查抓出的高严重度项）
    if args.dual_judge_pairs > 0:
        if not args.second_judge_model:
            raise SystemExit("双 judge 需要第二评审模型（--second-judge-model，授权的是 deepseek-v4-flash）")
        if args.dual_judge_pairs > _DUAL_JUDGE_MAX_PAIRS:
            raise SystemExit(
                f"双 judge 子集 {args.dual_judge_pairs} 对超过授权上限 {_DUAL_JUDGE_MAX_PAIRS} 对"
                "（授权文档第一节：≤20 对 × 2 顺序）——超限是预算边界，不是调个参数就行。"
            )

    notes = preflight(
        healths["A"], healths["B"], cases,
        arm_a_url=args.arm_a_url, arm_b_url=args.arm_b_url,
        allow_semantic_cache=args.allow_semantic_cache,
        allow_stale_service=args.allow_stale_service,
        allow_ephemeral_data_dir=args.allow_ephemeral_data_dir,
        allow_degraded_probe=args.allow_degraded_probe,
    )

    lines = _arm_header_lines(healths, urls, judge_model)
    lines.extend(notes)
    plan = plan_ab_run(cases, args.k, args.pairing, seconds_per_intent=args.seconds_per_intent)
    lines.append("")
    lines.extend(_plan_lines(plan))
    print("\n".join(lines), flush=True)

    if args.dry_run:
        print("\n未发起任何模型调用。")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ground_truth = build_ground_truth()
    arm_config = {
        arm: {
            "fingerprint": str(healths[arm].get("prompt_fingerprint") or ""),
            "variant": str(healths[arm].get("prompt_variant") or ""),
            "model": str(healths[arm].get("model") or ""),
        }
        for arm in ("A", "B")
    }
    print(f"\n[ab] 开跑：judge={judge_model}，产物落 {args.eval_dir}（stamp {stamp}）", flush=True)
    # 两个 client 分开：两臂是本机服务，trust_env=False 绕过本机代理
    # （地雷 12：http_proxy 会把"连不上 127.0.0.1"伪装成代理 502）；
    # LLM 网关是远端，保持默认代理语义。
    async with httpx.AsyncClient(trust_env=False) as exec_client, httpx.AsyncClient() as judge_client:
        payload = await run_ab_pipeline(
            cases=cases, k=args.k, pairing=args.pairing, judge_model=judge_model,
            healths=healths, urls=urls,
            arm_lines={arm: describe_run(healths[arm], judge_model) for arm in ("A", "B")},
            arm_config=arm_config,
            ground_truth=ground_truth, eval_dir=Path(args.eval_dir), stamp=stamp,
            label=args.label, client=exec_client, judge_client=judge_client,
            second_judge_model=args.second_judge_model,
            dual_judge_pairs=args.dual_judge_pairs,
            resume_path=Path(args.resume) if args.resume else None,
            positive_control=args.positive_control,
            seconds_per_intent=args.seconds_per_intent,
        )
    print(f"\n报告已写入：{payload['report_path']}")
    print(f"结构化结果：{payload['run_json_path']}")
    win = payload.get("win_rate") or {}
    swap_rate = (payload.get("swap") or {}).get("rate")
    print(
        f"胜率 {_fmt_rate(win.get('win_rate'))}"
        f"（A 胜 {win.get('wins', 0)}/B 胜 {win.get('losses', 0)}/平局 {win.get('ties', 0)}）"
        f"｜互换一致率 {_fmt_rate(swap_rate)}"
        f"｜判定：{'显著' if (payload.get('significance') or {}).get('significant') else '未达显著'}",
    )
    for reason in (payload.get("significance") or {}).get("reasons") or []:
        print(f"  - {reason}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B 分流跑测器")
    parser.add_argument("--arm-a-url", default=os.environ.get("EVAL_BASE_URL_A", "http://127.0.0.1:8000"))
    parser.add_argument("--arm-b-url", default=os.environ.get("EVAL_BASE_URL_B", "http://127.0.0.1:8011"))
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "eval" / "cases.yaml"))
    parser.add_argument("--only", default=None, help="只跑指定 case id（逗号分隔）")
    parser.add_argument("--tag", default=None, help="只跑带该标签的用例（先导档 = --tag smoke）")
    parser.add_argument("--exclude-tag", default=None, help="剔除带该标签的用例（主线 = --exclude-tag redteam）")
    parser.add_argument("--k", type=int, default=2, help="每臂每用例采样次数（默认 2）")
    parser.add_argument("--pairing", default="diagonal", choices=("diagonal", "cross"),
                        help="配对方式（diagonal 对数 k/用例；cross 翻倍到 k²，被测成本不变）")
    parser.add_argument("--dry-run", action="store_true", help="只跑前置检查与算式，不发模型调用")
    parser.add_argument("--resume", default=None, metavar="AB_PARTIAL_JSON",
                        help="从 eval/ab-partial-*.json 续跑（跳过已完成样本，前置按同臂同样本补回）")
    parser.add_argument("--label", default="", help="报告标签（如 先导/全量/阳性对照/A-A 校验）")
    parser.add_argument("--second-judge-model", default="",
                        help="双 judge 一致率的第二评审模型（授权：deepseek-v4-flash）")
    parser.add_argument("--dual-judge-pairs", type=int, default=0,
                        help="双 judge 子集的对数上限（授权 ≤20 对，每对 ×2 顺序）")
    parser.add_argument("--positive-control", action="store_true",
                        help="本轮臂 B 是已知更差的提示词（阳性对照，报告按有效性自证渲染）")
    parser.add_argument("--eval-dir", default=str(EVAL_DIR), help="产物目录（默认 eval/，ab- 前缀分桶）")
    parser.add_argument("--seconds-per-intent", type=float, default=None,
                        help="墙钟估算的秒/意图系数（默认 R7 实测 51.6，模型或用例集变了用实测值覆盖）")
    parser.add_argument("--allow-semantic-cache", action="store_true")
    parser.add_argument("--allow-stale-service", action="store_true")
    parser.add_argument("--allow-ephemeral-data-dir", action="store_true")
    parser.add_argument("--allow-degraded-probe", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    raise SystemExit(asyncio.run(main_async(parse_args())))


if __name__ == "__main__":
    main()
