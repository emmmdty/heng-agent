# -*- coding: utf-8 -*-
"""记忆回放对照（#12 任务 B，M0-b）：同一批含偏好的会话回放两遍。

臂 A = 记忆注入关（服务以 PREFERENCE_INJECTION_ENABLED=0 起服），臂 B = 开；
两臂同为基线提示词，唯一差异是注入开关——prompt_fingerprint 必须相同，
prompt_variant 用 mem-inject-off / mem-inject-on 自报臂身份。判段复用 A/B
工具链零新造：run_ab_pipeline（product_prefix="mem" → 产物 eval/mem-* 前缀
分桶）+ judge_pair_rows / ab_stats / make_judge_call 全套。

臂拓扑（设计决策回写在二十六期任务书第四节；两实例而非同服务两轮——
注入开关是进程级的，切开关必须重启 = 两实例）：

    # 臂 B（注入开，8014）
    VECTOR_STORE_DIR=/tmp/opencode/qdrant-memb PROMPT_VARIANT=mem-inject-on \
        uv run uvicorn app.presentation.server:app --port 8014
    # 臂 A（注入关，8015）
    VECTOR_STORE_DIR=/tmp/opencode/qdrant-mema PROMPT_VARIANT=mem-inject-off \
        PREFERENCE_INJECTION_ENABLED=0 \
        uv run uvicorn app.presentation.server:app --port 8015

两臂 DATA_DIR 都用仓库默认（流水不能落临时目录——二十三期教训）；
per-arm buyer 派生（-ab{arm}k{n}）已隔离两臂的记忆写入。

用法（烧 token 顺序：--only 先导 → 小样本 → 放大）：
    uv run python scripts/eval/mem_replay.py --dry-run            # 零模型调用
    uv run python scripts/eval/mem_replay.py --only memory-write,memory-recall --k 1 --judge-votes 3
    uv run python scripts/eval/mem_replay.py --k 2 --judge-votes 3 --judge-concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# 导入 settings 是为了它的副作用：模块顶层会 load_dotenv(.env)（地雷 8）。
from app.infrastructure.settings import load_settings  # noqa: E402,F401
from app.application.harness.run_identity import describe_run  # noqa: E402
from scripts.eval_regression import build_ground_truth, resolve_judge_model  # noqa: E402
from scripts.eval.ab_run import (  # noqa: E402
    _arm_header_lines,
    _fetch_health,
    _plan_lines,
    ab_participant_ids,
    guard_judge_models,
    plan_ab_run,
    preflight,
    run_ab_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"

# 预登记用例子集 = 主指标的人群（冻结在交接文档「五之一」任务 B 指标表）：
# 写入 / 跨会话读取 / 会话内偏好冲突 / 撤回链（撤回前写两条 + 撤回后验证）。
# 增删走回写通道，不在脚本里悄悄改。
PREFERENCE_PRESET_IDS = (
    "memory-write",
    "memory-recall",
    "preference-conflict-cheapest-vs-dislike",
    "memory-forget-setup",
    "memory-forget",
)

# 臂语义（写反 = 整轮读数方向反）：A=注入关、B=注入开
ARM_EXPECT = {
    "A": {"variant": "mem-inject-off"},
    "B": {"variant": "mem-inject-on"},
}

# 阳性对照臂语义（已知更差——工具有效性自证，模式照抄二十五期）：
# A=矛盾注入（取反偏好 seed + 正常注入；变体名含 weaker 供渲染器数据驱动
# 识别弱臂），B=正常注入。
CONTROL_ARM_EXPECT = {
    "A": {"variant": "mem-weaker-contradiction"},
    "B": {"variant": "mem-inject-on"},
}

# 矛盾注入的 seed 内容：与注入链会写入的真偏好（dislike 不要塑料材质）取反。
NEGATED_PREFERENCE = {"kind": "like", "statement": "喜欢塑料材质"}


def seed_contradiction_preferences(data_dir: str | Path, cases: list[dict], k: int) -> list[str]:
    """给**对照臂（A）**的每个买家预写取反偏好，制造已知更差的矛盾记忆状态。

    buyer id 按臂后缀派生（-abak{n}），seed 只会被对照臂读到；跑中 memory-write
    会再写入真偏好（store 幂等不冲突）→ 该买家同时持有两条矛盾偏好。
    重复 seed 幂等（整文件重写）。文件 schema 与 JsonFilePreferenceStore 一致，
    schema 不兼容 = 对照臂静默回到无偏好态 = 自证轮白跑（有测试钉住）。
    """
    prefs_dir = Path(data_dir) / "preferences"
    prefs_dir.mkdir(parents=True, exist_ok=True)
    seeded: list[str] = []
    for case in cases:
        for sample_index in range(k):
            buyer_id = ab_participant_ids(case, "A", sample_index)[1]
            if buyer_id in seeded:  # 多个用例声明同一 buyer_id 时只 seed 一次
                continue
            payload = [{
                **NEGATED_PREFERENCE,
                "buyer_id": buyer_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }]
            (prefs_dir / f"{buyer_id}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8",
            )
            seeded.append(buyer_id)
    return seeded


def find_eval_preference_leftovers(data_dir: str | Path) -> list[Path]:
    """data/preferences/ 下的 eval-* 残留（历期评测写入的买家偏好）。

    回放的买家 id 全部以 eval- 开头（ab_participant_ids 派生），真实买家
    不会撞这个前缀。
    """
    prefs = Path(data_dir) / "preferences"
    if not prefs.is_dir():
        return []
    return sorted(prefs.glob("eval-*.json"))


def purge_eval_preference_leftovers(data_dir: str | Path, stamp: str) -> tuple[list[str], str]:
    """把 eval-* 残留移入备份目录（不删——评测状态可恢复、误伤可回滚）。

    返回 (移走的文件名列表, 备份目录)。没有残留时返回 ([], "")。
    """
    leftovers = find_eval_preference_leftovers(data_dir)
    if not leftovers:
        return [], ""
    backup = Path(data_dir) / f"preferences-leftovers-backup-{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for path in leftovers:
        target = backup / path.name
        if target.exists():  # 同 stamp 重跑：备份里已有同名，直接覆盖移动
            target.unlink()
        path.rename(target)
        moved.append(path.name)
    return moved, str(backup)


def select_replay_cases(cases: list[dict], only: str | None = None) -> list[dict]:
    """按预登记子集挑用例；--only 在子集内取交（M1 先导档挑 2 条用）。

    预登记 id 在 cases.yaml 里找不到 = 用例被改名/删除——报错退出而不是
    静默少跑：指标人群缩水后读数外观完全正常，污染只会下结论时才显形。
    """
    cases_by_id = {case["id"]: case for case in cases}
    missing = [cid for cid in PREFERENCE_PRESET_IDS if cid not in cases_by_id]
    if missing:
        raise SystemExit(
            f"预登记用例在 cases.yaml 里找不到（被改名或删除？）：{'、'.join(missing)}\n"
            "人群冻结在指标表——要改先回写二十六期任务书第四节。"
        )
    selected_ids = list(PREFERENCE_PRESET_IDS)
    if only:
        wanted = [item.strip() for item in only.split(",") if item.strip()]
        outside = [cid for cid in wanted if cid not in PREFERENCE_PRESET_IDS]
        if outside:
            raise SystemExit(
                f"--only 指到预登记子集外（{'、'.join(outside)}）——记忆回放的人群是冻结口径，"
                f"要改走回写通道。可用：{'、'.join(PREFERENCE_PRESET_IDS)}"
            )
        selected_ids = [cid for cid in PREFERENCE_PRESET_IDS if cid in wanted]
    if not selected_ids:
        raise SystemExit("选出的用例集为空——0 条用例的回放是假跑")
    return [cases_by_id[cid] for cid in selected_ids]


def preflight_arms(healths: dict, arm_expect: dict = ARM_EXPECT) -> None:
    """记忆回放特有的两臂语义预检（ab_run.preflight 之上的第二道闸）。

    必须在 ab_run.preflight 之后跑：后者已拦探活降级/临时目录/同 URL 等项，
    本函数不再重复；单独复用本函数时那些闸不在。

    A/B 提示词对比量的是"两臂不同"，记忆回放量的是"两臂同稿、只差注入
    开关"——指纹必须相同、变体必须互异且与臂语义一致。混装（一臂跑错
    代码 / 漏设 env）在这里拦下，而不是变成一轮无法归因的读数。
    """
    problems: list[str] = []
    for arm in ("A", "B"):
        health = healths.get(arm) or {}
        if not health:
            problems.append(f"臂 {arm} 的 /health 不可用（服务没起或端口不对）")
            continue
        code = health.get("code") or {}
        if code.get("stale"):
            files = "、".join(code.get("stale_files") or []) or "若干文件"
            problems.append(f"臂 {arm} 跑的是旧代码（stale：{files}）——先重启再跑")
        if health.get("semantic_cache"):
            problems.append(f"臂 {arm} 语义缓存开着——回复会命中缓存，评的是缓存不是 Agent")

    fingerprints = {arm: (healths.get(arm) or {}).get("prompt_fingerprint") for arm in ("A", "B")}
    if not all(fingerprints.values()):
        problems.append(
            f"两臂指纹信息不全（A={fingerprints['A']} / B={fingerprints['B']}）——先确认服务是当前代码"
        )
    elif len(set(fingerprints.values())) != 1:
        problems.append(
            f"两臂提示词指纹不同（A={fingerprints['A']} / B={fingerprints['B']}）——"
            "记忆回放要求两臂同稿（只差注入开关）；指纹不同量到的是提示词差异不是注入差异"
        )

    for arm, expect in arm_expect.items():
        actual = (healths.get(arm) or {}).get("prompt_variant") or ""
        if actual != expect["variant"]:
            problems.append(
                f"臂 {arm} 的变体自报是 {actual!r}，应为 {expect['variant']!r}——"
                "起服时 PROMPT_VARIANT 没设对（注入关的臂还要 PREFERENCE_INJECTION_ENABLED=0）"
            )

    models = {arm: (healths.get(arm) or {}).get("model") for arm in ("A", "B")}
    if models["A"] and models["B"] and models["A"] != models["B"]:
        problems.append(f"两臂被测模型不同（A={models['A']} / B={models['B']}）——读数混入模型差异")

    if problems:
        raise SystemExit("记忆回放两臂预检未过：\n  - " + "\n  - ".join(problems))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="记忆回放对照（#12 任务 B）")
    parser.add_argument("--arm-a-url", default=os.environ.get("EVAL_BASE_URL_A", "http://127.0.0.1:8015"),
                        help="臂 A（注入关）服务地址，默认 8015")
    parser.add_argument("--arm-b-url", default=os.environ.get("EVAL_BASE_URL_B", "http://127.0.0.1:8014"),
                        help="臂 B（注入开）服务地址，默认 8014")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "eval" / "cases.yaml"))
    parser.add_argument("--only", default=None,
                        help="只跑指定 case id（逗号分隔；必须在预登记子集内，M1 先导档用）")
    parser.add_argument("--k", type=int, default=1,
                        help="每臂每用例采样次数（默认 1；M1 先导 1，M2 放大 2）")
    parser.add_argument("--pairing", default="diagonal", choices=("diagonal", "cross"))
    parser.add_argument("--judge-votes", type=int, default=3,
                        help="每对每序投票次数取众数（指标口径默认 3）")
    parser.add_argument("--judge-concurrency", type=int, default=4,
                        help="judge 按对并发上界（授权口径 ~4）")
    parser.add_argument("--seconds-per-intent", type=float, default=None,
                        help="墙钟估算的秒/意图系数（默认按 R7 实测）")
    parser.add_argument("--resume", default=None, metavar="MEM_PARTIAL_JSON",
                        help="从 eval/mem-partial-*.json 续跑（跳过已完成样本，前置按同臂同样本补回）")
    parser.add_argument("--label", default="", help="报告标签（如 M1 先导/M2 全量）")
    parser.add_argument("--positive-control", action="store_true",
                        help="阳性对照模式：臂 A=矛盾注入（已知更差，起服前 seed 取反偏好、"
                             "变体须为 mem-weaker-contradiction），臂 B=正常注入；"
                             "报告按有效性自证渲染。子集建议 memory-write,memory-recall,"
                             "preference-conflict-cheapest-vs-dislike（写入用例是被测链路的前置）")
    parser.add_argument("--eval-dir", default=str(EVAL_DIR), help="产物目录（mem- 前缀分桶）")
    parser.add_argument("--dry-run", action="store_true", help="只跑前置检查与账本，不发模型调用")
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    urls = {"A": args.arm_a_url, "B": args.arm_b_url}
    healths = {}
    for arm in ("A", "B"):
        try:
            healths[arm] = await _fetch_health(urls[arm])
        except Exception as err:  # noqa: BLE001 —— 探活失败必须留名（哪个臂、什么原因）
            raise SystemExit(f"臂 {arm}（{urls[arm]}）的 /health 不可达：{err}") from err

    with open(args.cases, encoding="utf-8") as handle:
        cases = yaml.safe_load(handle)["cases"]
    cases = select_replay_cases(cases, only=args.only)

    # judge 模型与被测模型都必须过闸：模型三选纪律（mimo 被测 / longcat judge）
    judge_model = resolve_judge_model()
    guard_judge_models(judge_model, healths, second_judge_model="")

    # 两道闸：ab_run 既有前置（探活降级/临时目录/故障支持）+ 记忆回放特有语义
    notes = preflight(
        healths["A"], healths["B"], cases,
        arm_a_url=urls["A"], arm_b_url=urls["B"],
    )
    arm_expect = CONTROL_ARM_EXPECT if args.positive_control else ARM_EXPECT
    preflight_arms(healths, arm_expect)

    # 两臂必须共用仓库 data/（流水与偏好不能分家——mem_deposit 对账依赖）
    data_dirs = {str(h.get("data_dir") or "") for h in healths.values()}
    data_dirs.discard("")
    if len(data_dirs) != 1:
        raise SystemExit(
            f"两臂 DATA_DIR 不一致（{sorted(data_dirs)}）——记忆回放要求两臂共用仓库 data/"
        )
    data_dir = next(iter(data_dirs))

    # 跨期残留清洗：历期评测的 eval-* 买家偏好会让注入臂的**写入用例**看到
    # '已存档'而跳过工具调用（M1 先导实测），污染沉淀提取与写入对读数。
    leftovers = find_eval_preference_leftovers(data_dir)
    if args.dry_run:
        if leftovers:
            print(
                f"⚠️ 检测到 {len(leftovers)} 个历期 eval-* 偏好残留（真实跑测会自动移入备份目录）：\n"
                + "\n".join(f"  - {p.name}" for p in leftovers),
            )
        if args.positive_control:
            plan_buyers = [
                ab_participant_ids(case, "A", i)[1]
                for case in cases for i in range(args.k)
            ]
            print(
                f"[对照模式] dry-run 预览：开跑前将给 {len(plan_buyers)} 个对照臂买家 seed 取反偏好"
                "（喜欢塑料材质）：\n  " + "、".join(plan_buyers),
            )
    else:
        moved, backup_dir = purge_eval_preference_leftovers(data_dir, stamp=datetime.now().strftime("%Y%m%d-%H%M%S"))
        if moved:
            print(f"[mem] 已清洗 {len(moved)} 个 eval-* 偏好残留 → {backup_dir}：{'、'.join(moved)}", flush=True)
        if args.positive_control:
            # seed 必须在 purge 之后：先清洗历期残留、再种矛盾偏好，顺序反了 seed 会被清掉
            seeded = seed_contradiction_preferences(data_dir, cases, args.k)
            print(
                f"[对照模式] 已给 {len(seeded)} 个对照臂买家 seed 取反偏好"
                "（喜欢塑料材质）——已知更差的矛盾记忆状态",
                flush=True,
            )

    lines = _arm_header_lines(healths, urls, judge_model)
    lines.extend(notes)
    plan = plan_ab_run(cases, args.k, args.pairing,
                       seconds_per_intent=args.seconds_per_intent, votes=args.judge_votes)
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
    print(f"\n[mem] 开跑：judge={judge_model}，产物落 {args.eval_dir}（mem- 前缀，stamp {stamp}）", flush=True)
    # 两个 client 分开：两臂是本机服务，trust_env=False 绕本机代理（地雷 12）；
    # LLM 网关是远端，保持默认代理语义——与 ab_run 同一条纪律。
    async with httpx.AsyncClient(trust_env=False) as exec_client, httpx.AsyncClient() as judge_client:
        payload = await run_ab_pipeline(
            cases=cases, k=args.k, pairing=args.pairing, judge_model=judge_model,
            healths=healths, urls=urls,
            arm_lines={arm: describe_run(healths[arm], judge_model) for arm in ("A", "B")},
            arm_config=arm_config,
            ground_truth=ground_truth, eval_dir=Path(args.eval_dir), stamp=stamp,
            label=args.label or ("M2 阳性对照（矛盾注入）" if args.positive_control else "记忆回放对照"),
            client=exec_client, judge_client=judge_client,
            resume_path=Path(args.resume) if args.resume else None,
            seconds_per_intent=args.seconds_per_intent,
            votes=args.judge_votes,
            judge_concurrency=args.judge_concurrency,
            product_prefix="mem",
            positive_control=args.positive_control,
            progress=print,
        )
    from scripts.eval.ab_report import _fmt_rate

    print(f"\n报告已写入：{payload['report_path']}")
    print(f"结构化结果：{payload['run_json_path']}")
    win = payload.get("win_rate") or {}
    print(
        f"胜率 {_fmt_rate(win.get('win_rate'))}"
        f"（A={win.get('wins', 0)} / B={win.get('losses', 0)} / 平={win.get('ties', 0)}，"
        f"decisive {win.get('n_decisive', 0)}）"
        f"｜B=注入开：显著>50% 才算记忆层自进化有效",
    )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async(parse_args())))


if __name__ == "__main__":
    main()
