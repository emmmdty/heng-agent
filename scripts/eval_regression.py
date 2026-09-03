# -*- coding: utf-8 -*-
"""评测回归脚本

用法（先启动服务）：
    uv run uvicorn app.presentation.server:app --port 8000
    uv run python scripts/eval_regression.py [--cases eval/cases.yaml] [--only case_id]

judge 模型建议**与被测模型不同**（`EVAL_JUDGE_MODEL=<另一个模型>`）：
同模型自评存在自我偏好偏差，判出来的分数会系统性偏高。

流程：逐 case 顺序打 POST /commerce/intents（同 case 多轮复用会话）→
LLM judge 按 Rubric（P0 数字事实 / P1 行为命中 / P2 表达）逐条打分 →
输出 eval/report-{时间戳}.md。

case 可选 prior_context 字段：告知 judge 本会话之前已成立的事实（如跨会话写入的长期偏好），
否则 judge 只看本会话 transcript，会把"正确应用了历史偏好"误判为"无据添加"。

评分口径：P0 任一不过 = 该 case 直接 FAIL；总分 = P0 0.5 / P1 0.35 / P2 0.15 加权。
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

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# 导入 settings 是为了它的副作用：模块顶层会 load_dotenv(.env)。
# 本脚本的 judge 直连 httpx、绕过模型层，直接读 os.environ["LLM_BASE_URL"]；
# 而 .env 只有在 settings 被导入时才会进环境。缺了这一行，按 README 用 .env
# 配置的人跑本脚本必然 KeyError，且原先该异常被吞进 transcript，13 条全 ERROR
# 也看不出原因——两个问题叠在一起，排查成本远高于缺陷本身。
from app.infrastructure.settings import load_settings  # noqa: E402,F401
from app.application.harness.run_identity import describe_run  # noqa: E402
from app.infrastructure.transient import is_transient_error  # noqa: E402

# 被测服务地址。默认 8000；`EVAL_BASE_URL` 可指到别的端口——
# 起第二个实例做 --dry-run 体检（或服务本来就不在 8000）时需要它。
BASE_URL = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# judge 不经模型层闸门（直连 httpx），自己退避重试，避免主模型限流时整轮评测报废
_JUDGE_MAX_RETRIES = 3
_JUDGE_RETRY_BASE_SECONDS = 8.0

JUDGE_SYSTEM_PROMPT = """你是严格的电商 Agent 评测员。给你一段"买家多轮提问与 Agent 回复"的对话记录、
商品库事实表（ground truth），以及分级评分细则（P0 数字事实与安全底线 / P1 行为与命中 / P2 表达）。
部分 case 会额外给出"会话前置事实"（如买家在早先会话里已写入的长期偏好）：这些事实真实有效，
即使它不出现在本段对话记录里，Agent 引用或应用它也不算编造。
逐条判断细则是否满足：数字事实类细则以商品库事实表为基准比对（回复中的价格/库存与事实表一致即通过，
运费关税等衍生数字只要金额自洽且未与事实表矛盾即通过）；行为类细则以对话记录与会话前置事实为依据，
拿不准按不通过处理。
每条细则先在 reason 里完成推理，再给出 pass 定论；pass 必须与 reason 的最终结论一致。
只输出 JSON（字段顺序固定：criterion → reason → pass）：
{"p0": [{"criterion": "...", "reason": "...", "pass": true/false}],
 "p1": [...], "p2": [...]}"""


def build_ground_truth() -> str:
    """从种子商品数据与汇率表生成事实表，供 judge 校验数字事实。"""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from app.domain.catalog.exchange_rate import ExchangeRateTable
    from app.infrastructure.persistence.seed_products import build_seed_products

    lines = ["| product_id | 标题 | 品类 | sku | 价格 | 库存 |", "|---|---|---|---|---|---|"]
    for product in build_seed_products():
        for sku in product.skus:
            lines.append(
                f"| {product.product_id} | {product.title} | {product.category} "
                f"| {sku.sku_id}({sku.spec}) | {sku.price} | {sku.stock} |",
            )
    rates = ", ".join(f"1 {cur} = {rate} CNY" for cur, rate in ExchangeRateTable().rates_to_cny.items())
    lines.append("")
    lines.append(f"系统汇率表（到手价工具按此折算目标币种，折算后的价格属于工具返回，不算自行估算）：{rates}")
    lines.append("")
    lines.extend(_landed_price_rules())
    return "\n".join(lines)


def _landed_price_rules() -> list[str]:
    """到手价规则表。

    不给这一段的话，judge 手上只有商品价与汇率，对运费/关税/免税额度
    **只能验自洽、验不了正确**——Agent 自圆其说就能过。实测代价：
    模型写出"1,199 × 12% ≈ ¥3.48"（计税基数错，最终数字碰巧对），
    judge 的判词是"与商品库价格及自洽运费/关税一致"。

    与金额出处校验互补：出处校验管"数字有没有来源"，这段管"数字对不对"。
    """
    from app.domain.catalog.exchange_rate import ExchangeRateTable
    from app.domain.shipping.tariff_schedule import (
        _BASE_FREIGHT_CNY_MINOR,
        _TARIFF_RATES,
        TariffSchedule,
    )

    # 免税额度改走公开方法（十一期）：原先直接 import `_DE_MINIMIS_CNY_MINOR`，
    # 规则表一改存储结构这里就断——事实基准依赖领域层私有常量本身就是缝。
    tariff = TariffSchedule(rates=ExchangeRateTable())

    lines = ["到手价规则表（工具按此计算；Agent 报的运费/关税/免税额度必须与本表推出的结果一致）：", ""]
    lines.append("    到手价 = 商品小计 + 运费 + 关税")
    lines.append("    关税   = **超出免税额度的部分** × 品类费率（小计未超额度则为 0）")
    lines.append("             即：应税基数 = max(0, 小计 − 免税额度)，关税 = 应税基数 × 费率。")
    lines.append("             写成「整单金额 × 费率」是错的，哪怕最终数字碰巧对得上")
    lines.append("    运费   = 基础运费 × (1 + 0.6 × (总件数 - 1))，即首件全价 + 每件续件 60%；")
    lines.append("             多个商品合并一单时按**整批总件数**算一次，不是各单品运费相加")
    lines.append("")
    lines.append("| 目的国 | 基础运费(CNY) | 免税额度(原生口径) | 免税额度(CNY) | 品类关税费率 |")
    lines.append("|---|---|---|---|---|")
    for dest in sorted(_TARIFF_RATES):
        freight = _BASE_FREIGHT_CNY_MINOR[dest] / 100
        # 两个口径都给：额度本来是各国用自己货币定义的（US 800 USD、EU 150 EUR），
        # 只给 CNY 的话，Agent 跨币种表述"美国免税门槛 $800"时 judge 无从核对。
        native = tariff.de_minimis_native(dest)
        de_minimis = tariff.de_minimis(dest, "CNY").to_major_units()
        # 精度不能丢：US 旅行装备是 7.5%，`:.0%` 会显示成 8%——
        # judge 拿着 8% 去核对，正确的关税反而会被判错。
        rates_text = "、".join(
            f"{cat} {rate * 100:g}%" for cat, rate in _TARIFF_RATES[dest].items()
        ).replace("*", "其他")
        lines.append(
            f"| {dest} | {freight:g} | {native.to_major_units():g} {native.currency} "
            f"| {de_minimis:g} | {rates_text} |",
        )
    lines.append("")
    lines.append(
        "注：免税额度按**整批小计**判定（不是逐件）；混合品类超额时，"
        "超出部分按各行金额占比分摊后各按自己的品类费率计征。",
    )
    return lines


async def call_judge(
    client: httpx.AsyncClient,
    transcript: str,
    rubric: dict,
    ground_truth: str,
    prior_context: str = "",
) -> dict:
    prior_block = f"## 会话前置事实\n{prior_context}\n\n" if prior_context else ""
    payload = {
        # judge 可独立指定模型：主模型切新版/被限流时，评分基准不跟着飘
        "model": os.environ.get("EVAL_JUDGE_MODEL") or os.environ.get("LLM_MODEL", "qwen-plus"),
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## 商品库事实表\n{ground_truth}\n\n"
                    f"{prior_block}"
                    f"## 对话记录\n{transcript}\n\n"
                    f"## 评分细则\n{json.dumps(rubric, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }

    last_error: Exception | None = None
    for attempt in range(_JUDGE_MAX_RETRIES):
        try:
            response = await client.post(
                f"{os.environ['LLM_BASE_URL'].rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception as err:  # noqa: BLE001
            if not is_transient_error(err) or attempt == _JUDGE_MAX_RETRIES - 1:
                raise
            last_error = err
            delay = _JUDGE_RETRY_BASE_SECONDS * (2**attempt)
            print(f"   judge 遇限流，{delay:.0f}s 后重试：{err}", flush=True)
            await asyncio.sleep(delay)
    raise last_error if last_error else RuntimeError("judge 重试耗尽")


def score_case(judged: dict) -> tuple[float, bool]:
    """返回 (加权得分, p0全过)。空档位按满分处理。"""

    def ratio(items: list) -> float:
        return sum(1 for item in items if item.get("pass")) / len(items) if items else 1.0

    p0_ratio, p1_ratio, p2_ratio = ratio(judged.get("p0", [])), ratio(judged.get("p1", [])), ratio(judged.get("p2", []))
    weighted = 0.5 * p0_ratio + 0.35 * p1_ratio + 0.15 * p2_ratio
    return round(weighted, 3), p0_ratio == 1.0


def partial_path(stamp: str) -> Path:
    return PROJECT_ROOT / "eval" / f"partial-{stamp}.json"


def write_partial(path: Path, run_line: str, results: list[dict]) -> None:
    """每条用例跑完就把**当前全部结果**重写一遍。

    为什么不是"异常时才保存"：中断最常见的形态是 Ctrl-C 和进程被杀，
    那两种情况下没有机会执行保存逻辑。唯一可靠的做法是每条跑完就写。
    为什么是全量覆盖而不是 append：JSON 数组 append 要么写坏格式、
    要么得自己维护括号；而整份重写的代价相对一条用例 2-3 分钟的模型调用可以忽略。
    """
    path.write_text(
        json.dumps({"run_line": run_line, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_partial(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"续跑文件不存在：{path}（eval/partial-*.json 是整轮跑测过程中落的）")
    return json.loads(path.read_text(encoding="utf-8"))


def plan_resume(cases: list[dict], completed: set[str]) -> tuple[list[dict], list[str]]:
    """续跑要跑哪些用例。

    跳过已完成的，但**待跑用例声明的前置若也在已完成列表里，要一并重跑**：
    `memory-recall` 依赖 `memory-write` 先把偏好写进去，跳过前置直接跑后继，
    评的是一个不成立的前提，而分数看上去完全正常——这类错误没有任何东西会报警。

    返回的顺序与 cases.yaml 一致：用例之间的依赖靠顺序执行保证。
    """
    pending = [case for case in cases if case["id"] not in completed]
    prerequisites = {
        req
        for case in pending
        for req in (case.get("requires") or [])
        if req in completed
    }
    run_ids = {case["id"] for case in pending} | prerequisites
    todo = [case for case in cases if case["id"] in run_ids]
    skipped = [case["id"] for case in cases if case["id"] not in run_ids]
    return todo, skipped


def merge_results(
    cases: list[dict], previous: list[dict], fresh: list[dict],
) -> list[dict]:
    """重跑过的取新结果、没跑的沿用旧结果，并按用例顺序排好。

    只保留**本轮选中**的用例：用 --tag 缩小范围后续跑时，
    旧 partial 里多出来的用例不能混进报告，否则总览的分母是错的。
    """
    by_id = {result["id"]: result for result in previous}
    by_id.update({result["id"]: result for result in fresh})
    return [by_id[case["id"]] for case in cases if case["id"] in by_id]


def declared_fault_components(cases: list[dict]) -> set[str]:
    """本轮选中的用例一共声明了哪些故障组件。"""
    components: set[str] = set()
    for case in cases:
        components.update(case.get("faults") or [])
    return components


def guard_fault_support(cases: list[dict], health: dict) -> None:
    """开跑前拦截：有用例声明了故障，而服务没启用注入。

    不拦的后果很安静：那条用例会在**一切正常**的情况下跑完，然后大概率 PASS——
    判据成了绿色装饰，还烧了一轮配额。与 CI 里"没数据就当通过"是同一个陷阱：
    一个永远绿的判据比没有判据更坏，它让人以为这块被覆盖了。

    没有用例声明故障时不做任何检查——绝大多数轮次都不注入，
    把它做成硬前置等于给所有人加一道无谓的门槛。
    """
    needed = declared_fault_components(cases)
    if not needed:
        return
    injection = health.get("fault_injection")
    if isinstance(injection, dict) and injection.get("enabled"):
        return
    raise SystemExit(
        f"本轮有用例声明了故障注入（{sorted(needed)}），但服务未启用。\n"
        f"请以 FAULT_INJECTION_ENABLED=1 重启服务后再跑：\n"
        f"    FAULT_INJECTION_ENABLED=1 uv run uvicorn app.presentation.server:app --port 8000\n"
        f"（不拦下来的话，这些用例会在精排/向量库完全正常的情况下跑完并大概率 PASS）",
    )


async def apply_faults(client: httpx.AsyncClient, components: list[str]) -> None:
    """设置当前进程的故障注入；空列表 = 清空。

    失败**必须抛**：吞掉异常继续跑，等于在没有故障的情况下评一条故障用例，
    结论是假的而且看不出来。
    """
    response = await client.post(
        f"{BASE_URL}/debug/faults", json={"components": components}, timeout=30,
    )
    response.raise_for_status()


async def run_case(client: httpx.AsyncClient, case: dict, ground_truth: str) -> dict:
    session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    buyer_id = case.get("buyer_id") or f"eval-buyer-{case['id']}"
    faults = list(case.get("faults") or [])
    if faults:
        # 注入失败就让异常冒到上层，把这条用例判 ERROR——不能在无故障的情况下评它
        await apply_faults(client, faults)
    transcript_lines: list[str] = []
    for query in case["queries"]:
        response = await client.post(
            f"{BASE_URL}/commerce/intents",
            json={
                "shopping_session_id": session_id,
                "buyer_id": buyer_id,
                "locale": "zh-CN",
                "currency": "CNY",
                "raw_query": query,
            },
            timeout=600,
        )
        response.raise_for_status()
        final_text = response.json()["final_text"]
        transcript_lines.append(f"[买家] {query}\n[Agent] {final_text}")

    if faults:
        # 无论判分成功与否都要清干净：漏清会让**后面每一条用例**都带着故障跑，
        # 而报告里只有这一条写着 faults——那种读数没人能解释
        await apply_faults(client, [])

    transcript = "\n\n".join(transcript_lines)
    judged = await call_judge(client, transcript, case["rubric"], ground_truth, case.get("prior_context", ""))
    score, p0_all_pass = score_case(judged)
    return {
        "id": case["id"],
        # 报告要记下这一轮落的是哪份流水：金额出处门禁据此把扫描范围收敛到本轮，
        # 否则它扫的是累积目录里的全部历史，读数只会越积越高（见
        # scripts/eval/audit_number_provenance.py 的「扫描范围」）。
        "session_id": session_id,
        "description": case["description"],
        # 报告要写明这条是在什么故障下跑的，否则"检索档位不对"会被归因到检索参数
        "faults": faults,
        "score": score,
        "p0_pass": p0_all_pass,
        "verdict": "PASS" if p0_all_pass and score >= 0.7 else "FAIL",
        "judged": judged,
        "transcript": transcript,
    }


def render_report(results: list[dict], run_line: str = "") -> str:
    lines = [
        f"# Globex 评测回归报告（{datetime.now().strftime('%Y-%m-%d %H:%M')}）",
        "",
        # 配置行紧跟标题：分数变了要能先排除"是不是换配置了"，再去改 Agent
        f"跑测配置：{run_line}" if run_line else "跑测配置：未知（服务未报）",
        "",
        f"总览：{sum(1 for r in results if r['verdict'] == 'PASS')}/{len(results)} PASS，"
        f"平均分 {sum(r['score'] for r in results) / len(results):.3f}",
        "",
        "| case | 描述 | 得分 | P0 | 故障注入 | 结果 |",
        "|------|------|------|-----|------|------|",
    ]
    for r in results:
        # 故障注入单列一栏：不写的话，"这条的检索档位不对"会被归因到检索参数，
        # 而真相是这一条本来就是在精排被人为打挂的情况下跑的
        faults = "/".join(r.get("faults") or []) or "—"
        lines.append(
            f"| {r['id']} | {r['description']} | {r['score']} | "
            f"{'通过' if r['p0_pass'] else '不通过'} | {faults} | {r['verdict']} |",
        )
    lines.append("")
    for r in results:
        lines.append(f"## {r['id']}（{r['verdict']}，{r['score']}）")
        for level in ("p0", "p1", "p2"):
            for item in r["judged"].get(level, []):
                mark = "PASS" if item.get("pass") else "FAIL"
                lines.append(f"- [{level.upper()}][{mark}] {item['criterion']}：{item.get('reason', '')}")
        lines.append("")
        lines.append("<details><summary>对话记录</summary>\n")
        lines.append(r["transcript"])
        lines.append("\n</details>\n")
    return "\n".join(lines)


_TAG_ALL = "full"  # 隐含标签：所有用例都属于它，无需逐条标注


async def _guard_semantic_cache(allow: bool) -> dict:
    """语义缓存开着时拒绝跑回归，并把 /health 原样带回去写进报告。

    实测踩过：一条 case 的错误回复进了缓存，之后改 prompt 重跑，回复一字不差——
    评测彻底失去了检验能力。回归必须评 Agent 真实行为。

    顺路返回 health：报告需要记下"这个读数是哪套配置跑出来的"（模型、提示词版本、
    精排开关、字面门限），而这一次请求本来就要发，等于零额外成本。
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            health = (await client.get(f"{BASE_URL}/health")).json()
    except Exception as err:  # noqa: BLE001 —— 拿不到 health 不阻断，后续请求自会报错
        print(f"警告：无法读取 /health（{err}），跳过缓存检查", flush=True)
        return {}
    if health.get("semantic_cache") and not allow:
        raise SystemExit(
            "拒绝跑回归：服务端语义缓存处于开启状态，评分会变成评缓存。\n"
            "请用 SEMANTIC_CACHE_ENABLED=0 重启服务后重试，例如：\n"
            "  SEMANTIC_CACHE_ENABLED=0 docker compose -f docker/docker-compose.yaml up -d app worker\n"
            "确认要带缓存跑则加 --allow-semantic-cache。",
        )
    return health


def select_cases(cases: list[dict], only: str | None = None, tag: str | None = None) -> list[dict]:
    """按 --only / --tag 挑用例。

    分层的目的是让扩容后的用例集还跑得起：日常 smoke（8-10 条，10 分钟内），
    发版前 full（全部）。`full` 不需要逐条标注——所有用例隐含属于它，
    否则新增用例漏标 tag 就会永远不被跑到，而"静默不跑"和真绿外观完全一样。

    选空了一律报错退出：跑 0 条会产出一份"全过"的报告，
    它和真的全过在报告里长得一模一样。
    """
    selected = cases
    if only is not None:
        selected = [c for c in selected if c["id"] == only]
    elif tag is not None and tag != _TAG_ALL:
        selected = [c for c in selected if tag in (c.get("tags") or [])]

    if not selected:
        known = sorted({t for c in cases for t in (c.get("tags") or [])} | {_TAG_ALL})
        raise SystemExit(
            f"没有用例匹配（--only={only} --tag={tag}）。\n"
            f"  可用标签：{'、'.join(known)}\n"
            f"  可用用例：{'、'.join(c['id'] for c in cases)}",
        )
    return selected


def _guard_stale_service(health: dict, allow: bool) -> None:
    """服务进程比磁盘上的代码旧时拒绝跑回归。

    九期实测踩过，代价是一轮白跑的定向回归：uvicorn 16:43:06 启动，
    `tariff_schedule.py` 16:49:20 修完（给 to_dict 加 de_minimis_threshold_major），
    进程没重启。之后重跑的两条用例打的都是装着旧代码的服务，
    新字段一次也没出现在工具返回里——而 408 单测全绿（单测读磁盘），
    /health 报的配置行与新服务一字不差，报告也照样 PASS。

    没有这道拦截，这种轮次唯一的症状是"修了但读数没变"，
    而交接文档会把人引向"再去查代码里第三条没覆盖的路径"——代码是对的。
    整轮 13 条 25-40 分钟真金白银，值得在开跑前花这一次 /health 拦下来。
    """
    code = health.get("code")
    if not isinstance(code, dict) or not code.get("stale") or allow:
        return
    listed = "、".join(code.get("stale_files") or []) or "若干文件"
    raise SystemExit(
        f"拒绝跑回归：被测服务跑的是旧代码。\n"
        f"  服务启动于 {code.get('started_at')}，"
        f"但 {listed} 在那之后被改过（最新 {code.get('source_mtime')}）。\n"
        f"评的会是修复前的行为，而单测和 /health 都不会报警。\n"
        f"请重启服务后重试：\n"
        f"  uv run uvicorn app.presentation.server:app --port 8000\n"
        f"确认要带旧代码跑则加 --allow-stale-service。",
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "eval" / "cases.yaml"))
    parser.add_argument("--only", default=None, help="只跑指定 case id")
    parser.add_argument(
        "--tag",
        default=None,
        help=f"只跑带该标签的用例（smoke 为日常档；{_TAG_ALL} 或不传为全部）",
    )
    parser.add_argument(
        "--allow-semantic-cache",
        action="store_true",
        help="允许在语义缓存开启的环境下跑（不推荐，评的会是缓存而不是 Agent）",
    )
    parser.add_argument(
        "--allow-stale-service",
        action="store_true",
        help="允许在服务代码比磁盘旧的情况下跑（不推荐，评的会是修复前的行为）",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PARTIAL_JSON",
        help="从 eval/partial-*.json 续跑：跳过已完成的用例，前置（requires）自动补回",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只跑前置检查并打印本轮将跑哪些用例，一次模型调用都不发",
    )
    args = parser.parse_args()

    health = await _guard_semantic_cache(args.allow_semantic_cache)
    _guard_stale_service(health, args.allow_stale_service)
    run_line = describe_run(health, os.environ.get("EVAL_JUDGE_MODEL", ""))
    print(f"跑测配置：{run_line}\n", flush=True)

    with open(args.cases, encoding="utf-8") as f:
        cases = yaml.safe_load(f)["cases"]
    cases = select_cases(cases, only=args.only, tag=args.tag)
    guard_fault_support(cases, health)

    previous: list[dict] = []
    todo = cases
    if args.resume:
        payload = load_partial(Path(args.resume))
        previous = payload.get("results", [])
        todo, skipped = plan_resume(cases, {result["id"] for result in previous})
        rerun_prereq = [
            case["id"] for case in todo
            if case["id"] in {result["id"] for result in previous}
        ]
        print(f"续跑：跳过 {len(skipped)} 条已完成，待跑 {len(todo)} 条", flush=True)
        if rerun_prereq:
            print(f"      其中 {rerun_prereq} 是待跑用例的前置，已完成但会重跑", flush=True)

    ground_truth = build_ground_truth()

    if args.dry_run:
        # 把"能不能跑"的判断提前到 5 秒内：白等 90 分钟才发现服务跑着旧代码，
        # 是这套流程里最贵的一种失败。这里不加新判据，只是把已有判据挪到前面。
        with_faults = [case["id"] for case in todo if case.get("faults")]
        print(f"\n[dry-run] 前置检查全部通过，本轮将跑 {len(todo)} 条：", flush=True)
        for case in todo:
            mark = f"  [故障注入 {'/'.join(case['faults'])}]" if case.get("faults") else ""
            print(f"  - {case['id']}{mark}", flush=True)
        print(f"\n事实表 {len(ground_truth)} 字符，带故障注入 {len(with_faults)} 条。"
              f"未发起任何模型调用。", flush=True)
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    partial = partial_path(stamp)
    fresh: list[dict] = []
    async with httpx.AsyncClient() as client:
        for case in todo:  # 顺序执行：memory-recall 依赖 memory-write
            print(f"== 评测 {case['id']} ...", flush=True)
            try:
                result = await run_case(client, case, ground_truth)
            except Exception as err:  # noqa: BLE001 —— 单条失败不中断整轮回归
                # 异常必须当场打出来：只塞进 transcript 的话，13 条全错也要等整轮
                # 跑完才知道原因，而每一条都真金白银烧了网关配额。
                print(f"   [异常] {type(err).__name__}: {str(err)[:400]}", flush=True)
                result = {
                    "id": case["id"], "session_id": None,
                    "description": case["description"],
                    "score": 0.0, "p0_pass": False, "verdict": "ERROR",
                    "judged": {}, "transcript": f"执行异常：{type(err).__name__}: {err}",
                }
            print(f"   -> {result['verdict']}（{result['score']}）", flush=True)
            fresh.append(result)
            # 每条跑完就落一次盘：整轮 80-120 分钟真金白银，
            # 第 39 条崩了不该把前面 38 条的结果一起赔进去
            write_partial(partial, run_line, merge_results(cases, previous, fresh))

    results = merge_results(cases, previous, fresh)
    report = render_report(results, run_line)
    report_path = PROJECT_ROOT / "eval" / f"report-{stamp}.md"
    report_path.write_text(report, encoding="utf-8")
    # 同时落一份机器可读的：bad case 采集要按判据逐条读失败项，
    # 让它去正则解析 markdown 会在报告样式一改的时候静默失灵。
    json_path = PROJECT_ROOT / "eval" / f"report-{stamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "run": run_line,
                "health": health,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # 正式报告落盘之后才删增量文件：顺序反了的话，报告写失败时两份都没了
    partial.unlink(missing_ok=True)

    print(f"\n报告已写入：{report_path}")
    print(f"结构化结果：{json_path}")
    print(report.split("\n\n")[1])


if __name__ == "__main__":
    asyncio.run(main())
