# -*- coding: utf-8 -*-
"""回放执行产物 → 结构化沉淀（#12 任务 B，M1 接线）。

提取与验证全部确定性、零 LLM：
  1. extract_remember_calls：从会话流水提取 remember_preference_tool 的写入——
     只认 tool.result 里 saved 命中的成功写入（失败的写入不产生沉淀）；
  2. build_deposit：写入事件 → 沉淀条目；验证器按预登记映射构造，映射里
     没有的 kind 直接拒绝（"不可验证 = 不许写入"的延伸：没有预登记验证器
     形态的写入也不入库，宁可少一条不留假凭证）；
  3. verify_against_run：沉淀的断言对照回放产物里同 sample 的两臂下游
     transcript 逐一判定，产出可验证率与行为差异证据。

先导档（M1）预登记：memory-write 的写入 → memory-recall 行为面的验证；
下游映射与验证器形态要扩，走回写通道。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.application.memory.deposit_store import DepositStore
from app.domain.buyer.deposit import MemoryDeposit, VerificationResult, build_verifier

REMEMBER_TOOL = "remember_preference_tool"

# 先导档预登记：触发用例 → 验证下游用例（沉淀改变的行为在哪个用例里显形）。
# preference-conflict 是会话内自写自用（第 1 轮写、第 2 轮推荐），下游 = 本用例
# 自己（on/off 是两次独立执行，对照仍成立）；memory-forget 链的撤回语义
# （撤回后旧行为应恢复）与验证器方向相反，留 M3 回写后再接。
DOWNSTREAM_CASE = {
    "memory-write": "memory-recall",
    "preference-conflict-cheapest-vs-dislike": "preference-conflict-cheapest-vs-dislike",
}

# 商品库事实（cases.yaml preference-conflict 注释）：Voyager 旅行三件套记忆棉款
# （涤纶外套）是商品库里唯一的塑料纤维旅行套装——dislike 注入生效的遵从判定
# 就盯它。材质事实词取自商品库属性（确定性），不是猜测。
DISLIKE_PRODUCT_KEYWORD = "Voyager"
DISLIKE_MATERIAL_MARKERS = ("涤纶", "化纤", "塑料")

# 冲突说明的连接语（2026-09-06 用户裁量的两级判定口径）：材质事实 + 把选择权
# 交回买家的话，两样齐才算"显式说明"；中性词（"如果"）单挑不算。
DISLIKE_CHOICE_MARKERS = ("没意见", "不介意", "介意", "偏好", "冲突", "避开", "慎选")
MAIN_REC_MARKERS = ("最推荐", "首推", "主推", "综合来看", "综合推荐")


def _dislike_verifier_spec() -> dict:
    return {
        "kind": "recommendation_compliance",
        "product": DISLIKE_PRODUCT_KEYWORD,
        "material_markers": list(DISLIKE_MATERIAL_MARKERS),
        "choice_markers": list(DISLIKE_CHOICE_MARKERS),
        "main_rec_markers": list(MAIN_REC_MARKERS),
        "require_contrast": True,
    }


def extract_remember_calls(session_id: str, data_dir: str | Path) -> list[dict]:
    """会话流水 → [{kind, statement, trigger_query}]。

    只收成功写入：tool.invoke 之后必须等到 tool.result 的 saved 才算数，
    error 的写入如实丢弃（没写进去的偏好不存在行为影响，编一条沉淀出来
    反而污染对账）。trigger_query 取写入前最近一条买家发言。
    """
    path = Path(data_dir) / "conversations" / f"{session_id}.jsonl"
    if not path.is_file():
        raise SystemExit(
            f"会话流水不存在：{path}——执行记录声明了这个 session，"
            "流水却找不到（DATA_DIR 对不上？），提取无从对账"
        )
    writes: list[dict] = []
    pending: dict | None = None
    trigger_query = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        kind = record.get("kind")
        if kind == "turn" and record.get("role") == "buyer":
            trigger_query = str(record.get("content") or "")
            continue
        if kind != "event":
            continue
        payload = record.get("payload") or {}
        if payload.get("tool") != REMEMBER_TOOL:
            continue
        if record.get("type") == "tool.invoke":
            args = payload.get("args") or {}
            pending = {
                "kind": str(args.get("kind") or ""),
                "statement": str(args.get("statement") or ""),
                "trigger_query": trigger_query,
            }
        elif record.get("type") == "tool.result" and pending is not None:
            if payload.get("saved"):
                writes.append(pending)
            pending = None
    return writes


def build_deposit(case_id: str, buyer_id: str, session_id: str, write: dict) -> MemoryDeposit:
    """一次成功写入 → 沉淀条目。验证器形态按 kind 预登记，未登记的 kind 拒绝。"""
    write_kind = write["kind"]
    statement = write["statement"]
    if write_kind == "dislike":
        verifier_spec = _dislike_verifier_spec()
        precondition = f"买家询问会命中「{DISLIKE_PRODUCT_KEYWORD}」所在品类的商品时"
        assertion = (
            f"两级遵从判定（2026-09-06 裁量口径）：注入开的主推荐不得是「{DISLIKE_PRODUCT_KEYWORD}」，"
            f"且它出现处必须伴随材质冲突说明（材质事实+交还选择权）；注入关按同一判据应判违规——"
            "两臂判定不同才算这条偏好改变了行为"
        )
    elif write_kind == "like":
        verifier_spec = {
            "kind": "preference_mention",
            "keywords": [statement],
        }
        precondition = "买家询问与偏好相关的推荐时"
        assertion = (
            f"注入开的回复体现偏好「{statement}」、注入关不体现（无注入来源）；"
            "两臂都提或都不提 = 注入没有改变行为，不作数"
        )
    else:
        raise SystemExit(
            f"写入 kind={write_kind!r} 没有预登记的验证器形态——"
            "没有确定性验证器的写入不入库（不可验证 = 不许写入），要扩先回写任务书"
        )
    return MemoryDeposit(
        buyer_id=buyer_id,
        kind=write_kind,
        statement=statement,
        trigger_session_id=session_id,
        trigger_query=write["trigger_query"],
        behavior_surface="recommendation",
        precondition=precondition,
        assertion=assertion,
        verifier_spec=verifier_spec,
    )


def _paired_transcripts(run_json: dict, case_id: str, sample_index: int) -> tuple[str, str, str]:
    """取同 (case, sample) 的两臂 transcript：返回 (transcript_on, transcript_off, 问题)。

    transcript_on = 臂 B（注入开）、transcript_off = 臂 A（注入关）——臂语义
    与 mem_replay.ARM_EXPECT 一致，写反读数方向就反。
    """
    by_key = {
        (r["case_id"], r["arm"], r["sample_index"]): r
        for r in (run_json.get("results") or [])
    }
    on = by_key.get((case_id, "B", sample_index))
    off = by_key.get((case_id, "A", sample_index))
    problem = ""
    if not on or not on.get("ok"):
        problem = f"臂 B（注入开）{case_id} 采样 {sample_index} 的产物缺失或失败"
    elif not off or not off.get("ok"):
        problem = f"臂 A（注入关）{case_id} 采样 {sample_index} 的产物缺失或失败"
    if problem:
        return "", "", problem
    return on["transcript"], off["transcript"], ""


def _buyer_from_session(session_id: str, data_dir: str | Path) -> str:
    """从会话流水首行取真实 buyer_id——执行记录不落 buyer，流水才是事实源。"""
    path = Path(data_dir) / "conversations" / f"{session_id}.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("kind") == "session":
                return str(record.get("buyer_id") or "")
    raise SystemExit(f"会话流水 {path} 里没有 session 行——流水不完整，提取无从对账")


def verify_against_run(
    run_json: dict, data_dir: str | Path, store: DepositStore | None = None,
) -> dict:
    """跑测产物 → 沉淀提取（**只取臂 B**——注入开臂才是记忆自进化闭环所在，
    臂 A 是注入关的反事实基线，它的写入不参与对账）+ 逐条验证 + 入库。
    返回对账账本（也供落盘）。

    只认标准注入对照轮（臂 A 变体 = mem-inject-off）：阳性对照轮的臂 A 是
    矛盾注入臂，拿它当"注入关"基线，对账语义整个错位——必须拦下。

    可验证率的口径：每条沉淀都必须有确定性验证器且真的跑过——验证器跑不了
    的沉淀在写入门就该被拒，出现在这里是管线缺陷，直接把 rate 打穿报错。
    """
    arm_variant = str(((run_json.get("arm_config") or {}).get("A") or {}).get("variant") or "")
    if arm_variant != "mem-inject-off":
        raise SystemExit(
            f"该产物不是标准注入对照轮（臂 A 变体 = {arm_variant!r}，应为 mem-inject-off）——"
            "阳性对照轮的臂 A 是矛盾注入臂，沉淀验证语义不适用，别喂对照轮产物"
        )
    executions = [
        r for r in (run_json.get("results") or [])
        if r.get("ok") and r.get("arm") == "B" and r.get("case_id") in DOWNSTREAM_CASE
    ]
    ledger: list[dict] = []
    for trigger_case, downstream in DOWNSTREAM_CASE.items():
        for record in executions:
            if record["case_id"] != trigger_case:
                continue
            session_id = record.get("session_id") or ""
            buyer_id = _buyer_from_session(session_id, data_dir)
            writes = extract_remember_calls(session_id, data_dir)
            for write in writes:
                try:
                    deposit = build_deposit(trigger_case, buyer_id, session_id, write)
                except SystemExit as err:
                    ledger.append({
                        "case_id": trigger_case, "sample_index": record["sample_index"],
                        "statement": write["statement"], "verifiable": False,
                        "ok": False, "detail": str(err),
                    })
                    continue
                transcript_on, transcript_off, problem = _paired_transcripts(
                    run_json, downstream, record["sample_index"],
                )
                if problem:
                    ledger.append({
                        "case_id": trigger_case, "sample_index": record["sample_index"],
                        "deposit_id": deposit.deposit_id, "statement": deposit.statement,
                        "verifiable": True, "ok": False, "detail": problem,
                    })
                    continue
                result: VerificationResult = build_verifier(deposit.verifier_spec).check(
                    transcript_on, transcript_off,
                )
                ledger.append({
                    "case_id": trigger_case, "sample_index": record["sample_index"],
                    "deposit_id": deposit.deposit_id, "statement": deposit.statement,
                    "kind": deposit.kind, "verifiable": True,
                    "ok": result.ok, "detail": result.detail,
                    "assertion": deposit.assertion,
                })
                if store is not None and result.ok:
                    store.append(deposit)
    total = len(ledger)
    verifiable = sum(1 for item in ledger if item["verifiable"])
    if verifiable != total:
        raise SystemExit(
            f"存在没有确定性验证器的写入（{total - verifiable}/{total}）——"
            "写入门被绕过是管线缺陷，先修管线再谈读数"
        )
    passed = sum(1 for item in ledger if item["ok"])
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "deposits": ledger,
        "n_deposits": total,
        "n_verified": verifiable,
        "verifiable_rate": 1.0 if total == 0 else verifiable / total,
        "n_behavior_confirmed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="记忆回放沉淀提取与验证（#12 任务 B）")
    parser.add_argument("run_json", help="eval/mem-run-*.json（mem_replay 的产物）")
    parser.add_argument("--data-dir", default="data", help="会话流水目录（默认 data；run json 里有 healths 时优先）")
    args = parser.parse_args(argv)

    run_json = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
    healths = run_json.get("healths") or {}
    data_dirs = {str(h.get("data_dir") or "") for h in healths.values() if isinstance(h, dict)}
    data_dirs.discard("")
    if len(data_dirs) > 1:
        raise SystemExit(
            f"两臂 DATA_DIR 不一致（{sorted(data_dirs)}）——流水分家，沉淀提取无从对账，"
            "先查两臂起服配置（记忆回放要求两臂共用仓库 data/）"
        )
    data_dir = next(iter(data_dirs), args.data_dir)
    report = verify_against_run(run_json, data_dir, store=DepositStore(data_dir=data_dir))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.run_json).parent / f"mem-deposit-{stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"沉淀 {report['n_deposits']} 条（可验证率 {report['verifiable_rate']:.0%}，"
        f"行为差异确认 {report['n_behavior_confirmed']} 条）→ {out_path}"
    )
    for item in report["deposits"]:
        mark = "PASS" if item["ok"] else "FAIL"
        print(f"  [{mark}] {item['statement']}（样本 {item['sample_index']}）：{item['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
