# -*- coding: utf-8 -*-
"""bad_case_pool —— 失败用例的标注池

飞轮的中间环节：**发现 → 去重入池 → 人工定级 → 升级成回归用例**。

两个采集口：
    from_provenance   金额出处扫描的发现（离线补判 + 运行时告警）
    from_report       Rubric 评测里 FAIL/ERROR 的 case

池子落在 `eval/bad_cases.jsonl`，一行一条，靠 fingerprint 幂等。
fingerprint 刻意**不含分数、时间、会话 id** —— 同一个失败在不同轮次
跑出 0.5 和 0.6 是同一个问题，指纹必须相同，否则每跑一轮就多一批"新"条目，
池子会被最容易复现的那几条淹掉。

`status` 是人工分诊结果（new / promoted / wontfix），重扫只刷新
`last_seen_at`，绝不回写 status：分诊做过一次就不该被自动化推翻。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# 会话 id 形如 eval-{case_id}-{6位随机}，去掉前后缀即可还原 case 名
_SESSION_PREFIX = "eval-"
_EXCERPT_LIMIT = 400

STATUS_NEW = "new"
STATUS_PROMOTED = "promoted"    # 已升级成回归用例
STATUS_WONTFIX = "wontfix"      # 确认不修（判据写歪了 / 观测问题 / 模型抖动）
STATUS_FIXED = "fixed"          # 根因已修，留档待下一轮验证
VALID_STATUSES = (STATUS_NEW, STATUS_PROMOTED, STATUS_WONTFIX, STATUS_FIXED)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


@dataclass
class BadCase:
    fingerprint: str
    source: str          # provenance / rubric
    session_id: str
    case_id: str
    buyer_query: str
    agent_excerpt: str
    reason: str
    status: str = STATUS_NEW
    # 分诊留言：为什么定这个级。没有它，三周后没人记得当初为什么标了 wontfix
    triage_note: str = ""
    first_seen_at: str = field(default_factory=_now)
    last_seen_at: str = field(default_factory=_now)


def _case_id_from_session(session_id: str) -> str:
    """eval-compare-two-6d0690 → compare-two；非评测会话原样返回。"""
    if not session_id.startswith(_SESSION_PREFIX):
        return session_id
    stem = session_id[len(_SESSION_PREFIX):]
    head, _, tail = stem.rpartition("-")
    return head if head and len(tail) <= 8 else stem


def load_pool(path: Path) -> dict[str, BadCase]:
    pool: dict[str, BadCase] = {}
    if not Path(path).exists():
        return pool
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue  # 单行损坏不该让整个池子读不出来
        pool[row["fingerprint"]] = BadCase(**row)
    return pool


def write_pool(path: Path, pool: dict[str, BadCase]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(pool.values(), key=lambda case: (case.first_seen_at, case.fingerprint))
    Path(path).write_text(
        "\n".join(json.dumps(asdict(case), ensure_ascii=False) for case in ordered) + "\n",
        encoding="utf-8",
    )


def merge(path: Path, cases: Iterable[BadCase]) -> tuple[int, int]:
    """入池，返回 (新增数, 已存在数)。已存在的只刷新 last_seen_at。"""
    pool = load_pool(path)
    added = updated = 0
    for case in cases:
        existing = pool.get(case.fingerprint)
        if existing is None:
            pool[case.fingerprint] = case
            added += 1
        else:
            existing.last_seen_at = case.last_seen_at
            existing.reason = case.reason  # 成因描述可能变准，但 status 不动
            updated += 1
    write_pool(path, pool)
    return added, updated


def from_provenance(audits: Iterable) -> list[BadCase]:
    """金额出处扫描的发现 → bad case（一个会话一条，不按金额拆）。

    不按金额拆是刻意的：同一轮里 5 个金额全无出处，通常是**同一个**根因
    （少了一个工具、或者轨迹漏发），拆成 5 条只会让池子看起来问题很多。
    """
    cases: list[BadCase] = []
    for audit in audits:
        if not audit.unsourced:
            continue
        kinds = sorted({item.kind for item in audit.unsourced})
        samples = "、".join(dict.fromkeys(item.raw for item in audit.unsourced))[:120]
        cases.append(
            BadCase(
                # 指纹用"会话所属 case + 成因类型"，不含金额值：
                # 同一个 case 每次跑出的具体数字都不一样，含进去等于不去重
                fingerprint=fingerprint("provenance", _case_id_from_session(audit.session_id), *kinds),
                source="provenance",
                session_id=audit.session_id,
                case_id=_case_id_from_session(audit.session_id),
                # 多轮会话取首轮问句：它定的调，后续轮次都挂在它上面
                buyer_query=(getattr(audit, "buyer_texts", None) or [""])[0],
                agent_excerpt=samples,
                reason=(
                    f"{len(audit.unsourced)}/{audit.total_amounts} 处金额无工具出处"
                    f"（{'、'.join(kinds)}）：{samples}"
                ),
            ),
        )
    return cases


def from_report(report: dict) -> list[BadCase]:
    """Rubric 评测报告（JSON 形态）里 FAIL/ERROR 的 case → bad case。"""
    cases: list[BadCase] = []
    for result in report.get("results", []):
        if result.get("verdict") == "PASS":
            continue
        failed = [
            item.get("criterion", "")
            for level in ("p0", "p1", "p2")
            for item in (result.get("judged") or {}).get(level, [])
            if not item.get("pass")
        ]
        reason = "；".join(failed) or f"verdict={result.get('verdict')}"
        cases.append(
            BadCase(
                # 指纹只含 case 名与失败判据：同一条判据反复不过是同一个问题，
                # 分数每轮都在小幅浮动，含进去就永远去不了重
                fingerprint=fingerprint("rubric", result["id"], *sorted(failed)),
                source="rubric",
                session_id="",
                case_id=result["id"],
                buyer_query="",
                agent_excerpt=(result.get("transcript") or "")[:_EXCERPT_LIMIT],
                reason=reason,
            ),
        )
    return cases


def triage(
    path: Path, selector: str, status: str, note: str = "", force: bool = False,
) -> tuple[int, list["BadCase"]]:
    """把池子里匹配 `selector` 的条目定级，返回 (改动条数, 被跳过的已定级条目)。

    分诊必须有命令可用：这一环若只能手改 JSONL，实际上没人会去改，
    飞轮就永远停在"发现"这一步，池子越攒越大直到被忽略。

    `selector` 三种形态：完整 fingerprint、fingerprint 前缀（`--list` 显示的就是前缀，
    照着敲必须能用）、case_id。

    **按 case_id 批量定级时不覆盖已定级的条目**（除非 `force=True`）。
    真实踩到的：一个 case_id 命中该用例的全部指纹，把之前人工定为 wontfix 的另一条
    也一并改掉、备注一并冲掉，而输出只说"已把 2 条标为 fixed"——
    看不出改了哪两条，更看不出原来是什么。**池子是人工判断的载体，
    工具不该让人一条命令静默冲掉它。** 按指纹点名则不受此限：
    那本来就是"我知道我在改哪一条"。
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"未知状态 {status!r}，可选：{VALID_STATUSES}")
    pool = load_pool(path)

    exact = [case for case in pool.values() if case.fingerprint == selector]
    by_prefix = [case for case in pool.values() if case.fingerprint.startswith(selector)]
    if not exact and len(by_prefix) > 1:
        raise ValueError(
            f"指纹前缀 {selector!r} 匹配到 {len(by_prefix)} 条，无法确定改哪一条："
            f"{[case.fingerprint for case in by_prefix]}",
        )
    targeted = exact or by_prefix
    if targeted:
        # 按指纹点名：明确知道改的是哪一条，不需要保护
        matched, protected = targeted, []
    else:
        matched = [case for case in pool.values() if case.case_id == selector]
        protected = (
            [] if force else [case for case in matched if case.status != STATUS_NEW]
        )
        matched = [case for case in matched if case not in protected]

    for case in matched:
        case.status = status
        if note:
            case.triage_note = note
    if matched:
        write_pool(path, pool)
    return len(matched), protected
