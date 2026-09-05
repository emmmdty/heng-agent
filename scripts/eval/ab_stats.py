# -*- coding: utf-8 -*-
"""A/B 统计函数（任务 A 第 2 项）：位置互换一致率 / 胜率 / 符号检验 / bootstrap CI。

口径冻结在交接文档「五之一」指标表，本模块是口径的编码，不是新口径：
  - 位置互换自一致率 ≥ 90%，不达标该轮 judge 读数作废重跑（MT-Bench 的位置偏差控制）；
  - 显著性 = 去平局符号检验 p < 0.05 且按用例重采样的 bootstrap 95% CI 不含 0.5；
  - decisive pairs ≥ 30 是下结论的最低门槛。

两个如实性的硬点：
  - 样本无从判定时输出 None（rate/p/CI），**不伪造 0% 或 100%**——
    空judge样本算出的 100% 一致率是假绿（同踩坑 33"0 处金额算出的 0%"）；
  - "未达显著"与"没有差异"是两个陈述。significance() 的 significant=False
    永远附带 reasons 说明卡在哪一道门，给上游留如实表述的通道。
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

_VALID_VERDICTS = {"a", "b", "tie"}


def position_swap_consistency(rows: list[dict]) -> dict:
    """位置互换自一致率：同一对样本正反两个顺序各判一次，裁决一致的比例。

    rows 元素：{"case_id", "pair_index", "verdict_ab", "verdict_ba"}，
    两个 verdict 是映射到臂名后的结果（"a"/"b"/"tie"），None 表示该方向
    judge 失败（error 行）。error 行进 n_error、不进分母——
    塌缩进分母会把"judge 挂了一半"洗成"一致率 100%"。
    """
    n_pairs = 0
    n_consistent = 0
    n_error = 0
    for row in rows:
        ab, ba = row.get("verdict_ab"), row.get("verdict_ba")
        if ab is None or ba is None:
            n_error += 1
            continue
        n_pairs += 1
        if ab == ba:
            n_consistent += 1
    return {
        "n_pairs": n_pairs,
        "n_consistent": n_consistent,
        "rate": (n_consistent / n_pairs) if n_pairs else None,
        "n_error": n_error,
    }


def win_rate_summary(verdicts: list) -> dict:
    """A 臂视角的胜率汇总。verdicts 元素："a"(A 胜) / "b"(A 败) / "tie" / None(error)。

    非法元素 raise ValueError：拼错的臂名静默进某个分类，就是塌缩的另一种形态。
    """
    wins = losses = ties = errors = 0
    for verdict in verdicts:
        if verdict is None:
            errors += 1
            continue
        if verdict not in _VALID_VERDICTS:
            raise ValueError(f"非法 verdict：{verdict!r}（应为 'a'/'b'/'tie'/None）")
        if verdict == "a":
            wins += 1
        elif verdict == "b":
            losses += 1
        else:
            ties += 1
    n = len(verdicts)
    n_decisive = wins + losses
    n_valid = n - errors  # error 行不是裁决，不进任何分母（同 position_swap 的规矩）
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_error": errors,
        "n_decisive": n_decisive,
        "win_rate": (wins / n_valid) if n_valid else None,
        "win_rate_excl_ties": (wins / n_decisive) if n_decisive else None,
    }


def sign_test_p(wins: int, losses: int) -> float:
    """去平局后的精确二项符号检验（双侧）：p = 2·Σ_{k≤min} C(n,k)/2ⁿ。

    小样本必须走精确式：n=3 全一边时精确 p=0.25，正态近似会显著低估——
    而 A/B 的样本量注定小，近似在这里不是风格问题，是读数真错。
    """
    if wins < 0 or losses < 0:
        raise ValueError(f"wins/losses 不能为负：{wins}, {losses}")
    n = wins + losses
    if n == 0:
        raise ValueError("没有决定性对（wins+losses=0），符号检验无从计算")
    smaller = min(wins, losses)
    tail = sum(math.comb(n, k) for k in range(smaller + 1))
    return min(1.0, 2 * tail / 2 ** n)


def bootstrap_ci_win_rate(
    pairs: list[tuple[str, int]],
    n_boot: int = 10000,
    seed: int = 42,
    level: float = 0.95,
) -> dict:
    """按用例重采样的 win-rate bootstrap CI（冻结口径：重采样单位是用例）。

    pairs = (case_id, indicator)，indicator ∈ {1,0}——只收决定性对，
    平局与 error 行由调用方过滤后再传入（本函数不代过滤：过滤口径本身
    是指标定义的一部分，静默代劳等于改口径）。

    为什么按用例不按对：同一用例的多次采样共享同一个买家请求与判据，
    不是独立样本；按对重采样会把 CI 压窄、显著性虚高。
    """
    if not pairs:
        raise ValueError("决定性对为空，CI 无从计算")
    indicators = [value for _, value in pairs]
    if any(value not in (0, 1) for value in indicators):
        raise ValueError(f"indicator 只能是 0/1，收到：{sorted({v for v in indicators if v not in (0, 1)})[:5]}")
    if n_boot < 1:
        raise ValueError(f"n_boot 至少为 1，收到 {n_boot}")
    if not 0 < level < 1:
        raise ValueError(f"level 必须在 (0,1)，收到 {level}")

    by_case: dict[str, list[int]] = defaultdict(list)
    for case_id, value in pairs:
        by_case[case_id].append(value)
    case_ids = sorted(by_case)

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sampled = [by_case[rng.choice(case_ids)] for _ in case_ids]
        flat = [value for group in sampled for value in group]
        means.append(sum(flat) / len(flat))
    means.sort()

    alpha = (1 - level) / 2
    lo_index = max(0, int(math.floor(alpha * n_boot)))
    hi_index = min(n_boot - 1, int(math.ceil((1 - alpha) * n_boot)) - 1)
    return {
        "point": sum(indicators) / len(indicators),
        "lo": means[lo_index],
        "hi": means[hi_index],
        "n_boot": n_boot,
        "n_pairs": len(pairs),
        "n_cases": len(case_ids),
        "level": level,
    }


def significance(
    summary: dict,
    swap: dict,
    p_value: float | None,
    ci: dict | None,
    min_decisive: int = 30,
    min_swap_rate: float = 0.9,
) -> dict:
    """显著性判定：significant = judge 有效 ∧ 决定性对达标 ∧ p<0.05 ∧ CI 不含 0.5。

    位置互换一致率不达标时 judge 读数整体作废（指标表口径），一票否决。
    p/CI 无从计算（无决定性对）时置 None 并写明"样本不足"——
    **不是"没有差异"**，这两个陈述的区别就是本函数存在的原因。
    """
    rate = swap.get("rate")
    judge_valid = rate is not None and rate >= min_swap_rate
    n_decisive = summary["n_decisive"]
    enough_pairs = n_decisive >= min_decisive

    reasons: list[str] = []
    if not judge_valid:
        reasons.append(
            f"位置互换一致率 {rate if rate is not None else '无从判定'} < {min_swap_rate}，"
            "该轮 judge 读数作废重跑"
        )
    if not enough_pairs:
        reasons.append(f"decisive pairs {n_decisive} < {min_decisive}，样本不足，未达显著")
    if n_decisive == 0:
        p_ok = False
        ci_ok = None
        reasons.append("无决定性对，p 值与 CI 无从计算")
    else:
        p_ok = p_value is not None and p_value < 0.05
        if not p_ok:
            reasons.append(f"符号检验 p={p_value} 不满足 p<0.05")
        if ci is None:
            ci_ok = None
            reasons.append("bootstrap CI 缺失，无从判定")
        else:
            ci_ok = not (ci["lo"] <= 0.5 <= ci["hi"])
            if not ci_ok:
                reasons.append(f"bootstrap 95% CI [{ci['lo']}, {ci['hi']}] 包含 0.5")

    return {
        "judge_valid": judge_valid,
        "enough_pairs": enough_pairs,
        "p_value": p_value,
        "ci_excludes_half": ci_ok,
        "significant": judge_valid and enough_pairs and p_ok and bool(ci_ok),
        "reasons": reasons,
    }


def decisive_pairs_gate(summary: dict, min_pairs: int = 30) -> dict:
    """decisive pairs ≥ 30 的最低门槛（单独导出：报告里要单列这一行）。"""
    return {
        "n_decisive": summary["n_decisive"],
        "min_pairs": min_pairs,
        "sufficient": summary["n_decisive"] >= min_pairs,
    }


def decisive_indicators(rows: list[dict]) -> tuple[list[str], list[tuple[str, int]], int]:
    """从成对判行提取三样读数，只取**位置互换一致**的对：

    - win feed：一致对的 verdict_ab（"a"/"b"/"tie"）——正序读数，反序只作
      一致性校验（MT-Bench 口径：正反两判一致才可信）；
    - CI 对：一致对里的决定性对 (case_id, 1 if a else 0)，平局不进
      （符号检验与 CI 都是去平局口径）；
    - n_flip：两序都判了但结论相反的对数。噪声对等权计入会污染读数，
      丢掉不许静默——调用方必须在报告里点名这个数。

    error 行（任一方向失败）这里直接跳过：它们由 position_swap_consistency
    计数，两处不重复计。所有行 verdict 都缺失时返回空表——空读数由上游
    如实渲染"无从判定"，本函数不伪造。
    """
    feed: list[str] = []
    ci_pairs: list[tuple[str, int]] = []
    n_flip = 0
    for row in rows:
        verdict_ab, verdict_ba = row.get("verdict_ab"), row.get("verdict_ba")
        if verdict_ab is None or verdict_ba is None:
            continue
        if verdict_ab != verdict_ba:
            n_flip += 1
            continue
        feed.append(verdict_ab)
        if verdict_ab in ("a", "b"):
            ci_pairs.append((row["case_id"], 1 if verdict_ab == "a" else 0))
    return feed, ci_pairs, n_flip


def judge_agreement(rows: list[dict]) -> dict:
    """双 judge 一致率：同一对同一顺序，两个 judge 的裁决一致的比例。

    只作 judge 可信度证据（指标表：不参与胜负判定、不与 rubric 分数横比）。
    任一方向失败的对进 n_error 不进分母——"judge 挂了一半"洗不成"一致率高"，
    同 position_swap_consistency 的规矩。
    """
    n = 0
    agree = 0
    n_error = 0
    for row in rows:
        first, second = row.get("verdict_first"), row.get("verdict_second")
        if first is None or second is None:
            n_error += 1
            continue
        n += 1
        if first == second:
            agree += 1
    return {"n_pairs": n, "n_agree": agree, "rate": (agree / n) if n else None, "n_error": n_error}
