# -*- coding: utf-8 -*-
"""A/B 报告渲染器：统计读数 + 配置行 + 护栏 → eval/ab-report-*.md。

纯渲染，零网络、零文件 IO：输入是 run_ab_pipeline 组装好的 payload，
输出是一篇 markdown。所有口径数字来自 ab_stats（同一份算式），本模块
只负责把它们说清楚——不许在这里重算任何统计量。

如实性硬点：
  - 护栏没读数就渲染"未测定"，绝不渲染成通过——没有读数的护栏不是护栏，
    "无从判定"与"全对"必须在字面上可区分（踩坑 33 的同一条）；
  - "样本不足，未达显著"原样进报告——这不是失败，是口径；
  - 互换翻转的对被丢弃必须点名（decisive_indicators 的 n_flip），
    静默丢弃就是塌缩的另一种形态。
"""
from __future__ import annotations

# 一票否决护栏的标准五项（交接文档「五之一」指标表）。
# guardrails 传空时按"未测定"逐项列出——缺读数本身就是要被看见的读数。
_STANDARD_GUARDRAILS = (
    ("① make check 八项", "全绿"),
    ("② full 44 结算 PASS 率", "≥ 44/44"),
    ("③ 无出处金额率", "≤ 8%"),
    ("④ judge 均分", "≥ 0.9826 − 0.038"),
    ("⑤ completion P50 / 轮延迟 P50", "劣化 ≤ 10%"),
)


def _fmt_rate(rate: float | None) -> str:
    return "无从判定" if rate is None else f"{rate:.1%}"


def render_ab_report(payload: dict) -> str:
    label = payload.get("label") or "A/B 成对比较"
    lines: list[str] = [
        f"# A/B 成对比较报告：{label}（{payload['stamp']}）",
        "",
    ]

    lines += ["## 两臂配置", ""]
    for arm in ("A", "B"):
        config = (payload.get("arm_config") or {}).get(arm) or {}
        lines.append(
            f"- **臂 {arm}**：{payload['arm_lines'][arm]}"
            f"（指纹 {config.get('fingerprint', '未知')}"
            f"｜变体 {config.get('variant') or '(基线)'}"
            f"｜模型 {config.get('model', '未知')}）",
        )
    lines.append("")

    plan = payload["plan"]
    lines += [
        "## 计划账本（与 dry-run 同一份算式：plan_ab_run）",
        "",
        f"- 用例 {plan['n_cases']} × k={plan['k']} × 2 臂（{payload['pairing']} 配对）："
        f"执行 {plan['executions']}｜意图 {plan['intents']}｜成对 {plan['pairs']}"
        f"｜judge 调用 {plan['judge_calls']}（含位置互换）",
        f"- decisive 上限 {plan['decisive_ceiling']}（门槛 {plan['decisive_gate']}）",
        f"- 墙钟假设：{payload['wall_clock_assumption']}",
        "",
    ]

    executions = payload.get("executions") or {"total": 0, "ok": 0, "failed": []}
    lines += [
        "## 执行段",
        "",
        f"- 执行 {executions['total']}：成功 {executions['ok']}｜失败 {len(executions.get('failed') or [])}",
    ]
    for item in executions.get("failed") or []:
        lines.append(
            f"  - ❌ {item['case_id']} 臂 {item['arm']} 采样 {item['sample_index']}：{item['error']}",
        )
    for item in executions.get("fault_clear_failures") or []:
        lines.append(
            f"  - ⚠️ {item['case_id']} 臂 {item['arm']} 采样 {item['sample_index']}"
            f"执行成功但故障清理失败（后续用例可能带着故障跑）：{item['error']}",
        )
    lines.append("")

    swap = payload["swap"]
    swap_ok = swap.get("rate") is not None and swap["rate"] >= 0.9
    lines += [
        "## judge 有效性",
        "",
        f"- 位置互换一致率：{_fmt_rate(swap.get('rate'))}"
        f"（{swap.get('n_consistent', 0)}/{swap.get('n_pairs', 0)}）｜error 行 {swap.get('n_error', 0)}",
        f"- 互换翻转（两序结论相反，不计入读数）：{payload.get('n_flip', 0)}",
        "",
    ]
    if swap_ok:
        lines.append("- ✅ 达标（≥90%）")
    else:
        lines.append("- ❌ 未达标：该轮 judge 读数作废重跑")
    lines.append("")

    win = payload["win_rate"]
    n_decisive = win["n_decisive"]
    lines += [
        "## 胜负读数（A 臂视角）",
        "",
        f"- 互换一致对 {win['n']}：A 胜 {win['wins']} / B 胜 {win['losses']} / 平局 {win['ties']}",
        f"- 胜率 {_fmt_rate(win['win_rate'])}（{win['wins']}/{win['n']}）"
        f"｜去平局胜率 {_fmt_rate(win['win_rate_excl_ties'])}"
        f"（决定性对 {n_decisive}，即 A 胜 {win['wins']}/{n_decisive}）",
    ]
    gate = payload.get("decisive_gate") or {"n_decisive": n_decisive, "min_pairs": 30, "sufficient": n_decisive >= 30}
    if gate["sufficient"]:
        lines.append(f"- ✅ decisive pairs {n_decisive} ≥ {gate['min_pairs']}（达到下结论门槛）")
    else:
        lines.append(f"- ⚠️ decisive pairs {n_decisive} < {gate['min_pairs']}（未达下结论门槛）")

    p_value = payload.get("p_value")
    ci = payload.get("ci")
    p_text = "无从计算" if p_value is None else f"p={p_value:.4g}"
    if ci is None:
        ci_text = "CI 无从计算"
    else:
        ci_text = (f"bootstrap 95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}]"
                   f"（按用例重采样，{ci['n_cases']} 例 / {ci['n_pairs']} 对）")
    lines.append(f"- 符号检验 {p_text}｜{ci_text}")
    lines.append("")

    sig = payload["significance"]
    verdict = "**显著（候选更优）**" if sig["significant"] else "**未达显著**"
    lines.append(f"- 判定：{verdict}")
    for reason in sig.get("reasons") or []:
        lines.append(f"  - {reason}")
    lines.append("")

    dual = payload.get("dual_judge")
    lines += ["## 双 judge 一致率（judge 可信度证据，不参与胜负判定）", ""]
    if dual:
        lines.append(
            f"- 第二评审 {dual['model']}：{dual['n_agree']}/{dual['n_pairs']} = {_fmt_rate(dual['rate'])}"
            f"｜error 行 {dual.get('n_error', 0)}",
        )
    else:
        lines.append("- 未执行")
    lines.append("")

    if payload.get("positive_control"):
        lines += ["## 阳性对照（有效性自证）", ""]
        arm_config = payload.get("arm_config") or {}
        weaker = next(
            (
                arm
                for arm in ("A", "B")
                if "weaker" in str((arm_config.get(arm) or {}).get("variant") or "")
            ),
            None,
        )
        if weaker is None:
            lines.append(
                "- ❌ 找不到已知更差臂（arm_config 变体名需含 'weaker'，如 control-weaker-confirm）"
                "——臂设置与报告口径不符，人工核对两臂变体后再下结论。"
            )
        else:
            win_rate = win.get("win_rate") or 0
            # win_rate 是 A 臂视角；已知更差臂被判更差 = 负向显著（方向随 weaker 换位）。
            # 判定看统计量本身（p<0.05 + CI 不含 0.5），**不吃 significant 旗标**——
            # 它被 90% 互换门槛的 judge_valid 污染（M3 对照轮 p=0.0115、CI 不含 0.5
            # 的真实负向显著被误报成"区分度缺陷"的实测教训）。
            statistically_negative = (
                (sig.get("p_value") is not None and sig["p_value"] < 0.05)
                and bool(sig.get("ci_excludes_half"))
            )
            weaker_won = (win_rate > 0.5) if weaker == "A" else (win_rate < 0.5)
            lines.append(
                f"- 已知更差臂 = **臂 {weaker}**（变体 {(arm_config.get(weaker) or {}).get('variant')}），"
                "预期被判负向显著。"
            )
            if statistically_negative and not weaker_won:
                lines.append(
                    f"- ✅ 负向显著：已知更差的臂 {weaker} 被判显著更差——工具有区分度，有效性自证通过。"
                )
            elif statistically_negative:
                lines.append(
                    f"- ❌ 方向反了：已知更差的臂 {weaker} 被判更优——读数有效性存疑，先查工具再谈胜负。"
                )
            else:
                lines.append("- ❌ 未判出显著差异——工具有区分度缺陷，先修工具再谈胜负。")
        lines.append("")

    lines += ["## 护栏读数（一票否决，与胜率无关）", ""]
    lines += ["| 护栏 | 读数 | 门槛 | 判定 |", "|---|---|---|---|"]
    provided = {g["name"]: g for g in payload.get("guardrails") or []}
    names = [g["name"] for g in payload.get("guardrails") or []] or [name for name, _ in _STANDARD_GUARDRAILS]
    thresholds = dict(_STANDARD_GUARDRAILS)
    for name in names:
        guard = provided.get(name)
        if guard is None:
            lines.append(f"| {name} | 未测定 | {thresholds.get(name, '—')} | **未测定（不构成通过）** |")
        else:
            mark = "✅ 通过" if guard["pass"] else "❌ 未达标"
            lines.append(f"| {name} | {guard['value']} | {guard['threshold']} | {mark} |")
    lines.append("")

    cost = payload.get("cost_latency")
    if cost is not None:
        lines += ["## 成本与延迟（两臂，来自会话流水）", ""]
        for arm in ("A", "B"):
            arm_cost = cost.get(arm) or {}
            completion = arm_cost.get("completion_p50")
            latency = arm_cost.get("latency_p50_s")
            if not arm_cost:
                lines.append(f"- 臂 {arm}：未测定（见附注）")
                continue
            lines.append(
                f"- 臂 {arm}：completion P50 "
                f"{completion if completion is None else f'{completion:.0f}'}"
                f"｜轮延迟 P50 {latency if latency is None else f'{latency:.1f}'}s",
            )
        lines.append("")

    notes = payload.get("notes") or []
    # 错误明细：配对错误与判词错误逐条点名——error 行只留计数不留名字，
    # 是塌缩的另一种形态（本仓纪律：宁可少一对，不进一条假读数）
    error_lines: list[str] = []
    for item in payload.get("pair_errors") or []:
        error_lines.append(f"- ❌ [配对] {item['reason']}")
    for row in payload.get("rows") or []:
        if row.get("error_ab"):
            error_lines.append(f"- ❌ [判词·正序] {row['error_ab']}")
        if row.get("error_ba"):
            error_lines.append(f"- ❌ [判词·反序] {row['error_ba']}")
    if error_lines:
        lines += ["## 错误明细（全部点名，不塌缩）", ""]
        lines.extend(error_lines[:30])
        if len(error_lines) > 30:
            lines.append(f"- …其余 {len(error_lines) - 30} 条见 ab-run-*.json")
        lines.append("")
    if notes:
        lines += ["## 附注", ""]
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)
