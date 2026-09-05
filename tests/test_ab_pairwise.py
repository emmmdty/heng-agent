# -*- coding: utf-8 -*-
"""成对比较器（任务 A 第 1 项）。

指标口径冻结在交接文档「五之一」：A/B 两臂各采 k 次，judge 盲判成对比较，
报 win/tie/loss。本模块只做两件确定性的事：
  1. 构造盲判提示词（judge 看不到版本身份——盲判破了，胜率就是自我偏好读数）；
  2. 解析判词（脏输出必须报错留名，不许静默塌缩）。

**为什么脏输出不能塌缩**：rejudge 的教训（二十三期）——脏判词塌缩成少计的
一致条目，读数看着正常、其实不可比（踩坑 32 同族：观测链先坏，被测者背锅）。
成对比较里塌缩的形态更隐蔽：把"裁决: 3"当成平局，胜率就悄悄偏向某一侧。
所以一切解析不了的输出一律 VerdictParseError，由上层记 error 行（带用例名）。
"""
import pytest

from scripts.eval.ab_pairwise import (
    VerdictParseError,
    build_pairs,
    build_pair_prompt,
    judge_pair,
    majority_verdict,
    map_winner,
    parse_verdict,
)


class TestBuildPairPrompt:
    def test_contains_both_transcripts_and_case(self):
        prompt = build_pair_prompt("买家：找个露营灯", "[买家] 要灯\n[Agent] 回复左", "[买家] 要灯\n[Agent] 回复右")
        assert "找个露营灯" in prompt
        assert "[Agent] 回复左" in prompt
        assert "[Agent] 回复右" in prompt

    def test_is_blind_no_variant_identity(self):
        """盲判是指标定义的一部分：提示词不得出现臂名/版本身份字样。"""
        prompt = build_pair_prompt("x", "t1", "t2")
        for banned in ("variant", "Variant", "候选版", "A 臂", "B 臂", "A臂", "B臂", "旧版", "新版"):
            assert banned not in prompt
        assert "回复1" in prompt and "回复2" in prompt

    def test_requires_verdict_format_instruction(self):
        prompt = build_pair_prompt("x", "t1", "t2")
        assert "裁决" in prompt and "理由" in prompt


class TestParseVerdict:
    # ---- 合法形态（容错面） ----
    def test_plain_halfwidth(self):
        assert parse_verdict("裁决: 1\n理由: 回复1 更贴合预算") == {
            "winner": "1", "rationale": "回复1 更贴合预算",
        }

    def test_fullwidth_colon_and_upper_label(self):
        assert parse_verdict("WINNER：2\n理由：略")["winner"] == "2"

    def test_tie_variants(self):
        for text in ("裁决: 平局\n理由: 相当", "裁决: 平\n理由: 相当", "裁决: tie\n理由: 相当"):
            assert parse_verdict(text) == {"winner": "tie", "rationale": "相当"}

    def test_case_insensitive_and_whitespace(self):
        assert parse_verdict("  verdict:   1  \nRationale: ok")["winner"] == "1"

    def test_verdict_not_on_last_line(self):
        text = "分析如下。\n裁决: 2\n理由: 回复2 的金额有工具出处\n（完）"
        assert parse_verdict(text)["winner"] == "2"

    def test_duplicate_same_value_tolerated(self):
        """同一份输出两处裁决但值一致——容错通过，不算脏。"""
        text = "WINNER: 1\n裁决: 1\n理由: 一致"
        assert parse_verdict(text)["winner"] == "1"

    def test_rationale_quoting_other_verdict_is_not_contradiction(self):
        """理由里**引用**另一个裁决字样不算矛盾——审查发现会误丢对，
        decisive pairs 无端缩水且 n_error 里看不出是误丢。"""
        text = "裁决: 1\n理由: 我本来想写 裁决: 2，但更正为回复1"
        parsed = parse_verdict(text)
        assert parsed["winner"] == "1"
        assert "裁决: 2" in parsed["rationale"]

    def test_fullwidth_digits(self):
        assert parse_verdict("裁决: ２\n理由: ok")["winner"] == "2"

    def test_reply_prefix_value(self):
        assert parse_verdict("裁决: 回复2\n理由: ok")["winner"] == "2"

    def test_trailing_text_after_value(self):
        assert parse_verdict("裁决: 1 更好\n理由: ok")["winner"] == "1"

    # ---- 脏输出（必须报错留名，不许塌缩） ----
    def test_none_raises(self):
        with pytest.raises(VerdictParseError):
            parse_verdict(None)

    def test_empty_raises(self):
        for raw in ("", "   \n  "):
            with pytest.raises(VerdictParseError):
                parse_verdict(raw)

    def test_no_verdict_label_raises(self):
        with pytest.raises(VerdictParseError):
            parse_verdict("回复1 比较好，理由是它更便宜。")

    def test_invalid_value_raises(self):
        """'裁决: 3'、'裁决: 都行'——解析不了就报错，绝不悄悄归入平局。"""
        for raw in ("裁决: 3\n理由: x", "裁决: 都行\n理由: x"):
            with pytest.raises(VerdictParseError):
                parse_verdict(raw)

    def test_contradictory_values_raise(self):
        with pytest.raises(VerdictParseError):
            parse_verdict("裁决: 1\n理由: x\n裁决: 2")
        with pytest.raises(VerdictParseError):
            parse_verdict("裁决: 1\n理由: x\nWINNER: TIE")

    def test_two_digit_run_is_dirty_not_silent_one(self):
        """'裁决: 12' 不得被截断成 1 静默通过。"""
        with pytest.raises(VerdictParseError):
            parse_verdict("裁决: 12\n理由: x")

    def test_missing_rationale_raises(self):
        for raw in ("裁决: 1", "裁决: 1\n理由:", "裁决: 1\n理由:   "):
            with pytest.raises(VerdictParseError):
                parse_verdict(raw)

    def test_error_message_is_actionable(self):
        """报错要能直接进 error 行留名（带判词原文），不能只给一句'解析失败'。"""
        with pytest.raises(VerdictParseError) as err:
            parse_verdict("裁决: 3\n理由: x")
        assert "3" in str(err.value)


class TestMapWinner:
    def test_maps_positions_to_arms(self):
        assert map_winner("1", ("a", "b")) == "a"
        assert map_winner("2", ("a", "b")) == "b"
        assert map_winner("tie", ("a", "b")) == "tie"

    def test_swapped_order_maps_opposite(self):
        """位置互换的机制基础：同样的原始裁决，顺序对调后映射成相反臂名。"""
        assert map_winner("1", ("b", "a")) == "b"
        assert map_winner("2", ("b", "a")) == "a"

    def test_invalid_winner_raises(self):
        with pytest.raises(ValueError):
            map_winner("3", ("a", "b"))


class TestBuildPairs:
    def test_diagonal_zips(self):
        pairs = build_pairs(["a1", "a2"], ["b1", "b2"])
        assert pairs == [("a1", "b1", 0), ("a2", "b2", 1)]

    def test_diagonal_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_pairs(["a1"], ["b1", "b2"])

    def test_cross_product(self):
        pairs = build_pairs(["a1", "a2"], ["b1", "b2"], mode="cross")
        assert pairs == [("a1", "b1", 0), ("a1", "b2", 1), ("a2", "b1", 2), ("a2", "b2", 3)]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_pairs([], [])
        with pytest.raises(ValueError):
            build_pairs(["a1"], [], mode="cross")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            build_pairs(["a"], ["b"], mode="nope")


class TestJudgePairContext:
    """事实表与会话前置事实进判词提示词（授权文档 M1 的实现决策）。

    judge 要判"两份回复哪个事实更可靠"，手上得有工具口径的事实基准
    （本仓 judge 一贯喂事实表；build_ground_truth 的产出）。盲判约束
    **只针对版本身份**，不针对事实表——不喂事实表，judge 只能比文笔。
    会话前置事实同理：memory-recall 应用历史偏好若不告知 judge，
    会被误判成无据编造（eval_regression 同一条先例）。
    """

    async def test_ground_truth_block_included(self):
        captured = {}

        async def fake_judge(prompt: str) -> str:
            captured["prompt"] = prompt
            return "裁决: 平局\n理由: 相当"

        await judge_pair(fake_judge, "x", "t1", "t2", ground_truth="| product_id | 价格 |")
        assert "商品库事实表" in captured["prompt"]
        assert "| product_id | 价格 |" in captured["prompt"]

    async def test_prior_context_block_included(self):
        captured = {}

        async def fake_judge(prompt: str) -> str:
            captured["prompt"] = prompt
            return "裁决: 平局\n理由: 相当"

        await judge_pair(fake_judge, "x", "t1", "t2", prior_context="买家已写入偏好：不要塑料")
        assert "会话前置事实" in captured["prompt"]
        assert "买家已写入偏好：不要塑料" in captured["prompt"]

    def test_ground_truth_and_prior_sections_omitted_when_empty(self):
        """不传就一个字都不出现——不给 judge 留'事实表：未知'这种空段落。"""
        prompt = build_pair_prompt("x", "t1", "t2")
        assert "事实表" not in prompt
        assert "前置事实" not in prompt

    def test_context_sections_stay_blind(self):
        """加段不许破盲判：臂名/变体身份不得借道进入提示词。"""
        prompt = build_pair_prompt(
            "x", "t1", "t2",
            ground_truth="| product_id | 价格 |",
            prior_context="买家已写入偏好",
        )
        for banned in ("variant", "候选版", "A 臂", "B 臂", "A臂", "B臂"):
            assert banned not in prompt

    def test_ground_truth_block_comes_before_transcripts(self):
        """事实表在回复之前给出：judge 先建立事实基准，再读两份回复。"""
        prompt = build_pair_prompt("买家请求x", "回复左", "回复右", ground_truth="事实表内容")
        assert prompt.index("事实表内容") < prompt.index("回复左")


class TestJudgePair:
    async def test_legal_judge_maps_to_arm(self):
        async def fake_judge(prompt: str) -> str:
            return "裁决: 1\n理由: 回复1 给出了工具出处"

        result = await judge_pair(fake_judge, "x", "ta", "tb")
        assert result == {
            "winner": "a", "rationale": "回复1 给出了工具出处",
            "raw": "裁决: 1\n理由: 回复1 给出了工具出处",
        }

    async def test_swapped_order_flips_arm(self):
        async def fake_judge(prompt: str) -> str:
            return "裁决: 1\n理由: 位置在前的更好"

        result = await judge_pair(fake_judge, "x", "ta", "tb", order=("b", "a"))
        assert result["winner"] == "b"

    async def test_dirty_judge_raises_not_collapses(self):
        async def fake_judge(prompt: str) -> str:
            return "都差不多"

        with pytest.raises(VerdictParseError):
            await judge_pair(fake_judge, "x", "ta", "tb")

    async def test_tie_passes_through(self):
        async def fake_judge(prompt: str) -> str:
            return "裁决: 平局\n理由: 两份回复事实一致"

        assert (await judge_pair(fake_judge, "x", "ta", "tb"))["winner"] == "tie"


class TestPairPromptEffectDimensions:
    """M2'-b（根因 2.2-3 的放大器修复）：成对判据补处理效应维度。

    泛 smoke 先导 10/24 对稳定平局——两臂行为本就该一样的用例只贡献
    tie，稀释一致率分母；候选的真实效应维度（确认卡完整性、算式展示）
    不在原通用四条判据的视野里，效应可见性为零。新判据 ①②③ 各自锚定
    已冻结的本仓标准：judge rubric P0 数字事实 / 无出处金额率门禁 ≤8%
    （provenance 把解释性算式计入暴露面）/ R8 判据对复述数量的扣分
    先例——不是为臂 B 发明的标准。盲判约束与空段省略行为不变
    （TestBuildPairPrompt / TestJudgePairContext 既有测试钉住）。
    """

    def test_fabrication_counts_as_fact_error(self):
        """判据①：凭空添加商品库没有的参数同样算事实错误（分布级编造）。"""
        prompt = build_pair_prompt("x", "t1", "t2")
        assert "凭空" in prompt
        assert "事实错误" in prompt

    def test_number_provenance_criterion(self):
        """判据②：金额应来自工具返回，自行计算的过程式算式算无出处数字。"""
        prompt = build_pair_prompt("x", "t1", "t2")
        assert "算式" in prompt
        assert "无出处" in prompt

    def test_order_completeness_criterion(self):
        """判据③：订单写操作前要素完整度（商品/数量/单价/金额带币种）。"""
        prompt = build_pair_prompt("x", "t1", "t2")
        assert "要素完整" in prompt
        for token in ("数量", "单价", "币种"):
            assert token in prompt

    def test_criteria_stay_arm_blind(self):
        """新判据文本必须臂中立：不得借道引入版本身份。"""
        prompt = build_pair_prompt("x", "t1", "t2")
        for banned in ("variant", "Variant", "候选版", "A 臂", "B 臂", "A臂", "B臂", "旧版", "新版"):
            assert banned not in prompt

    def test_per_reply_analysis_instruction_kept(self):
        """CoT 迭代无效但无回归（格式修复保留）：先逐条评估再裁决。"""
        prompt = build_pair_prompt("x", "t1", "t2")
        assert "分别" in prompt and "评估" in prompt
        assert "再给" in prompt


class TestPairPromptReasoningSpace:
    """互换一致率 78.3%（先导档）的判段修复：给 judge 推理空间。

    翻转全部落在模糊对上，判词形态是"先裁决后一句理由"——没有逐条独立
    评估的空间，位置敏感性高（MT-Bench/Arena-Hard 的已知问题，标准缓解是
    CoT 先评再裁 + 显式位置中性声明）。这是工具侧修复，指标口径不变。
    """

    def test_prompt_asks_for_per_reply_analysis_before_verdict(self):
        prompt = build_pair_prompt("x", "t1", "t2")
        assert "分别" in prompt and "评估" in prompt  # 先逐条独立评估
        assert "再给" in prompt  # 然后才给裁决

    def test_prompt_declares_position_neutrality(self):
        prompt = build_pair_prompt("x", "t1", "t2")
        assert "顺序" in prompt and ("无关" in prompt or "不影响" in prompt)

    def test_verdict_still_required_with_rationale(self):
        prompt = build_pair_prompt("x", "t1", "t2")
        assert "裁决" in prompt and "理由" in prompt


class TestMajorityVerdict:
    """M2'-d 步骤 2（多数投票）：每序 ×3 取众数，压 judge 采样噪声。

    口径不变（90% 互换门槛 / decisive ≥30 原样）——投票只降单次裁决的
    方差，不改读数定义。无众数（1-1-1）返回 None 由调用方记 error 行：
    宁可少一对，不进一条编造的读数。
    """

    def test_unanimous(self):
        assert majority_verdict(["a", "a", "a"]) == "a"

    def test_two_of_three_majority(self):
        assert majority_verdict(["a", "b", "a"]) == "a"
        assert majority_verdict(["tie", "tie", "b"]) == "tie"

    def test_split_three_ways_has_no_majority(self):
        assert majority_verdict(["a", "b", "tie"]) is None

    def test_even_split_has_no_majority(self):
        assert majority_verdict(["a", "b"]) is None

    def test_two_votes_agree(self):
        assert majority_verdict(["b", "b"]) == "b"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            majority_verdict([])


class TestMajorityVerdictGuardrails:
    """plan 级护栏：votes 超界在烧任何配额之前拒绝。"""

    def test_votes_bounds_in_plan(self):
        from scripts.eval.ab_run import plan_ab_run
        case = {"id": "c1", "queries": ["q1"]}
        with pytest.raises(ValueError, match="votes"):
            plan_ab_run([case], 1, votes=0)
        with pytest.raises(ValueError, match="votes"):
            plan_ab_run([case], 1, votes=6)
        plan = plan_ab_run([case], 1, votes=3)
        assert plan["judge_calls"] == 6  # 1 对 × 2 序 × 3 票
