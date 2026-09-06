# -*- coding: utf-8 -*-
"""结构化记忆沉淀（#12 任务 B，M0-c）。

范围约束（交接文档「五之一」任务 B）：沉淀条目 = 结构化字段而非自由文本，
每条必须能回答"回放里它改变了哪个行为"；不可验证的进化等于不可信的进化。
本模块钉三件事：
  1. MemoryDeposit schema：触发会话 / 偏好断言 / 行为面 / 生效条件 / 可验证断言，
     全字段校验，缺一不许构造；
  2. 确定性验证器：对照两份回放 transcript（注入开 vs 关）断言行为差异——
     验证器是纯文本判定，零 LLM；assertion 人读，verifier_spec 机检；
  3. 写入门：verifier_spec 构造不出验证器的条目在写入时被拒（不许写入）。
存储走 memory 层（JSON 文件，与 PreferenceStore 同一目录约定）。
"""
import json
from pathlib import Path

import pytest

from app.domain.buyer.deposit import (
    MemoryDeposit,
    PreferenceMentionVerifier,
    PreferenceRecallRestoredVerifier,
    ProductPresenceVerifier,
    RecommendationComplianceVerifier,
    VerificationResult,
    build_verifier,
)
from app.application.memory.deposit_store import DepositStore


def _deposit(**overrides) -> MemoryDeposit:
    fields = dict(
        buyer_id="b1",
        kind="dislike",
        statement="不要塑料材质",
        trigger_session_id="ab-b-k0-memory-write-abc123",
        trigger_query="记住：我以后买东西都不要塑料材质的，我对塑料过敏。",
        behavior_surface="recommendation",
        precondition="推荐含材质属性的商品时",
        assertion="注入开时推荐不含 Voyager（涤纶）旅行三件套；注入关时无此约束",
        verifier_spec={
            "kind": "product_presence",
            "product": "Voyager",
            "expect_on": False,
            "require_contrast": True,
        },
    )
    fields.update(overrides)
    return MemoryDeposit(**fields)


class TestMemoryDepositSchema:
    def test_valid_deposit_constructs(self):
        deposit = _deposit()
        assert deposit.statement == "不要塑料材质"
        assert deposit.created_at  # 缺省自动打时间戳

    def test_required_fields_reject_empty(self):
        for field in ("buyer_id", "statement", "trigger_session_id", "trigger_query",
                      "precondition", "assertion"):
            with pytest.raises(ValueError):
                _deposit(**{field: ""})
            with pytest.raises(ValueError):
                _deposit(**{field: "   "})

    def test_kind_must_be_like_or_dislike(self):
        with pytest.raises(ValueError):
            _deposit(kind="habit")

    def test_behavior_surface_must_be_known(self):
        """行为面是封闭枚举——开放字符串会让'影响的行为面'退化成自由文本。"""
        with pytest.raises(ValueError):
            _deposit(behavior_surface="whatever")

    def test_verifier_spec_required(self):
        with pytest.raises(ValueError):
            _deposit(verifier_spec={})

    def test_dict_round_trip_preserves_fields(self):
        deposit = _deposit()
        restored = MemoryDeposit.from_dict(deposit.to_dict())
        assert restored == deposit

    def test_deposit_id_is_deterministic_for_dedup(self):
        """同 buyer + 同 statement + 同触发会话 = 同一条沉淀（重跑不重复入库）。"""
        assert _deposit().deposit_id == _deposit().deposit_id
        assert _deposit().deposit_id != _deposit(trigger_session_id="another-session")

    def test_json_round_trip_preserves_verifier_spec(self):
        deposit = _deposit()
        payload = json.loads(json.dumps(deposit.to_dict(), ensure_ascii=False))
        assert MemoryDeposit.from_dict(payload) == deposit


class TestBuildVerifier:
    def test_builds_product_presence(self):
        verifier = build_verifier({"kind": "product_presence", "product": "Voyager", "expect_on": False})
        assert isinstance(verifier, ProductPresenceVerifier)

    def test_builds_preference_mention(self):
        verifier = build_verifier({"kind": "preference_mention", "keywords": ["不要塑料", "塑料过敏"]})
        assert isinstance(verifier, PreferenceMentionVerifier)

    def test_builds_recommendation_compliance(self):
        verifier = build_verifier({
            "kind": "recommendation_compliance",
            "product": "Voyager",
            "material_markers": ["涤纶"],
            "choice_markers": ["没意见"],
            "main_rec_markers": ["最推荐"],
        })
        assert isinstance(verifier, RecommendationComplianceVerifier)

    def test_unknown_kind_raises_with_name(self):
        with pytest.raises(ValueError) as err:
            build_verifier({"kind": "vibes"})
        assert "vibes" in str(err.value)

    def test_missing_params_raise(self):
        with pytest.raises(ValueError):
            build_verifier({"kind": "product_presence"})
        with pytest.raises(ValueError):
            build_verifier({"kind": "preference_mention", "keywords": []})
        with pytest.raises(ValueError):
            build_verifier({"kind": "recommendation_compliance", "product": "Voyager"})


class TestProductPresenceVerifier:
    def test_excluded_product_with_contrast_passes(self):
        """注入开不含 Voyager、注入关含 Voyager——行为差异可归因到注入，验证通过。"""
        verifier = ProductPresenceVerifier(product="Voyager", expect_on=False, require_contrast=True)
        result = verifier.check(
            transcript_on="[Agent] 为您推荐 Nomadica 旅行三件套……",
            transcript_off="[Agent] 为您推荐 Voyager 旅行三件套 记忆棉款，139 元……",
        )
        assert result.ok
        assert result.detail

    def test_no_contrast_fails_when_required(self):
        """两臂都含该商品 = 注入没有改变行为——这条沉淀回答不了'改变了哪个行为'。"""
        verifier = ProductPresenceVerifier(product="Voyager", expect_on=False, require_contrast=True)
        result = verifier.check(
            transcript_on="[Agent] 推荐 Voyager 记忆棉款",
            transcript_off="[Agent] 推荐 Voyager 记忆棉款",
        )
        assert not result.ok

    def test_no_contrast_tolerated_when_not_required(self):
        verifier = ProductPresenceVerifier(product="Voyager", expect_on=False, require_contrast=False)
        result = verifier.check(
            transcript_on="[Agent] 推荐 Nomadica",
            transcript_off="[Agent] 推荐 Voyager 与 Nomadica",
        )
        assert result.ok

    def test_expectation_violation_fails_with_detail(self):
        """注入开了反而还在推 Voyager——断言被违反，验证必须留名失败原因。"""
        verifier = ProductPresenceVerifier(product="Voyager", expect_on=False, require_contrast=False)
        result = verifier.check(
            transcript_on="[Agent] 推荐 Voyager 记忆棉款",
            transcript_off="[Agent] 推荐 Voyager 记忆棉款",
        )
        assert not result.ok and "Voyager" in result.detail

    def test_expect_on_true_passes_when_present_on_only(self):
        verifier = ProductPresenceVerifier(product="Nomadica", expect_on=True, require_contrast=True)
        assert verifier.check("[Agent] 推荐 Nomadica", "[Agent] 推荐 Voyager").ok


class TestRecommendationComplianceVerifier:
    """两级判定（2026-09-06 用户裁量，对齐 cases.yaml preference-conflict 先例）：

    ① 主推荐不得是 dislike 命中的商品；② 该商品出现处必须伴随材质冲突说明
    （材质事实词 + 把选择权交回买家的话，两样齐才算"显式说明"）。
    测试形态按 M1 复跑两臂 transcript 的真实结构造（markdown 分节）。
    """

    _SPEC = {
        "product": "Voyager",
        "material_markers": ["涤纶", "化纤"],
        "choice_markers": ["没意见", "介意", "偏好", "冲突", "避开", "慎选"],
        "main_rec_markers": ["最推荐", "首推", "综合来看", "主推"],
    }

    def _verifier(self, **overrides) -> RecommendationComplianceVerifier:
        return RecommendationComplianceVerifier(**{**self._SPEC, **overrides})

    # ---- 臂 B 形态（M1 实测）：主推荐 Nomadica + Voyager 备选带材质说明 → 合规
    _ARM_B_ON = """### ① Nomadica 旅行三件套 ⭐ 首推
帆布+再生尼龙，你之前偏好避开塑料，这款完全契合。

### 备选：Voyager 旅行三件套 记忆棉款
| 材质 | 记忆棉 + 涤纶外套 |
> **说明**：但外套为涤纶（化纤），不属于帆布/天然材质那档。如果你对涤纶没意见，这款性价比更高。

**综合来看，最推荐 Nomadica 军绿色款**。"""

    # ---- 臂 A 形态（M1 实测）：Voyager 平级列出，只有材质表格行、无冲突连接语 → 违规
    _ARM_A_OFF = """### ① Nomadica 旅行三件套
帆布+再生尼龙是知识库点名的"非塑料"优选材质。

### ② Voyager 旅行三件套 记忆棉款 —— 预算更友好
| 材质 | 记忆棉 + 涤纶外套 |
> **推荐理由**：如果颈枕舒适度是第一诉求，记忆棉的包裹感更好；价格入门，适合尝鲜。"""

    def test_arm_b_shape_passes_and_arm_a_shape_violates(self):
        """M1 复跑的两臂形态：注入开合规、注入关违规——注入效应量得出来。"""
        verifier = self._verifier()
        on_result = verifier._judge(self._ARM_B_ON)
        off_result = verifier._judge(self._ARM_A_OFF)
        assert on_result[0] is True, on_result[1]
        assert off_result[0] is False, off_result[1]

    def test_full_check_confirms_with_contrast(self):
        result = self._verifier().check(
            transcript_on=self._ARM_B_ON, transcript_off=self._ARM_A_OFF,
        )
        assert result.ok and "注入关" in result.detail

    def test_main_recommendation_is_disliked_product_violates(self):
        transcript = "### 推荐\n综合来看，最推荐 Voyager 记忆棉款，139 元。"
        ok, detail = self._verifier()._judge(transcript)
        assert not ok and "主推荐" in detail

    def test_bare_mention_without_conflict_note_violates(self):
        """产品列了、材质词也有（表格行），但没有把材质与偏好的冲突挑明 = 未说明。"""
        transcript = "### ① Voyager 旅行三件套\n| 材质 | 记忆棉 + 涤纶外套 |\n价格 139 元，库存 60 件。"
        ok, detail = self._verifier()._judge(transcript)
        assert not ok and "未说明" in detail

    def test_material_word_alone_is_not_a_conflict_note(self):
        """只有材质事实（表格行有涤纶）没有交还选择权的话——不构成显式说明。
        '如果'这类中性词不算连接语，防误放行。"""
        transcript = "### ② Voyager 记忆棉款\n涤纶外套，舒适导向。适合尝鲜。"
        ok, detail = self._verifier()._judge(transcript)
        assert not ok

    def test_product_absent_on_and_off_fails_contrast(self):
        """两臂都没出现该商品 = 注入没改变行为——合规但无对比，不作数。"""
        result = self._verifier().check(
            transcript_on="### 推荐\n最推荐 Nomadica。", transcript_off="### 推荐\n最推荐 Nomadica。",
        )
        assert not result.ok and "注入" in result.detail

    def test_product_removed_by_injection_passes(self):
        """注入开彻底不提、注入关列了（裸列或说明都算）——行为被注入改变，确认。"""
        result = self._verifier().check(
            transcript_on="### 推荐\n最推荐 Nomadica。",
            transcript_off=self._ARM_A_OFF,
        )
        assert result.ok

    def test_no_main_rec_marker_falls_back_to_mention_check(self):
        """没有主推荐结构词时，退化为逐处提及检查——裸列照样违规。"""
        transcript = "Voyager 记忆棉款，139 元，适合尝鲜。"
        ok, detail = self._verifier()._judge(transcript)
        assert not ok and "未说明" in detail


class TestPreferenceMentionVerifier:
    def test_mentioned_on_but_not_off_passes(self):
        verifier = PreferenceMentionVerifier(keywords=["不要塑料", "塑料过敏"])
        result = verifier.check(
            transcript_on="[Agent] 考虑到您不要塑料材质的偏好，推荐帆布款……",
            transcript_off="[Agent] 为您推荐最便宜的旅行三件套……",
        )
        assert result.ok

    def test_mentioned_in_both_fails(self):
        """注入关也提了偏好 = 注入没产生差异（或偏好从别的渠道泄漏）——不许记成功。"""
        verifier = PreferenceMentionVerifier(keywords=["不要塑料"])
        result = verifier.check(
            transcript_on="[Agent] 您说过不要塑料材质",
            transcript_off="[Agent] 您说过不要塑料材质",
        )
        assert not result.ok

    def test_not_mentioned_on_fails_with_detail(self):
        verifier = PreferenceMentionVerifier(keywords=["不要塑料"])
        result = verifier.check(transcript_on="[Agent] 随便推荐", transcript_off="[Agent] 随便推荐")
        assert not result.ok and "不要塑料" in result.detail

    def test_any_keyword_match_counts(self):
        verifier = PreferenceMentionVerifier(keywords=["不要塑料", "塑料过敏"])
        assert verifier.check(
            transcript_on="[Agent] 您对塑料过敏，已避开相关材质", transcript_off="[Agent] 推荐如下",
        ).ok


class TestPreferenceRecallRestoredVerifier:
    """B3（二十七期）：forget 链的沉淀验证——方向与遵从判定相反。

    语义：撤回生效 = 注入开臂的下游行为**恢复**（dislike 命中的商品回到
    可推荐 + 无"偏好仍生效"声称）；注入开仍挡着它或声称排除 = 撤回链残留
    （0 容忍，任务 B 指标表"记忆污染回归"）。注入关臂只是参考基线：
    两臂都恢复才是正确行为，**不吃 contrast 作废**（对比纪律在这里方向相反）。
    """

    def _verifier(self):
        return build_verifier({
            "kind": "recall_restored",
            "product": "Voyager",
            "exclusion_markers": ["已排除", "为您排除", "仍然生效", "继续避开"],
        })

    def test_builds_recall_restored(self):
        assert isinstance(self._verifier(), PreferenceRecallRestoredVerifier)

    def test_product_back_in_recommendation_passes(self):
        on = "[Agent] 最划算的是 Voyager 旅行三件套 记忆棉款（139 元）"
        off = "[Agent] 推荐 Voyager 旅行三件套（139 元）"
        result = self._verifier().check(on, off)
        assert result.ok is True

    def test_product_blocked_after_forget_fails(self):
        on = "[Agent] 推荐非塑料的 Nomadica 旅行三件套（189 元）"
        off = "[Agent] Voyager 旅行三件套（139 元）也不错"
        result = self._verifier().check(on, off)
        assert result.ok is False
        assert "撤回" in result.detail or "未出现" in result.detail

    def test_exclusion_claim_fails_even_if_product_mentioned(self):
        on = "[Agent] 已排除塑料材质商品；Voyager 那款就不看了"
        off = "[Agent] Voyager（139 元）"
        result = self._verifier().check(on, off)
        assert result.ok is False
        assert "仍生效" in result.detail or "排除" in result.detail

    def test_both_arms_restored_passes_no_contrast_invalidation(self):
        """两臂都恢复 = 撤回链的正确终态——contrast 纪律在这里方向相反，
        同判不定罪（对照 recommendation_compliance 的"两臂相同不作数"）。"""
        on = "[Agent] Voyager 旅行三件套（139 元）最便宜"
        off = "[Agent] Voyager 旅行三件套（139 元）"
        assert self._verifier().check(on, off).ok is True

    def test_missing_params_raise(self):
        with pytest.raises(ValueError):
            build_verifier({"kind": "recall_restored"})


class TestForgetChainDownstream:
    """B3：DOWNSTREAM_CASE 增 memory-forget-setup → memory-forget，
    验证器方向取反（dislike → recall_restored；未撤回的 like 仍查不误伤）。"""

    def test_forget_setup_is_registered_with_reversed_direction(self):
        from scripts.eval.mem_deposit import DOWNSTREAM_CASE
        assert DOWNSTREAM_CASE.get("memory-forget-setup") == "memory-forget"

    def test_dislike_deposit_from_forget_chain_gets_recall_restored_spec(self):
        from scripts.eval.mem_deposit import build_deposit
        write = {"kind": "dislike", "statement": "不要塑料材质",
                 "trigger_query": "记住两件事：我不要塑料材质的东西"}
        deposit = build_deposit("memory-forget-setup", "b1", "sess-1", write)
        assert deposit.verifier_spec["kind"] == "recall_restored"
        assert "撤回" in deposit.assertion

    def test_like_deposit_from_forget_chain_keeps_mention_semantics(self):
        """未撤回的那条偏好（军绿色）不该被一起丢掉——mention 语义不变。"""
        from scripts.eval.mem_deposit import build_deposit
        write = {"kind": "like", "statement": "偏好军绿色",
                 "trigger_query": "另外我偏好军绿色"}
        deposit = build_deposit("memory-forget-setup", "b1", "sess-1", write)
        assert deposit.verifier_spec["kind"] == "preference_mention"


class TestVerificationResult:
    def test_result_carries_detail_both_ways(self):
        assert VerificationResult(ok=True, detail="x").ok
        assert VerificationResult(ok=False, detail="y").detail == "y"


class TestDepositStore:
    def _store(self, tmp_path: Path) -> DepositStore:
        return DepositStore(data_dir=str(tmp_path))

    def test_append_then_list_round_trips(self, tmp_path):
        store = self._store(tmp_path)
        store.append(_deposit())
        stored = store.list_by_buyer("b1")
        assert len(stored) == 1
        assert stored[0].assertion == _deposit().assertion

    def test_append_is_idempotent_by_deposit_id(self, tmp_path):
        store = self._store(tmp_path)
        store.append(_deposit())
        store.append(_deposit())
        assert len(store.list_by_buyer("b1")) == 1

    def test_unverifiable_deposit_is_rejected_and_not_written(self, tmp_path):
        """写入门：构造不出确定性验证器的条目不许落盘——'不可验证 = 不许写入'
        是冻结红线，宁可少一条沉淀，不留一条没法对账的。"""
        store = self._store(tmp_path)
        with pytest.raises(ValueError) as err:
            store.append(_deposit(verifier_spec={"kind": "vibes"}))
        assert "不可验证" in str(err.value) or "vibes" in str(err.value)
        assert store.list_by_buyer("b1") == []

    def test_buyers_are_isolated(self, tmp_path):
        store = self._store(tmp_path)
        store.append(_deposit())
        store.append(_deposit(buyer_id="b2", trigger_session_id="s2"))
        assert len(store.list_by_buyer("b1")) == 1
        assert len(store.list_by_buyer("b2")) == 1

    def test_state_survives_reinstantiation(self, tmp_path):
        self._store(tmp_path).append(_deposit())
        stored = self._store(tmp_path).list_by_buyer("b1")
        assert len(stored) == 1
