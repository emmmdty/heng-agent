# -*- coding: utf-8 -*-
"""记忆回放对照（#12 任务 B，M0-b）。

同一批含偏好的会话回放两遍：臂 A = 记忆注入关、臂 B = 开。判段复用 A/B
工具链零新造——本模块自有的确定性逻辑只有三块，全部在这里钉住：
  1. 预登记用例子集的选择（id 对不上 = 用例改名了，报错退出而非静默少跑）；
  2. 两臂配置预检（同指纹、异变体、同模型、非 stale、缓存关）——防静默混装
     （陷阱 1 同族：两臂实际跑的不是声明的那套配置，读数整轮作废且无告警）；
  3. --only 在预登记子集内取交（先导档 M1 挑 2 条），取空报错。
执行段/判段/统计/报告全部委托 run_ab_pipeline（另有测试钉住），此处不重复。
"""
import json
from pathlib import Path

import pytest

from scripts.eval.mem_replay import (
    ARM_EXPECT,
    CONTROL_ARM_EXPECT,
    PREFERENCE_PRESET_IDS,
    find_eval_preference_leftovers,
    preflight_arms,
    purge_eval_preference_leftovers,
    seed_contradiction_preferences,
    select_replay_cases,
)


def _case(case_id: str) -> dict:
    return {"id": case_id, "queries": ["q"], "rubric": {"p0": []}}


def _health(fingerprint="a0915fac", variant="mem-inject-off", model="mimo-v2.5", stale=False):
    return {
        "prompt_fingerprint": fingerprint,
        "prompt_variant": variant,
        "model": model,
        "code": {"stale": stale, "stale_files": []},
        "semantic_cache": False,
    }


def _healths_ok():
    return {
        "A": _health(variant="mem-inject-off"),
        "B": _health(variant="mem-inject-on"),
    }


class TestSelectReplayCases:
    def test_preset_ids_are_pre_registered_and_complete(self):
        """预登记子集钉死：写入/读取/会话内冲突/撤回链五条——指标表的人群，
        改动走回写通道，不在脚本里悄悄增删。"""
        assert set(PREFERENCE_PRESET_IDS) == {
            "memory-write", "memory-recall",
            "preference-conflict-cheapest-vs-dislike",
            "memory-forget-setup", "memory-forget",
        }

    def test_selects_preset_from_cases(self):
        cases = [_case(cid) for cid in
                 ("memory-write", "memory-recall", "no-fabrication", *PREFERENCE_PRESET_IDS)]
        selected = select_replay_cases(cases)
        assert [c["id"] for c in selected] == list(PREFERENCE_PRESET_IDS)

    def test_renamed_or_missing_case_fails_loudly(self):
        """预登记 id 在 cases.yaml 里找不到 = 用例被改名/删除——静默少跑会让
        指标人群缩水而读数外观正常。必须报错留名。"""
        cases = [_case("memory-write"), _case("memory-recall"), _case("memory-forget")]
        with pytest.raises(SystemExit) as err:
            select_replay_cases(cases)
        assert "preference-conflict-cheapest-vs-dislike" in str(err.value)

    def test_only_narrows_within_preset(self):
        """--only 是预登记子集内的取交（M1 先导档挑 2 条），不是任意挑选。"""
        cases = [_case(cid) for cid in PREFERENCE_PRESET_IDS] + [_case("no-fabrication")]
        selected = select_replay_cases(cases, only="memory-write,memory-recall")
        assert [c["id"] for c in selected] == ["memory-write", "memory-recall"]

    def test_only_outside_preset_is_empty_error(self):
        """--only 指到子集外的用例 = 评的人群偏离冻结口径，报错而非跑空。"""
        cases = [_case(cid) for cid in PREFERENCE_PRESET_IDS]
        with pytest.raises(SystemExit):
            select_replay_cases(cases, only="no-fabrication")


class TestPreflightArms:
    def test_ok_config_passes(self):
        preflight_arms(_healths_ok())

    def test_fingerprint_mismatch_raises(self):
        """两臂指纹不同 = 跑的不是同一份基线——读数量到的是提示词差异不是注入差异。"""
        healths = _healths_ok()
        healths["B"]["prompt_fingerprint"] = "b2222222"
        with pytest.raises(SystemExit) as err:
            preflight_arms(healths)
        assert "fingerprint" in str(err.value) or "指纹" in str(err.value)

    def test_wrong_variant_raises_with_arm_named(self):
        healths = _healths_ok()
        healths["A"]["prompt_variant"] = ""
        with pytest.raises(SystemExit) as err:
            preflight_arms(healths)
        assert "臂 A" in str(err.value) and "mem-inject-off" in str(err.value)

    def test_model_mismatch_raises(self):
        healths = _healths_ok()
        healths["B"]["model"] = "another-model"
        with pytest.raises(SystemExit) as err:
            preflight_arms(healths)
        assert "模型" in str(err.value)

    def test_stale_service_raises(self):
        healths = _healths_ok()
        healths["B"]["code"]["stale"] = True
        with pytest.raises(SystemExit) as err:
            preflight_arms(healths)
        assert "stale" in str(err.value)

    def test_semantic_cache_on_raises(self):
        """缓存会把臂 B 上一次的回复喂给臂 A 的评测——评测纪律沿用 eval_regression。"""
        healths = _healths_ok()
        healths["A"]["semantic_cache"] = True
        with pytest.raises(SystemExit) as err:
            preflight_arms(healths)
        assert "语义缓存" in str(err.value)

    def test_unreachable_arm_raises_named(self):
        healths = _healths_ok()
        healths["B"] = {}
        with pytest.raises(SystemExit) as err:
            preflight_arms(healths)
        assert "臂 B" in str(err.value)

    def test_arm_expect_semantics_pinned(self):
        """臂语义钉死：A=关、B=开。写反了整轮读数方向就反了。"""
        assert ARM_EXPECT["A"]["variant"] == "mem-inject-off"
        assert ARM_EXPECT["B"]["variant"] == "mem-inject-on"


class TestEvalPreferenceLeftovers:
    """跨期残留清洗（M1 先导挖出的真缺陷）。

    历期评测写入的 eval-* 买家偏好留在 data/preferences/ 里，下一轮回放的
    注入臂会把残留偏好注入**写入用例本身**——模型看到'已存档'就跳过工具
    调用（M1 实测：臂 B 写入零 tool.invoke，回复声称'之前也已经存档过了'，
    被 judge 双序判为不实陈述）。清洗是回放可重复性的前置，不是可选卫生。
    """

    def test_finds_eval_prefixed_files_only(self, tmp_path):
        prefs = tmp_path / "preferences"
        prefs.mkdir()
        (prefs / "eval-memory-buyer-abbk0.json").write_text("[]", encoding="utf-8")
        (prefs / "real-buyer.json").write_text("[]", encoding="utf-8")
        (prefs / "sub").mkdir()
        found = find_eval_preference_leftovers(tmp_path)
        assert [p.name for p in found] == ["eval-memory-buyer-abbk0.json"]

    def test_purge_moves_to_backup_and_keeps_non_eval(self, tmp_path):
        prefs = tmp_path / "preferences"
        prefs.mkdir()
        (prefs / "eval-a.json").write_text("[]", encoding="utf-8")
        (prefs / "real-buyer.json").write_text("[]", encoding="utf-8")
        moved, backup_dir = purge_eval_preference_leftovers(tmp_path, stamp="20260906-120000")
        assert moved == ["eval-a.json"]
        assert (Path(backup_dir) / "eval-a.json").is_file()
        assert not (prefs / "eval-a.json").exists()
        assert (prefs / "real-buyer.json").is_file()

    def test_purge_without_leftovers_is_noop(self, tmp_path):
        moved, backup_dir = purge_eval_preference_leftovers(tmp_path, stamp="s")
        assert moved == []
        assert backup_dir == ""


class TestPositiveControl:
    """阳性对照（矛盾注入臂，已知更差——工具有效性自证，模式照抄二十五期）。

    给对照臂买家预写一条**取反偏好**（like 喜欢塑料材质），跑中写入用例再写
    入真偏好 → 买家同时持有矛盾记忆 → 已知更差的记忆状态。纯数据层 seed，
    与注入开关正交、零 app 改动；buyer id 按臂后缀隔离，seed 只影响对照臂。
    """

    def test_control_arm_expect_pinned(self):
        """对照臂语义钉死：A=矛盾注入（已知更差，变体名含 weaker 供渲染器识别）、
        B=正常注入。写反了自证读数方向就反了。"""
        assert CONTROL_ARM_EXPECT["A"]["variant"] == "mem-weaker-contradiction"
        assert CONTROL_ARM_EXPECT["B"]["variant"] == "mem-inject-on"

    def test_seed_targets_arm_a_buyers_only(self, tmp_path):
        cases = [
            {"id": "memory-write", "buyer_id": "eval-memory-buyer", "queries": ["q"]},
            {"id": "memory-recall", "buyer_id": "eval-memory-buyer", "queries": ["q"]},
        ]
        seeded = seed_contradiction_preferences(tmp_path, cases, k=2)
        prefs = tmp_path / "preferences"
        names = sorted(p.name for p in prefs.glob("*.json"))
        assert names == ["eval-memory-buyer-abak0.json", "eval-memory-buyer-abak1.json"]
        assert seeded == ["eval-memory-buyer-abak0", "eval-memory-buyer-abak1"]
        payload = json.loads((prefs / "eval-memory-buyer-abak0.json").read_text(encoding="utf-8"))
        assert payload[0]["kind"] == "like"
        assert payload[0]["statement"] == "喜欢塑料材质"
        assert payload[0]["buyer_id"] == "eval-memory-buyer-abak0"

    async def test_seed_readable_by_preference_store(self, tmp_path):
        """seed 文件必须能被服务的 JsonFilePreferenceStore 读出——schema 不兼容
        的 seed = 对照臂静默回到无偏好态，自证轮白跑。"""
        from app.infrastructure.persistence.json_file_stores import JsonFilePreferenceStore

        cases = [{"id": "memory-recall", "buyer_id": "eval-memory-buyer", "queries": ["q"]}]
        seed_contradiction_preferences(tmp_path, cases, k=1)
        seed_contradiction_preferences(tmp_path, cases, k=1)  # 幂等
        store = JsonFilePreferenceStore(tmp_path)
        prefs = await store.list_by_buyer("eval-memory-buyer-abak0")
        assert [(p.kind, p.statement) for p in prefs] == [("like", "喜欢塑料材质")]
