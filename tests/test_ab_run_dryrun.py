# -*- coding: utf-8 -*-
"""ab_run 的 dry-run 前置与算式（任务 A 第 4 项，本期只做 dry-run 路径）。

要防的问题（都是踩过或险些踩的）：
  - 预算算式与授权口径对不上（P1-4：原文"2 整轮 full/220 次意图"与 k=2 双臂矛盾）；
  - 在降级态跑 A/B：一臂降级，两臂读数一起作废，而报告上看不出来
    （前置 P0-2 的空索引陷阱的同族）；
  - 两臂配错指向同一个服务——A/B 变成 A/A，烧完才发现。

dry-run 的输出必须自带算式（意图数 / judge 调用数 / 墙钟估算），
让"跑之前先报成本"有据可依，而不是拍脑袋报个数。

真实跑测路径本期**刻意不开**：候选提示词还没立项产出，预算方案待拍板。
main() 的真实档直接退出并说明原因——逃生阀不留，防止误触发烧钱。
"""
import pytest

from scripts.eval.ab_run import (
    plan_ab_run,
    run_dry_run,
)


def _case(case_id: str, queries: int, tags=None, faults=None) -> dict:
    return {
        "id": case_id,
        "queries": ["q"] * queries,
        "tags": tags or [],
        "faults": faults,
    }


def _health(url_stale=False, cache=False, variant="", fingerprint="a0915fac", probe=None):
    return {
        "model": "mimo-v2.5",
        "prompt_fingerprint": fingerprint,
        "prompt_variant": variant,
        "semantic_cache": cache,
        "data_dir": "/repo/data",
        "code": {"stale": url_stale, "started_at": "x", "source_mtime": "y", "stale_files": []},
        "retrieval": {
            "reranker": True,
            "lexical_index": True,
            "lexical_gate": 4.0,
            "probe": probe if probe is not None else {"embedding": "ok", "reranker": "ok"},
        },
        "fault_injection": {"enabled": True, "active": []},
    }


class TestPlan:
    """预算算式：k=2 双臂的账本。这是 P1-4 重算的代码化，数字必须能对上手算。"""

    def test_k2_full_mainline_math(self):
        cases = [_case(f"c{i}", 1) for i in range(44)]
        plan = plan_ab_run(cases, k=2)
        assert plan["n_cases"] == 44
        assert plan["executions"] == 176  # 2 臂 × 44 条 × k=2
        assert plan["intents"] == 176  # 每条 1 轮
        assert plan["pairs"] == 88  # diagonal：每用例 k=2 对
        assert plan["judge_calls"] == 176  # 每对正反两个顺序各判一次
        assert plan["pairing"] == "diagonal"

    def test_intents_count_queries_not_cases(self):
        """意图数按 queries 数（R7 口径 64/44），不按用例数。"""
        cases = [_case("a", 3), _case("b", 2)]
        plan = plan_ab_run(cases, k=2)
        assert plan["intents"] == (3 + 2) * 2 * 2  # Σqueries × 2 臂 × k

    def test_cross_pairing(self):
        cases = [_case("a", 1)]
        plan = plan_ab_run(cases, k=2, pairing="cross")
        assert plan["pairs"] == 4 and plan["pairing"] == "cross"

    def test_wall_clock_estimate_present_and_positive(self):
        cases = [_case(f"c{i}", 1) for i in range(44)]
        plan = plan_ab_run(cases, k=2)
        assert plan["estimated_minutes"] > 0
        # R7 实测 64 意图 ≈ 55 分钟 → ~51.6s/意图；176 意图 ≈ 151 分钟
        assert 100 <= plan["estimated_minutes"] <= 200

    def test_decisive_gate_math_is_reported_not_promised(self):
        """决定性对是跑出来才知道的量——算式只报上限与门槛占比，不许许诺达标。"""
        cases = [_case(f"c{i}", 1) for i in range(44)]
        plan = plan_ab_run(cases, k=2)
        assert plan["decisive_ceiling"] == 88
        assert plan["decisive_gate"] == 30
        assert plan["decisive_needed_ratio"] == pytest.approx(30 / 88)

    def test_empty_cases_raise(self):
        with pytest.raises(ValueError):
            plan_ab_run([], k=2)

    def test_k_below_one_raises(self):
        with pytest.raises(ValueError):
            plan_ab_run([_case("a", 1)], k=0)


class TestDryRun:
    def test_happy_path_reports_both_arms_and_plan(self):
        cases = [_case("a", 1), _case("b", 2)]
        text = run_dry_run(
            health_a=_health(variant=""),
            health_b=_health(variant="candidate-x", fingerprint="b2222222"),
            cases=cases,
            k=2,
            judge_model="longcat-2.0",
        )
        assert "臂 A" in text and "臂 B" in text
        assert "a0915fac" in text and "b2222222" in text
        assert "提示词变体 candidate-x" in text
        assert "意图" in text and "judge" in text and "分钟" in text

    def test_aa_pairing_warns(self):
        """两臂指纹相同 = A/A 对照：量的是 judge 噪声，必须当场挑明。"""
        text = run_dry_run(
            health_a=_health(variant="base"),
            health_b=_health(variant="also-base"),
            cases=[_case("a", 1)],
            k=2,
            judge_model="j",
        )
        assert "A/A" in text

    def test_fingerprint_missing_is_not_reported_as_aa(self):
        """两臂都没报指纹（旧服务）→ 写'无从判定'，不许误报成 A/A。"""
        health_a = _health(variant="a")
        health_b = _health(variant="b")
        for h in (health_a, health_b):
            h.pop("prompt_fingerprint")
        text = run_dry_run(
            health_a=health_a,
            health_b=health_b,
            cases=[_case("a", 1)],
            k=2,
            judge_model="j",
        )
        assert "A/A" not in text
        assert "无从判定" in text

    def test_same_url_rejected(self):
        with pytest.raises(SystemExit):
            run_dry_run(
                health_a=_health(),
                health_b=_health(variant="x"),
                cases=[_case("a", 1)],
                k=2,
                judge_model="j",
                arm_a_url="http://127.0.0.1:8000",
                arm_b_url="http://127.0.0.1:8000",
            )

    def test_same_url_different_spelling_rejected(self):
        """localhost 与 127.0.0.1 字面不同、指向相同——字符串比较会漏。"""
        with pytest.raises(SystemExit):
            run_dry_run(
                health_a=_health(),
                health_b=_health(variant="x"),
                cases=[_case("a", 1)],
                k=2,
                judge_model="j",
                arm_a_url="http://127.0.0.1:8000",
                arm_b_url="http://localhost:8000",
            )

    def test_semantic_cache_on_rejects(self):
        with pytest.raises(SystemExit) as err:
            run_dry_run(
                health_a=_health(cache=True),
                health_b=_health(variant="x"),
                cases=[_case("a", 1)],
                k=2,
                judge_model="j",
            )
        assert "语义缓存" in str(err.value)

    def test_stale_service_rejects(self):
        with pytest.raises(SystemExit) as err:
            run_dry_run(
                health_a=_health(url_stale=True),
                health_b=_health(variant="x"),
                cases=[_case("a", 1)],
                k=2,
                judge_model="j",
            )
        assert "旧代码" in str(err.value)

    def test_degraded_probe_rejects(self):
        """降级态不做 A/B：一臂降级两臂作废，且 A/B 报告不像主线报告那样显眼。"""
        with pytest.raises(SystemExit) as err:
            run_dry_run(
                health_a=_health(probe={"embedding": "error: 502", "reranker": "ok"}),
                health_b=_health(variant="x"),
                cases=[_case("a", 1)],
                k=2,
                judge_model="j",
            )
        assert "降级" in str(err.value) or "embedding" in str(err.value)

    def test_disabled_probe_passes_with_note(self):
        """disabled 与 error 含义相反（retrieval_probe 的契约）：未配精排是合法配置，
        不拦，但要在输出里点名——A/B 的归因需要知道两臂检索档位。"""
        text = run_dry_run(
            health_a=_health(probe={"embedding": "ok", "reranker": "disabled"}),
            health_b=_health(variant="x"),
            cases=[_case("a", 1)],
            k=2,
            judge_model="j",
        )
        assert "disabled" in text or "未启用" in text

    def test_probe_missing_field_still_rejects(self):
        """探活结果整体缺失（旧服务/deep 未生效）→ 无从判定降级与否 → 拦。"""
        health = _health()
        del health["retrieval"]["probe"]
        with pytest.raises(SystemExit):
            run_dry_run(
                health_a=health,
                health_b=_health(variant="x"),
                cases=[_case("a", 1)],
                k=2,
                judge_model="j",
            )

    def test_faults_declared_without_service_support_rejects(self):
        """与 eval_regression 同一道闸：故障用例跑在没开注入的服务上 = 假 PASS。"""
        no_fault_support = _health(variant="x")
        no_fault_support["fault_injection"] = {"enabled": False, "active": []}
        with pytest.raises(SystemExit):
            run_dry_run(
                health_a=_health(),
                health_b=no_fault_support,
                cases=[_case("a", 1, faults=["reranker"])],
                k=2,
                judge_model="j",
            )

    def test_variant_missing_field_is_flagged(self):
        """旧代码服务不报 prompt_variant 字段 → 两臂归因缺主键，必须点名。"""
        health = _health()
        health.pop("prompt_variant")
        text = run_dry_run(
            health_a=health,
            health_b=_health(variant="x"),
            cases=[_case("a", 1)],
            k=2,
            judge_model="j",
        )
        assert "prompt_variant" in text
