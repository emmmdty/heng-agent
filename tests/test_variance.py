# -*- coding: utf-8 -*-
"""variance 工具的分组键

**为什么需要这组测试**：方差工具建成以来从未真正出过组——
`config_key()` 把配置行**整行**当键，而配置行里有
`代码 新鲜(服务启动于 09-04 10:00:00)` 这样一段：**每次重启服务，
启动时刻都变，同配置的两轮就永远进不了同一组**。
`make variance` 于是只能对"恰好同一次重启里跑了几轮"的罕见情形出读数，
实际上等于不可用。

修法是解析配置行的分段结构，剔除易变段（启动时刻、过期文件清单），
保留语义段（模型 / 评审模型 / 提示词指纹 / 精排与探活 / 语义缓存 / 故障注入）。
代码新鲜度只保留"新鲜 / 已过期"这个语义标记本身——跑在旧代码上的轮次
不该和新代码的混在一起，但"几点启动的"不是配置。
"""
from scripts.eval.variance import config_key, collect_scores, run_level_means

FRESH_A = (
    "被测模型 mimo-v2.5｜评审模型 longcat-2.0｜提示词 3a9d8a99"
    "｜精排 开｜字面索引 开｜字面门限 4.0｜语义缓存 关"
    "｜代码 新鲜(服务启动于 09-04 10:00:00)"
)
FRESH_B = (
    "被测模型 mimo-v2.5｜评审模型 longcat-2.0｜提示词 3a9d8a99"
    "｜精排 开｜字面索引 开｜字面门限 4.0｜语义缓存 关"
    "｜代码 新鲜(服务启动于 09-04 17:32:41)"
)
STALE = (
    "被测模型 mimo-v2.5｜评审模型 longcat-2.0｜提示词 3a9d8a99"
    "｜精排 开｜字面索引 开｜字面门限 4.0｜语义缓存 关"
    "｜代码 ⚠️已过期(服务启动于 09-04 08:00:00，但 settings.py 在那之后被改过"
    "，最新 09-04 09:00:00)"
)
OTHER_PROMPT = FRESH_A.replace("3a9d8a99", "b7c1d2e3")
OTHER_JUDGE = FRESH_A.replace("longcat-2.0", "deepseek-v4-flash")


class TestConfigKeyIgnoresRestartTime:
    def test_same_config_different_start_time_same_group(self):
        """核心红测试：仅启动时刻不同的两轮必须进同一组。

        现状（整行当键）下它们是两组，方差对"同配置多轮"永远量不出来。"""
        assert config_key({"run": FRESH_A}) == config_key({"run": FRESH_B})

    def test_stale_and_fresh_are_different_groups(self):
        """启动时刻剔除，但"跑在旧代码上"这个语义必须保留——
        过期服务的读数评的是修复前的行为，混进来方差就没了意义。"""
        assert config_key({"run": FRESH_A}) != config_key({"run": STALE})

    def test_different_prompt_fingerprint_different_group(self):
        assert config_key({"run": FRESH_A}) != config_key({"run": OTHER_PROMPT})

    def test_different_judge_different_group(self):
        """换 judge 等于换尺子（配置行自己的注释口径），保守起见不混组。"""
        assert config_key({"run": FRESH_A}) != config_key({"run": OTHER_JUDGE})

    def test_missing_run_line_is_its_own_group(self):
        assert config_key({}) == config_key({})
        assert config_key({}) != config_key({"run": FRESH_A})


def _report(run_line, case_id, score):
    return {
        "run": run_line,
        "results": [{"id": case_id, "score": score, "verdict": "PASS"}],
    }


class TestGroupingEndToEnd:
    def test_collect_scores_groups_across_restarts(self):
        reports = [
            _report(FRESH_A, "compare-two", 1.0),
            _report(FRESH_B, "compare-two", 0.75),
            _report(OTHER_PROMPT, "compare-two", 0.5),
        ]
        scores = collect_scores(reports)
        same_config = [k for k in scores if "3a9d8a99" in k[0]]
        assert len(same_config) == 1, f"同配置应合并成一组，实际 {same_config}"
        assert scores[same_config[0]] == [1.0, 0.75]

    def test_run_level_means_groups_across_restarts(self):
        reports = [_report(FRESH_A, "a", 1.0), _report(FRESH_B, "b", 0.8)]
        means = run_level_means(reports)
        assert len(means) == 1
        assert means[next(iter(means))] == [1.0, 0.8]
