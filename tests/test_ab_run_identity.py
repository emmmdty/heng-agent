# -*- coding: utf-8 -*-
"""run_identity 的 prompt_variant 字段（任务 A 第 3 项）。

要防的问题：**A/B 两臂的读数说不清自己是哪一臂跑出来的。**

提示词指纹（prompt_fingerprint）只在两版提示词内容不同时才不同；
A/B 分流还需要一个"这一臂是哪个候选"的显式标记——同一次跑测里
变体名是归因的主键，指纹是内容完整性的校验码，两者缺一不可。

接线测试钉住三处（防"写完没接上"，七期的教训：判据不接线等于没做）：
  1. Settings 能从 PROMPT_VARIANT 环境变量读到变体名（默认空 = 基线）；
  2. /health 返回体里有 prompt_variant 字段（空也报——"字段缺席"与
     "空字符串=基线"必须可区分，前者说明服务是旧代码）；
  3. 配置行在变体非空时渲染出来。

配置行的渲染遵循 _describe_faults 的先例：**只在真的设了变体时才占一格**。
恒定不变的空格子每多一个，真正变了的那个就更难被看见；
而"没设变体 = 基线"本身就是真话，不是缺信息——这与"缺字段写未知"不冲突：
那个规矩管的是"这项状态无从判定"，prompt_variant 空时状态是明确的。
"""
import inspect

from app.application.harness.run_identity import describe_run


class TestDescribeRunVariant:
    def test_variant_set_is_rendered_next_to_fingerprint(self):
        line = describe_run(
            {
                "model": "mimo-v2.5",
                "prompt_fingerprint": "a0915fac",
                "prompt_variant": "candidate-no-confirm",
                "retrieval": {"reranker": True, "lexical_index": True, "lexical_gate": 4.0},
            },
            judge_model="longcat-2.0",
        )
        assert "提示词 a0915fac" in line
        assert "提示词变体 candidate-no-confirm" in line
        # 变体紧跟指纹：归因时这两个字段是一对
        assert line.index("提示词变体") - line.index("提示词 a0915fac") < 20

    def test_variant_empty_is_base_not_unknown(self):
        """空 = 基线（真话），不渲染占位格，也不写"未知"误导。"""
        for health in (
            {"model": "m", "prompt_fingerprint": "a0915fac"},
            {"model": "m", "prompt_fingerprint": "a0915fac", "prompt_variant": ""},
        ):
            line = describe_run(health, judge_model="j")
            assert "提示词变体" not in line
            assert "未知" not in line.split("提示词")[1].split("｜")[0]

    def test_old_server_without_field_still_renders(self):
        """老服务不报这个字段 → 不渲染；报了空 → 同样不渲染。两者外观一致。"""
        line = describe_run({"model": "m", "prompt_fingerprint": "a0915fac"}, judge_model="j")
        assert "提示词变体" not in line


class TestSettingsVariant:
    def test_default_is_empty_base(self):
        from app.infrastructure.settings import Settings

        assert Settings.__dataclass_fields__["prompt_variant"].default == ""

    def test_load_settings_reads_env(self, monkeypatch):
        from app.infrastructure.settings import load_settings

        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("PROMPT_VARIANT", "candidate-x")
        assert load_settings().prompt_variant == "candidate-x"

        monkeypatch.setenv("PROMPT_VARIANT", "")
        assert load_settings().prompt_variant == ""


class TestWiring:
    """钉住接线：/health 与 Container 都要有 prompt_variant，且源头是 settings。"""

    def test_health_payload_includes_variant_field(self):
        from app.presentation import server

        source = inspect.getsource(server)
        assert '"prompt_variant"' in source, (
            "/health 必须报出 prompt_variant（空也报）——不接线的话 A/B 两臂"
            "的读数无法归因到变体，指纹相同与否会变成唯一线索"
        )

    def test_container_sources_variant_from_settings(self):
        from app import composition

        source = inspect.getsource(composition)
        assert "prompt_variant=settings.prompt_variant" in source, (
            "Container 的 prompt_variant 必须来自 settings——写死空串的话"
            "PROMPT_VARIANT 环境变量永远到不了 /health"
        )
