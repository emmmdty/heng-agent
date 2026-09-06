# -*- coding: utf-8 -*-
"""Skill 打包单元与注册表（#14 任务 C，C2）。

任务书口径（交接文档「五之一」任务 C + 二十七期 C1 设计笔记）：
Skill = 提示词片段 + 工具子集 + 判据的打包单元，按任务阶段渐进注入。
本模块钉纯逻辑（schema 校验 / 注册表 / 定义装载），**不碰 composition.py、
不动 heng.yml**——接线在 C3，指纹变更与"新指纹下重取基线"前置纪律一起走。

设计输入（C1 证据，见二十七期skill化设计笔记.md）：
- 工具子集按调用证据分三档：常驻检索 / 交易 / 记忆；Task* 四件套 0 调用不进任何 skill；
- 阶段路由必须保守（宁可多带，C4 的 PASS 不回退是一票否决）；
- 片段缺省不发：registry 对没有 skill 覆盖的阶段返回空集，不假装有内容。
"""
from pathlib import Path

import pytest

from app.skills.loader import load_skill_definitions
from app.skills.registry import SkillRegistry
from app.skills.schema import Stage, SkillSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _spec(skill_id="search-basics", stages=(Stage.SEARCH,), tools=("product_search_tool",),
          fragments=("只推商品库里存在的商品。",), criteria=("价格必须来自工具返回",)):
    return SkillSpec(
        skill_id=skill_id, stages=stages, tools=tools,
        prompt_fragments=fragments, criteria=criteria,
    )


class TestSkillSpec:
    def test_valid_spec_is_frozen(self):
        spec = _spec()
        with pytest.raises(Exception):
            spec.skill_id = "mutated"

    def test_empty_id_rejected(self):
        with pytest.raises(ValueError, match="skill_id"):
            SkillSpec(skill_id="  ", stages=(Stage.SEARCH,), tools=(), prompt_fragments=("x",), criteria=())

    def test_skill_must_pack_something(self):
        """既无片段也无工具也无判据的 skill 是空壳——注册了也不会有任何
        效果，静默存在等于骗自己，构造期就拒。"""
        with pytest.raises(ValueError, match="空壳"):
            SkillSpec(skill_id="empty", stages=(Stage.SEARCH,), tools=(),
                      prompt_fragments=(), criteria=())

    def test_unknown_stage_type_rejected(self):
        with pytest.raises(ValueError):
            SkillSpec(skill_id="s", stages=("nonsense",), tools=("t",),
                      prompt_fragments=("f",), criteria=())

    def test_whitespace_fragment_rejected(self):
        """空白片段会被拼进 system prompt 却什么都没说——还占 token。"""
        with pytest.raises(ValueError, match="片段"):
            SkillSpec(skill_id="s", stages=(Stage.SEARCH,), tools=(),
                      prompt_fragments=("   ",), criteria=())


class TestSkillRegistry:
    def test_register_and_lookup_by_stage(self):
        reg = SkillRegistry(tool_universe={"product_search_tool", "quote_basket_tool"})
        reg.register(_spec())
        reg.register(_spec(skill_id="trade-basics", stages=(Stage.TRADE,),
                           tools=("quote_basket_tool",)))
        assert [s.skill_id for s in reg.for_stage(Stage.SEARCH)] == ["search-basics"]
        assert [s.skill_id for s in reg.for_stage(Stage.TRADE)] == ["trade-basics"]

    def test_common_stage_always_included(self):
        """common 阶段的 skill 在任何阶段都生效（能力边界声明一类）。"""
        reg = SkillRegistry(tool_universe=set())
        reg.register(_spec(skill_id="always", stages=(Stage.COMMON,), tools=()))
        assert [s.skill_id for s in reg.for_stage(Stage.SEARCH)] == ["always"]
        assert [s.skill_id for s in reg.for_stage(Stage.TRADE)] == ["always"]

    def test_duplicate_skill_id_rejected(self):
        reg = SkillRegistry(tool_universe={"product_search_tool"})
        reg.register(_spec())
        with pytest.raises(ValueError, match="search-basics"):
            reg.register(_spec())

    def test_unknown_tool_rejected_with_name(self):
        """工具名拼错 = 静默缺工具 = 行为回退——注册期报错留名，
        不等 C4 护栏轮的'工具调用率下降'才炸。"""
        reg = SkillRegistry(tool_universe={"product_search_tool"})
        with pytest.raises(ValueError, match="no_such_tool"):
            reg.register(_spec(tools=("no_such_tool",)))

    def test_for_stage_order_is_registration_order(self):
        reg = SkillRegistry(tool_universe={"a", "b"})
        reg.register(_spec(skill_id="first", stages=(Stage.SEARCH,), tools=("a",)))
        reg.register(_spec(skill_id="second", stages=(Stage.SEARCH,), tools=("b",)))
        assert [s.skill_id for s in reg.for_stage(Stage.SEARCH)] == ["first", "second"]

    def test_render_fragments_deterministic_with_stage_header(self):
        """渐进注入的产物 = 该阶段全部片段的确定性拼接；
        同一注册状态渲染两次逐字节一致（否则 prompt 缓存/指纹都乱）。"""
        reg = SkillRegistry(tool_universe={"a"})
        reg.register(_spec(skill_id="s1", tools=("a",),
                           fragments=("片段一。", "片段二。")))
        reg.register(_spec(skill_id="common", stages=(Stage.COMMON,), tools=(),
                           fragments=("常驻片段。",)))
        text_a = reg.render_prompt(Stage.SEARCH)
        text_b = reg.render_prompt(Stage.SEARCH)
        assert text_a == text_b
        assert "片段一。" in text_a and "常驻片段。" in text_a

    def test_render_empty_stage_returns_empty_string(self):
        """没有 skill 覆盖的阶段返回空串——调用方据此整段省略，
        不给 prompt 留'技能：未知'的空段落（对齐 build_pair_prompt 的空段纪律）。"""
        reg = SkillRegistry(tool_universe=set())
        assert reg.render_prompt(Stage.TRADE) == ""

    def test_tool_subset_for_stage(self):
        """渐进加载的另一半：该阶段允许的工具并集（不含 Task* 死重）。"""
        reg = SkillRegistry(tool_universe={"a", "b", "c"})
        reg.register(_spec(skill_id="s1", stages=(Stage.SEARCH,), tools=("a",)))
        reg.register(_spec(skill_id="s2", stages=(Stage.SEARCH,), tools=("b",)))
        assert reg.tool_subset(Stage.SEARCH) == ("a", "b")


class TestLoader:
    def test_loads_packaged_definitions(self):
        """仓内 definitions.yml 必须能装载且过校验——工具名对不上运行时
        全集会在 C3 接线时再核（这里核结构与字面）。"""
        skills = load_skill_definitions()
        assert skills, "定义文件为空 = C2 没交付"
        ids = [s.skill_id for s in skills]
        assert len(ids) == len(set(ids)), "skill_id 重复"

    def test_definitions_cover_three_delivery_stages(self):
        """C1 设计笔记的三档：检索 / 交易 / 记忆 + common 常驻。
        每档至少一个 skill，且工具子集互不包含死重工具。"""
        skills = load_skill_definitions()
        by_stage = set()
        for s in skills:
            by_stage |= set(s.stages)
        assert {Stage.SEARCH, Stage.TRADE, Stage.MEMORY, Stage.COMMON} <= by_stage
        dead = {"TaskUpdate", "TaskCreate", "TaskList", "TaskGet"}
        for s in skills:
            assert not dead & set(s.tools), f"{s.skill_id} 带了 0 调用死重工具"

    def test_definitions_yaml_exists_and_parses(self):
        path = PROJECT_ROOT / "app" / "skills" / "definitions.yml"
        assert path.exists(), "定义文件缺失"
        assert path.read_text(encoding="utf-8").strip()

    def test_bare_string_fragments_rejected_not_char_split(self):
        """loader 不得把裸字符串预转换成 tuple——那会把片段拆成单字序列
        静默通过全链路，渲染成单字换行拼进 prompt（烧前评审 A1）。"""
        import tempfile

        bad = """
skills:
  - id: broken
    stages: [search]
    tools: [product_search_tool]
    prompt_fragments: "只推真实商品"
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(bad)
            path = Path(fh.name)
        with pytest.raises(ValueError, match="字符串序列"):
            load_skill_definitions(path)

    def test_duplicate_ids_rejected_at_load(self):
        """loader 自己声称 id 唯一校验（模块 docstring）——单独使用 loader
        （不经 registry）时重复 id 也必须报错留名（烧前评审 A2）。"""
        import tempfile

        bad = """
skills:
  - id: dup
    stages: [search]
    tools: [product_search_tool]
    prompt_fragments: ["a"]
  - id: dup
    stages: [trade]
    tools: [quote_basket_tool]
    prompt_fragments: ["b"]
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(bad)
            path = Path(fh.name)
        with pytest.raises(ValueError, match="dup"):
            load_skill_definitions(path)

    def test_top_level_non_mapping_rejected_with_name(self):
        """顶层是 list 的畸形 YAML 报带文件名的 ValueError，
        不是裸 AttributeError（降级路径必须留因——硬纪律五.4）。"""
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False,
                                         encoding="utf-8") as fh:
            fh.write("- just\n- a\n- list\n")
            path = Path(fh.name)
        with pytest.raises(ValueError, match="顶层"):
            load_skill_definitions(path)

    def test_resident_search_tool_in_common_stage(self):
        """product_search_tool 是常驻档（C1：R8 109 次；heng.yml 交易流也要求
        先检索候选）——必须挂在 common，任何阶段都不可被渐进加载过滤掉
        （烧前评审 B1：保守铁律 = 宁可多带）。"""
        skills = {s.skill_id: s for s in load_skill_definitions()}
        common_tools = set()
        for s in skills.values():
            if Stage.COMMON in s.stages:
                common_tools |= set(s.tools)
        assert "product_search_tool" in common_tools


class TestRegistryReadAccess:
    def test_get_by_id(self):
        reg = SkillRegistry(tool_universe={"product_search_tool"})
        reg.register(_spec())
        assert reg.get("search-basics").tools == ("product_search_tool",)
        assert reg.get("nope") is None

    def test_render_criteria_dedupes_keeping_order(self):
        reg = SkillRegistry(tool_universe={"a"})
        reg.register(_spec(skill_id="s1", tools=("a",),
                           criteria=("数字必须有出处", "如实说明能力边界")))
        reg.register(_spec(skill_id="s2", stages=(Stage.SEARCH,), tools=(),
                           criteria=("数字必须有出处",)))
        assert reg.render_criteria(Stage.SEARCH) == ["数字必须有出处", "如实说明能力边界"]

    def test_tool_subset_dedupes_overlapping_tools(self):
        reg = SkillRegistry(tool_universe={"a", "b"})
        reg.register(_spec(skill_id="s1", stages=(Stage.SEARCH,), tools=("a", "b")))
        reg.register(_spec(skill_id="s2", stages=(Stage.SEARCH,), tools=("b",)))
        assert reg.tool_subset(Stage.SEARCH) == ("a", "b")
