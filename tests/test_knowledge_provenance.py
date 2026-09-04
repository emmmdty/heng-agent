# -*- coding: utf-8 -*-
"""知识库出处校验（knowledge_provenance）单测

判据只有一条：**回复里声称"来自知识库 / 品类洞察"的内容，本会话必须真的
有过一次成功的知识库返回。**声称有而工具报错 / 没调用 / 零命中，即为张冠李戴。

来源（交接文档"第一点五优先"，欠了半个期）：二十期分诊 `category-insight`
时确认，judge 结构上看不到工具返回——"知识库当时可不可用"它判不了，
判词原文是"由于无法验证知识库实际状态……拿不准按不通过处理"。
Agent 如实标注了出处却必然判 FAIL，这已经是那条判据的第三个补丁。
拆法各归其位：**数值对不对**归 judge（写死区间拿 transcript 核），
**出处属不属实**看轨迹，归本判据。

范围刻意收窄（宁可漏报不误报）：

    1. 只认**显式**出处声明（"知识库 / 品类洞察"字样）。泛泛的"根据我们的
       经验"不算——把口语化的归因全扫进来，判据会失去可用性。
    2. **诚实降级不算声明**："知识库暂时不可用，我先按常识给您讲"恰恰是
       工具报错时唯一正确的行为，同一行里出现降级措辞就不判。
    3. 反方向（工具返回了、回复没提）不是错——出处校验只罚"声称有而无据"。
"""
from app.application.harness.knowledge_provenance import (
    check_knowledge,
    collect_knowledge_sources,
)


def _sources(tool_results=()):
    return collect_knowledge_sources(tool_results)


_KB_SUCCESS = {
    "tool": "category_insight_tool",
    "hit_count": 2,
    "insights": [{"content": "旅行收纳袋选 PU 涤纶面料，价格区间 60-160 元", "source": "travel-gear.md"}],
}
_KB_EMPTY = {"tool": "category_insight_tool", "hit_count": 0, "insights": []}
_KB_ERROR = {"tool": "category_insight_tool", "error": "知识库索引未就绪"}
_OTHER_TOOL = {"tool": "product_search_tool", "hits": [{"product_id": "P1001"}]}


class TestClaimDetection:
    def test_claim_without_any_kb_call_is_unsourced(self):
        report = check_knowledge("根据知识库，这个品类最看重材质。", _sources())
        assert report.claims == 1
        assert not report.clean

    def test_claim_with_a_successful_return_is_clean(self):
        sources = _sources([_KB_SUCCESS])
        assert check_knowledge("知识库里说这个品类先看材质。", sources).clean

    def test_claim_over_an_error_result_is_unsourced(self):
        """工具报错时"知识库说…"是最典型的张冠李戴——内容只能来自模型自身。"""
        report = check_knowledge("知识库显示这个品类看价格区间。", _sources([_KB_ERROR]))
        assert not report.clean

    def test_claim_over_zero_hits_is_unsourced(self):
        """检索成功但零命中，同样给不出"知识库里说"的出处。"""
        report = check_knowledge("品类洞察里讲要看材质。", _sources([_KB_EMPTY]))
        assert not report.clean

    def test_other_tool_results_do_not_count_as_kb_provenance(self):
        report = check_knowledge("根据知识库，先看材质。", _sources([_OTHER_TOOL]))
        assert not report.clean

    def test_no_claim_word_is_no_claim(self):
        assert check_knowledge("这个品类一般先看材质和容量。", _sources()).clean


class TestHonestDegradation:
    def test_unavailable_wording_is_not_a_claim(self):
        """工具报错后如实告知"知识库暂时不可用"是唯一正确的行为，不能反罚。"""
        reply = "知识库暂时不可用，我先按常识给您讲：这个品类看材质。"
        assert check_knowledge(reply, _sources([_KB_ERROR])).clean

    def test_degradation_and_claim_in_different_lines_are_judged_separately(self):
        """降级说明在上一行，下一行仍声称"知识库说"——按行判，第二行该抓。"""
        reply = "知识库查询失败了。\n不过知识库里说这个品类看材质，您参考一下。"
        report = check_knowledge(reply, _sources([_KB_ERROR]))
        assert len(report.unsourced) == 1


class TestScope:
    def test_reverse_direction_is_not_an_error(self):
        """工具返回了、回复没引用——出处校验只罚"声称有而无据"。"""
        assert check_knowledge("这个品类先看材质和容量。", _sources([_KB_SUCCESS])).clean

    def test_claim_counted_per_line(self):
        """声明按行计：同一行里提到两次"知识库 / 品类洞察"是同一处声明，
        拆成两条只会让读报告的人以为有两处独立的编造。"""
        reply = "知识库里说看材质；另外品类洞察显示价格区间 60-160。"
        report = check_knowledge(reply, _sources())
        assert report.claims == 1
        assert len(report.unsourced) == 1


class TestCalibratedOnRealCorpus:
    """模式在 120 份真实流水上校准过（15 处声明行逐条分诊）。
    这里的每条样本都取自真实回复原文，不是构造的理想输入。"""

    def test_capability_offer_is_not_a_claim(self):
        """真实误报（eval-chitchat-boundary）："我可以提供品类洞察"是能力提议，
        没有把任何内容归因给知识库。"""
        reply = "💡 **选购建议** — 不确定怎么挑？我可以提供品类洞察、热门款型、避坑指南"
        assert check_knowledge(reply, _sources()).clean

    def test_absence_observation_is_not_a_claim(self):
        """真实误报（eval-long-context-memory）："知识库里这条数据不完整，
        我不能替它编一个数字"是诚实行为，反过来罚它，下次它就会编。"""
        reply = "商品卡片上的信息只标注了\"太阳能 + USB 双模供电\"——说明知识库里这条数据不完整，我不能替它编一个数字。"
        assert check_knowledge(reply, _sources([_KB_ERROR])).clean

    def test_absence_with_kb_as_subject_is_not_a_claim(self):
        """"知识库里没有这款"是缺失观察——知识库是"没有"的主语。"""
        assert check_knowledge("很抱歉，知识库里没有这款的防水数据。", _sources()).clean

    def test_attribution_of_a_negative_attribute_is_a_claim(self):
        """"知识库里说这款没有快充"是归因——"没有"属于商品不属于知识库，
        中间隔着字，不得按缺失排除（这条是编造的高发形状）。"""
        report = check_knowledge("知识库里说这款没有快充，只有普通充电。", _sources())
        assert not report.clean

    def test_honest_disclaimer_is_not_a_claim(self):
        """真实原文（eval-no-fabrication-0e2263）：明确标注"非知识库数据"
        是反向诚实，恰恰是判据想鼓励的行为。"""
        assert check_knowledge("### 关于 4K 无人机选购的通用建议（非知识库数据）", _sources()).clean

    def test_kb_lacks_this_category_is_not_a_claim(self):
        """真实原文（eval-no-fabrication-dda02f）："暂无……专项选购指南"是缺失观察。"""
        reply = "- 品类洞察知识库中暂无无人机品类的专项选购指南；"
        assert check_knowledge(reply, _sources([_KB_EMPTY])).clean

    def test_attribution_shapes_from_the_corpus_are_claims(self):
        """真实有据声明的各种构型——判据必须认得它们（在该抓的地方不漏）。"""
        lines = [
            "好的，以下是从知识库拉回来的旅行装备选购口径：",
            "## 📊 品类知识库提示",
            "### 📊 价格参考（来自品类洞察）",
            "根据品类洞察和商品检索，帮你筛选出两款：",
            "帆布+再生尼龙是品类洞察中公认的优质材质。",
        ]
        for line in lines:
            report = check_knowledge(line, _sources())
            assert report.claims == 1, f"应当识别为出处声明：{line}"
