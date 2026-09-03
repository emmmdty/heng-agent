# -*- coding: utf-8 -*-
"""用例指代歧义自检的判据单测

自检脚本本身也会错，而它错的方式很有害：**误报会逼人去改本来没问题的用例**。
原判据只看品牌词——"LumenGo" 匹配两个 SPU 就报歧义，哪怕 query 里写的是
"LumenGo 露营灯"、自然语言早把歧义消掉了。按那个判据去"修"，只会把
query 改得越来越啰嗦，真正的歧义（AeroHush 耳机 = Pro 还是 Lite？）反而
淹没在一堆误报里。

所以判据必须和真实的消歧机制对齐：看**整句 query** 能不能把候选收敛到一个。
"""
from scripts.eval.audit_cases import Candidate, resolve_brand


def _candidates(*titles):
    return [Candidate(product_id=f"P{i:04d}", title=title) for i, title in enumerate(titles, start=1)]


class TestResolveBrand:
    def test_single_candidate_is_never_ambiguous(self):
        resolution = resolve_brand("LumenGo 露营灯", _candidates("LumenGo 便携露营灯 可充电"))
        assert resolution.ambiguous is False

    def test_category_word_in_query_disambiguates(self):
        """LumenGo 匹配两个 SPU，但 query 里的"露营灯"只命中其中一个。"""
        resolution = resolve_brand(
            "帮我买 1 个 LumenGo 露营灯军绿色",
            _candidates("LumenGo 便携露营灯 可充电", "LumenGo Mini 钥匙扣手电"),
        )
        assert resolution.ambiguous is False
        assert resolution.winner == "P0001"

    def test_shared_category_word_stays_ambiguous(self):
        """AeroHush 耳机：Pro 和 Lite 都是耳机，"耳机"消不掉歧义。"""
        resolution = resolve_brand(
            "AeroHush 耳机寄到美国多少钱",
            _candidates("AeroHush 主动降噪蓝牙耳机 Pro", "AeroHush Lite 半入耳蓝牙耳机"),
        )
        assert resolution.ambiguous is True
        assert resolution.winner is None

    def test_model_number_disambiguates(self):
        resolution = resolve_brand(
            "VoltTrek 30W 迷你充电器多少钱",
            _candidates(
                "VoltTrek 65W 氮化镓旅行充电器（全球插脚）",
                "VoltTrek 30W 迷你充电器",
                "VoltTrek 100W 四口氮化镓充电器",
            ),
        )
        assert resolution.ambiguous is False
        assert resolution.winner == "P0002"

    def test_bare_brand_stays_ambiguous(self):
        resolution = resolve_brand(
            "VoltTrek 充电器哪个划算",
            _candidates("VoltTrek 65W 氮化镓旅行充电器", "VoltTrek 30W 迷你充电器"),
        )
        assert resolution.ambiguous is True


class TestExplicitComparison:
    """一条 query 故意点名同品牌的多个变体，是**有意的对比**，不是歧义。

    九期写 `de-minimis-boundary-eu` 时撞上：免税额度临界最干净的对照就是
    同系列两款箱子（TrailOx 20寸 899 元未超额度、24寸 1199 元超了），
    变量只差价格一项。query 里 "20寸"/"24寸" 都写全了、rubric 也把两组数字
    都钉住了，但原判据只认"收敛到唯一一个"，于是报歧义。

    这跟本脚本要防的风险是正交的：它防的是"Agent 合理地选了另一个变体却被判
    FAIL"，而两个变体都被要求出现时，这种误判根本不可能发生。
    照着误报去改 query，只会把用例改弱——正是本脚本开头警告过的那种"修法"。

    放行的边界要卡死：必须**每个**候选都命中了自己的独有辨识词才算显式对比；
    只点名一部分时剩下那些仍是歧义。
    """

    def test_query_naming_every_variant_is_a_comparison_not_ambiguity(self):
        resolution = resolve_brand(
            "TrailOx 20寸登机箱和 TrailOx 24寸托运箱，都寄到欧盟，分别到手多少钱？",
            _candidates("TrailOx 20寸登机行李箱 铝框款", "TrailOx 24寸托运行李箱 铝框款"),
        )
        assert resolution.ambiguous is False
        assert resolution.explicit_all is True
        assert resolution.winner is None, "对比场景没有唯一赢家，不能硬选一个"

    def test_naming_only_some_variants_is_still_ambiguous(self):
        """三个变体只点名两个，剩下那个仍可能被 Agent 合理选中。"""
        resolution = resolve_brand(
            "VoltTrek 30W 和 VoltTrek 100W 哪个划算",
            _candidates(
                "VoltTrek 65W 氮化镓旅行充电器（全球插脚）",
                "VoltTrek 30W 迷你充电器",
                "VoltTrek 100W 四口氮化镓充电器",
            ),
        )
        assert resolution.explicit_all is False

    def test_bare_brand_is_not_a_comparison(self):
        """所有候选都没被点名 ≠ 全都被点名。这条防的是把 0 命中错当成全命中。"""
        resolution = resolve_brand(
            "VoltTrek 充电器哪个划算",
            _candidates("VoltTrek 65W 氮化镓旅行充电器", "VoltTrek 30W 迷你充电器"),
        )
        assert resolution.ambiguous is True
        assert resolution.explicit_all is False


class TestMissingPins:
    """显式对比时，rubric 必须把被点名的变体都钉住。

    只钉一个的话，Agent 老老实实答了两个，判据却只检验一个——
    另一个答错也照样 PASS。这是"能被跳过的判据等于没有判据"的另一种形态。
    """

    def test_partial_pins_are_reported(self):
        from scripts.eval.audit_cases import missing_pins

        assert missing_pins({"P1002", "P1015"}, {"P1002"}) == {"P1015"}

    def test_all_pinned_is_clean(self):
        from scripts.eval.audit_cases import missing_pins

        assert missing_pins({"P1002", "P1015"}, {"P1002", "P1015"}) == set()

    def test_no_pins_at_all_is_not_a_gap(self):
        """整条 rubric 都不钉 product_id（判据用价格数字写死）是合法写法，
        不能因为"没钉"就报缺口——那会逼人往 rubric 里塞无关的 id。"""
        from scripts.eval.audit_cases import missing_pins

        assert missing_pins({"P1002", "P1015"}, set()) == set()

    def test_pins_from_other_cases_do_not_count(self):
        """rubric 钉的是别的商品时，与这组候选无关。"""
        from scripts.eval.audit_cases import missing_pins

        assert missing_pins({"P1002", "P1015"}, {"P1008"}) == set()
