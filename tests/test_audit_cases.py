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
