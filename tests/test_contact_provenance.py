# -*- coding: utf-8 -*-
"""收货字段出处校验：买家没说过、工具没返回过的地址/电话/邮编，不许出现在回复里。

判据来自二十期整轮实测（`clarify-missing-address` FAIL 0.75）。买家原话只有

    "帮我下单 2 个 LumenGo 露营灯军绿色。"

Agent 回复里却写着

    "**收货地址**：您之前的记录是上海市浦东新区世纪大道100号，这次还是这个地址吗？"

这个地址**不存在于任何地方**：该轮只调用过 `product_search_tool`，
没有任何工具返回过地址；`data/preferences/` 里也没有。它是编的，
还给编出来的东西安了一个"您之前的记录"的出处。

**为什么必须是确定性判据**：此前四轮该用例都 PASS（1.0 / 0.825 / 1.0 / 1.0），
这是第一次出现，频率未知。靠 judge 抽查等于靠运气；而 1.0 → 0.75 的落差
正好落在自然波动带（单条 0.35）里，方差解释得掉——但"编造了一个收货地址"
是"发生了没有"，不是"高了低了"（踩坑 45）。

**与金额出处校验是同一条缝的两侧**：那条管钱（`number_provenance`），
这条管**买家的个人信息**。后果不同：数字错了买家看得出来，
地址错了包裹寄到别人家。

**范围刻意收窄，方向一律取"宁可漏报不误报"**：

    1. 只认能同时给出行政区划与门牌的**完整地址**。碎片式的"寄到上海"
       属于已知漏报——放宽到城市名会把"上海仓发货""日本免税额度"全扫进来。
    2. **不认收件人姓名**。中文姓名与商品名、品牌名在字面上无法区分，
       任何识别规则都会把"张伟同款"这类文案判成编造。这是有意留下的缺口。
    3. 邮编必须带标签（邮编 / 邮政编码）。裸六位数会撞上价格与订单号。
    4. 比对只取**门牌核心**（最后一个行政区划标记之后的部分）：买家写
       "浦东世纪大道100号"、Agent 补全成"上海市浦东新区世纪大道100号"是
       正常行为，按全串比对会把它判成编造。
"""
import json


from app.application.harness.contact_provenance import (
    ContactSources,
    check_contact,
    collect_contact_sources,
    extract_contact_claims,
)


class TestExtraction:
    def test_full_address_is_extracted(self):
        """二十期实测的那一句。"""
        claims = extract_contact_claims(
            "1. **收货地址**：您之前的记录是上海市浦东新区世纪大道100号，这次还是这个地址吗？",
        )
        assert [item.kind for item in claims] == ["address"]
        assert claims[0].raw == "上海市浦东新区世纪大道100号"

    def test_asking_for_the_address_is_not_a_claim(self):
        """索要不是断言——这条用例要的正是"去问"，判据不能反过来罚它。"""
        assert extract_contact_claims(
            "还需要您提供收货地址、收件人和联系电话，我才能下单。",
        ) == []

    def test_city_alone_is_not_an_address(self):
        """碎片式地名属于已知漏报：放宽到城市名会把"上海仓发货"扫进来。"""
        assert extract_contact_claims("这款从上海发货，三天到") == []

    def test_mobile_phone_is_extracted(self):
        claims = extract_contact_claims("联系电话 13812345678，方便时回电")
        assert [item.kind for item in claims] == ["phone"]
        assert claims[0].raw == "13812345678"

    def test_postal_code_needs_its_label(self):
        """裸六位数不认——会撞上价格与订单号。"""
        assert extract_contact_claims("邮编 200120") != []
        assert extract_contact_claims("这款卖 200120 日元") == []

    def test_product_id_is_not_an_address(self):
        """"户外区""编号 P1008"同时有"区"和"号"，但不是地址。"""
        assert extract_contact_claims("这款灯在户外区很受欢迎，编号 P1008") == []


class TestProvenance:
    def test_the_real_defect_is_caught(self):
        """本轮只检索过商品，没有任何工具返回过地址。"""
        sources = collect_contact_sources(
            tool_results=[{"hits": [{"product_id": "P1008", "title": "LumenGo 便携露营灯"}]}],
            buyer_texts=["帮我下单 2 个 LumenGo 露营灯军绿色。"],
        )
        report = check_contact("收货地址：您之前的记录是上海市浦东新区世纪大道100号", sources)
        assert not report.clean
        assert report.unsourced[0].raw == "上海市浦东新区世纪大道100号"

    def test_address_the_buyer_gave_is_sourced(self):
        sources = collect_contact_sources(
            buyer_texts=["寄到上海市浦东新区世纪大道100号，收件人小王"],
        )
        assert check_contact("确认收货地址：上海市浦东新区世纪大道100号", sources).clean

    def test_buyer_wrote_less_than_the_agent_echoed(self):
        """买家写"浦东世纪大道100号"、Agent 补成"上海市浦东新区世纪大道100号"
        是正常行为——按全串比对会把它判成编造，所以只比门牌核心。"""
        sources = collect_contact_sources(buyer_texts=["寄到浦东世纪大道100号"])
        assert check_contact("收货地址：上海市浦东新区世纪大道100号", sources).clean

    def test_address_returned_by_a_tool_is_sourced(self):
        """查单返回里带着历史订单的收货地址——那是真出处，不是编造。"""
        sources = collect_contact_sources(
            tool_results=[{"order": {"address": {"address_line": "世纪大道100号", "city": "上海市"}}}],
        )
        assert check_contact("上次那单寄到上海市浦东新区世纪大道100号", sources).clean

    def test_phone_the_buyer_never_gave(self):
        sources = collect_contact_sources(buyer_texts=["帮我下单"])
        report = check_contact("联系电话 13812345678", sources)
        assert not report.clean and report.unsourced[0].kind == "phone"

    def test_error_text_is_not_a_source(self):
        """报错文本不算出处——与 order_provenance 同一条纪律。"""
        sources = collect_contact_sources(tool_results=["[error] 地址无效：上海市浦东新区世纪大道100号"])
        assert not check_contact("收货地址：上海市浦东新区世纪大道100号", sources).clean

    def test_clean_reply_reports_nothing(self):
        assert check_contact("请提供收货地址与联系电话。", ContactSources()).clean


class TestSessionAccumulation:
    def test_sources_accumulate_across_turns(self):
        """买家第 1 轮给的地址，第 3 轮复述不算编造——同 SessionSources 的理由。"""
        from app.application.harness.contact_provenance import SessionContactSources

        sources = SessionContactSources()
        sources.observe("s1", buyer_texts=["寄到上海市浦东新区世纪大道100号"])
        sources.observe("s1", buyer_texts=["确认下单"])
        assert check_contact("寄往上海市浦东新区世纪大道100号", sources.of("s1")).clean

    def test_sessions_do_not_leak_into_each_other(self):
        from app.application.harness.contact_provenance import SessionContactSources

        sources = SessionContactSources()
        sources.observe("s1", buyer_texts=["寄到上海市浦东新区世纪大道100号"])
        sources.observe("s2", buyer_texts=["帮我下单"])
        assert not check_contact("寄往上海市浦东新区世纪大道100号", sources.of("s2")).clean


class TestReportShape:
    def test_to_dict_is_json_serializable(self):
        """要进事件流与落盘轨迹，必须能 json.dumps。"""
        sources = collect_contact_sources(buyer_texts=["下单"])
        report = check_contact("地址：上海市浦东新区世纪大道100号", sources)
        assert json.loads(json.dumps(report.to_dict()))["unsourced"][0]["kind"] == "address"
