# -*- coding: utf-8 -*-
"""judge 的事实基准

要防的问题：**judge 手上没有规则，就只能验"自洽"，Agent 自圆其说就能过。**

九期实测：`de-minimis-boundary-eu` 那轮 Agent 写出"1,199 × 12% ≈ ¥3.48"——
计税基数是错的（1199 × 12% = 143.88），只是最终数字碰巧对。
judge 当时手上只有商品表和汇率表，判词写的是"与商品库价格及自洽运费/关税一致"，
它根本没有办法独立验证运费 75、关税 3.48、免税额度 1170 这三个数对不对。

补规则表进 ground_truth 的成本极低（静态、几十行），换来的是 judge 从
"验自洽"升级到"验正确"。这与金额出处校验是两条互补的线：
出处校验管**数字有没有来源**，ground_truth 管**数字对不对**。
"""
from scripts.eval_regression import build_ground_truth


class TestGroundTruth:
    def test_includes_product_facts(self):
        text = build_ground_truth()
        assert "P1008" in text and "LumenGo" in text

    def test_includes_exchange_rates(self):
        assert "1 USD = 7.1 CNY" in build_ground_truth()

    def test_includes_tariff_rates_per_destination(self):
        """没有费率表，judge 判不出"欧盟 12%"是对是错。"""
        text = build_ground_truth()
        assert "0.12" in text or "12%" in text, "欧盟费率必须可查"
        assert "数码配件" in text, "US 数码配件 0% 这类品类差异必须可查"

    def test_includes_de_minimis_thresholds(self):
        """免税额度是这批新用例的核心判据，judge 必须能独立核对。"""
        text = build_ground_truth()
        assert "1170" in text, "EU 免税额度 1170 元（150 EUR）"
        assert "5680" in text, "US 免税额度 5680 元（800 USD）"

    def test_includes_freight_rule_with_the_multi_item_formula(self):
        """组合运费"首件全价 + 续件 60%"是模型最常算错的一条，
        judge 手上没有这条公式就只能跟着模型的算法走。"""
        text = build_ground_truth()
        assert "75" in text, "EU 基础运费"
        assert "60%" in text or "0.6" in text, "续件系数必须可查"

    def test_states_the_taxable_base_explicitly(self):
        """关税只对**超出免税额度的部分**计征——judge 要能据此判出
        "1199 × 12%" 这种写法是错的，哪怕最终数字碰巧对。"""
        text = build_ground_truth()
        assert "超出" in text and "免税额度" in text

    def test_fractional_rates_keep_their_precision(self):
        """US 旅行装备是 7.5%，四舍五入成 8% 会让 judge 拿错标准去核对——
        正确的关税反而会被判错。"""
        text = build_ground_truth()
        assert "7.5%" in text
        assert "旅行装备 8%" not in text
