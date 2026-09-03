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
import ast
from pathlib import Path

from scripts.eval_regression import build_ground_truth

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

    def test_includes_native_de_minimis_kinds(self):
        """免税额度要给**原生口径**，否则 judge 核不了"美国免税门槛 $800"这句话。

        十一期：额度本来就是各国用自己货币定义的（US 800 USD、EU 150 EUR），
        只给 CNY 折算值时，Agent 的跨币种表述 judge 只能"拿不准按不通过"，
        而那个数其实是对的。
        """
        text = build_ground_truth()
        assert "800 USD" in text, "US 免税额度的原生口径"
        assert "150 EUR" in text, "EU 免税额度的原生口径"

    def test_states_taxable_base_formula(self):
        """光说"超出部分"不够：judge 要能拿公式独立算出应税基数，
        才判得了「1,199 × 12%」这种基数错误。"""
        text = build_ground_truth()
        assert "应税基数" in text
        assert "整单金额 × 费率" in text, "必须明确点出错误写法，judge 才好照着判"

    def test_does_not_import_private_de_minimis_constant(self):
        """事实基准不该依赖领域层私有常量——十期就是那么写的，
        规则表一改存储结构（十一期改成原生口径）这里立刻 ImportError。

        判据按 AST 看**导入了什么名字**，不按文本 grep：注释里提一句常量名
        （说明为什么不再用它）是正常的，grep 会把它误报成违规。
        """
        source = (PROJECT_ROOT / "scripts" / "eval_regression.py").read_text(encoding="utf-8")
        imported = {
            alias.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not [name for name in imported if name.startswith("_DE_MINIMIS")]
