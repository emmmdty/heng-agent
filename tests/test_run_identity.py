# -*- coding: utf-8 -*-
"""跑测身份（run identity）单测

要防的问题：**一个读数说不清它是哪套配置跑出来的**。

设计演进记录里已经写了"评测分数与所用模型绑定，换模型必须重跑并在报告里标注"，
但报告本身不记模型——全靠跑的人当时记得。同理不记的还有：提示词版本
（改一句 prompt 分数就会动）、精排是否可用、字面门限取值。
过两周回头看一份报告，只剩一个数字和一堆无法归因的差异。

所以配置要**由被测服务自己报**，评测脚本原样抄进报告，而不是靠人填。
脚本本来就为了拦语义缓存去读一次 /health，顺路把整份配置留下即可，零额外成本。
"""
from app.application.harness.run_identity import describe_run


class TestDescribeRun:
    def test_renders_the_fields_that_explain_a_score(self):
        line = describe_run(
            {
                "model": "mimo-v2.5",
                "prompt_fingerprint": "a1b2c3d4",
                "retrieval": {"reranker": True, "lexical_index": True, "lexical_gate": 4.0},
            },
            judge_model="deepseek-v4-flash",
        )
        for expected in ("mimo-v2.5", "deepseek-v4-flash", "a1b2c3d4", "4.0"):
            assert expected in line, f"报告里必须能看到 {expected}"

    def test_missing_fields_are_marked_unknown_not_dropped(self):
        """老版本服务不报这些字段时要显式写"未知"，不能悄悄少一行——
        少一行会被读成"这项没启用"，比写"未知"更误导。"""
        line = describe_run({}, judge_model="")
        assert "未知" in line

    def test_reranker_off_is_stated_explicitly(self):
        line = describe_run(
            {"model": "m", "retrieval": {"reranker": False, "lexical_index": True, "lexical_gate": 4.0}},
            judge_model="j",
        )
        assert "精排" in line and "关" in line
