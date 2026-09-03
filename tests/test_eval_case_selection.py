# -*- coding: utf-8 -*-
"""用例分层选择（--only / --tag）单测

要防的问题：**扩容后的用例集因为太贵而没人跑。**

13 条整轮已经要 25-40 分钟；交接文档主线一要把用例扩到 40+ 条，
一轮就是 80-120 分钟。没有分层的话，结果不是"跑得更全"，
而是"日常根本不跑"，扩容反而让回归失效。

所以约定：日常改代码跑 smoke（8-10 条，10 分钟内），发版前跑 full。
`full` 不需要在每条用例上标注——**所有用例隐含属于 full**，
否则新增用例时漏标一个 tag，它就会永远不被跑到，
而这种"静默不跑"和七期 BM25 忘接线是同一类故障：外观上一切正常。
"""
import pytest

from scripts.eval_regression import select_cases


CASES = [
    {"id": "a", "tags": ["smoke"]},
    {"id": "b", "tags": ["smoke", "pricing"]},
    {"id": "c"},                      # 没标 tags
    {"id": "d", "tags": ["pricing"]},
]


class TestSelectCases:
    def test_no_filter_runs_everything(self):
        assert [c["id"] for c in select_cases(CASES)] == ["a", "b", "c", "d"]

    def test_only_picks_one_case(self):
        assert [c["id"] for c in select_cases(CASES, only="c")] == ["c"]

    def test_tag_filters_to_that_layer(self):
        assert [c["id"] for c in select_cases(CASES, tag="smoke")] == ["a", "b"]
        assert [c["id"] for c in select_cases(CASES, tag="pricing")] == ["b", "d"]

    def test_full_means_everything_including_untagged(self):
        """漏标 tag 的用例必须仍然出现在 full 里，否则它会永远不被跑到。"""
        assert [c["id"] for c in select_cases(CASES, tag="full")] == ["a", "b", "c", "d"]

    def test_only_wins_over_tag(self):
        """--only 是定向验证链路用的，不该被分层挡住。"""
        assert [c["id"] for c in select_cases(CASES, only="d", tag="smoke")] == ["d"]

    def test_unknown_selector_fails_loudly(self):
        """打错标签名不能静默跑 0 条——那会被读成"全过"。

        实测代价：一轮"0 条用例、13/13 空集通过"的报告，和真绿一模一样。
        """
        with pytest.raises(SystemExit, match="没有用例"):
            select_cases(CASES, tag="smok")
        with pytest.raises(SystemExit, match="没有用例"):
            select_cases(CASES, only="zzz")
