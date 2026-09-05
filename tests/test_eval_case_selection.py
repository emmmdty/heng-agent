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


class TestOnlyAcceptsSeveralIds:
    """`--only` 支持逗号分隔。

    定向验证一处改动往往要看两三条相关用例（十六期验 quote_basket 时就是），
    分几次跑意味着几份报告、几次前置检查，也没法一眼看到对比。
    """

    def _cases(self):
        return [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    def test_single_id_still_works(self):
        from scripts.eval_regression import select_cases

        assert [c["id"] for c in select_cases(self._cases(), only="b")] == ["b"]

    def test_comma_separated_ids(self):
        from scripts.eval_regression import select_cases

        assert [c["id"] for c in select_cases(self._cases(), only="c,a")] == ["a", "c"]

    def test_order_follows_the_file_not_the_argument(self):
        """顺序按用例集来，不按参数——用例之间有顺序依赖（requires），
        让命令行决定执行顺序会把那层保证破坏掉。"""
        from scripts.eval_regression import select_cases

        assert [c["id"] for c in select_cases(self._cases(), only="c,b,a")] == ["a", "b", "c"]

    def test_whitespace_is_tolerated(self):
        from scripts.eval_regression import select_cases

        assert [c["id"] for c in select_cases(self._cases(), only=" a , c ")] == ["a", "c"]

    def test_unknown_id_still_reports_what_is_available(self):
        import pytest

        from scripts.eval_regression import select_cases

        with pytest.raises(SystemExit, match="可用用例"):
            select_cases(self._cases(), only="zzz-nope")

    def test_partially_unknown_ids_run_the_known_ones(self):
        """一个拼错不该让整轮跑不起来：已知的照跑，报告里看得出少了谁。

        （反过来做成硬失败也讲得通，但那会让"跑三条其中一条改了名"
        这种常见情况变成一次白等。）
        """
        from scripts.eval_regression import select_cases

        assert [c["id"] for c in select_cases(self._cases(), only="a,zzz-nope")] == ["a"]


class TestExcludeTag:
    """`--exclude-tag`：主线基线（44 条）要能一条命令复现出来。

    二十三期加进 11 条红队用例之后，用例集变成 55 条，而 `full` 是隐含标签
    （所有用例都属于它），于是**没有任何一个选择器能选出那 44 条主线**——
    交接文档与贡献证明里的 "full 44 条 44/44 PASS / 均分 0.993" 这条基线
    从此不可复现，而二十五期 A/B 的护栏正是拿它当门槛。

    修法刻意是"排除式"而不是给 44 条各标一个 `mainline`：
    逐条标注一旦漏标，那条用例就会静默掉出基线集，
    与本文件开头那条"漏标 tag 永远不被跑到"是同一类故障。
    """

    CASES = [
        {"id": "a", "tags": ["smoke"]},
        {"id": "b", "tags": ["smoke", "redteam"]},
        {"id": "c"},                       # 没标 tags
        {"id": "d", "tags": ["redteam"]},
    ]

    def test_exclude_removes_that_layer(self):
        assert [c["id"] for c in select_cases(self.CASES, exclude_tag="redteam")] == ["a", "c"]

    def test_untagged_cases_survive_exclusion(self):
        """漏标 tags 的用例不该被排除吞掉——它属于主线，不属于被排除的那层。"""
        picked = select_cases(self.CASES, exclude_tag="redteam")
        assert "c" in [c["id"] for c in picked]

    def test_composes_with_tag(self):
        assert [c["id"] for c in select_cases(self.CASES, tag="smoke", exclude_tag="redteam")] == ["a"]

    def test_only_wins_over_exclude(self):
        """--only 是定向验证链路用的，和压过 --tag 同一条理由。"""
        assert [c["id"] for c in select_cases(self.CASES, only="d", exclude_tag="redteam")] == ["d"]

    def test_comma_separated_tags(self):
        assert [c["id"] for c in select_cases(self.CASES, exclude_tag="redteam, smoke")] == ["c"]

    def test_excluding_everything_fails_loudly(self):
        """排空了必须报错：跑 0 条会产出一份和真绿一模一样的"全过"报告。"""
        with pytest.raises(SystemExit, match="没有用例"):
            select_cases([{"id": "a", "tags": ["redteam"]}], exclude_tag="redteam")

    def test_error_message_mentions_the_exclusion(self):
        """报错要说得出是谁把用例排空的，否则只能去猜 --tag 打错了没。"""
        with pytest.raises(SystemExit, match="exclude-tag"):
            select_cases([{"id": "a", "tags": ["redteam"]}], exclude_tag="redteam")


class TestMainlineBaselineIsReproducible:
    """对真实用例集断言不变式：主线 = 全部 − 红队，两边不重不漏。

    刻意**不**断言"恰好 44 条"：主线加用例是正常演进，
    而"这一轮跑的是不是同一份考卷"由 variance 的用例集身份
    （case id 集合的 sha1）保证，不需要在这里钉死一个会过期的数字。
    """

    def _real_cases(self):
        import yaml

        from scripts.eval_regression import PROJECT_ROOT

        return yaml.safe_load((PROJECT_ROOT / "eval" / "cases.yaml").read_text(encoding="utf-8"))["cases"]

    def test_mainline_and_redteam_partition_the_suite(self):
        cases = self._real_cases()
        mainline = {c["id"] for c in select_cases(cases, exclude_tag="redteam")}
        redteam = {c["id"] for c in select_cases(cases, tag="redteam")}
        assert not (mainline & redteam)
        assert mainline | redteam == {c["id"] for c in cases}
