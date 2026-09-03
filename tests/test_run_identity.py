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
from app.application.harness.run_identity import code_identity, describe_run


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


class TestCodeIdentity:
    """要防的问题：**服务进程比磁盘上的代码旧，而没有任何东西会报警。**

    九期实测踩到：uvicorn 16:43:06 启动，`tariff_schedule.py` 16:49:20 修完，
    进程再没重启过。之后跑的两条定向回归打的都是这个装着旧代码的服务，
    修复加的 `de_minimis_threshold_major` 一次也没出现在工具返回里——
    交接文档预告的"如果没变说明还有第三条路径"于是被误导向了代码，
    而代码是对的（408 单测全绿，因为单测读的是磁盘上的新代码）。

    单测绿 + 评测拿到旧行为可以同时成立，`/health` 也照样报着一模一样的配置行。
    所以"代码比进程新"必须成为服务自报的一部分，且判据要确定性：
    比 git sha 更可靠的是 **源码 mtime vs 进程启动时刻**——它能抓到未提交的改动，
    而本次正是未提交状态下踩的。
    """

    def test_source_modified_after_start_is_reported_stale(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        started = (tmp_path / "a.py").stat().st_mtime - 60  # 进程比源码早一分钟启动
        identity = code_identity(root=tmp_path, started_at=started)
        assert identity["stale"] is True
        assert "a.py" in " ".join(identity["stale_files"])

    def test_source_older_than_start_is_fresh(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        started = (tmp_path / "a.py").stat().st_mtime + 60  # 源码改完之后才启动
        identity = code_identity(root=tmp_path, started_at=started)
        assert identity["stale"] is False
        assert identity["stale_files"] == []

    def test_pycache_does_not_count_as_a_source_change(self, tmp_path):
        """`__pycache__` 在进程跑起来之后才写入是常态，不能据此判过期。"""
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        started = (tmp_path / "a.py").stat().st_mtime + 60
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "a.cpython-312.pyc").write_text("junk", encoding="utf-8")
        assert code_identity(root=tmp_path, started_at=started)["stale"] is False

    def test_stale_service_is_shouted_in_the_report_line(self):
        """报告开头那一行是"分数变了先看它"的地方，过期必须在这里就刺眼。"""
        line = describe_run(
            {"model": "m", "code": {"stale": True, "started_at": "09-03 16:43:06",
                                    "source_mtime": "09-03 16:49:20", "stale_files": ["app/x.py"]}},
            judge_model="j",
        )
        assert "过期" in line

    def test_fresh_service_states_its_start_time(self):
        line = describe_run(
            {"model": "m", "code": {"stale": False, "started_at": "09-03 16:43:06",
                                    "source_mtime": "09-03 16:40:00", "stale_files": []}},
            judge_model="j",
        )
        assert "16:43:06" in line

    def test_old_server_without_code_field_is_unknown_not_silently_fresh(self):
        """老版本服务不报 code 字段时写"未知"——不能默认当成新鲜的。"""
        line = describe_run({"model": "m"}, judge_model="j")
        assert "代码 未知" in line


class TestHealthWiring:
    """钉住接线本身。

    七期的教训（设计演进记录）：BM25 索引只在 `scripts/eval/*` 里构造、从没接进
    `composition.py`，评测选出的最优配置根本没上线，而"忘了接线"和"故意关掉"
    外观完全一样，没有任何告警。判据做得再对，不接进 /health 就等于没做。

    这里不起 TestClient：`/health` 走 lifespan 构建整个 container（embedding、
    Qdrant、网关），单测跑 8 秒且不碰这些外部依赖，为一条接线断言把它们全拉起来
    不划算。所以钉两件确定性的事：模块确实引用了 code_identity，
    且 /health 的返回体里确实有 `"code": code_identity()`。
    """

    def test_server_imports_code_identity(self):
        from app.presentation import server

        assert server.code_identity is code_identity

    def test_health_payload_includes_code_field(self):
        import inspect

        from app.presentation import server

        source = inspect.getsource(server)
        assert '"code": code_identity()' in source, (
            "/health 必须报出代码新鲜度——不接线的话，一个跑着旧代码的服务"
            "报的配置行和新服务一字不差"
        )


class TestStaleServiceGuard:
    """开跑前拦下"服务跑着旧代码"，别等烧完 25-40 分钟才发现。

    与 `_guard_semantic_cache` 同一类判据：两者都是"评的不是 Agent 真实行为"，
    区别只在一个评的是缓存、一个评的是修复前的代码。
    """

    def _guard(self):
        from scripts.eval_regression import _guard_stale_service

        return _guard_stale_service

    def test_stale_service_refuses_to_run(self):
        import pytest

        with pytest.raises(SystemExit) as err:
            self._guard()(
                {"code": {"stale": True, "started_at": "09-03 16:43:06",
                          "source_mtime": "09-03 16:49:20",
                          "stale_files": ["domain/shipping/tariff_schedule.py"]}},
                allow=False,
            )
        message = str(err.value)
        assert "旧代码" in message
        assert "tariff_schedule.py" in message
        assert "uvicorn" in message, "拦下来必须同时给出重启命令，否则只是挡路"

    def test_fresh_service_passes(self):
        self._guard()({"code": {"stale": False, "stale_files": []}}, allow=False)

    def test_allow_flag_is_the_escape_hatch(self):
        self._guard()(
            {"code": {"stale": True, "stale_files": ["a.py"]}}, allow=True,
        )

    def test_unreachable_health_does_not_block(self):
        """拿不到 /health 时不阻断——后续请求自会报错，这里多拦一道只会误伤。

        老版本服务不报 code 字段时同理：不能把"没这个字段"当成"过期"。
        """
        self._guard()({}, allow=False)
        self._guard()({"model": "m"}, allow=False)
