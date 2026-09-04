# -*- coding: utf-8 -*-
"""评测证据的持久性：向量库可以挪走，**流水不能跟着走**。

要防的问题：**报告活着，它引用的证据死了。**

Qdrant 本地嵌入模式是单进程文件锁，想在不打扰已有服务的前提下起第二个实例，
就得换一份 Qdrant 存储。此前唯一的办法是整个 `DATA_DIR` 换掉——
而流水（`DATA_DIR/conversations/`）也跟着换走了。

实测代价（2026-09-04）：某一轮把 `DATA_DIR` 指到会话级临时目录，
当轮一切正常、`make check` 也绿；会话结束后目录被清理，
留在仓库 `eval/` 里的报告记着一个**已经不存在的 data_dir**，
于是那一批读数（无出处金额率、算式自洽、bad case 采集）**一条都无法复算**。
下一个接手的人看到的是一句"流水目录里一份都找不到"，而代码全对。

所以两者必须解耦：
    DATA_DIR          流水、会话、偏好——**证据**，必须留在仓库里
    VECTOR_STORE_DIR  Qdrant 落盘——**可重建的缓存**，随便挪

顺带一道拦截：`DATA_DIR` 落在系统临时目录下时，评测开跑前直接拒绝——
同 `code.stale`，这类错误的唯一症状出现在几十分钟之后，且指向完全错误的方向。
"""
import inspect
from pathlib import Path

import pytest

from app.infrastructure.persistence.json_file_stores import JsonFileConversationStore
from app.infrastructure.settings import load_settings
from app.infrastructure.vector.qdrant_product_index import QdrantProductIndex
from scripts.eval_regression import _guard_ephemeral_data_dir


def _settings(data_dir: Path, vector_store: Path | None = None):
    from app.infrastructure.settings import Settings

    return Settings(
        llm_base_url="", llm_api_key="", llm_model="", port=8000, log_level="info",
        embedding_base_url="", embedding_api_key="", embedding_model="", embedding_dim=8,
        qdrant_url="", qdrant_collection="test_products",
        reranker_base_url="", reranker_model="", tavily_api_key="",
        otlp_endpoint="", data_dir=data_dir,
        category_kb_collection="test_category_kb",
        context_size=128000, tool_result_limit=20000, reply_token_budget=0,
        tool_failure_threshold=3, tool_circuit_reset_seconds=60.0,
        cors_origins=["http://localhost:5173"],
        vector_store_dir_override=vector_store,
    )


class TestVectorStoreDirResolution:
    def test_unset_override_follows_data_dir(self, tmp_path):
        """不配就跟随 data_dir——单实例的默认行为一字不变。"""
        assert _settings(tmp_path).vector_store_dir == tmp_path

    def test_override_moves_the_vector_store_only(self, tmp_path):
        """配了之后向量库走新路，**data_dir 原地不动**（流水靠它）。"""
        settings = _settings(tmp_path / "repo", tmp_path / "scratch")
        assert settings.vector_store_dir == tmp_path / "scratch"
        assert settings.data_dir == tmp_path / "repo"

    def test_env_var_is_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "placeholder")
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "repo"))
        monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "scratch"))
        settings = load_settings()
        assert settings.data_dir == tmp_path / "repo"
        assert settings.vector_store_dir == tmp_path / "scratch"

    def test_env_var_absent_keeps_them_together(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "placeholder")
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "repo"))
        monkeypatch.delenv("VECTOR_STORE_DIR", raising=False)
        settings = load_settings()
        assert settings.vector_store_dir == settings.data_dir


class TestLocalQdrantLandsInVectorStoreDir:
    async def test_index_files_follow_the_override_not_data_dir(self, tmp_path):
        """真起一个本地模式的索引，看文件落在哪。

        这是本文件里唯一能证明"锁真的被挪走了"的判据——
        断言配置字段相等只证明读对了值，证明不了 Qdrant 换了地方开锁。
        """
        data_dir = tmp_path / "repo"
        scratch = tmp_path / "scratch"
        data_dir.mkdir()
        index = QdrantProductIndex(_settings(data_dir, scratch))
        try:
            assert (scratch / "qdrant").exists(), "向量库应落在 VECTOR_STORE_DIR 下"
            assert not (data_dir / "qdrant").exists(), "不该再占用 DATA_DIR 的 Qdrant 路径"
        finally:
            await index.close()


class TestTracesStayWithDataDir:
    def test_conversation_store_is_wired_to_data_dir(self):
        """接线断言：三个 JSON 落盘仓储必须吃 data_dir。

        如果哪天有人图省事把它们也改成 vector_store_dir，
        外观是"评测照常跑完"，而证据会在下一个会话里消失——
        与本文件开头那次实测是同一种失败。
        """
        from app import composition

        source = inspect.getsource(composition)
        for store in ("JsonFilePreferenceStore", "JsonFileSessionStore", "JsonFileConversationStore"):
            assert f"{store}(settings.data_dir)" in source, f"{store} 必须挂在 data_dir 上"

    def test_traces_land_under_data_dir(self, tmp_path):
        settings = _settings(tmp_path / "repo", tmp_path / "scratch")
        store = JsonFileConversationStore(settings.data_dir)
        assert Path(store._dir) == tmp_path / "repo" / "conversations"


class TestEphemeralDataDirGuard:
    def test_refuses_when_traces_land_in_system_temp(self, tmp_path):
        health = {"data_dir": "/tmp/claude-1000/abc/scratchpad/data"}
        with pytest.raises(SystemExit, match="临时目录"):
            _guard_ephemeral_data_dir(health, allow=False)

    def test_error_names_the_offending_path_and_the_way_out(self):
        health = {"data_dir": "/tmp/session-xyz/data"}
        with pytest.raises(SystemExit) as excinfo:
            _guard_ephemeral_data_dir(health, allow=False)
        message = str(excinfo.value)
        assert "/tmp/session-xyz/data" in message
        assert "VECTOR_STORE_DIR" in message, "要告诉人正确的做法，不能只说不行"

    def test_repo_data_dir_passes(self):
        """注意不能拿 pytest 的 tmp_path 当"正常路径"——它自己就在 /tmp 下。"""
        from app.infrastructure.settings import PROJECT_ROOT

        _guard_ephemeral_data_dir({"data_dir": str(PROJECT_ROOT / "data")}, allow=False)

    def test_escape_hatch(self):
        _guard_ephemeral_data_dir({"data_dir": "/tmp/x/data"}, allow=True)

    def test_report_without_data_dir_is_not_blocked(self):
        """十五期之前的 /health 不报 data_dir——拿不到就别拦，别把老服务锁死。"""
        _guard_ephemeral_data_dir({}, allow=False)
