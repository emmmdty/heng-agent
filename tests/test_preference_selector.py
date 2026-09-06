# -*- coding: utf-8 -*-
"""PreferenceSelector 单测。

最重要的一条不是"相关性排得准不准"，而是 **dislike 永远不会被截断**——
「不要塑料材质」与「推荐个咖啡杯」的向量相似度很低，纯 top_k 会把它丢掉，
于是推出一只塑料杯。漏黑名单是安全问题，不是相关性问题。
"""
from app.application.memory.preference_selector import PreferenceSelector
from app.domain.buyer.preference import BuyerPreference
from app.domain.catalog.ports.retrieval_ports import EmbeddingClient

_TERMS = ("咖啡", "杯", "旅行", "箱", "塑料", "设计", "露营")


class AxisEmbeddingClient(EmbeddingClient):
    """确定性桩：按特征词命中构造向量，相似度可预测。"""

    async def embed(self, text: str) -> list[float]:
        return [1.0 if term in text else 0.0 for term in _TERMS]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class BrokenEmbeddingClient(EmbeddingClient):
    async def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding 服务不可用")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding 服务不可用")


class ShortVectorEmbeddingClient(EmbeddingClient):
    """返回条数与入参不匹配的坏实现，用于验证防御分支。"""

    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0]]


def _pref(kind: str, statement: str, created_at: str) -> BuyerPreference:
    return BuyerPreference(
        buyer_id="b1", kind=kind, statement=statement, created_at=created_at,
    )


def _statements(preferences) -> list[str]:
    return [p.statement for p in preferences]


class TestDislikeSafetyFloor:
    async def test_dislikes_never_truncated_by_top_k(self):
        """核心用例：4 条 dislike + top_k=1，dislike 一条都不能少。"""
        preferences = [
            _pref("dislike", "不要塑料材质", "2026-01-01"),
            _pref("dislike", "不要动物皮革", "2026-01-02"),
            _pref("dislike", "不要含镍配件", "2026-01-03"),
            _pref("dislike", "不要一次性包装", "2026-01-04"),
            _pref("like", "喜欢小众设计", "2026-01-05"),
        ]
        selector = PreferenceSelector(AxisEmbeddingClient(), relevance_enabled=True)

        selected = await selector.select(preferences, query="推荐个咖啡杯", top_k=1)
        assert len([p for p in selected if p.kind == "dislike"]) == 4

    async def test_irrelevant_dislike_survives_relevance_ranking(self):
        """与 query 毫不相关的 dislike 也必须保留。"""
        preferences = [
            _pref("dislike", "不要塑料材质", "2026-01-01"),
            _pref("like", "喜欢咖啡杯", "2026-01-02"),
            _pref("like", "喜欢旅行箱", "2026-01-03"),
        ]
        selector = PreferenceSelector(AxisEmbeddingClient(), relevance_enabled=True)

        selected = await selector.select(preferences, query="露营装备", top_k=1)
        assert "不要塑料材质" in _statements(selected)

    async def test_top_k_zero_keeps_only_dislikes(self):
        preferences = [
            _pref("dislike", "不要塑料材质", "2026-01-01"),
            _pref("like", "喜欢小众设计", "2026-01-02"),
        ]
        selector = PreferenceSelector()

        selected = await selector.select(preferences, query="随便看看", top_k=0)
        assert _statements(selected) == ["不要塑料材质"]

    async def test_dislikes_come_first(self):
        preferences = [
            _pref("like", "喜欢小众设计", "2026-01-01"),
            _pref("dislike", "不要塑料材质", "2026-01-02"),
        ]
        selector = PreferenceSelector()

        selected = await selector.select(preferences, query="q", top_k=5)
        assert selected[0].kind == "dislike"


class TestRelevanceRanking:
    async def test_picks_semantically_closest_likes(self):
        preferences = [
            _pref("like", "喜欢露营风格", "2026-01-01"),
            _pref("like", "喜欢咖啡杯", "2026-01-02"),
            _pref("like", "喜欢旅行箱", "2026-01-03"),
        ]
        selector = PreferenceSelector(AxisEmbeddingClient(), relevance_enabled=True)

        selected = await selector.select(preferences, query="想买个咖啡杯", top_k=1)
        assert _statements(selected) == ["喜欢咖啡杯"]

    async def test_no_truncation_when_likes_within_top_k(self):
        """条数没超上限时不该白调 embedding，直接全给。"""
        preferences = [
            _pref("like", "喜欢小众设计", "2026-01-01"),
            _pref("like", "喜欢露营风格", "2026-01-02"),
        ]
        selector = PreferenceSelector(BrokenEmbeddingClient(), relevance_enabled=True)

        selected = await selector.select(preferences, query="q", top_k=5)
        assert len(selected) == 2, "未超上限就不该走 embedding，坏 embedder 也不影响"

    async def test_output_preserves_original_relative_order(self):
        """注入块行序必须稳定，否则 orchestrator 的『变了才重注入』会每轮误判。"""
        preferences = [
            _pref("like", "喜欢咖啡杯", "2026-01-01"),
            _pref("like", "喜欢旅行箱", "2026-01-02"),
            _pref("like", "喜欢露营风格", "2026-01-03"),
        ]
        selector = PreferenceSelector(AxisEmbeddingClient(), relevance_enabled=True)

        selected = await selector.select(preferences, query="旅行箱和咖啡杯", top_k=2)
        assert _statements(selected) == ["喜欢咖啡杯", "喜欢旅行箱"]


class TestFallbacks:
    async def test_disabled_uses_latest_first(self):
        preferences = [
            _pref("like", "旧偏好", "2026-01-01"),
            _pref("like", "新偏好", "2026-06-01"),
        ]
        selector = PreferenceSelector(AxisEmbeddingClient(), relevance_enabled=False)

        selected = await selector.select(preferences, query="q", top_k=1)
        assert _statements(selected) == ["新偏好"]

    async def test_no_embedder_falls_back_even_if_enabled(self):
        preferences = [
            _pref("like", "旧偏好", "2026-01-01"),
            _pref("like", "新偏好", "2026-06-01"),
        ]
        selector = PreferenceSelector(embedder=None, relevance_enabled=True)

        selected = await selector.select(preferences, query="q", top_k=1)
        assert _statements(selected) == ["新偏好"]

    async def test_embedding_failure_degrades_not_raises(self):
        """选偏好失败不能把整轮对话搞挂。"""
        preferences = [
            _pref("like", "旧偏好", "2026-01-01"),
            _pref("like", "新偏好", "2026-06-01"),
        ]
        selector = PreferenceSelector(BrokenEmbeddingClient(), relevance_enabled=True)

        selected = await selector.select(preferences, query="q", top_k=1)
        assert _statements(selected) == ["新偏好"]

    async def test_vector_count_mismatch_degrades(self):
        preferences = [
            _pref("like", "旧偏好", "2026-01-01"),
            _pref("like", "新偏好", "2026-06-01"),
        ]
        selector = PreferenceSelector(ShortVectorEmbeddingClient(), relevance_enabled=True)

        selected = await selector.select(preferences, query="q", top_k=1)
        assert _statements(selected) == ["新偏好"]


class TestEdgeCases:
    async def test_empty_input(self):
        assert await PreferenceSelector().select([], query="q", top_k=5) == []

    async def test_all_dislikes_with_zero_likes(self):
        preferences = [_pref("dislike", "不要塑料材质", "2026-01-01")]
        selected = await PreferenceSelector().select(preferences, query="q", top_k=5)
        assert _statements(selected) == ["不要塑料材质"]


class TestInjectionSwitch:
    """记忆注入 kill switch（#12 回放对照，M0-b）：臂 A 注入关、臂 B 开。

    select() 是主 Agent（orchestrator）与检索子 Agent（task_dispatch_tool）
    两个注入面的公共漏斗，在这里关 = 一处开关覆盖全部注入；两处调用方对
    空选中集本来就都不注入，不需要改调用侧。默认开 = 现行为零变化。
    """

    _PREFERENCES = [
        _pref("dislike", "不要塑料材质", "2026-01-01"),
        _pref("like", "喜欢小众设计", "2026-01-02"),
    ]

    async def test_disabled_selector_selects_nothing_even_dislikes(self):
        """注入关时连 dislike 也不注入——黑名单底线约束的是『注入了但被截断』，
        不是『注入通道整体关闭』；关注入 = 记忆层不参与本轮，语义是全量。"""
        selector = PreferenceSelector(injection_enabled=False)
        assert await selector.select(self._PREFERENCES, query="推荐个咖啡杯", top_k=5) == []

    async def test_env_zero_disables(self, monkeypatch):
        monkeypatch.setenv("PREFERENCE_INJECTION_ENABLED", "0")
        selector = PreferenceSelector()
        assert await selector.select(self._PREFERENCES, query="q", top_k=5) == []

    async def test_env_unset_defaults_on(self, monkeypatch):
        monkeypatch.delenv("PREFERENCE_INJECTION_ENABLED", raising=False)
        selector = PreferenceSelector()
        selected = await selector.select(self._PREFERENCES, query="q", top_k=5)
        assert _statements(selected) == ["不要塑料材质", "喜欢小众设计"]

    async def test_env_one_enables(self, monkeypatch):
        monkeypatch.setenv("PREFERENCE_INJECTION_ENABLED", "1")
        selector = PreferenceSelector()
        selected = await selector.select(self._PREFERENCES, query="q", top_k=5)
        assert _statements(selected) == ["不要塑料材质", "喜欢小众设计"]

    async def test_env_truthy_enables_like_settings_convention(self, monkeypatch):
        """取值约定对齐 settings.py（not in ("0","false","False") 即开）：
        操作者按全仓惯例写 =true 不会静默跑成 A/A。"""
        monkeypatch.setenv("PREFERENCE_INJECTION_ENABLED", "true")
        selector = PreferenceSelector()
        selected = await selector.select(self._PREFERENCES, query="q", top_k=5)
        assert _statements(selected) == ["不要塑料材质", "喜欢小众设计"]

    async def test_env_false_variants_disable(self, monkeypatch):
        for value in ("false", "False"):
            monkeypatch.setenv("PREFERENCE_INJECTION_ENABLED", value)
            selector = PreferenceSelector()
            assert await selector.select(self._PREFERENCES, query="q", top_k=5) == []

    async def test_env_cannot_override_explicit_param(self, monkeypatch):
        """构造参数显式关 + env 开 → 仍关：显式禁用优先，env 只管默认态。"""
        monkeypatch.setenv("PREFERENCE_INJECTION_ENABLED", "1")
        selector = PreferenceSelector(injection_enabled=False)
        assert await selector.select(self._PREFERENCES, query="q", top_k=5) == []

    async def test_disabled_does_not_call_embedder(self):
        """关注入连相关性排序都不该跑——零额外调用。"""
        selector = PreferenceSelector(BrokenEmbeddingClient(), relevance_enabled=True, injection_enabled=False)
        assert await selector.select(self._PREFERENCES, query="q", top_k=5) == []


class TestCorruptInjectionMode:
    """污染注入（阳性对照专用 fault injection，M2 对照轮设计第二版）。

    第一版"矛盾注入"（store 里 seed 取反偏好）被 Agent 的 forget 工具自愈：
    M2 对照轮实测，模型看到注入的『喜欢塑料材质』+ 买家说『记住不要塑料』，
    主动撤回了 seed 再写真偏好——判定时刻两臂记忆状态已无差异，对照失效。
    第二版在**注入层**污染：store 保持真实（沉淀提取不受影响），每轮注入
    都被替换为错误偏好，Agent 无法用 forget 自愈（store 里没有那条）。
    """

    _PREFERENCES = [
        _pref("dislike", "不要塑料材质", "2026-01-01"),
        _pref("like", "喜欢小众设计", "2026-01-02"),
    ]

    async def test_corrupt_mode_replaces_all_preferences_with_wrong_one(self, monkeypatch):
        monkeypatch.setenv("PREFERENCE_INJECTION_CORRUPT", "1")
        selector = PreferenceSelector()
        selected = await selector.select(self._PREFERENCES, query="推荐个咖啡杯", top_k=5)
        assert [(p.kind, p.statement) for p in selected] == [
            ("like", "买家偏好塑料材质的商品，推荐时优先塑料材质"),
        ]

    async def test_corrupt_mode_overrides_dislike_safety_floor_and_top_k(self, monkeypatch):
        """污染模式是有意的 fault injection：无视 top_k 与 dislike 安全底线，
        恒返回单条合成错误偏好——底线约束的是正常注入路径。"""
        monkeypatch.setenv("PREFERENCE_INJECTION_CORRUPT", "1")
        selector = PreferenceSelector()
        selected = await selector.select(
            [_pref("dislike", "不要塑料材质", "2026-01-01")], query="q", top_k=0,
        )
        assert [(p.kind, p.statement) for p in selected] == [
            ("like", "买家偏好塑料材质的商品，推荐时优先塑料材质"),
        ]

    async def test_corrupt_mode_defaults_off(self, monkeypatch):
        monkeypatch.delenv("PREFERENCE_INJECTION_CORRUPT", raising=False)
        selector = PreferenceSelector()
        selected = await selector.select(self._PREFERENCES, query="q", top_k=5)
        assert [p.statement for p in selected] == ["不要塑料材质", "喜欢小众设计"]

    async def test_corrupt_mode_returns_single_synthetic_entry(self, monkeypatch):
        """注入的是**合成的单条错误偏好**，不是对 store 内容的改写——
        store 保持真实是沉淀对账的前提。"""
        monkeypatch.setenv("PREFERENCE_INJECTION_CORRUPT", "1")
        selector = PreferenceSelector()
        selected = await selector.select(self._PREFERENCES, query="q", top_k=5)
        assert all(p.buyer_id == "" for p in selected)
        assert len(selected) == 1
