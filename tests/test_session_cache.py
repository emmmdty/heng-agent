# -*- coding: utf-8 -*-
"""会话缓存的上限。

排查"按会话累积的判定器有没有人清理"时发现的一串：
`SessionRegistry._agents`（**一个 Agent + 整段对话上下文**）、
`SessionSources`（每会话最多 4000×2 个 float）、`SequencingTracker`、
`OrderProvenanceTracker` —— 四个都提供了 `reset(session_id)`，
但**没有任何地方调用它**。内存随"进程见过多少个不同会话"单调增长，直到重启。

本地看不出来（会话就那么几个），压测也看不出来（loadtest 用的会话数很少），
只有长跑的线上进程会慢慢涨——**而这类涨法没有任何一条告警会响**。
"""
import pytest


class _FakeStore:
    def __init__(self) -> None:
        self.saved: dict[str, str] = {}

    async def save(self, session_id: str, raw: str) -> None:
        self.saved[session_id] = raw

    async def load(self, session_id: str):
        return self.saved.get(session_id)


class _FakeFactory:
    """只造一个可辨认的假 Agent：这里测的是缓存行为，不是 Agent 本身。"""

    def __init__(self) -> None:
        self.built: list[str] = []

    def build(self, restored_state=None):
        self.built.append("build")
        return object()


def _registry(max_sessions=2, on_evict=None):
    from app.application.agents.main_agent import SessionRegistry

    return SessionRegistry(
        _FakeFactory(), _FakeStore(), max_sessions=max_sessions, on_evict=on_evict,
    )


class TestLruEviction:
    async def test_keeps_within_the_cap(self):
        registry = _registry(max_sessions=2)
        for session in ("s1", "s2", "s3"):
            await registry.get_or_create(session)
        assert registry.cached_sessions() == ["s2", "s3"]

    async def test_touching_a_session_makes_it_recent(self):
        """最近用过的不该被挤掉——否则活跃会话会被冷会话顶走，
        每轮都要从存储恢复一次上下文。"""
        registry = _registry(max_sessions=2)
        await registry.get_or_create("s1")
        await registry.get_or_create("s2")
        await registry.get_or_create("s1")   # s1 变成最近使用
        await registry.get_or_create("s3")
        assert registry.cached_sessions() == ["s1", "s3"]

    async def test_evicted_session_can_come_back(self):
        """淘汰不丢对话：AgentState 每轮落盘，下次再来会被恢复。"""
        registry = _registry(max_sessions=1)
        await registry.get_or_create("s1")
        await registry.persist("s1")
        await registry.get_or_create("s2")
        assert registry.cached_sessions() == ["s2"]
        await registry.get_or_create("s1")
        assert "s1" in registry.cached_sessions()

    async def test_zero_means_unbounded(self):
        """0 = 不限，保留旧行为，便于排查时对照。"""
        registry = _registry(max_sessions=0)
        for i in range(5):
            await registry.get_or_create(f"s{i}")
        assert len(registry.cached_sessions()) == 5


class TestEvictionCallback:
    """会话被挤出内存时，它的判定器状态要一起清。

    不让各个判定器各自设上限：那会出现"Agent 还在、它的出处记录已经被挤掉"
    的错配——出处校验会降级成警告（安全），但**判据静默变松了，没人知道**。
    """

    async def test_callback_fires_with_the_evicted_id(self):
        evicted: list[str] = []
        registry = _registry(max_sessions=1, on_evict=evicted.append)
        await registry.get_or_create("s1")
        await registry.get_or_create("s2")
        assert evicted == ["s1"]

    async def test_callback_failure_does_not_break_the_request(self):
        """清理失败不能把正在进行的这一轮对话搞挂——
        淘汰是运维动作，买家不该为它买单。"""
        def boom(_session_id: str) -> None:
            raise RuntimeError("清理失败")

        registry = _registry(max_sessions=1, on_evict=boom)
        await registry.get_or_create("s1")
        await registry.get_or_create("s2")   # 不应抛
        assert registry.cached_sessions() == ["s2"]


class TestTrackersExposeReset:
    """三个按会话累积的判定器都要能被一句话清干净——
    淘汰回调依赖它们，缺一个就等于那一个继续泄漏。"""

    @pytest.mark.parametrize("factory", [
        lambda: __import__("app.application.harness.assertions", fromlist=["x"]).SequencingTracker(),
        lambda: __import__("app.application.harness.number_provenance", fromlist=["x"]).SessionSources(),
        lambda: __import__("app.application.harness.order_provenance", fromlist=["x"]).OrderProvenanceTracker(),
    ])
    def test_reset_is_callable(self, factory):
        tracker = factory()
        tracker.reset("never-seen")   # 未知会话也不能抛


class TestCompositionWiresEviction:
    """接线判据：淘汰回调必须真的清到三处状态。

    写好了不接线，与"故意不做"外观完全一样（踩坑 37 的同一条），
    而这一处的表现是"内存照样涨"——最不容易被发现的那种。
    """

    def test_composition_resets_all_three_trackers(self):
        import inspect

        from app import composition

        source = inspect.getsource(composition)
        for call in (
            "sequencing_tracker.reset(session_id)",
            "order_provenance_tracker.reset(session_id)",
            "orchestrator.forget_session(session_id)",
        ):
            assert call in source, f"淘汰回调没清 {call}"

    def test_registry_gets_the_configured_cap(self):
        import inspect

        from app import composition

        source = inspect.getsource(composition)
        assert "max_sessions=settings.session_cache_max" in source
        assert "on_evict=_forget_session" in source

    def test_orchestrator_forget_clears_number_sources(self):
        from app.application.harness.number_provenance import SessionSources

        sources = SessionSources()
        sources.observe("s1", tool_results=[{"price_major": 89.0}], buyer_texts=[])
        assert sources.of("s1").numbers
        sources.reset("s1")
        assert not sources.of("s1").numbers

    def test_default_cap_is_set(self):
        """默认要有上限——默认不限等于这次修的东西对大多数部署无效。"""
        from app.infrastructure.settings import Settings

        assert Settings.session_cache_max > 0
