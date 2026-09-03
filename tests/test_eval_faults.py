# -*- coding: utf-8 -*-
"""评测脚本的故障注入支持。

要防的是一个很安静的失效：用例声明了 `faults: [reranker]`，而服务没开注入，
于是这条用例在**精排完全正常**的情况下跑完，然后大概率 PASS——
判据成了绿色装饰，还烧了一轮网关配额。

与 CI 里那条"没数据就当通过"是同一个陷阱：
**一个永远绿的判据比没有判据更坏**，因为它会让人以为这块被覆盖了。
"""
import pytest

from scripts.eval_regression import (
    apply_faults,
    declared_fault_components,
    guard_fault_support,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload=None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=None,  # type: ignore[arg-type]
            )


class _FakeClient:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code
        self.posts: list[tuple[str, dict]] = []

    async def post(self, url, json=None, timeout=None):
        self.posts.append((url, json or {}))
        return _FakeResponse(self.status_code, {"fault_injection": {"active": json["components"]}})


class TestDeclaredComponents:
    def test_collects_across_cases(self):
        cases = [
            {"id": "a", "faults": ["reranker"]},
            {"id": "b"},
            {"id": "c", "faults": ["embedding", "reranker"]},
        ]
        assert declared_fault_components(cases) == {"embedding", "reranker"}

    def test_empty_when_nobody_declares(self):
        assert declared_fault_components([{"id": "a"}]) == set()


class TestPreflightGuard:
    def test_blocks_when_service_has_injection_off(self):
        """开跑前就拦下，而不是让声明了故障的用例安静地跑成绿色。

        整轮 60-90 分钟真金白银，这一次 /health 判定值回票价。
        """
        with pytest.raises(SystemExit) as err:
            guard_fault_support([{"id": "a", "faults": ["reranker"]}], {"fault_injection": False})
        assert "FAULT_INJECTION_ENABLED" in str(err.value)

    def test_passes_when_service_supports_injection(self):
        guard_fault_support(
            [{"id": "a", "faults": ["reranker"]}],
            {"fault_injection": {"enabled": True, "active": []}},
        )

    def test_no_guard_needed_when_no_case_declares_faults(self):
        """没人声明故障时不该因为服务没开注入而拦下整轮——
        绝大多数轮次都不注入，把它做成硬前置等于给所有人加一道无谓的门槛。"""
        guard_fault_support([{"id": "a"}], {"fault_injection": False})


class TestApplyFaults:
    async def test_activates_and_clears(self):
        client = _FakeClient()
        await apply_faults(client, ["reranker"])
        await apply_faults(client, [])
        assert [payload["components"] for _, payload in client.posts] == [["reranker"], []]
        assert all(url.endswith("/debug/faults") for url, _ in client.posts)

    async def test_failure_raises_so_the_case_becomes_error(self):
        """注入失败必须抛，让上层把这条用例判 ERROR。

        吞掉异常继续跑 = 在没有故障的情况下评一条故障用例，
        结论是假的，而且看不出来。
        """
        client = _FakeClient(status_code=404)
        with pytest.raises(Exception):
            await apply_faults(client, ["reranker"])
