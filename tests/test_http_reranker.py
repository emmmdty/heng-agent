# -*- coding: utf-8 -*-
"""HttpReranker 的协议解析。

为什么值得单测：精排失败的表现是**静默降级**——`catalog_search` 捕获异常、
按一阶段召回分排序、把 `rerank_applied` 标成 false，然后正常返回。
所以协议对不上时不会有任何人报警，只有指标会悄悄掉下去一截
（105 条标注实测：精排贡献 Recall +2.4pt / MRR +6.2pt）。

服务端是自建的（`scripts/serve_reranker.py`），协议随时可能被改动，
而"改坏了"和"没配"在日志里长得一模一样。
"""
import pytest

from app.domain.catalog.ports.retrieval_ports import Reranker
from app.infrastructure.rerank.http_reranker import HttpReranker


class _Settings:
    def __init__(self, url="http://r", model="m") -> None:
        self.reranker_base_url = url
        self.reranker_model = model


class _Response:
    def __init__(self, body, status=200) -> None:
        self._body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


class _Client:
    def __init__(self, response) -> None:
        self._response = response
        self.posted: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        self.posted = {"url": url, "json": json}
        return self._response


def _patched(monkeypatch, response):
    client = _Client(response)
    monkeypatch.setattr(
        "app.infrastructure.rerank.http_reranker.httpx.AsyncClient",
        lambda **kwargs: client,
    )
    return client


class TestProtocol:
    async def test_parses_the_standard_shape(self, monkeypatch):
        """{results:[{index, relevance_score}]}——Jina / TEI / vLLM 通用形态。"""
        _patched(monkeypatch, _Response({"results": [
            {"index": 0, "relevance_score": 0.2},
            {"index": 1, "relevance_score": 0.9},
        ]}))
        scores = await HttpReranker(_Settings()).rerank("q", ["a", "b"])
        assert scores == [0.2, 0.9]

    async def test_scores_follow_index_not_position(self, monkeypatch):
        """结果乱序返回时必须按 index 归位。

        按位置读会把分数配错文档——而排序结果看上去仍然"像那么回事"，
        这是最难被发现的一类错。
        """
        _patched(monkeypatch, _Response({"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ]}))
        assert await HttpReranker(_Settings()).rerank("q", ["a", "b"]) == [0.1, 0.9]

    async def test_accepts_score_as_an_alias(self, monkeypatch):
        _patched(monkeypatch, _Response({"results": [{"index": 0, "score": 0.5}]}))
        assert await HttpReranker(_Settings()).rerank("q", ["a"]) == [0.5]

    async def test_length_mismatch_raises_rather_than_guessing(self, monkeypatch):
        """条数对不上时抛异常，由调用方降级——**不能补零**。

        补零等于把"精排没跑"伪装成"精排给了 0 分"，
        而后者会真的改变排序结果，且 rerank_applied 还标着 true。
        """
        _patched(monkeypatch, _Response({"results": [{"index": 0, "relevance_score": 0.5}]}))
        with pytest.raises(RuntimeError):
            await HttpReranker(_Settings()).rerank("q", ["a", "b"])

    async def test_non_list_body_raises(self, monkeypatch):
        _patched(monkeypatch, _Response({"error": "boom"}))
        with pytest.raises(RuntimeError):
            await HttpReranker(_Settings()).rerank("q", ["a"])

    async def test_empty_documents_short_circuits(self, monkeypatch):
        """没有候选就不该发请求：空请求在有的服务端上会 400，
        把一次"本来就没事"变成一次降级。"""
        client = _patched(monkeypatch, _Response({"results": []}))
        assert await HttpReranker(_Settings()).rerank("q", []) == []
        assert client.posted == {}

    async def test_trailing_slash_is_trimmed(self, monkeypatch):
        client = _patched(monkeypatch, _Response({"results": [{"index": 0, "score": 1.0}]}))
        await HttpReranker(_Settings(url="http://r/")).rerank("q", ["a"])
        assert client.posted["url"] == "http://r/rerank"

    def test_is_a_reranker_port(self):
        """端口实现要能被 catalog_search 当 Reranker 用。"""
        assert isinstance(HttpReranker(_Settings()), Reranker)
