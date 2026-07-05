"""Tests for the pluggable reranker (second-stage retrieval)."""

from __future__ import annotations

from collections.abc import Sequence

from nlql import Document, Engine
from nlql.embed import FakeEmbedder
from nlql.rerank import FakeReranker
from nlql.model import Payload, Unit


def _units(*contents: str) -> list[Unit]:
    return [
        Unit(id=str(i), doc_id=str(i), kind="sentence", payload=Payload.text(c))
        for i, c in enumerate(contents)
    ]


class TestFakeReranker:
    def test_reorders_by_query_overlap(self) -> None:
        units = _units("nothing relevant here", "machine learning models", "learning")
        out = FakeReranker().rerank("machine learning", units)
        assert out[0].content == "machine learning models"  # most query tokens
        assert all("rerank" in u.scores for u in out)
        assert out[0].scores["rerank"] >= out[-1].scores["rerank"]


class _Spy:
    """Records how many candidates it saw and reverses them."""

    def __init__(self) -> None:
        self.seen = 0
        self.query = ""

    def rerank(self, query: str, units: Sequence[Unit]) -> list[Unit]:
        self.seen = len(units)
        self.query = query
        reversed_units = list(reversed(units))
        for rank, unit in enumerate(reversed_units):
            unit.scores["rerank"] = float(len(reversed_units) - rank)
        return reversed_units


def _corpus(n: int = 15) -> list[Document]:
    return [Document.from_text(f"agent memory tool item number {i}", id=f"d{i}") for i in range(n)]


class TestEngineIntegration:
    def test_reranker_over_fetches_and_reorders(self) -> None:
        spy = _Spy()
        engine = Engine(FakeEmbedder(), reranker=spy, rerank_factor=5)
        engine.add_documents(_corpus(15))
        results = engine.search(
            'SELECT SENTENCE LET rel = SIMILARITY(content, "agent memory") ORDER BY rel DESC LIMIT 2'
        )
        assert spy.seen >= 10  # recall over-fetched (limit 2 × factor 5) before reranking
        assert len(results) == 2  # final limit applied after rerank
        assert all("rerank" in u.scores for u in results)

    def test_per_query_reranker_overrides_none(self) -> None:
        spy = _Spy()
        engine = Engine(FakeEmbedder())  # no engine-level reranker
        engine.add_documents(_corpus(8))
        engine.search(
            'SELECT SENTENCE LET rel = SIMILARITY(content, "agent") LIMIT 2', reranker=spy
        )
        assert spy.seen > 0  # the per-call reranker ran

    def test_rerank_query_override(self) -> None:
        spy = _Spy()
        engine = Engine(FakeEmbedder(), reranker=spy)
        engine.add_documents(_corpus(5))
        engine.search(
            'SELECT SENTENCE LET rel = SIMILARITY(content, "original") LIMIT 2',
            rerank_query="a different rerank question",
        )
        assert spy.query == "a different rerank question"

    def test_explain_shows_reranker(self) -> None:
        engine = Engine(FakeEmbedder(), reranker=FakeReranker())
        engine.add_text("agent memory", id="d1")
        plan = engine.explain('SELECT SENTENCE LET rel = SIMILARITY(content, "agent") LIMIT 1')
        assert plan["rerank"] == "FakeReranker"

    def test_no_reranker_leaves_order(self) -> None:
        engine = Engine(FakeEmbedder())
        engine.add_documents(_corpus(5))
        results = engine.search(
            'SELECT SENTENCE LET rel = SIMILARITY(content, "agent") ORDER BY rel DESC LIMIT 3'
        )
        assert all("rerank" not in u.scores for u in results)  # no rerank score attached


class TestLengthDilutionFix:
    def test_reranker_rescues_full_overlap_passage(self) -> None:
        # The bi-encoder dilutes a long passage that contains every query term; the reranker
        # (token overlap, like a cross-encoder) promotes it back to the top.
        engine = Engine(FakeEmbedder(), reranker=FakeReranker(), rerank_factor=10)
        engine.add_documents(
            [
                Document.from_text("agent memory planning tool retrieval vector index query model system", id="full"),
                Document.from_text("banana bread recipe", id="noise1"),
                Document.from_text("weather forecast today", id="noise2"),
                Document.from_text("agent only", id="partial"),
            ]
        )
        results = engine.search(
            'SELECT SENTENCE LET rel = SIMILARITY(content, "agent memory planning tool") ORDER BY rel DESC LIMIT 1'
        )
        assert results[0].doc_id == "full"  # all four query terms present → reranked to top
