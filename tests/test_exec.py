"""Tests for the evaluator, planner, and end-to-end executor."""

from __future__ import annotations

import pytest

from nlql.embed import CachedEmbedder, FakeEmbedder
from nlql.errors import NLQLPlanError, NLQLTypeError
from nlql.exec import Evaluator, Executor
from nlql.ingest import IngestionPipeline
from nlql.ir import Call, Compare, Literal, Path, Query, Ref, Select
from nlql.lang import parse
from nlql.model import Document, Payload, Unit
from nlql.plan import Planner, score_key
from nlql.registry import GLOBAL_REGISTRY
from nlql.store import LocalStore

DOCS = [
    Document.from_text(
        "Machine learning models require data. Neural networks learn patterns.",
        id="d1",
        metadata={"status": "published", "year": 2024},
    ),
    Document.from_text(
        "Banana bread needs flour and sugar. Bake for forty minutes.",
        id="d2",
        metadata={"status": "draft", "year": 2023},
    ),
    Document.from_text(
        "Deep learning is machine learning with many layers.",
        id="d3",
        metadata={"status": "published", "year": 2025},
    ),
]


def build_executor(docs=DOCS, granularity="sentence") -> Executor:
    embedder = CachedEmbedder(FakeEmbedder(dim=64))
    registry = GLOBAL_REGISTRY.child()
    pipe = IngestionPipeline(embedder, registry=registry, granularity=granularity)
    store = LocalStore()
    store.upsert(pipe.process(docs))
    store.add_documents(docs)
    return Executor(store, registry, embedder, granularity=granularity)


def _unit(content: str, **meta: object) -> Unit:
    return Unit(id="u", doc_id="d", kind="sentence", payload=Payload.text(content), metadata=dict(meta))


class TestEvaluator:
    def setup_method(self) -> None:
        self.ev = Evaluator(GLOBAL_REGISTRY.child())

    def test_path_content_and_meta(self) -> None:
        u = _unit("hello world", status="published")
        assert self.ev.eval(Path("content"), u, {}) == "hello world"
        assert self.ev.eval(Path("meta", ["status"]), u, {}) == "published"
        assert self.ev.eval(Path("status"), u, {}) == "published"  # bare shorthand
        assert self.ev.eval(Path("meta", ["missing"]), u, {}) is None

    def test_numeric_coercion(self) -> None:
        u = _unit("x", year=2024)
        assert self.ev.eval(Compare(">", Path("meta", ["year"]), Literal(2023)), u, {}) is True
        u2 = _unit("x", year="2024")  # string metadata compares numerically
        assert self.ev.eval(Compare(">", Path("meta", ["year"]), Literal(2023)), u2, {}) is True

    def test_date_coercion(self) -> None:
        u = _unit("x", date="2024-06-01")
        assert self.ev.eval(Compare(">", Path("meta", ["date"]), Literal("2024-01-01")), u, {}) is True
        assert self.ev.eval(Compare("<", Path("meta", ["date"]), Literal("2024-01-01")), u, {}) is False

    def test_null_handling(self) -> None:
        u = _unit("x")
        assert self.ev.eval(Compare(">", Path("meta", ["missing"]), Literal(1)), u, {}) is False
        assert self.ev.eval(Compare("==", Path("meta", ["missing"]), Literal(None)), u, {}) is True

    def test_incomparable_raises(self) -> None:
        u = _unit("x", label="abc")
        with pytest.raises(NLQLTypeError):
            self.ev.eval(Compare("<", Path("meta", ["label"]), Literal(5)), u, {})

    def test_builtin_calls(self) -> None:
        u = _unit("hello world")
        assert self.ev.eval(Call("LENGTH", [Path("content")]), u, {}) == 11
        assert self.ev.eval(Call("CONTAINS", [Path("content"), Literal("WORLD")]), u, {}) is True

    def test_provides_score_read(self) -> None:
        call = Call("SIMILARITY", [Path("content"), Literal("q")])
        u = _unit("x")
        u.scores[score_key(call)] = 0.75
        assert self.ev.eval(call, u, {}) == 0.75

    def test_logical_composition(self) -> None:
        u = _unit("planning and memory")
        expr = parse('SELECT CHUNK WHERE content CONTAINS "planning" AND NOT content CONTAINS "tools"').where
        assert self.ev.eval(expr, u, {}) is True


class TestPlanner:
    def test_dedup_identical_scorers(self) -> None:
        q = parse('SELECT SENTENCE LET a = SIMILARITY(content, "x"), b = SIMILARITY(content, "x") ORDER BY a DESC')
        plan = Planner(GLOBAL_REGISTRY.child()).plan(q, granularity="sentence")
        assert len(plan.scorers) == 1

    def test_distinct_scorers_kept(self) -> None:
        q = parse('SELECT SENTENCE LET a = SIMILARITY(content, "x"), b = SIMILARITY(content, "y") ORDER BY a DESC')
        plan = Planner(GLOBAL_REGISTRY.child()).plan(q, granularity="sentence")
        assert len(plan.scorers) == 2

    def test_unknown_function_rejected(self) -> None:
        q = parse("SELECT CHUNK WHERE NOPE(content) > 1")
        with pytest.raises(NLQLPlanError):
            Planner(GLOBAL_REGISTRY.child()).plan(q, granularity="sentence")

    def test_unknown_alias_rejected(self) -> None:
        q = Query(select=Select("sentence"), where=Compare(">", Ref("ghost"), Literal(1)))
        with pytest.raises(NLQLPlanError):
            Planner(GLOBAL_REGISTRY.child()).plan(q, granularity="sentence")

    def test_non_literal_similarity_query_rejected(self) -> None:
        q = parse("SELECT CHUNK WHERE SIMILARITY(content, content) > 0.5")
        with pytest.raises(NLQLPlanError):
            Planner(GLOBAL_REGISTRY.child()).plan(q, granularity="sentence")


class TestExecutorSemantics:
    def test_semantic_ranking(self) -> None:
        ex = build_executor()
        results = ex.execute(
            parse('SELECT SENTENCE LET rel = SIMILARITY(content, "machine learning") ORDER BY rel DESC LIMIT 2')
        )
        assert len(results) == 2
        # Both top hits are from the ML documents, not the banana one.
        assert all("learning" in r.content.lower() for r in results)
        assert all(r.doc_id in {"d1", "d3"} for r in results)

    def test_named_score_surfaced(self) -> None:
        ex = build_executor()
        results = ex.execute(
            parse('SELECT SENTENCE LET rel = SIMILARITY(content, "neural networks") ORDER BY rel DESC LIMIT 1')
        )
        assert "rel" in results[0].scores
        assert -1.0 <= results[0].scores["rel"] <= 1.0
        # Internal score-key entries are stripped from results.
        assert not any(k.startswith("{") for k in results[0].scores)

    def test_metadata_filter(self) -> None:
        ex = build_executor()
        results = ex.execute(parse('SELECT SENTENCE WHERE meta.status == "published"'))
        assert {r.doc_id for r in results} == {"d1", "d3"}
        assert len(results) == 3  # d1 has 2 sentences, d3 has 1

    def test_combined_semantic_and_metadata(self) -> None:
        ex = build_executor()
        results = ex.execute(
            parse(
                'SELECT SENTENCE LET rel = SIMILARITY(content, "machine learning") '
                'WHERE rel >= -1.0 AND meta.year >= 2024 ORDER BY rel DESC'
            )
        )
        assert all(r.metadata["year"] >= 2024 for r in results)

    def test_span_includes_context(self) -> None:
        ex = build_executor()
        results = ex.execute(
            parse('SELECT SPAN(SENTENCE, window => 1) LET rel = SIMILARITY(content, "neural networks") ORDER BY rel DESC LIMIT 1')
        )
        assert results[0].kind == "span"
        # The best match ("Neural networks…") is expanded with its sibling sentence.
        assert "Neural networks" in results[0].content
        assert "Machine learning models" in results[0].content
        assert results[0].span is not None

    def test_document_granularity(self) -> None:
        ex = build_executor()
        results = ex.execute(
            parse('SELECT DOCUMENT LET rel = SIMILARITY(content, "machine learning") ORDER BY rel DESC')
        )
        assert all(r.kind == "document" for r in results)
        assert len(results) <= 3
        # Top document is one of the ML docs and carries full document text.
        assert results[0].doc_id in {"d1", "d3"}
        assert "." in results[0].content

    def test_cross_granularity_chunk_from_sentence(self) -> None:
        """SELECT CHUNK from a sentence-indexed store now aggregates (used to raise)."""
        ex = build_executor()  # default granularity = "sentence"
        results = ex.execute(parse('SELECT CHUNK WHERE meta.status == "published"'))
        assert len(results) >= 1
        assert all(r.kind == "chunk" for r in results)

    def test_cross_granularity_sentence_from_chunk(self) -> None:
        """SELECT SENTENCE from a chunk-indexed store splits at query time."""
        ex = build_executor(granularity="chunk")
        results = ex.execute(parse("SELECT SENTENCE"))
        assert len(results) >= 1
        assert all(r.kind == "sentence" for r in results)

    def test_cross_granularity_document_any(self) -> None:
        """SELECT DOCUMENT works from both sentence and chunk stores."""
        for g in ("sentence", "chunk"):
            ex = build_executor(granularity=g)
            results = ex.execute(parse("SELECT DOCUMENT"))
            assert len(results) >= 1
            assert all(r.kind == "document" for r in results)

    def test_empty_store(self) -> None:
        ex = build_executor(docs=[])
        assert ex.execute(parse("SELECT SENTENCE")) == []
