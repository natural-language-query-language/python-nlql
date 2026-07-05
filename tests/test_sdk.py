"""Tests for the SDK: Engine, Query Builder, and LLM integration surface."""

from __future__ import annotations

import nlql
from nlql import Document, Engine, FakeEmbedder
from nlql.sdk.builder import F, Meta, select, similarity
from nlql.types import Signature, TypeTag

CORPUS = [
    Document.from_text("Autonomous agents plan and use tools.", id="a", metadata={"status": "published"}),
    Document.from_text("Banana bread recipe with flour.", id="b", metadata={"status": "draft"}),
]


def seeded_engine() -> Engine:
    eng = Engine(FakeEmbedder())
    eng.add_documents(CORPUS)
    return eng


class TestEngineIngestion:
    def test_add_text_returns_id(self) -> None:
        eng = Engine(FakeEmbedder())
        doc_id = eng.add_text("hello world", metadata={"k": "v"})
        assert isinstance(doc_id, str) and doc_id
        assert len(eng) == 1

    def test_add_text_explicit_id(self) -> None:
        eng = Engine(FakeEmbedder())
        assert eng.add_text("hi", id="doc-7") == "doc-7"

    def test_add_documents_and_len(self) -> None:
        eng = seeded_engine()
        assert len(eng) == 2  # one sentence per doc


class TestEngineQuery:
    def test_search_string(self) -> None:
        eng = seeded_engine()
        results = eng.search('SELECT SENTENCE WHERE meta.status == "published"')
        assert [u.doc_id for u in results] == ["a"]

    def test_search_semantic(self) -> None:
        eng = seeded_engine()
        results = eng.search(
            'SELECT SENTENCE LET rel = SIMILARITY(content, "autonomous agents tools") ORDER BY rel DESC LIMIT 1'
        )
        assert results[0].doc_id == "a"
        assert "rel" in results[0].scores

    def test_search_limit_override(self) -> None:
        eng = seeded_engine()
        assert len(eng.search("SELECT SENTENCE", limit=1)) == 1

    def test_explain(self) -> None:
        eng = seeded_engine()
        plan = eng.explain(
            'SELECT SENTENCE LET rel = SIMILARITY(content, "agents") WHERE rel >= 0.1 ORDER BY rel DESC LIMIT 3'
        )
        assert plan["select"]["unit"] == "sentence"
        assert plan["scores"][0]["query"] == "agents"
        assert plan["limit"] == 3
        assert plan["recall"] == "exact-flat"


class TestQueryBuilder:
    def test_builder_matches_string(self) -> None:
        built = (
            select("sentence")
            .let("rel", similarity("content", "AI agents"))
            .where(F("rel") >= 0.8, Meta("status") != "draft")
            .order_by("rel", desc=True)
            .limit(5)
            .build()
        )
        parsed = nlql.parse(
            'SELECT SENTENCE LET rel = SIMILARITY(content, "AI agents") '
            'WHERE rel >= 0.8 AND meta.status != "draft" ORDER BY rel DESC LIMIT 5'
        )
        assert built == parsed

    def test_builder_executes(self) -> None:
        eng = seeded_engine()
        q = select("sentence").where(Meta("status") == "published").build()
        assert [u.doc_id for u in eng.search(q)] == ["a"]

    def test_boolean_operators(self) -> None:
        from nlql.sdk.builder import content

        expr = (content.contains("agents") | content.contains("tools")).expr
        assert expr.KIND == "or"


class TestThreeEntryEquivalence:
    def test_string_builder_ir_agree(self) -> None:
        eng = seeded_engine()
        text = 'SELECT SENTENCE LET rel = SIMILARITY(content, "autonomous agents") ORDER BY rel DESC LIMIT 1'

        from_string = eng.search(text)
        from_builder = eng.search(
            select("sentence").let("rel", similarity("content", "autonomous agents")).order_by("rel", desc=True).limit(1).build()
        )
        ir_dict = nlql.parse(text).to_dict()
        from_ir = eng.search_ir(ir_dict)

        assert [u.id for u in from_string] == [u.id for u in from_builder] == [u.id for u in from_ir]


class TestLLMIntegration:
    def test_function_schema_constrains_names(self) -> None:
        eng = seeded_engine()
        schema = eng.function_schema()
        enum = schema["$defs"]["call"]["properties"]["name"]["enum"]
        assert "SIMILARITY" in enum and "CONTAINS" in enum

    def test_function_tool_shape(self) -> None:
        eng = seeded_engine()
        tool = eng.function_tool()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "nlql_query"
        assert tool["function"]["parameters"]["title"] == "NLQLQuery"

    def test_search_ir_from_dict(self) -> None:
        eng = seeded_engine()
        ir = {
            "select": {"unit": "sentence"},
            "where": {
                "node": "compare",
                "op": "==",
                "left": {"node": "path", "root": "meta", "segments": ["status"]},
                "right": {"node": "literal", "value": "published"},
            },
        }
        assert [u.doc_id for u in eng.search_ir(ir)] == ["a"]


class TestInstanceExtension:
    def test_register_function_instance_scoped(self) -> None:
        eng = Engine(FakeEmbedder())

        @eng.register_function("DOUBLELEN", signature=Signature((TypeTag.TEXT,), TypeTag.NUMBER))
        def doublelen(text: str) -> int:
            return len(text) * 2

        eng.add_text("hello", id="d1")
        assert len(eng.search("SELECT SENTENCE WHERE DOUBLELEN(content) > 8")) == 1
        # The function is not visible on a fresh global-scoped engine.
        other = Engine(FakeEmbedder())
        assert not other.registry.has("function", "DOUBLELEN")
