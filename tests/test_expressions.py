"""Nested expression evaluation, and SIMILARITY's vector-selector validation."""

from __future__ import annotations

import pytest

from nlql import Engine
from nlql.embed import FakeEmbedder
from nlql.errors import NLQLPlanError
from nlql.lang import parse
from nlql.plan import Planner
from nlql.registry import GLOBAL_REGISTRY
from nlql.types import Signature, TypeTag


def _engine_with_first5() -> Engine:
    engine = Engine(FakeEmbedder())

    @engine.register_function("FIRST5", signature=Signature((TypeTag.TEXT,), TypeTag.TEXT))
    def first5(text: str) -> str:
        return str(text)[:5]

    return engine


def _plan(query: str) -> None:
    Planner(GLOBAL_REGISTRY.child()).plan(parse(query), granularity="sentence")


class TestNestedExpressions:
    def test_nested_custom_function(self) -> None:
        engine = _engine_with_first5()
        engine.add_text("hello world foo bar", id="d1")
        results = engine.search('SELECT SENTENCE WHERE FIRST5(FIRST5(content)) == "hello"')
        assert [u.doc_id for u in results] == ["d1"]

    def test_nested_builtins(self) -> None:
        engine = Engine(FakeEmbedder())
        engine.add_text("HELLO WORLD", id="d1")
        assert len(engine.search("SELECT SENTENCE WHERE LENGTH(LOWER(content)) == 11")) == 1

    def test_deep_nesting_and_composition(self) -> None:
        engine = _engine_with_first5()
        engine.add_text("hello world", id="d1")
        query = 'SELECT SENTENCE WHERE LENGTH(FIRST5(content)) == 5 AND content CONTAINS "world"'
        assert [u.doc_id for u in engine.search(query)] == ["d1"]

    def test_custom_function_in_order_by(self) -> None:
        engine = _engine_with_first5()
        engine.add_text("bbbbb", id="short")
        engine.add_text("aaaaa aaaaa extra", id="long")
        results = engine.search("SELECT SENTENCE ORDER BY LENGTH(content) DESC")
        assert results[0].doc_id == "long"


class TestSimilaritySelector:
    def test_content_accepted(self) -> None:
        _plan('SELECT SENTENCE LET s = SIMILARITY(content, "q")')  # no error

    def test_vec_named_accepted(self) -> None:
        plan = Planner(GLOBAL_REGISTRY.child()).plan(
            parse('SELECT SENTENCE LET s = SIMILARITY(vec.image, "q")'), granularity="sentence"
        )
        assert plan.scorers[0].vector_name == "image"

    def test_transformed_first_arg_rejected(self) -> None:
        engine = _engine_with_first5()
        engine.add_text("x", id="d1")
        with pytest.raises(NLQLPlanError, match="vector selector"):
            engine.search('SELECT SENTENCE LET s = SIMILARITY(FIRST5(content), "q") ORDER BY s DESC')

    def test_metadata_first_arg_rejected(self) -> None:
        with pytest.raises(NLQLPlanError):
            _plan('SELECT SENTENCE LET s = SIMILARITY(meta.title, "q")')

    def test_literal_first_arg_rejected(self) -> None:
        with pytest.raises(NLQLPlanError):
            _plan('SELECT SENTENCE LET s = SIMILARITY("plain", "q")')
