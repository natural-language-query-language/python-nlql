"""Tests for named multi-vectors: one record, several vectors, queried via vec.<name>."""

from __future__ import annotations

import pytest

from nlql import Engine
from nlql.embed import FakeEmbedder
from nlql.embed import FakeMultimodalEmbedder
from nlql.errors import NLQLError
from nlql.lang import parse
from nlql.plan import Planner
from nlql.registry import GLOBAL_REGISTRY


def _engine() -> Engine:
    return Engine(
        FakeEmbedder(),
        granularity="chunk",
        named_embedders={"image": FakeMultimodalEmbedder(), "title": FakeEmbedder()},
    )


def _seed(engine: Engine) -> None:
    engine.add_multivector(
        "red_dress",
        content="a comfortable summer outfit",
        named={"image": b"a photo of a red dress", "title": "Red Summer Dress"},
    )
    engine.add_multivector(
        "blue_car",
        content="a fast vehicle for the open road",
        named={"image": b"a photo of a blue sports car", "title": "Blue Sports Car"},
    )


class TestNamedVectorQueries:
    def test_query_image_vector(self) -> None:
        engine = _engine()
        _seed(engine)
        results = engine.search(
            'SELECT CHUNK LET rel = SIMILARITY(vec.image, "red dress") ORDER BY rel DESC LIMIT 1'
        )
        assert results[0].id == "red_dress"

    def test_query_title_vector(self) -> None:
        engine = _engine()
        _seed(engine)
        results = engine.search(
            'SELECT CHUNK LET rel = SIMILARITY(vec.title, "sports car") ORDER BY rel DESC LIMIT 1'
        )
        assert results[0].id == "blue_car"

    def test_query_default_content_vector(self) -> None:
        engine = _engine()
        _seed(engine)
        results = engine.search(
            'SELECT CHUNK LET rel = SIMILARITY(content, "comfortable outfit") ORDER BY rel DESC LIMIT 1'
        )
        assert results[0].id == "red_dress"


class TestModel:
    def test_named_vectors_stored_on_unit(self) -> None:
        engine = _engine()
        engine.add_multivector("p", content="text", named={"image": b"img", "title": "T"})
        unit = engine.store.all_units()[0]
        assert set(unit.vectors) == {"image", "title"}
        assert unit.get_vector("default") is unit.vector
        assert unit.get_vector("content") is unit.vector
        assert unit.get_vector("image") is not None
        assert unit.get_vector("missing") is None

    def test_missing_named_embedder_raises(self) -> None:
        engine = Engine(FakeEmbedder(), granularity="chunk")  # no named embedders
        with pytest.raises(NLQLError):
            engine.add_multivector("x", content="c", named={"image": b"data"})


class TestPlanner:
    def test_extracts_vector_name(self) -> None:
        registry = GLOBAL_REGISTRY.child()
        plan = Planner(registry).plan(
            parse('SELECT CHUNK LET r = SIMILARITY(vec.image, "x") ORDER BY r DESC'),
            granularity="chunk",
        )
        assert plan.scorers[0].vector_name == "image"

    def test_content_is_default_vector(self) -> None:
        registry = GLOBAL_REGISTRY.child()
        plan = Planner(registry).plan(
            parse('SELECT CHUNK LET r = SIMILARITY(content, "x") ORDER BY r DESC'),
            granularity="chunk",
        )
        assert plan.scorers[0].vector_name == "default"
