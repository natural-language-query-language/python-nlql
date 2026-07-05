"""Tests for declared metadata field types and typed comparison (M4)."""

from __future__ import annotations

from nlql import Document, Engine, FakeEmbedder
from nlql.types import TypeTag
from nlql.types.coerce import compare_values


class TestCompareHint:
    def test_default_coerces_numeric_strings(self) -> None:
        assert compare_values("==", "007", 7) is True  # inference: "007" -> 7.0

    def test_text_hint_suppresses_numeric_coercion(self) -> None:
        assert compare_values("==", "007", 7, TypeTag.TEXT) is False  # compared as strings

    def test_number_hint(self) -> None:
        assert compare_values(">", "42", 10, TypeTag.NUMBER) is True

    def test_date_hint(self) -> None:
        assert compare_values(">", "2024-06-01", "2024-01-01", TypeTag.DATE) is True
        assert compare_values("<", "2024-06-01", "2024-01-01", TypeTag.DATE) is False


class TestEngineFieldTypes:
    def _engine(self, field_types: dict[str, TypeTag] | None = None) -> Engine:
        engine = Engine(FakeEmbedder(), field_types=field_types)
        engine.add_documents(
            [Document.from_text("area code lookup", id="d1", metadata={"code": "007"})]
        )
        return engine

    def test_text_field_prevents_false_numeric_match(self) -> None:
        typed = self._engine({"code": TypeTag.TEXT})
        assert typed.search('SELECT SENTENCE WHERE meta.code == 7') == []  # "007" != 7 as text
        untyped = self._engine()
        assert len(untyped.search('SELECT SENTENCE WHERE meta.code == 7')) == 1  # coerced

    def test_number_field(self) -> None:
        engine = Engine(FakeEmbedder(), field_types={"views": TypeTag.NUMBER})
        engine.add_text("popular post", id="d1", metadata={"views": "1500"})
        assert len(engine.search("SELECT SENTENCE WHERE meta.views > 1000")) == 1

    def test_date_field_ordering(self) -> None:
        engine = Engine(FakeEmbedder(), field_types={"published": TypeTag.DATE})
        engine.add_text("first", id="d1", metadata={"published": "2024-01-15"})
        engine.add_text("second", id="d2", metadata={"published": "2025-03-01"})
        engine.add_text("third", id="d3", metadata={"published": "2023-11-20"})
        results = engine.search("SELECT SENTENCE ORDER BY meta.published DESC")
        assert [u.doc_id for u in results] == ["d2", "d1", "d3"]  # chronological, newest first

    def test_typed_field_not_pushed_down(self) -> None:
        engine = Engine(FakeEmbedder(), field_types={"code": TypeTag.TEXT})
        engine.add_text("x", id="d1", metadata={"code": "007"})
        plan = engine.explain('SELECT SENTENCE WHERE meta.code == "007"')
        # LocalStore pushes metadata, but a declared-typed field stays residual (typed eval).
        assert plan["filter"]["pushed"] is None
        assert plan["filter"]["residual"] is not None
