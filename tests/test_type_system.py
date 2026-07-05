"""Tests for type syntax sugar + custom type registration (since v0.3.2)."""

from __future__ import annotations

import nlql
from nlql.embed import FakeEmbedder
from nlql.ir import Literal, query_json_schema
from nlql.lang import parse
from nlql.types import TypeHandler


class TestTypeSyntaxSugar:
    def test_date_sugar_parses(self):
        q = parse('SELECT SENTENCE WHERE meta.date > DATE "2024-01-01" LIMIT 1')
        assert q.where.right.type_hint == "date"
        assert q.where.right.value == "2024-01-01"

    def test_timestamp_sugar_parses(self):
        q = parse('SELECT SENTENCE WHERE meta.ts >= TIMESTAMP "2024-01-01 12:00:00"')
        assert q.where.right.type_hint == "timestamp"

    def test_text_sugar_parses(self):
        q = parse('SELECT SENTENCE WHERE meta.zip == TEXT "02134"')
        assert q.where.right.type_hint == "text"

    def test_date_comparison_without_field_types(self):
        e = nlql.Engine(FakeEmbedder())
        e.add_text("New.", metadata={"date": "2024-06-01"})
        e.add_text("Old.", metadata={"date": "2023-01-01"})
        r = e.search('SELECT SENTENCE WHERE meta.date > DATE "2024-01-01" LIMIT 3')
        assert len(r) == 1
        assert r[0].content == "New."

    def test_text_suppresses_number_coercion(self):
        """TEXT '02134' must not be treated as the number 2134."""
        e = nlql.Engine(FakeEmbedder())
        e.add_text("Boston.", metadata={"zip": "02134"})
        r = e.search('SELECT SENTENCE WHERE meta.zip == TEXT "02134" LIMIT 1')
        assert len(r) == 1


class TestCustomTypeRegistration:
    def test_decorator_class(self):
        @nlql.register_type("SEMVER")
        class Semver:
            def parse(self, s):
                return tuple(int(x) for x in s.split("."))

            def compare(self, a, b, op):
                return a >= b if op == ">=" else a == b if op == "==" else False

        e = nlql.Engine(FakeEmbedder(), field_types={"ver": "SEMVER"})
        e.add_text("v1.", metadata={"ver": "1.2.0"})
        e.add_text("v2.", metadata={"ver": "2.0.0"})
        r = e.search('SELECT SENTENCE WHERE meta.ver >= "1.5.0" LIMIT 3')
        assert len(r) == 1
        assert r[0].content == "v2."

    def test_decorator_function(self):
        @nlql.register_type("UPPER")
        def to_upper(s):
            return s.upper()

        e = nlql.Engine(FakeEmbedder(), field_types={"code": "UPPER"})
        e.add_text("C.", metadata={"code": "abc"})
        r = e.search('SELECT SENTENCE WHERE meta.code == "ABC" LIMIT 1')
        assert len(r) == 1

    def test_custom_type_syntax_sugar(self):
        nlql.register_type("EMAIL", TypeHandler(parse=lambda s: s.lower()))
        e = nlql.Engine(FakeEmbedder())
        e.add_text("Contact.", metadata={"email": "User@Example.COM"})
        r = e.search('SELECT SENTENCE WHERE meta.email == EMAIL "user@example.com" LIMIT 1')
        assert len(r) == 1

    def test_instance_level_not_global(self):
        e = nlql.Engine(FakeEmbedder())

        @e.register_type("INSTANCE_ONLY")
        def pio(s):
            return s.upper()

        e.add_text("I.", metadata={"v": "abc"})
        r = e.search('SELECT SENTENCE WHERE meta.v == INSTANCE_ONLY "ABC" LIMIT 1')
        assert len(r) == 1  # upper("abc") = "ABC" == "ABC"

        # another engine doesn't have INSTANCE_ONLY → fallback string compare "abc" != "ABC"
        e2 = nlql.Engine(FakeEmbedder())
        e2.add_text("I2.", metadata={"v": "abc"})
        r2 = e2.search('SELECT SENTENCE WHERE meta.v == INSTANCE_ONLY "ABC" LIMIT 1')
        assert len(r2) == 0


class TestLiteralTypeHintSerialization:
    def test_to_dict_with_hint(self):
        assert Literal("2024-01-01", type_hint="date").to_dict()["type_hint"] == "date"

    def test_from_dict_with_hint(self):
        assert Literal.from_dict({"node": "literal", "value": "x", "type_hint": "email"}).type_hint == "email"

    def test_from_dict_without_hint(self):
        assert Literal.from_dict({"node": "literal", "value": 42}).type_hint is None

    def test_schema_has_type_hint(self):
        assert "type_hint" in query_json_schema()["$defs"]["literal"]["properties"]
