"""Focused tests for the Query Builder DSL (operator overloading + constructors)."""

from __future__ import annotations

import pytest

from nlql.ir import And, Call, Compare, Not, Or, Path, Ref
from nlql.sdk.builder import (
    F,
    Meta,
    contains,
    content,
    field,
    length,
    select,
    similarity,
)


class TestConstructors:
    def test_content_and_fields(self) -> None:
        assert content.expr == Path("content")
        assert field("meta.status").expr == Path("meta", ["status"])
        assert field("title").expr == Path("title")
        assert Meta("status").expr == Path("meta", ["status"])
        assert F("rel").expr == Ref("rel")

    def test_similarity_from_str_and_e(self) -> None:
        assert similarity("content", "q").expr == Call("SIMILARITY", [Path("content"), _lit("q")])
        assert similarity(content, "q").expr == Call("SIMILARITY", [Path("content"), _lit("q")])

    def test_contains_and_length(self) -> None:
        assert contains("content", "x").expr == Call("CONTAINS", [Path("content"), _lit("x")])
        assert length("content").expr == Call("LENGTH", [Path("content")])
        assert length(content).expr == Call("LENGTH", [Path("content")])


class TestOperators:
    def test_comparisons(self) -> None:
        assert (F("r") >= 1).expr == Compare(">=", Ref("r"), _lit(1))
        assert (F("r") > 1).expr == Compare(">", Ref("r"), _lit(1))
        assert (F("r") <= 1).expr == Compare("<=", Ref("r"), _lit(1))
        assert (F("r") < 1).expr == Compare("<", Ref("r"), _lit(1))
        assert (F("r") == 1).expr == Compare("==", Ref("r"), _lit(1))
        assert (F("r") != 1).expr == Compare("!=", Ref("r"), _lit(1))

    def test_boolean_composition_flattens(self) -> None:
        expr = (content.contains("a") & content.contains("b") & content.contains("c")).expr
        assert isinstance(expr, And) and len(expr.operands) == 3
        or_expr = (content.contains("a") | content.contains("b") | content.contains("c")).expr
        assert isinstance(or_expr, Or) and len(or_expr.operands) == 3

    def test_not(self) -> None:
        assert (~content.contains("x")).expr == Not(Call("CONTAINS", [Path("content"), _lit("x")]))

    def test_predicate_helpers(self) -> None:
        assert content.matches("a.*").expr.name == "MATCH"
        assert content.like("draft%").expr.name == "LIKE"

    def test_e_is_unhashable(self) -> None:
        with pytest.raises(TypeError):
            {F("r")}  # __hash__ is None because == builds IR


class TestBuild:
    def test_no_where_is_none(self) -> None:
        assert select("chunk").build().where is None

    def test_single_where(self) -> None:
        q = select("chunk").where(Meta("s") == "x").build()
        assert isinstance(q.where, Compare)

    def test_multiple_where_becomes_and(self) -> None:
        q = select("chunk").where(Meta("a") == 1, Meta("b") == 2).build()
        assert isinstance(q.where, And) and len(q.where.operands) == 2

    def test_order_by_alias_and_expr(self) -> None:
        q = select("sentence").order_by("rel", desc=True).order_by(Meta("date")).build()
        assert q.order_by[0].expr == Ref("rel") and q.order_by[0].desc is True
        assert q.order_by[1].expr == Path("meta", ["date"]) and q.order_by[1].desc is False


def _lit(v: object):
    from nlql.ir import Literal

    return Literal(v)
