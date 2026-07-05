"""Tests for the NLQL/2 string front-end (grammar + parser -> IR)."""

import pytest

from nlql.errors import NLQLParseError
from nlql.ir import (
    And,
    Binding,
    Call,
    Compare,
    Literal,
    Not,
    Or,
    OrderKey,
    Path,
    Query,
    Ref,
    Select,
)
from nlql.lang import parse


class TestSelect:
    def test_plain_unit(self) -> None:
        assert parse("SELECT CHUNK").select == Select("chunk")

    def test_span_with_named_window(self) -> None:
        assert parse("SELECT SPAN(SENTENCE, window => 2)").select == Select("sentence", window=2)

    def test_span_with_positional_window(self) -> None:
        assert parse("SELECT SPAN(SENTENCE, 3)").select == Select("sentence", window=3)

    def test_span_default_window(self) -> None:
        assert parse("SELECT SPAN(SENTENCE)").select == Select("sentence", window=1)


class TestRefResolution:
    def test_let_alias_becomes_ref(self) -> None:
        q = parse('SELECT SENTENCE LET rel = SIMILARITY(content, "x") WHERE rel >= 0.5')
        assert q.where == Compare(">=", Ref("rel"), Literal(0.5))

    def test_content_stays_path(self) -> None:
        q = parse('SELECT SENTENCE WHERE content CONTAINS "x"')
        assert q.where == Call("CONTAINS", [Path("content"), Literal("x")])

    def test_meta_dotted_path(self) -> None:
        q = parse('SELECT CHUNK WHERE meta.status != "draft"')
        assert q.where == Compare("!=", Path("meta", ["status"]), Literal("draft"))

    def test_non_alias_bare_ident_is_path(self) -> None:
        # No LET binding named 'title' -> it is a path, not a ref.
        q = parse('SELECT CHUNK WHERE title CONTAINS "x"')
        assert isinstance(q.where, Call)
        assert q.where.args[0] == Path("title")


class TestExpressions:
    def test_infix_predicates_become_calls(self) -> None:
        assert parse('SELECT CHUNK WHERE content MATCH "a.*b"').where == Call(
            "MATCH", [Path("content"), Literal("a.*b")]
        )
        assert parse('SELECT CHUNK WHERE content LIKE "draft%"').where == Call(
            "LIKE", [Path("content"), Literal("draft%")]
        )

    def test_function_call_needs_no_grammar_change(self) -> None:
        # LENGTH is not a grammar keyword — it flows through NAME(args).
        q = parse("SELECT CHUNK WHERE LENGTH(content) > 100")
        assert q.where == Compare(">", Call("LENGTH", [Path("content")]), Literal(100))

    def test_and_or_flattened(self) -> None:
        q = parse('SELECT CHUNK WHERE content CONTAINS "a" AND content CONTAINS "b" AND content CONTAINS "c"')
        assert isinstance(q.where, And)
        assert len(q.where.operands) == 3

    def test_or_precedence_below_and(self) -> None:
        # a AND b OR c  ==  (a AND b) OR c
        q = parse('SELECT CHUNK WHERE content CONTAINS "a" AND content CONTAINS "b" OR content CONTAINS "c"')
        assert isinstance(q.where, Or)
        assert isinstance(q.where.operands[0], And)

    def test_not_and_parens(self) -> None:
        q = parse('SELECT CHUNK WHERE NOT (content CONTAINS "x")')
        assert isinstance(q.where, Not)
        assert isinstance(q.where.operand, Call)

    def test_negative_and_float_numbers(self) -> None:
        q = parse("SELECT CHUNK LET s = SIMILARITY(content, \"x\") WHERE s >= -0.25")
        assert q.where == Compare(">=", Ref("s"), Literal(-0.25))

    def test_boolean_and_null_literals(self) -> None:
        assert parse("SELECT CHUNK WHERE meta.flag == true").where.right == Literal(True)
        assert parse("SELECT CHUNK WHERE meta.x == null").where.right == Literal(None)

    def test_single_and_double_quotes(self) -> None:
        assert parse("SELECT CHUNK WHERE content CONTAINS 'x'").where.args[1] == Literal("x")


class TestClauses:
    def test_order_by_directions(self) -> None:
        q = parse("SELECT CHUNK LET s = SIMILARITY(content, \"x\") ORDER BY s DESC, meta.date ASC")
        assert q.order_by == [OrderKey(Ref("s"), desc=True), OrderKey(Path("meta", ["date"]), desc=False)]

    def test_order_by_default_asc(self) -> None:
        q = parse("SELECT CHUNK ORDER BY meta.date")
        assert q.order_by == [OrderKey(Path("meta", ["date"]), desc=False)]

    def test_limit(self) -> None:
        assert parse("SELECT CHUNK LIMIT 7").limit == 7

    def test_full_query_equivalence(self) -> None:
        # The DESIGN.md example: string -> IR equals the hand-built IR exactly.
        q = parse(
            """
            SELECT SPAN(SENTENCE, window => 1)
            LET   relevance = SIMILARITY(content, "AI Agent architecture"),
                  novelty   = SIMILARITY(content, "novel unpublished ideas")
            WHERE relevance >= 0.8
              AND meta.status != "draft"
              AND (content CONTAINS "planning" OR content CONTAINS "memory")
            ORDER BY relevance DESC, novelty DESC
            LIMIT 5
            """
        )
        expected = Query(
            select=Select("sentence", window=1),
            let=[
                Binding("relevance", Call("SIMILARITY", [Path("content"), Literal("AI Agent architecture")])),
                Binding("novelty", Call("SIMILARITY", [Path("content"), Literal("novel unpublished ideas")])),
            ],
            where=And(
                [
                    Compare(">=", Ref("relevance"), Literal(0.8)),
                    Compare("!=", Path("meta", ["status"]), Literal("draft")),
                    Or(
                        [
                            Call("CONTAINS", [Path("content"), Literal("planning")]),
                            Call("CONTAINS", [Path("content"), Literal("memory")]),
                        ]
                    ),
                ]
            ),
            order_by=[OrderKey(Ref("relevance"), desc=True), OrderKey(Ref("novelty"), desc=True)],
            limit=5,
        )
        assert q == expected


class TestErrors:
    def test_missing_select_raises(self) -> None:
        with pytest.raises(NLQLParseError):
            parse('WHERE content CONTAINS "x"')

    def test_garbage_raises(self) -> None:
        with pytest.raises(NLQLParseError):
            parse("SELECT CHUNK WHERE )(")

    def test_bad_select_unit_raises(self) -> None:
        with pytest.raises(NLQLParseError):
            parse("SELECT PARAGRAPH")

    def test_parse_error_has_location(self) -> None:
        try:
            parse("SELECT CHUNK WHERE @@")
        except NLQLParseError as e:
            assert e.line is not None
        else:
            pytest.fail("expected NLQLParseError")
