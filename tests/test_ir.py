"""Tests for the Query IR: construction, JSON round-trip, and schema export."""

import pytest

from nlql.errors import NLQLSchemaError
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
    expr_from_dict,
    query_json_schema,
)


def design_example() -> Query:
    """The canonical query from DESIGN.md, built directly as IR."""
    return Query(
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


class TestRoundTrip:
    def test_dict_round_trip_equal(self) -> None:
        q = design_example()
        assert Query.from_dict(q.to_dict()) == q

    def test_json_round_trip_equal(self) -> None:
        q = design_example()
        assert Query.from_json(q.to_json()) == q

    def test_minimal_query(self) -> None:
        q = Query(select=Select("chunk"))
        d = q.to_dict()
        assert d == {"select": {"unit": "chunk"}}
        assert Query.from_dict(d) == q

    def test_literal_types_preserved(self) -> None:
        for value in ["draft", 42, 0.8, True, False, None]:
            lit = Literal(value)
            assert expr_from_dict(lit.to_dict()) == lit

    def test_span_window_serialized(self) -> None:
        assert Select("sentence", window=2).to_dict() == {"unit": "sentence", "window": 2}
        assert Select("sentence").to_dict() == {"unit": "sentence"}


class TestValidation:
    def test_bad_compare_op_rejected(self) -> None:
        with pytest.raises(NLQLSchemaError):
            Compare("~=", Ref("x"), Literal(1))

    def test_bad_select_unit_rejected(self) -> None:
        with pytest.raises(NLQLSchemaError):
            Select("paragraph")

    def test_negative_window_rejected(self) -> None:
        with pytest.raises(NLQLSchemaError):
            Select("sentence", window=-1)

    def test_empty_path_root_rejected(self) -> None:
        with pytest.raises(NLQLSchemaError):
            Path("")

    def test_query_requires_select(self) -> None:
        with pytest.raises(NLQLSchemaError):
            Query.from_dict({"limit": 5})

    def test_unknown_expr_node_rejected(self) -> None:
        with pytest.raises(NLQLSchemaError):
            expr_from_dict({"node": "wormhole"})

    def test_non_object_expr_rejected(self) -> None:
        with pytest.raises(NLQLSchemaError):
            expr_from_dict("relevance")

    def test_not_node_round_trip(self) -> None:
        n = Not(Call("CONTAINS", [Path("content"), Literal("x")]))
        assert expr_from_dict(n.to_dict()) == n


class TestSchema:
    def test_schema_is_wellformed(self) -> None:
        schema = query_json_schema()
        assert schema["title"] == "NLQLQuery"
        assert "select" in schema["properties"]
        assert schema["$defs"]["select"]["properties"]["unit"]["enum"] == [
            "document",
            "chunk",
            "sentence",
        ]

    def test_all_refs_resolve(self) -> None:
        # Every "#/$defs/<x>" reference must point at a defined $def.
        schema = query_json_schema()
        defs = set(schema["$defs"])
        refs: list[str] = []

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "$ref" and isinstance(v, str):
                        refs.append(v)
                    else:
                        walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(schema)
        assert refs, "schema should contain $ref links"
        for ref in refs:
            assert ref.startswith("#/$defs/")
            assert ref.split("/")[-1] in defs, f"dangling ref {ref}"

    def test_function_names_constrained(self) -> None:
        schema = query_json_schema(function_names=["SIMILARITY", "CONTAINS"])
        assert schema["$defs"]["call"]["properties"]["name"]["enum"] == ["CONTAINS", "SIMILARITY"]

    def test_llm_style_ir_document_parses(self) -> None:
        # A hand-authored IR doc (as an LLM would emit) loads into an equivalent Query.
        doc = {
            "select": {"unit": "sentence"},
            "let": [
                {
                    "name": "relevance",
                    "expr": {
                        "node": "call",
                        "name": "SIMILARITY",
                        "args": [
                            {"node": "path", "root": "content", "segments": []},
                            {"node": "literal", "value": "AI agents"},
                        ],
                    },
                }
            ],
            "where": {
                "node": "compare",
                "op": ">=",
                "left": {"node": "ref", "name": "relevance"},
                "right": {"node": "literal", "value": 0.8},
            },
            "limit": 5,
        }
        q = Query.from_dict(doc)
        assert q.select.unit == "sentence"
        assert q.let[0].name == "relevance"
        assert q.limit == 5
