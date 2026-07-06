"""JSON Schema export for the Query IR.

This schema is the contract handed to an LLM (as a function-calling / structured-output
definition) so it can emit a valid Query IR directly — far more reliable than asking it
to produce an NLQL *string* that then has to parse. Passing ``function_names`` constrains
``Call.name`` to the functions actually registered, further reducing hallucination.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"


def query_json_schema(function_names: Iterable[str] | None = None) -> dict[str, Any]:
    """Return the JSON Schema (draft 2020-12) describing a Query IR document."""
    call_name: dict[str, Any] = {
        "type": "string",
        "description": "Function name. SIMILARITY(content, \"query\") = semantic relevance; CONTAINS(content, \"keyword\") = literal substring; LENGTH(content) = text length.",
    }
    if function_names is not None:
        call_name = {"type": "string", "enum": sorted(set(function_names))}

    expr_ref = {"$ref": "#/$defs/expr"}
    return {
        "$schema": _SCHEMA_URI,
        "title": "NLQLQuery",
        "description": "A semantic retrieval query. Use 'let' with SIMILARITY(content, \"query\") for relevance scoring, 'where' for filtering (metadata equality, CONTAINS for keyword matching, AND/OR for combining), 'order_by' for ranking, 'limit' to cap results.",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "select": {"$ref": "#/$defs/select"},
            "let": {
                "type": "array",
                "description": "Named scalar/score bindings referenced by WHERE and ORDER BY.",
                "items": {"$ref": "#/$defs/binding"},
            },
            "where": expr_ref,
            "order_by": {"type": "array", "items": {"$ref": "#/$defs/orderKey"}},
            "limit": {
                "type": ["integer", "null"],
                "minimum": 0,
                "description": "Max results. Pick by question breadth: 1-2 for specific facts, 5-10 for broad summaries. null/omit = return all matches ranked by relevance.",
            },
        },
        "required": ["select"],
        "$defs": {
            "select": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "unit": {"enum": ["document", "chunk", "sentence"], "description": "document = whole doc (meta/first page), chunk = passage (default), sentence = specific sentence"},
                    "window": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                        "description": "SPAN context radius; omit for the base unit.",
                    },
                },
                "required": ["unit"],
            },
            "binding": {
                "description": "Named binding: give SIMILARITY a name (e.g. 'rel') to reference in WHERE and ORDER BY.",
                "type": "object",
                "additionalProperties": False,
                "properties": {"name": {"type": "string"}, "expr": expr_ref},
                "required": ["name", "expr"],
            },
            "orderKey": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"expr": expr_ref, "desc": {"type": "boolean"}},
                "required": ["expr"],
            },
            "expr": {
                "oneOf": [
                    {"$ref": "#/$defs/literal"},
                    {"$ref": "#/$defs/path"},
                    {"$ref": "#/$defs/ref"},
                    {"$ref": "#/$defs/call"},
                    {"$ref": "#/$defs/compare"},
                    {"$ref": "#/$defs/and"},
                    {"$ref": "#/$defs/or"},
                    {"$ref": "#/$defs/not"},
                ]
            },
            "literal": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "node": {"const": "literal"},
                    "value": {"type": ["string", "number", "boolean", "null"]},
                    "type_hint": {
                        "type": "string",
                        "enum": ["date", "timestamp", "text", "number", "bool"],
                        "description": "Optional type hint for SQL-style typed literals (since v0.3.2).",
                    },
                },
                "required": ["node", "value"],
            },
            "path": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "node": {"const": "path"},
                    "root": {"type": "string", "description": "e.g. 'content' or 'meta'"},
                    "segments": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["node", "root"],
            },
            "ref": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"node": {"const": "ref"}, "name": {"type": "string"}},
                "required": ["node", "name"],
            },
            "call": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "node": {"const": "call"},
                    "name": call_name,
                    "args": {"type": "array", "items": expr_ref},
                },
                "required": ["node", "name"],
            },
            "compare": {
                "description": "Compare two values: meta.status == \"published\", meta.date >= \"2024-01-01\", ref(\"rel\") >= 0.5.",
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "node": {"const": "compare"},
                    "op": {"enum": ["==", "!=", "<", ">", "<=", ">="]},
                    "left": expr_ref,
                    "right": expr_ref,
                },
                "required": ["node", "op", "left", "right"],
            },
            "and": {
                "description": "All operands must match. Combine: relevance threshold + metadata filter + keyword.",
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "node": {"const": "and"},
                    "operands": {"type": "array", "items": expr_ref, "minItems": 2},
                },
                "required": ["node", "operands"],
            },
            "or": {
                "description": "Any operand can match. Use for alternatives: CONTAINS(content,\"x\") OR CONTAINS(content,\"y\").",
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "node": {"const": "or"},
                    "operands": {"type": "array", "items": expr_ref, "minItems": 2},
                },
                "required": ["node", "operands"],
            },
            "not": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"node": {"const": "not"}, "operand": expr_ref},
                "required": ["node", "operand"],
            },
        },
    }
