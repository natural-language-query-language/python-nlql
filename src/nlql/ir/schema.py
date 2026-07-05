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
        "description": "Registered function/operator name, e.g. SIMILARITY, CONTAINS, LENGTH.",
    }
    if function_names is not None:
        call_name = {"type": "string", "enum": sorted(set(function_names))}

    expr_ref = {"$ref": "#/$defs/expr"}
    return {
        "$schema": _SCHEMA_URI,
        "title": "NLQLQuery",
        "description": "A semantic retrieval query over an NLQL store.",
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
            "limit": {"type": ["integer", "null"], "minimum": 0},
        },
        "required": ["select"],
        "$defs": {
            "select": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "unit": {"enum": ["document", "chunk", "sentence"]},
                    "window": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                        "description": "SPAN context radius; omit for the base unit.",
                    },
                },
                "required": ["unit"],
            },
            "binding": {
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
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "node": {"const": "and"},
                    "operands": {"type": "array", "items": expr_ref, "minItems": 2},
                },
                "required": ["node", "operands"],
            },
            "or": {
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
