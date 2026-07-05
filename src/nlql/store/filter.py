"""In-memory metadata filter interpreter.

Applies the *pushed* subset of a WHERE clause — metadata comparisons composed with
AND/OR/NOT — to a unit's metadata. It is self-contained (depends only on the IR and
value coercion), so the store layer never imports the executor, and it reuses the shared
``compare_values`` so a pushed filter and an in-memory residual compare values identically.
"""

from __future__ import annotations

from typing import Any

from nlql.ir.nodes import And, Compare, Expr, Literal, Not, Or, Path
from nlql.types.coerce import compare_values


def _path_value(path: Path, metadata: dict[str, Any]) -> Any:
    segments = path.segments if path.root == "meta" else [path.root, *path.segments]
    current: Any = metadata
    for seg in segments:
        if isinstance(current, dict):
            current = current.get(seg)
        else:
            return None
    return current


def _operand(node: Expr, metadata: dict[str, Any]) -> Any:
    if isinstance(node, Literal):
        return node.value
    if isinstance(node, Path):
        return _path_value(node, metadata)
    return None  # pushed filters only ever contain literals and metadata paths


def matches_filter(expr: Expr | None, metadata: dict[str, Any]) -> bool:
    """Whether ``metadata`` satisfies a pushed metadata filter (``None`` matches all)."""
    if expr is None:
        return True
    if isinstance(expr, Compare):
        return compare_values(
            expr.op, _operand(expr.left, metadata), _operand(expr.right, metadata)
        )
    if isinstance(expr, And):
        return all(matches_filter(o, metadata) for o in expr.operands)
    if isinstance(expr, Or):
        return any(matches_filter(o, metadata) for o in expr.operands)
    if isinstance(expr, Not):
        return not matches_filter(expr.operand, metadata)
    return False
