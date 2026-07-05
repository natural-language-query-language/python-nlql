"""Filter pushdown analysis.

Splits a WHERE expression into a **pushed** part (translatable to a store's native
filter) and a **residual** part (evaluated in-memory on the returned candidates). The
split is conservative and correctness-preserving:

- Only conjuncts of the top-level AND are considered independently; an OR / NOT sub-tree
  is atomic (pushed whole or not at all), because you cannot split a disjunction across
  "store" and "memory" without changing its meaning.
- A conjunct is pushable only if the whole sub-tree is metadata comparisons composed with
  AND/OR/NOT. Anything touching ``content``, a LET score (``Ref``), or a function call
  (``SIMILARITY``/``CONTAINS``/custom) stays residual.

For a store that declares ``metadata_pushdown=False`` (e.g. a pure vector index like
Faiss), nothing is pushed and the whole filter is residual — still correct, just less
selective at the store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nlql.ir.nodes import And, Call, Compare, Expr, Literal, Not, Or, Path

if TYPE_CHECKING:
    # Only a type reference — the runtime code just duck-types ``caps.*_pushdown``.
    # Importing StoreCaps here at runtime would create a store→plan→store import cycle.
    from nlql.store.base import StoreCaps

# Comparison operators flip when the literal sits on the left of the metadata field.
_FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}


@dataclass(frozen=True, slots=True)
class FilterSplit:
    """A WHERE clause partitioned into store-native and in-memory parts."""

    pushed: Expr | None
    residual: Expr | None


def is_metadata_path(expr: Expr) -> bool:
    """A path addresses metadata iff its root is not the special ``content`` field."""
    return isinstance(expr, Path) and expr.root != "content"


def metadata_field(path: Path) -> str:
    """The dotted metadata key for a path (drops the optional ``meta`` prefix)."""
    segments = path.segments if path.root == "meta" else [path.root, *path.segments]
    return ".".join(segments)


def _metadata_literal_pair(compare: Compare) -> tuple[Path, Literal] | None:
    left, right = compare.left, compare.right
    if isinstance(left, Path) and left.root != "content" and isinstance(right, Literal):
        return left, right
    if isinstance(right, Path) and right.root != "content" and isinstance(left, Literal):
        return right, left
    return None


def normalized_compare(compare: Compare) -> tuple[str, str, Any]:
    """Return ``(field, op, value)`` with the metadata field on the left of ``op``.

    Only valid for a metadata comparison (see :func:`is_pushable`).
    """
    pair = _metadata_literal_pair(compare)
    if pair is None:
        raise ValueError("not a metadata comparison")
    path, literal = pair
    op = compare.op if compare.left is path else _FLIP[compare.op]
    return metadata_field(path), op, literal.value


def _is_pushable_subtree(expr: Expr, caps: StoreCaps, field_types: dict[str, Any]) -> bool:
    if isinstance(expr, Compare):
        if not caps.metadata_pushdown:
            return False
        pair = _metadata_literal_pair(expr)
        if pair is None:
            return False
        path, literal = pair
        if metadata_field(path) in field_types:
            return False  # a declared-typed field is evaluated in-memory with its type hint
        if literal.value is None:
            return False  # null comparisons have SQL-specific semantics — keep in-memory
        if literal.type_hint is not None:
            return False  # typed literal (DATE '...', EMAIL '...') → in-memory for type-aware comparison
        if expr.op in ("==", "!="):
            return True
        # Ordered comparisons push only for numeric literals — stores can range numbers,
        # but not arbitrary strings. This keeps native translation total for what we push.
        return isinstance(literal.value, (int, float)) and not isinstance(literal.value, bool)
    if isinstance(expr, Call):
        # CONTAINS(content, "x") pushes to a text-capable store (SQL ILIKE = substring).
        return caps.text_pushdown and is_content_contains(expr)
    if isinstance(expr, (And, Or)):
        return all(_is_pushable_subtree(o, caps, field_types) for o in expr.operands)
    if isinstance(expr, Not):
        return _is_pushable_subtree(expr.operand, caps, field_types)
    return False


def is_content_contains(expr: Expr) -> bool:
    """Whether ``expr`` is ``CONTAINS(content, "literal")`` — a pushable text predicate."""
    return (
        isinstance(expr, Call)
        and expr.name.upper() == "CONTAINS"
        and len(expr.args) == 2
        and isinstance(expr.args[0], Path)
        and expr.args[0].root == "content"
        and not expr.args[0].segments
        and isinstance(expr.args[1], Literal)
        and isinstance(expr.args[1].value, str)
    )


def is_pushable(
    expr: Expr, caps: StoreCaps, field_types: dict[str, Any] | None = None
) -> bool:
    """Whether ``expr`` can be pushed to a store with the given capabilities."""
    return _is_pushable_subtree(expr, caps, field_types or {})


def _combine(exprs: list[Expr]) -> Expr | None:
    if not exprs:
        return None
    if len(exprs) == 1:
        return exprs[0]
    return And(list(exprs))


def split_filter(
    where: Expr | None, caps: StoreCaps, field_types: dict[str, Any] | None = None
) -> FilterSplit:
    """Partition a WHERE clause into (pushed, residual) given store capabilities."""
    if where is None:
        return FilterSplit(None, None)
    types = field_types or {}
    conjuncts = list(where.operands) if isinstance(where, And) else [where]
    pushed: list[Expr] = []
    residual: list[Expr] = []
    for conjunct in conjuncts:
        (pushed if is_pushable(conjunct, caps, types) else residual).append(conjunct)
    return FilterSplit(_combine(pushed), _combine(residual))
