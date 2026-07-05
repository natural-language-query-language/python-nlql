"""Operand coercion for comparisons.

This is the wiring the reference implementation never had (its type system was dead code,
so dates were compared as raw strings). Before a comparison we coerce both operands: two
number-like values compare as numbers, two date-like strings compare as datetimes.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from nlql.errors import NLQLTypeError
from nlql.types.core import TypeTag, get_type_handler


def as_number(x: Any) -> float | None:
    """Return ``x`` as a float if it is a non-bool number or numeric string, else None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        # Fast reject before the expensive float() attempt: numbers start with a sign,
        # dot, or digit. This keeps per-row coercion cheap for plain string metadata.
        if not x or x[0] not in "+-.0123456789":
            return None
        try:
            return float(x)
        except ValueError:
            return None
    return None


def as_date(x: Any) -> datetime | None:
    """Return ``x`` as a datetime if it is a date/datetime or ISO date string, else None."""
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        return datetime(x.year, x.month, x.day)
    if isinstance(x, str):
        # Fast reject: ISO dates start with a digit. Avoids fromisoformat() + exception
        # on every non-date string (e.g. a "status" field) — the profiled hotspot.
        if not x or not x[0].isdigit():
            return None
        try:
            return datetime.fromisoformat(x)
        except ValueError:
            return None
    return None


def coerce_operands(a: Any, b: Any) -> tuple[Any, Any]:
    """Coerce a comparison pair to a common comparable type where possible."""
    an, bn = as_number(a), as_number(b)
    if an is not None and bn is not None:
        return an, bn
    ad, bd = as_date(a), as_date(b)
    if ad is not None and bd is not None:
        return ad, bd
    return a, b


def _coerce_with_hint(left: Any, right: Any, hint: Any, type_handlers: dict | None = None) -> tuple[Any, Any]:
    """Coerce a comparison pair, honoring a declared field type or literal type hint.

    ``TEXT`` forces string comparison; ``NUMBER`` / ``DATE`` force that coercion;
    ``TIMESTAMP`` uses date logic; custom registered types use their TypeHandler.parse;
    otherwise fall back to value-based inference.
    """
    hu = hint.upper() if isinstance(hint, str) else None
    if hint is TypeTag.TEXT or hu == "TEXT":
        return str(left), str(right)
    if hint is TypeTag.NUMBER or hu == "NUMBER":
        ln, rn = as_number(left), as_number(right)
        return (ln, rn) if ln is not None and rn is not None else (left, right)
    if hint is TypeTag.DATE or hu in ("DATE", "TIMESTAMP"):
        ld, rd = as_date(left), as_date(right)
        return (ld, rd) if ld is not None and rd is not None else (left, right)
    # custom registered type
    handler = get_type_handler(hint, type_handlers) if isinstance(hint, str) else None
    if handler:
        l = handler.parse(left) if isinstance(left, str) else left
        r = handler.parse(right) if isinstance(right, str) else right
        return l, r
    return coerce_operands(left, right)


def compare_values(op: str, left: Any, right: Any, hint: TypeTag | None = None, *, type_handlers: dict | None = None) -> bool:
    """Apply a comparison with NLQL semantics.

    Coerces numbers/dates (or the declared ``hint`` type), treats a null operand as excluded
    from ordered comparisons (SQL-like), and raises :class:`NLQLTypeError` on genuinely
    incomparable operands. Shared by the in-memory evaluator and the store-side metadata
    filter, so every store compares values identically.
    """
    if left is None or right is None:
        if op == "==":
            return left is right
        if op == "!=":
            return left is not right
        return False
    lhs, rhs = _coerce_with_hint(left, right, hint, type_handlers)
    # custom type compare override
    handler = get_type_handler(hint, type_handlers) if isinstance(hint, str) else None
    if handler and handler.compare is not None:
        return handler.compare(lhs, rhs, op)
    if op == "==":
        return bool(lhs == rhs)
    if op == "!=":
        return bool(lhs != rhs)
    try:
        if op == "<":
            return bool(lhs < rhs)
        if op == ">":
            return bool(lhs > rhs)
        if op == "<=":
            return bool(lhs <= rhs)
        return bool(lhs >= rhs)
    except TypeError as e:
        raise NLQLTypeError(f"cannot compare {left!r} and {right!r} with {op!r}") from e
