"""Operand coercion for comparisons.

This is the wiring the reference implementation never had (its type system was dead code,
so dates were compared as raw strings). Before a comparison we coerce both operands: two
number-like values compare as numbers, two date-like strings compare as datetimes.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from nlql.errors import NLQLTypeError
from nlql.types.core import TypeTag


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


def _coerce_with_hint(left: Any, right: Any, hint: TypeTag | None) -> tuple[Any, Any]:
    """Coerce a comparison pair, honoring a declared field type when given.

    ``TEXT`` forces a string comparison (so e.g. a zip code ``"02134"`` is never treated as
    the number 2134); ``NUMBER`` / ``DATE`` force that coercion; otherwise fall back to
    value-based inference.
    """
    if hint is TypeTag.TEXT:
        return str(left), str(right)
    if hint is TypeTag.NUMBER:
        ln, rn = as_number(left), as_number(right)
        return (ln, rn) if ln is not None and rn is not None else (left, right)
    if hint is TypeTag.DATE:
        ld, rd = as_date(left), as_date(right)
        return (ld, rd) if ld is not None and rd is not None else (left, right)
    return coerce_operands(left, right)


def compare_values(op: str, left: Any, right: Any, hint: TypeTag | None = None) -> bool:
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
    lhs, rhs = _coerce_with_hint(left, right, hint)
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
