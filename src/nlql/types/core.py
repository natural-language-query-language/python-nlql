"""Lightweight type tags and function signatures.

The type system serves three consumers: signature/arity checks at registration,
operand coercion before comparison (so dates compare as dates, numbers as numbers),
and pushdown analysis (external stores need correctly typed filter values).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class TypeTag(StrEnum):
    """Coarse value types flowing through query evaluation."""

    ANY = "any"
    TEXT = "text"
    NUMBER = "number"
    BOOL = "bool"
    DATE = "date"
    VECTOR = "vector"
    NULL = "null"


@dataclass(frozen=True, slots=True)
class Signature:
    """Declared argument and return types of a registered function.

    Args:
        args: Expected argument types, in order.
        returns: Result type.
        variadic: When ``True`` the last ``args`` entry may repeat zero or more
            times (so ``arity`` becomes "at least ``len(args) - 1``").
    """

    args: tuple[TypeTag, ...]
    returns: TypeTag
    variadic: bool = False

    def arity_ok(self, n: int) -> bool:
        """Whether ``n`` positional arguments satisfy this signature."""
        if self.variadic:
            return n >= max(0, len(self.args) - 1)
        return n == len(self.args)


# --------------------------------------------------------------------------- #
# Custom type registration (since v0.3.2)                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TypeHandler:
    """Parser + optional comparator for a user-registered type.

    ``parse`` converts a raw string (from ``DATE '...'`` or metadata) into the
    runtime value used in comparisons.  ``compare`` overrides the default
    ``compare_values`` when non-None, receiving ``(left, right, op)``.
    """

    parse: Callable[[str], Any]
    compare: Callable[[Any, Any, str], bool] | None = None


_TYPE_HANDLERS: dict[str, TypeHandler] = {}

_BUILTIN_TYPE_HINTS = frozenset({"date", "timestamp", "text", "number", "bool"})


def register_type(
    name: str, handler: TypeHandler | None = None, *, registry: dict[str, TypeHandler] | None = None
) -> Any:
    """Register a custom type. Two modes:

    1. Direct: ``register_type("EMAIL", TypeHandler(parse=...))``
    2. Decorator: ``@register_type("EMAIL")`` on a class (with parse/compare methods)
       or a bare function (used as parse).
    """
    store = registry if registry is not None else _TYPE_HANDLERS
    if handler is not None:
        store[name.upper()] = handler
        return handler

    def decorator(cls_or_fn: Any) -> Any:
        if isinstance(cls_or_fn, TypeHandler):
            h = cls_or_fn
        elif isinstance(cls_or_fn, type):
            obj = cls_or_fn()
            h = TypeHandler(parse=obj.parse, compare=getattr(obj, "compare", None))
        else:
            h = TypeHandler(parse=cls_or_fn)
        store[name.upper()] = h
        return cls_or_fn

    return decorator


def get_type_handler(
    name: str, type_handlers: dict[str, TypeHandler] | None = None
) -> TypeHandler | None:
    """Look up a registered TypeHandler by name (case-insensitive).
    Checks the optional instance-level dict first, then the global registry."""
    if name is None:
        return None
    key = name.upper() if isinstance(name, str) else str(name).upper()
    if type_handlers and key in type_handlers:
        return type_handlers[key]
    return _TYPE_HANDLERS.get(key)


def is_known_type_hint(name: str) -> bool:
    """Whether a type hint name is built-in or globally registered."""
    if name is None:
        return False
    lower = name.lower() if isinstance(name, str) else str(name).lower()
    return lower in _BUILTIN_TYPE_HINTS or get_type_handler(name) is not None
