"""Lightweight type tags and function signatures.

The type system serves three consumers: signature/arity checks at registration,
operand coercion before comparison (so dates compare as dates, numbers as numbers),
and pushdown analysis (external stores need correctly typed filter values).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


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
