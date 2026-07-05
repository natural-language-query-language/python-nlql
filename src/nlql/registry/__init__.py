"""Unified capability registry.

Importing this package seeds the built-in functions into :data:`GLOBAL_REGISTRY`.
"""

# Side effect: seed built-in functions (CONTAINS, SIMILARITY, LENGTH, …) into GLOBAL.
from nlql.registry import builtins as _builtins  # noqa: E402,F401
from nlql.registry.core import (
    CAPABILITY_KINDS,
    GLOBAL_REGISTRY,
    Capability,
    Registry,
    register_function,
    register_splitter,
)

__all__ = [
    "Registry",
    "Capability",
    "CAPABILITY_KINDS",
    "GLOBAL_REGISTRY",
    "register_function",
    "register_splitter",
]
