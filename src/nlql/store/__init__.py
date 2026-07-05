"""Store protocol and the built-in LocalStore (numpy flat index)."""

from nlql.store.base import Store, StoreCaps
from nlql.store.filter import matches_filter
from nlql.store.local import LocalStore

__all__ = ["Store", "StoreCaps", "LocalStore", "matches_filter"]
