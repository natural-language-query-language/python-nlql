# Store Interface

## One Interface, Many Backends

`Store` is the unified interface for storage and retrieval. `Engine` does not care where the data lives — the built-in in-memory index, a Faiss index, a Qdrant collection, or a Postgres table — as long as they implement the same set of methods, the engine drives them with the same query code.

```python
from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable
import numpy as np
from nlql.ir.nodes import Expr
from nlql.model import Document, Unit

@runtime_checkable
class Store(Protocol):
    def upsert(self, units: Sequence[Unit]) -> None: ...
    def add_documents(self, documents: Iterable[Document]) -> None: ...
    def get_document(self, doc_id: str) -> Document | None: ...
    def ann_search(
        self,
        vector: np.ndarray,
        k: int | None = None,
        *,
        filter: Expr | None = None,
    ) -> list[tuple[Unit, float]]: ...
    def scan(self, filter: Expr | None = None) -> list[Unit]: ...
    def all_units(self) -> list[Unit]: ...
    def neighbors(self, doc_id: str, ordinal: int, window: int) -> list[Unit]: ...
    def capabilities(self) -> StoreCaps: ...
    def __len__(self) -> int: ...
```

The two kinds of methods each have their own responsibility: `upsert` / `add_documents` handle writes, while `ann_search` / `scan` / `neighbors` handle queries. `ann_search` takes a `filter` parameter — this is IR data (the `WHERE` sub-expression), not a compiled Python predicate, so every backend can translate it into its own native query syntax.

## StoreCaps: Declaring What a Backend Can Do

Backends are not equally capable. Some support approximate nearest neighbors, others only exact search; some can filter metadata inside their own query engine, others cannot; some can handle full-text `CONTAINS` natively, others cannot. `StoreCaps` declares these differences explicitly, letting the Planner decide which conditions to hand off to the backend and which to keep for in-memory post-filtering.

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class StoreCaps:
    name: str = "local"
    vector_search: bool = True
    exact: bool = True               # exact (flat) or approximate (ANN) recall
    metadata_pushdown: bool = False  # whether it can filter metadata in its own query engine
    text_pushdown: bool = False      # whether it can handle CONTAINS natively (e.g. SQL ILIKE)
```

The `StoreCaps` a backend returns must be honest. Declaring `metadata_pushdown=True` is a commitment to translate metadata filtering into the backend's native query inside `ann_search` / `scan` and let the backend apply it. If you cannot deliver, do not declare it — the engine will make up the difference in memory, and the results remain correct.

## Capability Matrix of the Six Backends

| Backend | Vector search | Native metadata filter | Native full-text `CONTAINS` | Install |
|---|---|---|---|---|
| `LocalStore` | exact numpy dot product | yes (numpy mask) | in memory | built-in |
| `FaissStore` | exact Faiss | no (in-memory post-filter) | in memory | `pip install "python-nlql[faiss]"` |
| `HnswStore` | approximate hnswlib | yes (over-fetch + in-memory) | in memory | `pip install "python-nlql[hnsw]"` |
| `QdrantStore` | approximate Qdrant | yes (native Filter) | in memory | `pip install "python-nlql[qdrant]"` |
| `ChromaStore` | approximate Chroma | yes (native `where`) | in memory | `pip install "python-nlql[chroma]"` |
| `PgVectorStore` | Postgres + pgvector | yes (SQL `WHERE`) | yes (`ILIKE`) | `pip install "python-nlql[pgvector]"` |

Across the built-in store, Faiss, HnswLib, Qdrant, and Chroma, the same query returns identical candidates and scores (guaranteed by cross-backend tests). PgVector additionally translates `CONTAINS(content, "x")` into `content ILIKE '%x%'`, completing the full-text match on the database side.

## Writing a Custom Store

Adding a new backend means implementing the `Store` protocol. Shared logic can be inherited from `BaseUnitStore` — it unifies the "over-fetch candidates → in-memory post-filter" fallback path, so you only need to focus on what your backend can do.

```python
from nlql.store.common import BaseUnitStore
from nlql.store.base import StoreCaps

class MyBackendStore(BaseUnitStore):
    def capabilities(self) -> StoreCaps:
        return StoreCaps(
            name="my-backend",
            vector_search=True,
            exact=False,            # approximate ANN
            metadata_pushdown=True, # I translate metadata filtering into the backend query
            text_pushdown=False,
        )

    def ann_search(self, vector, k=None, *, filter=None):
        native_filter = self._translate(filter)     # IR → backend native syntax
        rows = self._backend.query(vector, k or 100, native_filter)
        return [(self._to_unit(r), float(r.score)) for r in rows]
```

Key points:

- `_translate` turns the IR-form `filter` into the backend's native query; this is the only way to hand filtering off to the backend.
- Conditions you cannot handle are left untouched in `filter`; `BaseUnitStore` runs them through the in-memory logic once more on the returned candidates.
- `capabilities()` must match actual behavior. Declaring a capability without implementing it in `ann_search` produces wrong results; doing it without declaring it only loses a performance optimization, while results remain correct.

## Next steps

- How this capability split affects the query plan: see [Hybrid Engine](./hybrid-pushdown.md)
- A runnable cross-backend example: see [Hybrid Backends Tutorial](../tutorials/hybrid-stores.md)
- Details of the Store protocol method signatures: see [Store Reference](../../reference/store.md)
