# Hybrid Engine

## Decoupling Queries from Backends

An NLQL query is semantically deterministic: which candidates to take, how to filter them, how to rank them. Which storage backend executes it is a separate concern — NLQL keeps these two layers apart. The same query can run on the built-in store, Faiss, Qdrant, Chroma, or PgVector with identical results, differing only in performance.

```python
from nlql import Engine, FakeEmbedder
from nlql.store.qdrant_store import QdrantStore

engine = Engine(FakeEmbedder(dim=64), store=QdrantStore(location=":memory:"))
```

Switching backends requires no change to the query code. `Engine` talks to any concrete store through a single `Store` interface; each backend implements that interface with its own native capability.

## Handed Off to the Backend, or Completed in Memory

A `WHERE` condition in a query has one of two fates:

- **What the backend can handle with its native capability**, the engine translates directly. For example, metadata filtering becomes a Qdrant `Filter` in Qdrant, an SQL `WHERE` in Postgres, and a numpy mask in the built-in store. The backend applies the filter inside its own query engine, so the candidates it returns already satisfy the condition.
- **What the backend cannot express**, the engine filters again in memory on the returned candidates. Typical cases are custom Python function predicates, complex regular expressions, and judgments that need business logic.

This split is decided automatically by the Planner based on the backend's capability declaration, and is invisible to the caller. When the entire query can be handled by the backend, it is a pure delegation with no in-memory stage.

## Inspecting the Split with explain

`engine.explain(query)` returns the query plan, whose `filter` section explicitly lists which conditions are handed off to the backend (`pushed`) and which remain for in-memory post-filtering (`residual`).

```python
from nlql import Engine, FakeEmbedder
from nlql.store import LocalStore
from nlql.store.faiss_store import FaissStore

query = """
    SELECT SENTENCE
    LET rel = SIMILARITY(content, "deep learning networks")
    WHERE meta.status == "published" AND meta.year >= 2024
    ORDER BY rel DESC
    LIMIT 3
"""

for name, store in [("LocalStore", LocalStore()), ("FaissStore", FaissStore())]:
    engine = Engine(FakeEmbedder(dim=64), store=store)
    engine.add_text("Neural networks power deep learning.", metadata={"status": "published", "year": 2025})
    plan = engine.explain(query)
    print(name, "pushed=", plan["filter"]["pushed"] is not None,
          "residual=", plan["filter"]["residual"] is not None)
```

The built-in store compiles metadata filtering into a numpy mask, so `pushed` is non-empty; Faiss itself has no metadata filtering capability, so `meta.status` and `meta.year` both fall into `residual` and are completed in memory. The two backends return the same final results.

!!! note "What the Store receives is data, not a closure"
    The filter handed off to the backend is an IR (the `WHERE` sub-expression of the query), not a compiled Python function. This is what lets each adapter translate it into its own backend's native query syntax — a Qdrant Filter, a Chroma `where` clause, an SQL `WHERE`. This is also why the split can be planned by the Planner before execution.

## Same Results, Different Performance

Cross-backend consistency is a contract. The same query returns identical candidates and scores on the built-in store, Faiss, HnswLib, Qdrant, and Chroma (guaranteed by cross-backend tests). Performance differences come from three sources:

- **Recall method**: the built-in store does exact dot products with numpy; HnswLib uses approximate nearest neighbors (suited to millions of items and above); Qdrant and Chroma use their own ANN implementations.
- **Where filtering happens**: filters handed off to the backend take effect during recall, yielding fewer candidates; in-memory filters must first pull back the over-fetched candidates and then sieve them.
- **Whether over-fetching happens**: when filtering stays in memory, the engine fetches more candidates from the backend (`scan` + in-memory filtering) to ensure no hits are missed.

So "letting the backend do more" is not only faster, it also reduces data transfer at scale. Pushing as much filtering as possible to the backend is one of the main levers for improving query throughput.

## Implementing a Custom Backend

Implement a `Store` and return an honest `StoreCaps` (declaring whether your backend supports vector search, whether it supports metadata filtering natively, whether it supports full-text search natively). The engine then decides which conditions to hand off to you and which to keep. Handle what you can in full; the engine picks up the rest automatically, with no fallback logic required on your side. See [Store Interface](./store-protocol.md).

## Next steps

- To run the cross-backend example hands-on: see [Hybrid Backends Tutorial](../tutorials/hybrid-stores.md)
- How backend capabilities are declared: see [Store Interface](./store-protocol.md)
- How the stages of a query execute: see [Design Overview](./overview.md)
