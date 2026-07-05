# Design

## A query, not a pile of code

NLQL organizes semantic retrieval into a single declarative query: relevance scoring, filtering, and sorting all live in one statement, instead of being scattered across business logic.

```sql
SELECT SENTENCE
LET   rel = SIMILARITY(content, "AI Agent")
WHERE rel >= 0.8 AND meta.status == "published"
ORDER BY rel DESC
LIMIT 5
```

The structure mirrors SQL: `SELECT` sets the return granularity, `LET` computes relevance, `WHERE` filters, `ORDER BY` and `LIMIT` sort and cap.

## Three ways to write the same query

NLQL supports three forms:

- **NLQL statement**: most readable, good for static queries
- **Python chained builder**: `select(...).let(...).where(...)`, good for programmatic assembly
- **JSON IR**: the structured form, a natural carrier for LLM tool-calling

All three compile to the same IR, so results are identical. This is also why NLQL fits AI agents: the model returns the query itself when calling a tool, with no extra protocol.

## How a query runs

`engine.search(query)` goes through:

1. **Recall**: fetch a batch of candidates by `SIMILARITY` from the index
2. **Filter**: apply `WHERE`
3. **Sort & limit**: `ORDER BY`, then `LIMIT`
4. **Rerank** (optional): if a reranker is configured, precise re-scoring

Vectors are computed at write time and stored in the index — queries never recompute them, which keeps latency low.

## Backends

NLQL is not tied to a specific database:

- The built-in store (pure Python, good for small-to-medium data) by default
- Swappable to Qdrant, Chroma, Faiss, HnswLib, Postgres + pgvector

```python
from nlql.store.qdrant_store import QdrantStore
engine = Engine(embedder, store=QdrantStore(location=":memory:"))
```

Every backend implements the same `Store` interface. The engine hands filtering off to the backend's native capability where possible and completes the rest in memory; results stay consistent across backends, only performance differs. See [Hybrid backends](../tutorials/hybrid-stores.md).

## Multimodal

The data model is uniform across text and images; both map to the same vector space. You can retrieve images with a text query, using the same query syntax as for text.

```python
mm = Engine(MultimodalEmbedder(), granularity="chunk")
mm.add_image(image_bytes, metadata={"kind": "photo"})
mm.search('SELECT CHUNK LET rel = SIMILARITY(content, "a fluffy cat") ORDER BY rel DESC')
```

## Summary

- Retrieval intent in one query, not scattered across code
- Three forms compile to the same IR, identical results
- Backends are swappable, query code unchanged
- Text and images, uniform

For lower-level details (IR node structure, evaluation model, type system), see `DESIGN.md` in the repository root.
