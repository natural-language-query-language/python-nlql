# Hybrid Backends

NLQL decouples queries from backends: the built-in store works out of the box, and you can also connect dedicated vector databases such as Qdrant or Faiss. The same query returns identical results across backends.

## Consistency

All backends conform to the same `Store` interface. The engine preferentially uses the backend's native capabilities for operations like filtering (faster) and completes any parts the backend cannot handle in memory. Regardless of which path is taken, the final results are identical; only performance differs.

## Example

```python
from nlql import Document, Engine, FakeEmbedder
from nlql.store import LocalStore

CORPUS = [
    Document.from_text("Machine learning models learn patterns.",
                       id="d1", metadata={"status": "published", "year": 2024}),
    Document.from_text("Neural networks power deep learning.",
                       id="d2", metadata={"status": "published", "year": 2025}),
    Document.from_text("Banana bread needs flour and sugar.",
                       id="d3", metadata={"status": "draft", "year": 2024}),
    Document.from_text("Reinforcement learning trains agents.",
                       id="d4", metadata={"status": "published", "year": 2020}),
]

QUERY = """
    SELECT SENTENCE
    LET rel = SIMILARITY(content, "deep learning networks")
    WHERE meta.status == "published" AND meta.year >= 2024
    ORDER BY rel DESC
    LIMIT 3
"""


def backends():
    stores = {"LocalStore": LocalStore()}
    try:
        from nlql.store.faiss_store import FaissStore
        stores["FaissStore"] = FaissStore()
    except Exception:
        print("(faiss not installed, skipping)")
    try:
        from nlql.store.qdrant_store import QdrantStore
        stores["QdrantStore"] = QdrantStore()
    except Exception:
        print("(qdrant not installed, skipping)")
    return stores


for name, store in backends().items():
    engine = Engine(FakeEmbedder(dim=64), store=store)
    engine.add_documents(CORPUS)
    hits = [(u.doc_id, round(u.scores["rel"], 3)) for u in engine.search(QUERY)]
    print(f"{name:12} → {hits}")
```

The `hits` from each backend are identical (`d2`, `d1`, with the same scores).

## Backend capabilities

| Backend | Vector search | Metadata filtering | Full-text search | Install |
|---|---|---|---|---|
| `LocalStore` | Built-in (exact) | Built-in | In-memory | Built-in |
| `FaissStore` | Faiss (exact) | In-memory | In-memory | `nlql[faiss]` |
| `HnswStore` | HnswLib (approximate, suited for large datasets) | Over-fetch + in-memory | In-memory | `nlql[hnsw]` |
| `QdrantStore` | Qdrant | Native | In-memory | `nlql[qdrant]` |
| `ChromaStore` | Chroma | Native | In-memory | `nlql[chroma]` |
| `PgVectorStore` | Postgres + pgvector | Native | Native (ILIKE) | `nlql[pgvector]` |

!!! note "Default backend"
    When no `store` is passed, `LocalStore` is used — pure Python, suitable for datasets in the tens of thousands to one hundred thousand. For larger datasets, switch to `HnswStore` or `FaissStore`.

!!! tip "Switching backends"
    ```python
    from nlql.store.qdrant_store import QdrantStore
    engine = nlql.Engine(embedder, store=QdrantStore(location=":memory:"))
    ```
    No changes are needed to ingestion or query code.

## Next steps

- [Store API](../../reference/store.md)
- [Quick start](quickstart.md)
- [Performance](../../performance.md)
- [Design overview](../concepts/overview.md)

---

**Full source**: [`examples/hybrid_stores.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/hybrid_stores.py) (requires `pip install "python-nlql[faiss,qdrant]"`)
