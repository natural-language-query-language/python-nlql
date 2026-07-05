# Performance

## Test Environment

- Dataset: 15,097 text segments (5,000 documents), vector dimension 384
- Runtime: Windows, Python 3.14
- Uses `FakeEmbedder`, measuring only NLQL's own retrieval latency, excluding embedding computation

## Measured Results

| Operation | Latency |
|---|---|
| Ingest | ~2,000 segments / second |
| Semantic query | 1.6 – 2 ms |
| Semantic query + metadata filter | ~4.4 ms |
| Metadata-only filter | ~6.2 ms |

At tens of thousands to one hundred thousand items, the built-in store (pure Python) keeps query latency in the millisecond range.

## Larger Datasets

Beyond one hundred thousand or millions of items, switch to a dedicated backend:

```python
from nlql.store.hnsw_store import HnswStore
engine = Engine(embedder, store=HnswStore())
```

Optional backends: `FaissStore`, `HnswStore`, `QdrantStore`, `ChromaStore`, `PgVectorStore`. Switching backends is a one-line change; query code stays the same. See [Hybrid Backends](content/tutorials/hybrid-stores.md).

## Running the Benchmark

```bash
python benchmarks/bench.py [n_docs]
```

The script lives in the repository's [`benchmarks/`](https://github.com/natural-language-query-language/python-nlql/blob/main/benchmarks/bench.py) directory.
