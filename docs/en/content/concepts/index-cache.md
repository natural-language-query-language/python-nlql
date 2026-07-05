# Index and Cache

Vectors are computed at ingestion time and stored in the index; they are not recomputed at query time. This is the fundamental reason NLQL query latency stays low, and it determines the range of similarity scores.

```python
import nlql

engine = nlql.Engine(nlql.embed.OpenAIEmbedder(base_url="...", api_key="..."))
engine.add_text("AI agents plan tasks and call external tools.")
engine.add_text("Banana bread is a quick loaf made with ripe bananas.")

results = engine.search(
    'SELECT SENTENCE LET rel = SIMILARITY(content, "artificial intelligence") ORDER BY rel DESC'
)
# rel falls in [-1, 1]; relevant documents around 0.3+, irrelevant ones near 0 or negative
```

## Computed at Ingest, Reused at Query

When a document is added, the engine splits it into units (by sentence by default), vectorizes each unit's text once, normalizes it, and writes it into the index. At query time only the query text is vectorized once; a single matrix multiplication then yields relevance for all candidates — the embedder is never re-invoked per record.

```python
engine.add_text("First sentence. Second sentence. Third sentence.")
# Internal flow: normalize → split → embed → index
# Three units are each embedded once, and the vectors enter the index
```

When subsequent queries hit these units, the vectors are already in the index and require no vectorization calls. Only the query text itself gets embedded.

## EmbeddingCache

When the same document is added repeatedly, or different documents share identical text fragments, redundant vectorization is wasteful. `EmbeddingCache` avoids this with content addressing.

```python
from nlql import EmbeddingCache, CachedEmbedder, OpenAIEmbedder

cache = EmbeddingCache()
embedder = CachedEmbedder(OpenAIEmbedder(base_url="...", api_key="..."), cache)

engine = nlql.Engine(embedder)
engine.add_text("Important fact repeated across documents.")
engine.add_text("Important fact repeated across documents.")  # second call: zero embedding calls
```

The cache key is the hash of `model_id + dim + modality + text content`. Changing the model or output dimension automatically invalidates the key, so a size-mismatched stale vector is never returned.

!!! note "Engine ships with caching by default"
    When you pass any embedder to `Engine`, if it is not already a `CachedEmbedder`, the engine wraps it automatically. In most cases you do not need to construct `CachedEmbedder` manually, unless you want explicit control over the cache instance.

### Persistence

The cache can be saved as an `.npz` file and reloaded after a process restart, avoiding paid API calls to re-embed the same batch of text on cold start.

```python
cache.save("embeddings.npz")          # flush to disk
cache.load("embeddings.npz")           # read back
```

!!! tip "Cache keys and text normalization"
    The cache key is based on normalized text. Extra whitespace or inconsistent line breaks do not produce different vectors — the same text hits the same cache entry regardless of how it is wrapped.

## Scores Are Raw Cosine

`SIMILARITY` returns the dot product of two normalized vectors — that is, the raw cosine, in the range `[-1, 1]`.

```python
for unit in engine.search(
    'SELECT SENTENCE LET rel = SIMILARITY(content, "machine learning") ORDER BY rel DESC'
):
    print(unit.scores["rel"], unit.content)
```

This is intentional. Some systems fold cosine into `(cos + 1) / 2` to land it in `[0, 1]`, but that makes thresholds unintuitive: two orthogonal vectors should be 0, yet after folding they become 0.5. NLQL uses the raw cosine directly:

- Strong relevance: about `0.3 ~ 0.6`
- Weak relevance: about `0.1 ~ 0.2`
- No relevance: near `0` or negative

When you write `WHERE rel >= 0.3`, the threshold corresponds to the actual geometric relationship. Both developers reading the query and an LLM producing IR via function-calling can set reasonable filter conditions on this basis.

## Index Backends

The default `LocalStore` uses pure numpy for exact nearest neighbor — a single matrix multiplication scores all candidates, with zero native dependencies, suited to small and medium datasets. As data volume grows, you can switch to an approximate-nearest-neighbor backend without changing query code:

```python
from nlql.store.hnsw_store import HnswStore
from nlql.store.faiss_store import FaissStore

engine = nlql.Engine(embedder, store=HnswStore())      # hnswlib, sublinear recall
engine = nlql.Engine(embedder, store=FaissStore())      # Faiss
```

All backends implement the same `Store` interface, share the same write path (vectorize → normalize → store), and share the same query path. Switching backends affects only performance characteristics, not results.

## Next steps

- For how these scores are used in the query stage, see [Execution Flow](./evaluation.md).
- For how documents are split into units, see [Ingestion](./ingestion.md).
- For practical guidance on multi-backend switching, see [Hybrid Backends](../tutorials/hybrid-stores.md).
- For a performance comparison across backends, see [Performance](../../performance.md).
