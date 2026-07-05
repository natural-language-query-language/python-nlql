# Execution Flow

A single `engine.search(query)` goes through four stages: recall, filter, sort with limit, and (optional) rerank. Understanding this flow helps you write queries with intuitive thresholds and better performance.

```python
engine.add_text("AI agents plan tasks and call external tools.", metadata={"status": "published"})
engine.add_text("Banana bread is a quick loaf made with ripe bananas.", metadata={"status": "draft"})

results = engine.search(
    'SELECT SENTENCE '
    'LET rel = SIMILARITY(content, "AI agents") '
    'WHERE rel >= 0.3 AND meta.status == "published" '
    'ORDER BY rel DESC LIMIT 5'
)
```

## Four Stages

### 1. Recall

The engine collects all `SIMILARITY` calls from the `LET` clause, vectorizes each query text once, and retrieves candidate units from the index. How candidates are retrieved depends on the index and backend:

- Built-in storage performs a single matrix multiplication (`matrix @ query_vector`), computing the cosine similarity of all candidate vectors against the query vector in one pass.
- External backends (Qdrant, Chroma, Faiss, HnswLib, pgvector) use their native ANN queries to fetch top-k.

```sql
LET rel = SIMILARITY(content, "AI agents")
```

The key point: relevance scores are computed once during recall and written into each unit's `scores` dictionary. Subsequent stages read this value; nothing is recomputed and the query text is never vectorized again.

If the query contains no `SIMILARITY` (pure metadata filtering), the engine skips vector recall and scans the store directly.

### 2. Filter

The `WHERE` clause is evaluated against each recall candidate. Comparisons and logical operators short-circuit, and cheap predicates (metadata, string matching) are evaluated first.

```sql
WHERE rel >= 0.3 AND meta.status == "published" AND content CONTAINS "tool"
```

Filtering is split into two parts: portions that can be delegated to the backend natively (such as Qdrant's metadata filters or pgvector's `WHERE`) are applied by the backend during recall; portions the backend cannot express (such as custom Python function predicates) are applied as an in-memory post-filter on the candidates. Both parts share the same semantics, so results are backend-independent.

Field comparisons follow type rules: numbers compare as numbers, dates as `datetime`, and rows with `null` in an ordered comparison drop out (SQL semantics).

### 3. Sort and Limit

`ORDER BY` sorts by the specified expression, which may reference aliases bound in `LET`. Multiple sort keys are applied from lowest to highest precedence (stable sort). `LIMIT` takes the first N rows.

```sql
ORDER BY rel DESC, meta.created_at DESC LIMIT 5
```

When `ORDER BY` is omitted, results are returned in descending order of the primary relevance score.

`SELECT` determines the return granularity. At ingestion time the engine indexes to a particular granularity (default `SENTENCE`); at query time larger units can be assembled on top of that granularity:

- `SELECT SENTENCE` — the matching sentence
- `SELECT SPAN(SENTENCE, window => 2)` — the matching sentence plus 2 sentences before and after
- `SELECT DOCUMENT` — one row per document, represented by its best match

### 4. Rerank (optional)

When a reranker is configured, the recall stage fetches more candidates (scaled by `rerank_factor`); the reranker rescore each `(query, passage)` pair and then applies `LIMIT`.

```python
from nlql import Engine, OpenAIEmbedder, CrossEncoderReranker

engine = Engine(
    OpenAIEmbedder(base_url="...", api_key="..."),
    reranker=CrossEncoderReranker(),
    rerank_factor=5,
)
```

Rerankers are pluggable: the built-ins are `FakeReranker` (deterministic, for testing) and `CrossEncoderReranker` (jointly encodes query and passage, addressing the length asymmetry of dual-tower models). You can also override it per query: `engine.search(query, reranker=my_reranker, rerank_query="...")`.

## Relevance Is Computed Once

The value of `SIMILARITY` is not recomputed per record. Its path is:

1. The engine collects all `SIMILARITY` calls and deduplicates them by their canonical form. `LET rel = SIMILARITY(content, "x")` and the inline form share the same computation and are evaluated only once.
2. During recall, candidate vectors are stacked into a matrix and multiplied once with the query vector, yielding the cosine for each record.
3. These values are written into the candidates' `scores` dictionary, where `WHERE` and `ORDER BY` read them.

So writing multiple `SIMILARITY` calls in a single query (multiple semantic angles) does not multiply vectorization cost — each distinct query text is embedded once, and each candidate participates in a single matrix multiplication.

```sql
LET   rel      = SIMILARITY(content, "AI agents"),
      novelty  = SIMILARITY(content, "novel unpublished ideas")
WHERE rel >= 0.3 AND novelty >= 0.2
ORDER BY rel DESC, novelty DESC
```

## Inspecting the Execution Plan

`engine.explain(query)` returns the query plan, including the parsed IR, collected scoring calls, filter split, and recall strategy. It is the tool for development and Agent self-inspection.

```python
plan = engine.explain(
    'SELECT SENTENCE LET rel = SIMILARITY(content, "x") '
    'WHERE rel >= 0.3 AND meta.status == "published" ORDER BY rel DESC LIMIT 5'
)
print(plan)
```

The fields in the returned structure (`scores`, `recall`, `pushed_filter`, `residual_filter`) make every step of the query explicit — this is the payoff of expressing retrieval as a single declarative query: execution details are explainable and verifiable.

## Next steps

- For the score semantics of relevance, see [Index and Cache](./index-cache.md).
- For custom scoring or filter functions, see [Registry and Extension](./registry.md).
- For a complete runnable retrieval example, see [Quickstart](../tutorials/quickstart.md).
- For the field definitions of the query IR, see [IR Reference](../../reference/ir.md).
