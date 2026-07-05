# Two-stage Reranking

Vector recall uses a dual-encoder model to encode the query and documents separately and then compute similarity, which is only a coarse match for "short document fragment vs. long query." The reranking stage jointly scores each recalled candidate against the query, reorders them, and returns the final results. NLQL over-fetches candidates by a `rerank_factor` multiple during recall and then refines them with a `Reranker`.

This example uses `FakeEmbedder` and `FakeReranker` for an offline demonstration.

```python
import nlql
from nlql.rerank import FakeReranker

DOCS = [
    # Contains all query terms but is very long; dual-encoder similarity gets diluted
    ("agent memory planning tool retrieval vector index query model system", "full"),
    ("banana bread recipe with flour and sugar", "noise"),
    ("agent", "partial"),
]
QUERY = 'SELECT SENTENCE LET rel = SIMILARITY(content, "agent memory planning tool") ORDER BY rel DESC LIMIT 3'
```

## Without a reranker

```python
engine = nlql.Engine(nlql.embed.FakeEmbedder(), reranker=None, rerank_factor=10)
for text, doc_id in DOCS:
    engine.add_text(text, id=doc_id)

print("== without reranking ==")
for unit in engine.search(QUERY):
    print(f"  ({unit.doc_id:8}) rel={unit.scores.get('rel', 0.0):+.3f}")
```

The `full` document covers every query term, but because the sentence is long, its similarity is averaged out and diluted, so it may not rank near the top.

## With a reranker

```python
engine = nlql.Engine(nlql.embed.FakeEmbedder(), reranker=FakeReranker(), rerank_factor=10)
for text, doc_id in DOCS:
    engine.add_text(text, id=doc_id)

print("== with FakeReranker ==")
for unit in engine.search(QUERY):
    rerank = unit.scores.get("rerank")
    tail = f"  rerank={rerank:.2f}" if rerank is not None else ""
    print(f"  ({unit.doc_id:8}) rel={unit.scores.get('rel', 0.0):+.3f}{tail}")
```

The `Reranker` protocol requires `rerank(query, units) -> units`: it takes the query text and the list of recalled candidates and returns a list ordered by the new score. In the results, `unit.scores["rerank"]` is the reranked score. The recall stage first takes `limit × rerank_factor` candidates, and after fine-ranking returns only the top `limit`.

## rerank_factor

`rerank_factor` controls the over-fetch multiple: the final `limit` multiplied by this factor gives the recall count. A larger factor yields more complete recall and leans more on the reranker for precision, but is also slower. Common values range from 5 to 20.

```python
nlql.Engine(nlql.embed.OpenAIEmbedder(), reranker=FakeReranker(), rerank_factor=5)
```

## CrossEncoder for production

`FakeReranker` is for demonstration only. In production, replace it with a real reranker:

```python
from nlql.rerank import CrossEncoderReranker

engine = nlql.Engine(
    nlql.embed.OpenAIEmbedder(),
    reranker=CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2"),
    rerank_factor=5,
)
```

`CrossEncoderReranker` is based on `sentence-transformers` and requires `pip install "python-nlql[local]"`.

!!! info "Overriding the rerank query per call"
    `engine.search(q, rerank_query="custom text")` overrides the query text used during reranking; it defaults to the primary `SIMILARITY` query. You can also pass a different `reranker` instance on a single call.

!!! tip "Reranking is visible in EXPLAIN"
    When a reranker is present, `engine.explain(q)` includes the reranker's class name, making it easy to confirm the configuration is in effect.

## Next steps

- [Quick start](./quickstart.md)
- [Multimodal search](./multimodal-search.md)
- [Reranker protocol](../../reference/rerank.md)
- [Performance benchmarks](../../performance.md)

---

**Full source**: [`examples/reranking.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/reranking.py)
