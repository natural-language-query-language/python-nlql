# Quick Start

The following example uses `FakeEmbedder`, which requires no network access or model downloads and runs directly.

!!! info "Prerequisites"
    `pip install python-nlql`, Python ≥ 3.11.

## Build the engine and ingest

```python
import nlql

engine = nlql.Engine(nlql.FakeEmbedder())

engine.add_text("AI agents plan tasks, keep memory, and call external tools.",
                id="doc-0", metadata={"status": "published", "topic": "agents"})
engine.add_text("Retrieval-augmented generation grounds LLM answers in your documents.",
                id="doc-1", metadata={"status": "published", "topic": "rag"})
engine.add_text("Banana bread needs flour, sugar, and about forty minutes to bake.",
                id="doc-2", metadata={"status": "draft", "topic": "cooking"})

print(f"ingested {len(engine)} sentences")
# → ingested 3 sentences
```

`Engine` takes a single embedder. Here `FakeEmbedder` is used for demonstration; swap in a real implementation such as `OpenAIEmbedder` and the rest of the code stays the same.

## Option 1: NLQL statement

```python
query = """
    SELECT SENTENCE
    LET relevance = SIMILARITY(content, "autonomous agents and tools")
    WHERE meta.status == "published"
    ORDER BY relevance DESC
    LIMIT 3
"""
for unit in engine.search(query):
    print(f"  [{unit.scores['relevance']:+.3f}] {unit.content}")
```

## Option 2: Python chaining

Suited for scenarios that assemble queries programmatically:

```python
from nlql.sdk.builder import select, similarity, Meta

built = (
    select("sentence")
    .let("relevance", similarity("content", "autonomous agents and tools"))
    .where(Meta("status") == "published")
    .order_by("relevance", desc=True)
    .limit(3)
    .build()
)
for unit in engine.search(built):
    print(f"  [{unit.scores['relevance']:+.3f}] {unit.content}")
```

!!! tip "Both forms return identical results"
    The NLQL statement and the chained builder compile to the same IR, so the returned results (order and scores) are exactly the same.

## Inspect the execution plan

```python
import json
print(json.dumps(engine.explain(query), indent=2, ensure_ascii=False))
```

`engine.explain()` outputs the query's execution plan: return granularity, relevance computation, filtering, and ordering. Use it to investigate query behavior.

## Option 3: LLM tool calling

Construct the JSON form of the query (the IR) directly — exactly what an LLM tool call returns:

```python
schema = engine.function_tool()   # tool description, hand to the LLM

results = engine.search_ir({
    "select": {"granularity": "sentence"},
    "let": [{"name": "relevance",
             "call": ["SIMILARITY", ["content", "autonomous agents and tools"]]}],
    "where": ["==", ["path", "meta", "status"], "published"],
    "order_by": [{"key": "relevance", "desc": True}],
    "limit": 3,
})
```

All three forms compile to the same IR and return identical results.

## Next steps

- [Hybrid backends](hybrid-stores.md)
- [Design overview](../concepts/overview.md)
- [Performance](../../performance.md)
- [Engine / QueryBuilder API](../../reference/sdk.md)

---

**Full source**: [`examples/quickstart.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/quickstart.py)
