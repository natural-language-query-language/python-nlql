# NLQL

NLQL is a semantic retrieval tool: use SQL-like statements to find relevant content from text.

The canonical form of a query is a serializable intermediate representation (IR). Three forms — NLQL strings, Python chained construction, and LLM tool calls — all compile to the same IR, so their results are identical.

## Example

```sql
SELECT SENTENCE
LET   relevance = SIMILARITY(content, "AI Agent architecture"),
      novelty   = SIMILARITY(content, "novel unpublished ideas")
WHERE relevance >= 0.8
  AND meta.status != "draft"
ORDER BY relevance DESC, novelty DESC
LIMIT 5
```

The statement structure mirrors SQL: `SELECT` specifies the return granularity, `LET` computes relevance, `WHERE` filters, and `ORDER BY` and `LIMIT` sort and cap results. Relevance scoring, filtering, and sorting are concentrated in a single statement instead of being scattered across business code.

## Three forms

The same query written three ways, with identical results:

=== "NLQL statement"

    ```python
    engine.search('''
        SELECT SENTENCE
        LET rel = SIMILARITY(content, "autonomous agents")
        WHERE rel >= 0.8 AND meta.status == "published"
        ORDER BY rel DESC
        LIMIT 5
    ''')
    ```

=== "Python chained"

    ```python
    from nlql.sdk.builder import select, similarity, F, Meta

    query = (select("sentence")
        .let("rel", similarity("content", "autonomous agents"))
        .where((F("rel") >= 0.8) & (Meta("status") == "published"))
        .order_by("rel", desc=True)
        .limit(5)
        .build())

    engine.search(query)
    ```

=== "LLM tool call"

    ```python
    schema = engine.function_tool()           # tool description, hand to the LLM
    engine.search_ir({ "select": ..., ... })  # execute the query returned by the LLM
    ```

## Installation

```bash
pip install python-nlql              # core
pip install "python-nlql[faiss]"     # optional: Faiss backend (also hnsw / qdrant / chroma / pgvector)
pip install "python-nlql[loaders]"   # optional: load DOCX / PDF
```

## Features

- **Declarative queries**: relevance, filtering, and sorting in a single statement
- **Pluggable backends**: built-in storage works out of the box; switching to Qdrant, Faiss, or others is a one-line change
- **Multimodal**: text and images share the same vector space; text can retrieve images
- **Explainable**: `engine.explain()` outputs the query execution plan

## Next steps

- [Quick start](content/tutorials/quickstart.md)
- [Design rationale](content/concepts/overview.md)
- [Hybrid backends](content/tutorials/hybrid-stores.md)
- [API reference](reference/sdk.md)
