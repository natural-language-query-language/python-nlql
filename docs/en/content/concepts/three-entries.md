# Three forms

NLQL provides three ways to construct a query: NLQL statements, the Python chained Builder, and JSON IR. All three compile to the same intermediate representation (IR — the structured form of a query), so they return identical results, row for row. Which one to choose depends on the use case: statements for static queries, the Builder for programmatic assembly, and JSON IR for LLM tool calls.

Below is the same query written three ways — recall by relevance, filter out drafts, sort by score, and take the top 3.

## NLQL statement

Most readable; well suited to static queries written into configuration or logs:

```sql
SELECT SENTENCE
LET   relevance = SIMILARITY(content, "autonomous agents and tools")
WHERE meta.status == "published"
ORDER BY relevance DESC
LIMIT 3
```

```python
for unit in engine.search(query):
    print(unit.content, unit.scores["relevance"])
```

## Python chained Builder

Suited to queries that business code assembles dynamically, with type hints:

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
    print(unit.content, unit.scores["relevance"])
```

Expressions like `F("relevance") >= 0.8` and `Meta("status") != "draft"` correspond to the infix comparisons in SQL. The Builder produces an IR object, which `engine.search()` also accepts.

## JSON IR

The structured form of a query — directly serializable, transmittable, and storable:

```python
ir = {
    "select": {"unit": "sentence"},
    "let": [
        {"name": "relevance",
         "expr": {"node": "call", "name": "SIMILARITY",
                  "args": [{"node": "path", "root": "content", "segments": []},
                           {"node": "literal", "value": "autonomous agents and tools"}]}}
    ],
    "where": {"node": "compare", "op": "==",
              "left": {"node": "path", "root": "meta", "segments": ["status"]},
              "right": {"node": "literal", "value": "published"}},
    "order_by": [{"expr": {"node": "ref", "name": "relevance"}, "desc": True}],
    "limit": 3,
}

for unit in engine.search_ir(ir):
    print(unit.content, unit.scores["relevance"])
```

`engine.search_ir(dict)` consumes this IR directly, skipping string parsing.

## Equivalence

The three forms produce the same IR, so recall, filtering, and sorting behave identically. `engine.explain()` returns the same parsed result and execution plan for any of them, which can be used for verification.

```python
assert engine.explain(nlql_query)["ir"] == engine.explain(built)["ir"]
```

## JSON IR and LLM tool calls

JSON IR is a natural carrier for LLM function-calling. `engine.function_tool()` generates a tool definition based on the IR JSON Schema; the model returns structured IR directly, and the engine executes it with `search_ir`:

```python
tool = engine.function_tool(name="nlql_query")
# tool["function"]["parameters"] is a JSON Schema, ready to pass to an OpenAI-compatible client
```

```python
# assume the LLM returned the ir dict above via a tool-call
results = engine.search_ir(ir)
```

Compared to having the model generate a query string and then parsing it, structured IR reduces syntax errors and eliminates the ambiguity of string concatenation. This is the recommended path for integrating NLQL into Agents / RAG.

!!! tip "Which one to choose"
    - Hard-coded in code or docs → NLQL statement
    - Assembled dynamically from conditions → Builder
    - Handed to an LLM → JSON IR + `function_tool`

## Next steps

- [Query syntax](./syntax.md): how to write each clause of an NLQL statement
- [Architecture](./architecture.md): how the three entry points converge on the same IR
- [Quick start](../tutorials/quickstart.md): runnable examples for all three entry points
- [API reference · Engine](../../reference/sdk.md): `search` / `search_ir` / `function_tool`
