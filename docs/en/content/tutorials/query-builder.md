# Query Builder

The Query Builder constructs queries through Python chaining and compiles to the same IR as NLQL strings. When you need to assemble queries at runtime based on conditions, it is safer than string concatenation.

The following example uses `FakeEmbedder`, which requires no network access or model downloads and runs directly.

```python
import nlql
from nlql.sdk.builder import select, similarity, Meta, F

engine = nlql.Engine(nlql.embed.FakeEmbedder())
engine.add_text("AI agents plan tasks, keep memory, and call external tools.",
                id="doc-0", metadata={"status": "published", "topic": "agents"})
engine.add_text("Retrieval-augmented generation grounds LLM answers in your documents.",
                id="doc-1", metadata={"status": "published", "topic": "rag"})
engine.add_text("Banana bread needs flour, sugar, and about forty minutes to bake.",
                id="doc-2", metadata={"status": "draft", "topic": "cooking"})
```

## Basic flow

Each builder method corresponds to a SQL clause: `select` sets the return granularity, `let` declares a named expression, `where` adds a filter, `order_by` sorts, `limit` caps the count, and `build()` produces the IR.

```python
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

`build()` returns the `Query` IR, which can be passed directly to `engine.search()` and returns results identical to the equivalent NLQL string.

## Fields and expressions

`Meta("status")` references a `metadata` field set at ingestion, and `F("relevance")` references an alias declared in `let`. The expression builders all return `E` objects and overload Python comparison and logical operators, letting you write readable predicates.

```python
from nlql.sdk.builder import F, Meta, similarity

# Reference aliases declared in let
F("relevance") >= 0.8
F("relevance") > 0.5

# Reference metadata fields
Meta("status") == "published"
Meta("status") != "draft"
Meta("topic") == "agents"
```

Beyond `similarity`, `contains` performs substring matching, `length` returns field length, and `field("a.b")` references any dotted path.

## Logical combinations

`&` denotes logical AND, `|` denotes logical OR, and `~` denotes logical NOT. When multiple conditions are passed to `where`, they are combined with AND by default.

```python
from nlql.sdk.builder import select, similarity, Meta, F

q = (
    select("sentence")
    .let("relevance", similarity("content", "agent memory and tools"))
    .where(
        (F("relevance") >= 0.3) & (Meta("status") == "published"),
        ~(Meta("topic") == "cooking"),
    )
    .order_by("relevance", desc=True)
    .limit(5)
    .build()
)

for unit in engine.search(q):
    print(f"  [{unit.scores['relevance']:+.3f}] {unit.content}")
```

## Assembling conditions dynamically

When filter conditions depend on runtime variables, user input, or permission context, use the builder to assemble them incrementally, avoiding the risks of string concatenation and escaping.

```python
from nlql.sdk.builder import select, similarity, Meta, F


def search_with(engine, query_text, status_filter, topic_filter):
    builder = (
        select("sentence")
        .let("relevance", similarity("content", query_text))
        .where(F("relevance") >= 0.2)
        .order_by("relevance", desc=True)
        .limit(5)
    )

    if status_filter is not None:
        builder = builder.where(Meta("status") == status_filter)
    if topic_filter is not None:
        builder = builder.where(Meta("topic") == topic_filter)

    return engine.search(builder.build())


for unit in search_with(engine, "agents", status_filter="published", topic_filter="agents"):
    print(f"  [{unit.scores['relevance']:+.3f}] {unit.content}")
```

`where` may be called multiple times; conditions accumulate in order.

## When to use the builder

NLQL strings suit fixed queries; the builder suits conditional branches, dynamic fields, and parameterized assembly. Both paths produce the same IR and can be mixed within the same project.

!!! tip "Pair with EXPLAIN"
    Pass any builder query to `engine.explain()` to inspect its execution plan and confirm that the relevance computation, filtering, and ordering behave as expected.

## Next steps

- [NLQL statement](./quickstart.md)
- [LLM tool calling](./llm-function-calling.md)
- [IR and field paths](../concepts/overview.md)
- [QueryBuilder API](../../reference/sdk.md)

---

**Full source**: [`examples/quickstart.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/quickstart.py)
