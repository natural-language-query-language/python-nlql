# LLM Tool Calling

Expose NLQL as a function-calling tool: `engine.function_tool()` produces an OpenAI-style tool definition to hand to the LLM, the LLM returns a structured Query IR, and `engine.search_ir()` executes it directly. This is the recommended retrieval path for agent and RAG applications — having the model output a structured IR is more reliable than generating a query string.

The following example uses `FakeEmbedder`, which requires no network access or model downloads and runs directly.

```python
import json
import nlql

engine = nlql.Engine(nlql.embed.FakeEmbedder())
engine.add_text("AI agents use planning, memory, and tool use.",
                metadata={"status": "published"})
engine.add_text("Vector databases store embeddings for similarity search.",
                metadata={"status": "published"})
```

## 1. Produce the tool definition

`function_tool()` returns a dictionary conforming to the OpenAI tool-calling spec; its `parameters` field is a JSON Schema that constrains the legal structure of the IR and the available function names.

```python
tool = engine.function_tool(name="nlql_query")
print(json.dumps({
    "type": tool["type"],
    "function": {"name": tool["function"]["name"]},
}, indent=2))
```

The `$defs` in `parameters` lists every IR node type (`select` / `let` / `where` / `order_by` / `limit`, etc.). The schema is generated dynamically from the functions actually registered in the engine, so you never have to maintain the tool definition by hand after extending functions.

## 2. Hand the tool to the LLM

Pass `tool` as the `tools` argument to any OpenAI-compatible chat endpoint:

```python
from openai import OpenAI

client = OpenAI(base_url="https://your-gateway/v1", api_key="sk-...")
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "how do agents use tools?"}],
    tools=[tool],
    tool_choice="auto",
)
args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
```

## 3. Execute the IR

The `arguments` returned by the LLM are a Query IR document; pass it directly to `search_ir()` for execution, with no string parsing. The IR below specifies: return sentence granularity, order by the similarity between `content` and the query text, keep only published content, and take the top 3.

```python
llm_output = {
    "select": {"unit": "sentence"},
    "let": [
        {
            "name": "relevance",
            "expr": {
                "node": "call",
                "name": "SIMILARITY",
                "args": [
                    {"node": "path", "root": "content", "segments": []},
                    {"node": "literal", "value": "how do agents work"},
                ],
            },
        }
    ],
    "where": {
        "node": "compare",
        "op": "==",
        "left": {"node": "path", "root": "meta", "segments": ["status"]},
        "right": {"node": "literal", "value": "published"},
    },
    "order_by": [{"expr": {"node": "ref", "name": "relevance"}, "desc": True}],
    "limit": 3,
}

for unit in engine.search_ir(llm_output):
    print(f"  [{unit.scores['relevance']:+.3f}] {unit.content}")
```

## JSON structure of the IR

Each IR node uses a `node` field to mark its type:

| `node` | Meaning |
|---|---|
| `path` | Field path; `root` is `content` / `meta`, etc., and `segments` is the sub-path |
| `literal` | Literal value |
| `ref` | Reference to an alias declared in `let` |
| `call` | Function call, `name` + `args` |
| `compare` | Comparison expression, `op` + `left` + `right` |

The five top-level fields `select` / `let` / `where` / `order_by` / `limit` correspond to a complete query. `engine.search_ir()` accepts this JSON and is equivalent to passing the corresponding NLQL string or Query Builder result to `engine.search()`.

!!! note "Three-entry equivalence"
    The NLQL string, Query Builder, and the IR returned by the LLM all compile to the same IR and return identical results. `search_ir` suits scenarios that produce IR programmatically; function calling is the most typical use case.

## Next steps

- [Quick start](./quickstart.md)
- [Query Builder](./query-builder.md)
- [IR node structure](../concepts/overview.md)
- [Engine API](../../reference/sdk.md)

---

**Full source**: [`examples/llm_function_calling.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/llm_function_calling.py)
