# LLM 工具调用

把 NLQL 暴露为函数调用工具：`engine.function_tool()` 产出 OpenAI 风格的工具定义交给 LLM，LLM 返回结构化的 Query IR，由 `engine.search_ir()` 直接执行。这是 Agent 与 RAG 应用的推荐检索路径——模型输出结构化 IR 比生成查询字符串更可靠。

以下示例使用 `FakeEmbedder`，无需网络与模型下载，可直接运行。

```python
import json
import nlql

engine = nlql.Engine(nlql.embed.FakeEmbedder())
engine.add_text("AI agents use planning, memory, and tool use.",
                metadata={"status": "published"})
engine.add_text("Vector databases store embeddings for similarity search.",
                metadata={"status": "published"})
```

## 1. 产出工具定义

`function_tool()` 返回符合 OpenAI 工具调用规范的字典，其 `parameters` 字段是一份 JSON Schema，约束了 IR 的合法结构与可用的函数名。

```python
tool = engine.function_tool(name="nlql_query")
print(json.dumps({
    "type": tool["type"],
    "function": {"name": tool["function"]["name"]},
}, indent=2))
```

`parameters` 中的 `$defs` 列出了所有 IR 节点类型（`select` / `let` / `where` / `order_by` / `limit` 等）。Schema 由本引擎实际注册的函数动态生成，因此扩展函数后无需手动维护工具定义。

## 2. 把工具交给 LLM

将 `tool` 作为 `tools` 参数传给 OpenAI 兼容的 chat 接口即可：

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

## 3. 执行 IR

LLM 返回的 `arguments` 即一份 Query IR 文档，直接交给 `search_ir()` 执行，无需字符串解析。下面这份 IR 描述：返回句子粒度，按 `content` 与查询文本的相似度排序，只看已发布内容，取前 3 条。

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

## IR 的 JSON 结构

每个 IR 节点用 `node` 字段标记类型：

| `node` | 含义 |
|---|---|
| `path` | 字段路径，`root` 为 `content` / `meta` 等，`segments` 为子路径 |
| `literal` | 字面量值 |
| `ref` | 引用 `let` 中声明的别名 |
| `call` | 函数调用，`name` + `args` |
| `compare` | 比较表达式，`op` + `left` + `right` |

顶层 `select` / `let` / `where` / `order_by` / `limit` 五个字段对应一条完整查询。`engine.search_ir()` 接受这份 JSON，等价于把对应的 NLQL 字符串或 Query Builder 结果交给 `engine.search()`。

!!! note "三入口等价"
    NLQL 字符串、Query Builder、LLM 返回的 IR 三者编译到同一份 IR，结果一致。`search_ir` 适合程序化产出 IR 的场景，函数调用是最典型的用例。

## 下一步

- [快速开始](./quickstart.md)
- [Query Builder](./query-builder.md)
- [IR 节点结构](../concepts/overview.md)
- [Engine API](../../reference/sdk.md)

---

**完整源码**：[`examples/llm_function_calling.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/llm_function_calling.py)
