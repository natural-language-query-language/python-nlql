# 三种写法

NLQL 提供三种构造查询的方式：NLQL 语句、Python 链式 Builder、JSON IR。三者编译到同一份中间表示（IR——查询的结构化形式），因此返回结果逐条一致。选择哪种取决于使用场景：静态查询用语句，程序化拼装用 Builder，LLM 工具调用用 JSON IR。

下面是同一个查询的三种写法——按相关度召回、过滤掉草稿、按分数排序取前 3 条。

## NLQL 语句

可读性最好，适合写在配置或日志里的静态查询：

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

## Python 链式 Builder

适合由业务代码动态拼装查询的场景，带类型提示：

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

`F("relevance") >= 0.8`、`Meta("status") != "draft"` 等表达式对应 SQL 中的中缀比较。Builder 产出的是一个 IR 对象，`engine.search()` 同样接受。

## JSON IR

查询的结构化形式，可直接序列化、传输、存储：

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

`engine.search_ir(dict)` 直接消费这份 IR，跳过字符串解析。

## 等价性

三种写法产出的 IR 相同，因此召回、过滤、排序的行为一致。`engine.explain()` 会对任意一种输入返回相同的解析结果与执行计划，可用于校验。

```python
assert engine.explain(nlql_query)["ir"] == engine.explain(built)["ir"]
```

## JSON IR 与 LLM 工具调用

JSON IR 是 LLM function-calling 的天然载体。`engine.function_tool()` 生成基于 IR JSON Schema 的工具定义，模型直接返回结构化 IR，引擎用 `search_ir` 执行：

```python
tool = engine.function_tool(name="nlql_query")
# tool["function"]["parameters"] 是 JSON Schema，可直接传给 OpenAI 兼容客户端
```

```python
# 假设 LLM 通过 tool-call 返回了上面的 ir 字典
results = engine.search_ir(ir)
```

相比让模型生成查询字符串再解析，结构化 IR 减少了语法错误，也省掉了字符串拼接的歧义。这是把 NLQL 接入 Agent / RAG 的推荐路径。

!!! tip "选择哪一种"
    - 写死在代码或文档里 → NLQL 语句
    - 由条件动态拼装 → Builder
    - 交给 LLM 调用 → JSON IR + `function_tool`

## 下一步

- [查询语法](./syntax.md)：NLQL 语句每个子句的写法
- [架构](./architecture.md)：三种入口如何汇入同一份 IR
- [快速上手](../tutorials/quickstart.md)：可运行的三入口示例
- [API 参考 · Engine](../../reference/sdk.md)：`search` / `search_ir` / `function_tool`
