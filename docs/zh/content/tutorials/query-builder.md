# Query Builder

Query Builder 用 Python 链式调用构造查询，与 NLQL 字符串编译到同一份 IR。需要在运行时根据条件拼装查询时，用它比拼接字符串更安全。

以下示例使用 `FakeEmbedder`，无需网络与模型下载，可直接运行。

```python
import nlql
from nlql.sdk.builder import select, similarity, Meta, F

engine = nlql.Engine(nlql.FakeEmbedder())
engine.add_text("AI agents plan tasks, keep memory, and call external tools.",
                id="doc-0", metadata={"status": "published", "topic": "agents"})
engine.add_text("Retrieval-augmented generation grounds LLM answers in your documents.",
                id="doc-1", metadata={"status": "published", "topic": "rag"})
engine.add_text("Banana bread needs flour, sugar, and about forty minutes to bake.",
                id="doc-2", metadata={"status": "draft", "topic": "cooking"})
```

## 基本流程

Builder 各方法对应一个 SQL 子句：`select` 指定返回粒度，`let` 声明命名表达式，`where` 加过滤条件，`order_by` 排序，`limit` 限量，`build()` 产出 IR。

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

`build()` 返回 `Query` IR，可直接传给 `engine.search()`，与等价的 NLQL 字符串返回完全一致的结果。

## 字段与表达式

`Meta("status")` 引用写入时的 `metadata` 字段，`F("relevance")` 引用 `let` 中声明的别名。表达式构造器都返回 `E` 对象，重载了 Python 比较与逻辑运算符，从而写出可读的谓词。

```python
from nlql.sdk.builder import F, Meta, similarity

# 引用 let 声明的别名
F("relevance") >= 0.8
F("relevance") > 0.5

# 引用元数据字段
Meta("status") == "published"
Meta("status") != "draft"
Meta("topic") == "agents"
```

除 `similarity` 外，`contains` 做子串匹配，`length` 取字段长度，`field("a.b")` 引用任意点分路径。

## 逻辑组合

`&` 表示逻辑与，`|` 表示逻辑或，`~` 表示逻辑非。多个条件传给 `where` 时默认按与组合。

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

## 动态拼装条件

当过滤条件依赖运行时变量、用户输入或权限上下文时，用 Builder 增量拼装，避免字符串拼接与转义风险。

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

`where` 可多次调用，条件按顺序累积。

## 何时用 Builder

NLQL 字符串适合写死的查询；Builder 适合条件分支、动态字段、参数化拼装。两条路径产出同一份 IR，可在同一项目里混用。

!!! tip "与 EXPLAIN 配合"
    传给 `engine.explain()` 查看任一 Builder 查询的执行计划，确认相关度计算、过滤与排序符合预期。

## 下一步

- [NLQL 语句](./quickstart.md)
- [LLM 工具调用](./llm-function-calling.md)
- [IR 与字段路径](../concepts/overview.md)
- [QueryBuilder API](../../reference/sdk.md)

---

**完整源码**：[`examples/quickstart.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/quickstart.py)
