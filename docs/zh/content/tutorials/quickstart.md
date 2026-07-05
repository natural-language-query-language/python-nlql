# 快速开始

以下示例使用 `FakeEmbedder`，无需网络与模型下载，可直接运行。

!!! info "环境"
    `pip install python-nlql`，Python ≥ 3.11。

## 建立引擎并写入

```python
import nlql

engine = nlql.Engine(nlql.embed.FakeEmbedder())

engine.add_text("AI agents plan tasks, keep memory, and call external tools.",
                id="doc-0", metadata={"status": "published", "topic": "agents"})
engine.add_text("Retrieval-augmented generation grounds LLM answers in your documents.",
                id="doc-1", metadata={"status": "published", "topic": "rag"})
engine.add_text("Banana bread needs flour, sugar, and about forty minutes to bake.",
                id="doc-2", metadata={"status": "draft", "topic": "cooking"})

print(f"已写入 {len(engine)} 个句子")
# → 已写入 3 个句子
```

`Engine` 接收一个 embedder。此处使用 `FakeEmbedder` 便于演示；替换为 `OpenAIEmbedder` 等真实实现后，其余代码不变。

## 方式一：NLQL 语句

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

## 方式二：Python 链式

适合需要程序化拼装查询的场景：

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

!!! tip "两种写法结果一致"
    NLQL 语句与链式构造编译到同一份 IR，返回结果（顺序与分数）完全相同。

## 查看执行计划

```python
import json
print(json.dumps(engine.explain(query), indent=2, ensure_ascii=False))
```

`engine.explain()` 输出查询的执行计划：返回粒度、相关度计算、过滤与排序。用于排查查询行为。

## 方式三：LLM 工具调用

直接构造查询的 JSON 形式（IR），即 LLM 工具调用返回的内容：

```python
schema = engine.function_tool()   # 工具描述，交给 LLM

results = engine.search_ir({
    "select": {"granularity": "sentence"},
    "let": [{"name": "relevance",
             "call": ["SIMILARITY", ["content", "autonomous agents and tools"]]}],
    "where": ["==", ["path", "meta", "status"], "published"],
    "order_by": [{"key": "relevance", "desc": True}],
    "limit": 3,
})
```

三种写法编译到同一份 IR，结果一致。

## 下一步

- [混合后端](hybrid-stores.md)
- [设计思路](../concepts/overview.md)
- [性能](../../performance.md)
- [Engine / QueryBuilder API](../../reference/sdk.md)

---

**完整源码**：[`examples/quickstart.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/quickstart.py)
