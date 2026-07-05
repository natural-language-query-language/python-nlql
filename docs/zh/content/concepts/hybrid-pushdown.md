# 混合引擎

## 查询与后端解耦

一条 NLQL 查询在语义层面是确定的：取哪些候选、按什么过滤、如何排序。但它交给哪个存储后端执行，是另一件事——NLQL 把这两层分开。同一句查询可以跑在内置存储、Faiss、Qdrant、Chroma、PgVector 上，结果逐条一致，区别只在于性能。

```python
from nlql import Engine, FakeEmbedder
from nlql.store.qdrant_store import QdrantStore

engine = Engine(FakeEmbedder(dim=64), store=QdrantStore(location=":memory:"))
```

后端切换不改查询代码。`Engine` 通过一个统一的 `Store` 接口与具体存储对话；每个后端用自己的原生能力实现这个接口。

## 交给后端，还是在内存里算

查询里的 `WHERE` 条件分两种命运：

- **能交给后端原生能力处理的**，引擎直接翻译过去。例如元数据过滤在 Qdrant 里变成 Qdrant 的 `Filter`，在 Postgres 里变成 SQL `WHERE`，在内置存储里变成 numpy 掩码。后端在自己的查询引擎里就完成过滤，返回的候选已经满足条件。
- **后端表达不了的**，引擎在返回的候选上再过一遍内存。典型情形是自定义 Python 函数谓词、复杂正则、需要业务逻辑的判断。

这个拆分由 Planner 根据后端的能力声明自动决定，调用方不感知。当整条查询都能由后端处理时，就是一次纯粹的委托，没有内存环节。

## 用 explain 看拆分

`engine.explain(query)` 返回查询计划，其中 `filter` 部分明确列出哪些条件交给了后端（`pushed`），哪些留作内存后置（`residual`）。

```python
from nlql import Engine, FakeEmbedder
from nlql.store import LocalStore
from nlql.store.faiss_store import FaissStore

query = """
    SELECT SENTENCE
    LET rel = SIMILARITY(content, "deep learning networks")
    WHERE meta.status == "published" AND meta.year >= 2024
    ORDER BY rel DESC
    LIMIT 3
"""

for name, store in [("LocalStore", LocalStore()), ("FaissStore", FaissStore())]:
    engine = Engine(FakeEmbedder(dim=64), store=store)
    engine.add_text("Neural networks power deep learning.", metadata={"status": "published", "year": 2025})
    plan = engine.explain(query)
    print(name, "pushed=", plan["filter"]["pushed"] is not None,
          "residual=", plan["filter"]["residual"] is not None)
```

内置存储把元数据过滤编成 numpy 掩码，`pushed` 非空；Faiss 自身没有元数据过滤能力，`meta.status` 与 `meta.year` 都进入 `residual`，在内存里完成。两个后端的最终结果相同。

!!! note "传给 Store 的是数据，不是闭包"
    交给后端的过滤条件是一段 IR（查询的 `WHERE` 子表达式），而不是一个编译好的 Python 函数。这样每个适配器都能把它翻译成自己后端的原生查询语法——Qdrant Filter、Chroma `where` 子句、SQL `WHERE`。这也是为什么拆分能由 Planner 在执行前就规划好。

## 结果一致，性能不同

跨后端的一致性是契约。同一句查询在内置存储、Faiss、HnswLib、Qdrant、Chroma 上返回的候选与分数逐条相同（由跨后端测试保证）。性能差异来自三处：

- **召回方式**：内置存储用 numpy 做精确点积；HnswLib 用近似最近邻（适合百万级以上）；Qdrant/Chroma 用各自的 ANN 实现。
- **过滤位置**：能交给后端的过滤在召回时就生效，候选更少；走内存的过滤要先把过取的候选拉回来再筛。
- **是否过取**：当过滤留在内存时，引擎会向后端多取一些候选（`scan` + 内存过滤），保证不漏命中。

所以"让后端多做"不只是更快，在大数据量下也更省传输。把过滤尽量交给后端是提升查询吞吐的主要杠杆之一。

## 自定义后端怎么办

实现一个 `Store`，返回诚实的 `StoreCaps`（声明你的后端支持向量检索、是否原生支持元数据过滤、是否原生支持全文）。引擎据此决定哪些条件交给你、哪些自己留。能整条处理就整条处理，处理不了的引擎自然接手，无需你写回退逻辑。详见 [Store 接口](./store-protocol.md)。

## 下一步

- 想动手跑跨后端示例：见 [混合后端教程](../tutorials/hybrid-stores.md)
- 后端能力如何声明：见 [Store 接口](./store-protocol.md)
- 查询各阶段如何执行：见 [设计思路](./overview.md)
