# 设计思路

## 把检索写成一句查询

NLQL 把语义检索组织成一条声明式查询：相关度计算、条件过滤、排序集中在同一条语句中，避免逻辑散落在业务代码里。

```sql
SELECT SENTENCE
LET   rel = SIMILARITY(content, "AI Agent")
WHERE rel >= 0.8 AND meta.status == "published"
ORDER BY rel DESC
LIMIT 5
```

语句结构与 SQL 对应：`SELECT` 指定返回粒度，`LET` 计算相关度，`WHERE` 过滤，`ORDER BY` 与 `LIMIT` 排序限量。

## 三种写法

NLQL 支持三种构造查询的方式：

- **NLQL 语句**：可读性好，适合静态查询
- **Python 链式**：`select(...).let(...).where(...)`，适合程序化拼装
- **JSON IR**：结构化形式，作为 LLM 工具调用的载体

三种写法编译到同一份中间表示（IR），结果一致。这也是 NLQL 适合作为 Agent 工具的原因：模型调用工具时返回的就是查询本身，无需额外协议。

## 执行流程

`engine.search(query)` 的执行流程：

1. **召回**：根据 `SIMILARITY` 从索引中取出一批候选
2. **过滤**：应用 `WHERE` 条件
3. **排序限量**：按 `ORDER BY` 排序，取 `LIMIT` 条
4. **重排**（可选）：若配置了 reranker，对候选精排

向量在写入时计算并落入索引，查询时不重复计算，因此查询延迟较低。

## 后端

NLQL 不绑定特定数据库：

- 默认使用内置存储（纯 Python，适合中小数据量）
- 可切换为 Qdrant、Chroma、Faiss、HnswLib、Postgres + pgvector

```python
from nlql.store.qdrant_store import QdrantStore
engine = Engine(embedder, store=QdrantStore(location=":memory:"))
```

所有后端实现同一套 `Store` 接口。引擎优先用后端自带的能力处理过滤等操作，后端无法处理的部分在内存中完成；不同后端下结果一致，仅性能存在差异。详见[混合后端](../tutorials/hybrid-stores.md)。

## 多模态

数据模型对文本与图像通用，二者映射到同一向量空间。因此可以用文字检索图像，查询语句与文本检索一致。

```python
mm = Engine(MultimodalEmbedder(), granularity="chunk")
mm.add_image(image_bytes, metadata={"kind": "photo"})
mm.search('SELECT CHUNK LET rel = SIMILARITY(content, "一只毛茸茸的猫") ORDER BY rel DESC')
```

## 小结

- 检索意图集中在一条查询中
- 三种写法编译到同一 IR，结果一致
- 后端可切换，查询代码不变
- 文本与图像通用

更底层的细节（IR 节点结构、求值模型、类型系统）见仓库根目录的 `DESIGN.md`。
