# 执行流程

一次 `engine.search(query)` 走四步：召回、过滤、排序限量、（可选）重排。理解这个流程，有助于写出阈值更直观、性能更好的查询。

```python
engine.add_text("AI agents plan tasks and call external tools.", metadata={"status": "published"})
engine.add_text("Banana bread is a quick loaf made with ripe bananas.", metadata={"status": "draft"})

results = engine.search(
    'SELECT SENTENCE '
    'LET rel = SIMILARITY(content, "AI agents") '
    'WHERE rel >= 0.3 AND meta.status == "published" '
    'ORDER BY rel DESC LIMIT 5'
)
```

## 四个阶段

### 1. 召回

引擎从 `LET` 子句里收集所有 `SIMILARITY` 调用，把每个查询文本向量化一次，再从索引中取出候选单元。取候选的方式由索引和后端决定：

- 内置存储做一次矩阵乘法（`matrix @ query_vector`），把全部候选向量与查询向量的余弦一次算完。
- 外部后端（Qdrant、Chroma、Faiss、HnswLib、pgvector）用各自的原生 ANN 查询取 top-k。

```sql
LET rel = SIMILARITY(content, "AI agents")
```

关键点：相关度分数在召回阶段一次性算出，写入每个单元的 `scores` 字典。后续阶段读取这个值，不再重新计算、不再重新向量化查询文本。

如果查询里没有 `SIMILARITY`（纯元数据过滤），引擎跳过向量召回，直接扫描存储。

### 2. 过滤

`WHERE` 子句对召回候选逐条求值。比较和逻辑短路求值，便宜的谓词（元数据、字符串匹配）先算。

```sql
WHERE rel >= 0.3 AND meta.status == "published" AND content CONTAINS "tool"
```

过滤分两部分：能交给后端原生的部分（如 Qdrant 的元数据过滤、pgvector 的 `WHERE`），在召回阶段就由后端完成；后端表达不了的部分（如自定义 Python 函数谓词），在内存里对候选做后置过滤。两部分的语义一致，结果与后端无关。

字段比较按类型规约：数值按数值比，日期按 `datetime` 比，`null` 参与有序比较时该行落选（SQL 语义）。

### 3. 排序限量

`ORDER BY` 按指定表达式排序，可引用 `LET` 绑定的别名。多个排序键从低到高依次应用（稳定排序）。`LIMIT` 取前 N 条。

```sql
ORDER BY rel DESC, meta.created_at DESC LIMIT 5
```

不写 `ORDER BY` 时，按主相关度分数降序返回。

`SELECT` 决定返回粒度。引擎在写入时索引到某个粒度（默认 `SENTENCE`）；查询时可在该粒度上组装更大的单元：

- `SELECT SENTENCE` — 命中句
- `SELECT SPAN(SENTENCE, window => 2)` — 命中句及其前后各 2 句
- `SELECT DOCUMENT` — 每个文档返回一条，取其最佳命中代表

### 4. 重排（可选）

配置了 reranker 时，召回阶段会过取更多候选（倍数由 `rerank_factor` 控制），重排器对每对 `(query, passage)` 重新打分，再取 `LIMIT`。

```python
from nlql import Engine, OpenAIEmbedder, CrossEncoderReranker

engine = Engine(
    OpenAIEmbedder(base_url="...", api_key="..."),
    reranker=CrossEncoderReranker(),
    rerank_factor=5,
)
```

重排器是可插拔的：内置 `FakeReranker`（确定性，测试用）和 `CrossEncoderReranker`（联合编码 query 与 passage，治双塔模型长度不对称）。也可以在单次查询上临时指定：`engine.search(query, reranker=my_reranker, rerank_query="...")`。

## 相关度只算一次

`SIMILARITY` 的值不随每条记录重复计算。它的处理路径是：

1. 引擎把所有 `SIMILARITY` 调用收集起来，按其规范化形式去重。`LET rel = SIMILARITY(content, "x")` 与内联写法共享同一份计算，只算一次。
2. 召回阶段，候选向量堆成矩阵，与查询向量做一次乘法，得到每条记录的余弦值。
3. 这些值写入候选单元的 `scores` 字典，`WHERE` 与 `ORDER BY` 读取它们。

因此一条查询里写多个 `SIMILARITY`（多个语义角度）也不会成倍增加向量化开销——每个独立的查询文本只 embed 一次，每个候选只参与一次矩阵乘法。

```sql
LET   rel      = SIMILARITY(content, "AI agents"),
      novelty  = SIMILARITY(content, "novel unpublished ideas")
WHERE rel >= 0.3 AND novelty >= 0.2
ORDER BY rel DESC, novelty DESC
```

## 查看执行计划

`engine.explain(query)` 返回查询计划，包含解析后的 IR、收集到的打分调用、过滤拆分和召回策略。开发期和 Agent 自检都靠它。

```python
plan = engine.explain(
    'SELECT SENTENCE LET rel = SIMILARITY(content, "x") '
    'WHERE rel >= 0.3 AND meta.status == "published" ORDER BY rel DESC LIMIT 5'
)
print(plan)
```

返回结构里的字段（`scores`、`recall`、`pushed_filter`、`residual_filter`）能让查询的每一步显式可见——这是 NLQL 把检索写成一条声明式查询的回报：执行细节可解释、可验证。

## 下一步

- 相关度的分数语义见 [索引与缓存](./index-cache.md)。
- 自定义打分或过滤函数见 [注册与扩展](./registry.md)。
- 完整可运行的检索示例见 [快速开始](../tutorials/quickstart.md)。
- 查询 IR 的字段定义见 [IR 参考](../../reference/ir.md)。
