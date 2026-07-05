# 性能

## 测试环境

- 数据量：15,097 个文本片段（5,000 篇文档），向量维度 384
- 运行环境：Windows，Python 3.14
- 使用 `FakeEmbedder`，仅测量 NLQL 自身的检索耗时，不含 embedding 计算

## 实测结果

| 操作 | 延迟 |
|---|---|
| 写入 | ~2,000 片段 / 秒 |
| 语义查询 | 1.6 – 2 ms |
| 语义查询 + 元数据过滤 | ~4.4 ms |
| 纯元数据过滤 | ~6.2 ms |

在万级到十万级数据量下，内置存储（纯 Python）的查询延迟在毫秒量级。

## 大数据量

数据量超过十万、百万级时，可切换专用后端：

```python
from nlql.store.hnsw_store import HnswStore
engine = Engine(embedder, store=HnswStore())
```

可选后端：`FaissStore`、`HnswStore`、`QdrantStore`、`ChromaStore`、`PgVectorStore`。切换后端仅改动一行，查询代码不变。详见[混合后端](content/tutorials/hybrid-stores.md)。

## 运行基准

```bash
python benchmarks/bench.py [n_docs]
```

脚本位于仓库的 [`benchmarks/`](https://github.com/natural-language-query-language/python-nlql/blob/main/benchmarks/bench.py) 目录。
