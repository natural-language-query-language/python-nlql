# 两段式重排

向量召回用双塔模型对查询与文档分别编码再算相似度，对“短文档片段 vs 长查询”只是粗匹配。重排阶段对召回的每一条候选与查询做联合打分，重新排序后取最终结果。NLQL 在召回阶段按 `rerank_factor` 倍数过取候选，再用 `Reranker` 精排。

本例用 `FakeEmbedder` 与 `FakeReranker` 离线演示。

```python
import nlql
from nlql import FakeReranker

DOCS = [
    # 包含全部查询词但很长，双塔相似度被稀释
    ("agent memory planning tool retrieval vector index query model system", "full"),
    ("banana bread recipe with flour and sugar", "noise"),
    ("agent", "partial"),
]
QUERY = 'SELECT SENTENCE LET rel = SIMILARITY(content, "agent memory planning tool") ORDER BY rel DESC LIMIT 3'
```

## 不加重排器

```python
engine = nlql.Engine(nlql.FakeEmbedder(), reranker=None, rerank_factor=10)
for text, doc_id in DOCS:
    engine.add_text(text, id=doc_id)

print("== without reranking ==")
for unit in engine.search(QUERY):
    print(f"  ({unit.doc_id:8}) rel={unit.scores.get('rel', 0.0):+.3f}")
```

`full` 文档虽然覆盖了所有查询词，但因为句子长，相似度被平均稀释，可能排不到前面。

## 加重排器

```python
engine = nlql.Engine(nlql.FakeEmbedder(), reranker=FakeReranker(), rerank_factor=10)
for text, doc_id in DOCS:
    engine.add_text(text, id=doc_id)

print("== with FakeReranker ==")
for unit in engine.search(QUERY):
    rerank = unit.scores.get("rerank")
    tail = f"  rerank={rerank:.2f}" if rerank is not None else ""
    print(f"  ({unit.doc_id:8}) rel={unit.scores.get('rel', 0.0):+.3f}{tail}")
```

`Reranker` 协议要求 `rerank(query, units) -> units`：接收查询文本与召回候选列表，返回按新分数排序的列表。结果中 `unit.scores["rerank"]` 即重排后的分数。召回阶段先取 `limit × rerank_factor` 条候选，精排后只返回前 `limit` 条。

## rerank_factor

`rerank_factor` 控制过取倍数：最终需要的 `limit` 乘以这个倍数得到召回数量。倍数越大召回越全、精度越依赖重排器，但也越慢。常用值在 5 到 20 之间。

```python
nlql.Engine(nlql.OpenAIEmbedder(), reranker=FakeReranker(), rerank_factor=5)
```

## 生产用 CrossEncoder

`FakeReranker` 仅用于演示。生产中替换为真实重排器：

```python
from nlql import CrossEncoderReranker

engine = nlql.Engine(
    nlql.OpenAIEmbedder(),
    reranker=CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2"),
    rerank_factor=5,
)
```

`CrossEncoderReranker` 基于 `sentence-transformers`，需安装 `pip install "python-nlql[local]"`。

!!! info "按查询覆盖重排文本"
    `engine.search(q, rerank_query="自定义文本")` 可覆盖重排阶段使用的查询文本，默认取主 `SIMILARITY` 查询。也可在单次调用上传入不同的 `reranker` 实例。

!!! tip "EXPLAIN 中可见重排"
    `engine.explain(q)` 在存在重排器时会附上重排器类名，便于确认配置生效。

## 下一步

- [快速开始](./quickstart.md)
- [多模态检索](./multimodal-search.md)
- [Reranker 协议](../../reference/rerank.md)
- [性能基准](../../performance.md)

---

**完整源码**：[`examples/reranking.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/reranking.py)
