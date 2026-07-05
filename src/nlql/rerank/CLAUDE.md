[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > [nlql](.) > **rerank**

# rerank — 两段式检索（召回 + 精排）

## 模块职责

双塔向量召回对"短 chunk vs 长查询"只是粗匹配。本模块提供可插拔 `Reranker` 协议，在召回过取候选后做第二段联合精排。

## 入口与对外接口

```python
from nlql import Reranker, FakeReranker, CrossEncoderReranker

engine = nlql.Engine(embedder, reranker=CrossEncoderReranker(), rerank_factor=5)
engine.search('SELECT SENTENCE LET rel = SIMILARITY(content, "…") ORDER BY rel DESC LIMIT 5')
# 按查询覆盖：
engine.search(q, reranker=my_reranker, rerank_query="...")
```

- `Reranker` Protocol：`rerank(query: str, units: list[Unit]) -> list[Unit]`（重排序后返回）。
- `FakeReranker` — 确定性，离线测试。
- `CrossEncoderReranker` — 联合打分 (query, passage)；治双塔长度不对称；可选（`nlql[local]`）。

## 关键设计点

- 流程：**召回（过取 `limit × rerank_factor`）→ 精排 → 取 top `limit`**，EXPLAIN 中可见。
- 可插拔 = 协议 oriented；自定义重排器实现 `Reranker` 即可。
- `rerank_query` 覆盖精排所用文本（默认用主 `SIMILARITY` 查询）。

## 关键依赖与配置

- 可选：`sentence-transformers`（CrossEncoder，`nlql[local]`，懒加载）。
- 上游：`exec.Executor` 调用；下游：消费 `model.Unit`。

## 测试与质量

- `tests/test_rerank.py`（流程 / 过取 / limit / 自定义 reranker）。

## 相关文件清单

- [`base.py`](base.py) — `Reranker` Protocol / `FakeReranker`
- [`cross_encoder.py`](cross_encoder.py) — `CrossEncoderReranker`

## 变更记录

- 2026-07-05：首次生成。
