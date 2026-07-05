[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > [nlql](.) > **ingest**

# ingest — 写入管线

## 模块职责

写入时完成 `Normalizer → Splitter(可插拔) → Embedder(带缓存) → Indexer`，把 `Document` 变成带向量的 `Unit` 并写入 Store。统一数据源无关——不再散落在某个 adapter。

## 入口与对外接口

```python
from nlql.ingest import IngestionPipeline, DefaultNormalizer, split_sentences, split_chunks, LanguageRouter, detect_language, make_pysbd_splitter
```

- `IngestionPipeline(embedder, registry=..., normalizer=..., granularity=...)` — `.process(documents) -> list[Unit]`。
- `DefaultNormalizer` — 文本规整（空白/换行）。
- `split_sentences` / `split_chunks` — 默认规则分词（内置，注册为 `SENTENCE` / `CHUNK`）。
- `LanguageRouter` — 语言路由：按检测到的语言分派到对应 splitter，可注册覆盖 `SENTENCE`。
- `detect_language(text)` + `make_pysbd_splitter(lang)` — 可选 pysbd（缩写鲁棒，`nlql[segment]`）。

## 关键设计点

- **Splitter 可插拔**：默认规则分词覆盖中/英/日 + CJK 标点；可注册 pysbd / spaCy / jieba / nltk。
- **写入与查询共用同一套 splitter** → 避免 v1 "查询时临时重切"。
- **粒度种类**：`document / chunk / sentence / span`（span 由 Executor 用 `Store.neighbors` 组装，不在写入期产生）。
- 导入包即副作用注册默认 `SENTENCE` / `CHUNK` 到 `GLOBAL_REGISTRY`。

## 关键依赖与配置

- 上游：`embed`（CachedEmbedder）/ `model`（Document/Unit/Payload）/ `registry`（splitter 注册）。
- 可选：`pysbd>=0.3.4`（`nlql[segment]`）。

## 测试与质量

- `tests/test_ingest.py`（管线）、`tests/test_language.py`（中/英/日分词 + 语言路由 + pysbd）。

## 相关文件清单

- [`pipeline.py`](pipeline.py) — `IngestionPipeline`
- [`normalize.py`](normalize.py) — `Normalizer` / `DefaultNormalizer`
- [`splitters.py`](splitters.py) — `split_sentences` / `split_chunks`（默认规则分词）
- [`language.py`](language.py) — `LanguageRouter` / `detect_language` / `make_pysbd_splitter`

## 变更记录

- 2026-07-05：首次生成。
