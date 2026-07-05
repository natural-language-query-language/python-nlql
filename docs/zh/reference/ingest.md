# Ingestion

`nlql.ingest` —— 写入管线：`Normalizer` → 可插拔 `Splitter` → `Embedder`（带缓存）→ `Indexer`。内置规则句子分割（中/英/日 + CJK 标点），可选 `pysbd`（`nlql[segment]`）。

::: nlql.ingest
