# Store

`nlql.store` —— `Store` 接口、`StoreCaps` 能力描述，以及内置的 `LocalStore`（基于 numpy 的精确检索）。各后端实现 `Store` 接口；引擎将过滤等操作尽量交给后端的原生能力处理，其余在内存中完成。

::: nlql.store

## 后端适配器 { #backends }

以下适配器实现 `Store` 接口，需安装对应 extras：

| 适配器 | 模块 | extras |
|---|---|---|
| `LocalStore` | `nlql.store.local` | 内置 |
| `FaissStore` | `nlql.store.faiss_store` | `nlql[faiss]` |
| `HnswStore` | `nlql.store.hnsw_store` | `nlql[hnsw]` |
| `QdrantStore` | `nlql.store.qdrant_store` | `nlql[qdrant]` |
| `ChromaStore` | `nlql.store.chroma_store` | `nlql[chroma]` |
| `PgVectorStore` | `nlql.store.pgvector_store` | `nlql[pgvector]` |

各后端的能力差异（向量检索类型、元数据与全文过滤）见 [混合后端](../content/tutorials/hybrid-stores.md)。
