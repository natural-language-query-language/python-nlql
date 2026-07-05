# API 参考

NLQL 的 public API 按 9 个分组组织，均可从 `import nlql` 顶层获取。下面列出各组的核心入口，完整签名见各模块页。

## 引擎与构造

[`Engine`](sdk.md#nlql.sdk.Engine) 是应用主入口，[`QueryBuilder`](sdk.md#nlql.sdk.QueryBuilder) 是链式构造器。

- `Engine`, `QueryBuilder`, `select`, `F`, `Meta`, `content`, `field`, `similarity`, `contains`, `length`
- 模块：[`nlql.sdk`](sdk.md)

## 数据模型

[`Document`](model.md#nlql.model.Document) / [`Payload`](model.md#nlql.model.Payload) / [`Unit`](model.md#nlql.model.Unit) / `Modality`。

- `Document`, `Payload`, `Unit`, `Modality`, `UnitKind`, `Span`, `Vector`
- 模块：[`nlql.model`](model.md)

## 语言与 IR

[`parse`](lang.md#nlql.lang.parse) 把 NLQL 语句编译到 [`Query`](ir.md#nlql.ir.Query) IR；[`query_json_schema`](ir.md#nlql.ir.query_json_schema) 导出 JSON Schema。

- `parse`, `NLQLParser` · 模块：[`nlql.lang`](lang.md)
- `Query`, `Select`, `Binding`, `Literal`, `Path`, `Ref`, `Call`, `Compare` 等 · 模块：[`nlql.ir`](ir.md)

## Embedder

[`Embedder`](embed.md#nlql.embed.Embedder) 协议与后端实现、[`MultimodalEmbedder`](embed.md#nlql.embed.MultimodalEmbedder)（文本与图像同空间）、[`EmbeddingCache`](embed.md#nlql.embed.EmbeddingCache)。

- `Embedder`, `FakeEmbedder`, `OpenAIEmbedder`, `CachedEmbedder`, `EmbeddingCache`, `MultimodalEmbedder`, `FakeMultimodalEmbedder`, `DoubaoVisionEmbedder`
- 模块：[`nlql.embed`](embed.md)

## Reranker

[`Reranker`](rerank.md#nlql.rerank.Reranker) 协议，第二段精排。

- `Reranker`, `FakeReranker`, `CrossEncoderReranker`
- 模块：[`nlql.rerank`](rerank.md)

## Store

[`Store`](store.md#nlql.store.Store) 接口与 [`StoreCaps`](store.md#nlql.store.StoreCaps) 能力描述，内置 [`LocalStore`](store.md#nlql.store.LocalStore)。后端适配器见 [Store 页底部](store.md#backends)。

- `Store`, `StoreCaps`, `LocalStore`, `matches_filter`
- 模块：[`nlql.store`](store.md)

## Registry

[`Registry`](registry.md#nlql.registry.Registry) 统一注册算子、分词器、embedder、模态。

- `Registry`, `GLOBAL_REGISTRY`, `Capability`, `register_function`, `register_splitter`
- 模块：[`nlql.registry`](registry.md)

## Ingestion

[`IngestionPipeline`](ingest.md#nlql.ingest.IngestionPipeline) 写入流水线与分词器。

- `IngestionPipeline`, `Normalizer`, `DefaultNormalizer`, `split_sentences`, `split_chunks`, `make_pysbd_splitter`
- 模块：[`nlql.ingest`](ingest.md)

## Planner / Executor / Loaders / Types / Errors

- [`Planner`, `QueryPlan`, `split_filter`](plan.md) · [`nlql.plan`](plan.md)
- [`Executor`, `Evaluator`](exec.md) · [`nlql.exec`](exec.md)
- [`Loader`, `load_documents`, `TextLoader`, `DocxLoader`, `PdfLoader`](loaders.md) · [`nlql.loaders`](loaders.md)
- [`TypeTag`, `Signature`](types.md) · [`nlql.types`](types.md)
- [`NLQLError`](errors.md#nlql.errors.NLQLError) 及子类 · [`nlql.errors`](errors.md)
