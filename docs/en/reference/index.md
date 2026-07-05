# API reference

NLQL's public API is organized into 9 groups, all importable from the top-level `import nlql`. The reference is rendered from the Python docstrings (which are in English) on the default-language pages linked below.

## Engine & construction

[`Engine`](/reference/sdk.md#nlql.sdk.Engine) is the main application entry point; [`QueryBuilder`](/reference/sdk.md#nlql.sdk.QueryBuilder) is the fluent constructor.

- `Engine`, `QueryBuilder`, `select`, `F`, `Meta`, `content`, `field`, `similarity`, `contains`, `length`
- Module: [`nlql.sdk`](/reference/sdk.md)

## Data model

[`Document`](/reference/model.md#nlql.model.Document) / [`Payload`](/reference/model.md#nlql.model.Payload) / [`Unit`](/reference/model.md#nlql.model.Unit) / `Modality`.

- `Document`, `Payload`, `Unit`, `Modality`, `UnitKind`, `Span`, `Vector`
- Module: [`nlql.model`](/reference/model.md)

## Language & IR

[`parse`](/reference/lang.md#nlql.lang.parse) compiles an NLQL statement into the [`Query`](/reference/ir.md#nlql.ir.Query) IR; [`query_json_schema`](/reference/ir.md#nlql.ir.query_json_schema) exports the JSON Schema.

- `parse`, `NLQLParser` · module: [`nlql.lang`](/reference/lang.md)
- `Query`, `Select`, `Binding`, `Literal`, `Path`, `Ref`, `Call`, `Compare` · module: [`nlql.ir`](/reference/ir.md)

## Embedder

The [`Embedder`](/reference/embed.md#nlql.embed.Embedder) protocol and backends, [`MultimodalEmbedder`](/reference/embed.md#nlql.embed.MultimodalEmbedder), and [`EmbeddingCache`](/reference/embed.md#nlql.embed.EmbeddingCache).

- `Embedder`, `FakeEmbedder`, `OpenAIEmbedder`, `CachedEmbedder`, `EmbeddingCache`, `MultimodalEmbedder`, `FakeMultimodalEmbedder`, `DoubaoVisionEmbedder`
- Module: [`nlql.embed`](/reference/embed.md)

## Reranker

The [`Reranker`](/reference/rerank.md#nlql.rerank.Reranker) protocol.

- `Reranker`, `FakeReranker`, `CrossEncoderReranker`
- Module: [`nlql.rerank`](/reference/rerank.md)

## Store

The [`Store`](/reference/store.md#nlql.store.Store) interface and [`StoreCaps`](/reference/store.md#nlql.store.StoreCaps); built-in [`LocalStore`](/reference/store.md#nlql.store.LocalStore). Backend adapters: [bottom of the Store page](/reference/store.md#backends).

- `Store`, `StoreCaps`, `LocalStore`, `matches_filter`
- Module: [`nlql.store`](/reference/store.md)

## Registry

[`Registry`](/reference/registry.md#nlql.registry.Registry).

- `Registry`, `GLOBAL_REGISTRY`, `Capability`, `register_function`, `register_splitter`
- Module: [`nlql.registry`](/reference/registry.md)

## Ingestion

[`IngestionPipeline`](/reference/ingest.md#nlql.ingest.IngestionPipeline).

- `IngestionPipeline`, `Normalizer`, `DefaultNormalizer`, `split_sentences`, `split_chunks`, `make_pysbd_splitter`
- Module: [`nlql.ingest`](/reference/ingest.md)

## Planner / Executor / Loaders / Types / Errors

- [`Planner`, `QueryPlan`, `split_filter`](/reference/plan.md) · [`nlql.plan`](/reference/plan.md)
- [`Executor`, `Evaluator`](/reference/exec.md) · [`nlql.exec`](/reference/exec.md)
- [`Loader`, `load_documents`, `TextLoader`, `DocxLoader`, `PdfLoader`](/reference/loaders.md) · [`nlql.loaders`](/reference/loaders.md)
- [`TypeTag`, `Signature`](/reference/types.md) · [`nlql.types`](/reference/types.md)
- [`NLQLError`](/reference/errors.md#nlql.errors.NLQLError) and subclasses · [`nlql.errors`](/reference/errors.md)
