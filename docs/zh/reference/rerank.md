# Reranker

`nlql.rerank` —— 两段式检索的第二段：可插拔 `Reranker` 协议。召回过取候选后，对每个 `(query, passage)` 联合精排。内置 `FakeReranker` 与 `CrossEncoderReranker`。

::: nlql.rerank
