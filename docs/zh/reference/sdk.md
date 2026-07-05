# SDK

`nlql.sdk` —— `Engine`（应用主入口）与流式 `QueryBuilder`。NLQL 语句、链式构造、LLM IR 三种查询入口均通过 `Engine` 执行，编译到同一份 IR。

`Engine` 通过构造参数接收 embedder、store、reranker 等组件；OpenAI 兼容渠道直接使用 `OpenAIEmbedder(base_url=...)`。

::: nlql.sdk
