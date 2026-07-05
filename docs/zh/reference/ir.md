# Query IR

`nlql.ir` —— Query IR：规范、可 JSON 序列化的查询 AST（`Select` / `Binding` / `Expr` 家族），三种入口的公共归宿。`query_json_schema()` 导出 JSON Schema，可直接作为 LLM function-calling 的参数 schema。

::: nlql.ir
