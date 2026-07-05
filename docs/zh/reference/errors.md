# Errors

`nlql.errors` —— 类型化异常层级，根 `NLQLError`，让调用方能按错误阶段精确捕获（解析 / Schema / 注册 / 类型 / 计划 / 执行）。所有 public 错误派生自 `NLQLError`。

::: nlql.errors
