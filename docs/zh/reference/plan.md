# Planner

`nlql.plan` —— 查询规划：相关度提取（`Scorer` / `score_key`）、`QueryPlan`，以及过滤操作的拆分（`split_filter` / `is_pushable`），决定哪些操作可交给后端、哪些在内存执行。

::: nlql.plan
