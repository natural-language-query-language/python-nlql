[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > [nlql](.) > **types**

# types — 类型系统

## 模块职责

类型标签（`TypeTag`）+ 函数签名（`Signature`）。供 Registry arity/类型检查、Planner 下推分析、求值期类型规约使用。**v1 整套类型系统是死代码——v2 真正接线进比较与下推**。

## 入口与对外接口

```python
from nlql import TypeTag, Signature
Signature((TypeTag.TEXT,), TypeTag.NUMBER)
engine = nlql.Engine(embedder, field_types={"published": TypeTag.DATE, "code": TypeTag.TEXT})
```

- `TypeTag`（StrEnum）：`TEXT / NUMBER / DATE / BOOL / VECTOR / ...`。
- `Signature(args: tuple[TypeTag, ...], returns: TypeTag)`。
- `types/coerce.py`：`as_number` / `as_date` 等规约函数，带**快速拒绝**（M6 性能优化：跳过对非数值/非日期字符串的 parse-and-fail，`datetime.fromisoformat` 调用从 ~3M 降到必要次数）。

## 关键设计点（DESIGN §12）

- **声明式字段类型驱动带提示比较**：`field_types={"published": DATE}` → 日期按日期比；`TEXT` 抑制数值强转（避免 `"2023"` 被当成数字）；`NUMBER` 数值比。
- **类型信息服务于下推**：外部库需要正确类型的 filter（如 pgvector 的类型转换）。
- **类型字段留内存求值不下推**（与 metadata_pushdown 区分）。

## 测试与质量

- `tests/test_types.py`（TypeTag / Signature / 规约）、`tests/test_columns.py`（列式过滤中的类型规约与逐行一致）。

## 相关文件清单

- [`core.py`](core.py) — `TypeTag` / `Signature`
- [`coerce.py`](coerce.py) — `as_number` / `as_date` 等规约（带快速拒绝）

## 变更记录

- 2026-07-05：首次生成。
