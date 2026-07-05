# 字段类型

## 让比较带上类型

元数据字段的值通常以字符串形式存进 `metadata`，但比较时它们常常是别的类型：日期、数值、布尔。如果不告诉引擎字段的类型，它只能靠猜——而猜错会得到错误的顺序。

`field_types` 在构造 `Engine` 时声明每个元数据字段的类型：

```python
import nlql
from nlql import TypeTag

engine = nlql.Engine(
    embedder,
    field_types={
        "published": TypeTag.DATE,
        "views":     TypeTag.NUMBER,
        "code":      TypeTag.TEXT,
        "featured":  TypeTag.BOOL,
    },
)
```

声明之后，比较在执行前会做类型规约：日期按日期比、数值按数值比、文本按字符串比。比较语义对内置存储和所有外部后端一致。

## TypeTag 枚举

`TypeTag` 是一个字符串枚举，覆盖查询中流动的值类型：

```python
from enum import StrEnum

class TypeTag(StrEnum):
    ANY = "any"
    TEXT = "text"
    NUMBER = "number"
    BOOL = "bool"
    DATE = "date"
    VECTOR = "vector"
    NULL = "null"
```

除了驱动元数据比较，`TypeTag` 还用于注册函数的签名校验（参数个数与类型）和向外部后端传递正确类型的过滤值。一条 `Signature((TypeTag.TEXT,), TypeTag.NUMBER)` 描述"接受一个文本、返回数值"的函数。

## 为什么需要类型声明

考虑一个 `published` 字段，存的是 ISO 日期字符串 `"2024-03-01"`。没有类型声明时：

```sql
WHERE meta.published > "2024-01-01"
```

字符串比较按字典序进行。`"2024-03-01" > "2024-01-01"` 在字典序下恰好成立，但一旦日期格式变成 `"2024-3-1"`（无前导零），字典序就崩了——`"2024-3-1"` 会排在 `"2024-12-01"` 后面。

声明 `TypeTag.DATE` 后，两边先规约成 `datetime` 再比较，与格式无关：

```python
engine = nlql.Engine(embedder, field_types={"published": TypeTag.DATE})
# meta.published > "2024-01-01"  →  datetime 比较，结果正确
```

数值字段同理。`meta.views > 1000` 如果 `views` 存成字符串 `"999"`，字符串比较会得出 `"999" > "1000"`（字典序），但数值比较不会。

## TEXT 抑制数值强转

类型声明不只是"加上"类型，有时是"挡掉"错误的自动规约。一个产品编号、邮政编码、版本号看起来像数字，但应当作为字符串比较。

```python
engine_a = nlql.Engine(embedder)                      # 无声明，自动推断
engine_b = nlql.Engine(embedder, field_types={"zip": TypeTag.TEXT})
```

| 字段值 | `meta.zip > "02100"` 的结果 | 原因 |
|---|---|---|
| `"02134"` | 引擎 A：`False`（2134 不大于 2100） | 自动推断成数值 |
| `"02134"` | 引擎 B：`True`（字典序 `"02134" > "02100"`） | 强制字符串比较 |

`TypeTag.TEXT` 明确告诉引擎：不要把这个字段当数字。这对身份证号、订单号、SKU 等字段很重要。

## 规约的实际行为

比较前，引擎按声明的类型规约操作数；规约不了（比如对非日期字符串声明了 `DATE`）就退回原始值比较。`null` 参与有序比较时该行落选（SQL 语义），参与 `==` / `!=` 时按是否相等判断。

```python
from nlql.types.coerce import compare_values
from nlql import TypeTag

compare_values(">", "2024-03-01", "2024-01-01", hint=TypeTag.DATE)  # True
compare_values(">", "02134", "02100", hint=TypeTag.TEXT)            # True
```

这段比较逻辑被内置存储和外部后端共享，因此同一条 `WHERE` 在不同后端下行为一致。

!!! tip "声明越多，行为越可预测"
    只有被比较的字段才需要声明类型。给每个会出现在 `WHERE` / `ORDER BY` 里的元数据字段标上类型，能让查询结果在不同后端、不同数据格式下都稳定。未声明的字段走值推断：两个看起来都像数字的值按数值比，否则按字符串。

## 与后端过滤的关系

声明了类型的字段仍参与元数据过滤。引擎把类型信息一并传给后端，让外部后端用正确的类型翻译过滤条件（例如 pgvector 需要知道某列是日期还是数字）。具体哪些条件能交给后端，由后端的 `StoreCaps` 决定，详见 [Store 接口](./store-protocol.md)。

## 下一步

- 类型如何影响过滤位置：见 [Store 接口](./store-protocol.md)
- 字段在查询里的写法：见 [快速开始](../tutorials/quickstart.md)
- `TypeTag` 与 `Signature` 的完整签名：见 [类型参考页](../../reference/types.md)
