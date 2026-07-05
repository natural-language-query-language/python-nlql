# 类型系统

> since v0.3.2

NLQL 的类型系统支持两种方式驱动类型化比较：

1. **预声明** `field_types`（Engine 构造时）
2. **查询时显式转型**（SQL 风格语法糖，无需预声明）

## 语法糖：查询时显式转型

在 WHERE 的值前加类型前缀，NLQL 自动按该类型比较：

```sql
WHERE meta.published > DATE '2024-01-01'
WHERE meta.created >= TIMESTAMP '2024-01-01 12:00:00'
WHERE meta.zip == TEXT '02134'
WHERE meta.views > NUMBER '1000'
```

这些语法糖等价于声明 `field_types`，但只在当前查询生效。

## 自定义类型注册

注册自定义类型（parse + 可选 compare 重载），支持装饰器：

```python
import nlql

@nlql.register_type("SEMVER")
class Semver:
    def parse(self, s):
        return tuple(int(x) for x in s.split("."))
    def compare(self, left, right, op):
        if op == ">=": return left >= right
        return False

@nlql.register_type("UPPER")
def to_upper(s):
    return s.upper()
```

注册后可在查询中使用：

```sql
WHERE meta.email == EMAIL 'user@example.com'
WHERE meta.ver >= '1.5.0'
```

## 实例级注册

`Engine.register_type` 不污染全局：

```python
@engine.register_type("INSTANCE_ONLY")
def my_parse(s): return s + "!"
```

## 内置类型

| 前缀 | 行为 |
|---|---|
| `DATE` | ISO 日期比较 |
| `TIMESTAMP` | 含时间的日期 |
| `TEXT` | 强制字符串（抑制数值强转） |
| `NUMBER` | 强制数值 |
| `BOOL` | 布尔值 |
