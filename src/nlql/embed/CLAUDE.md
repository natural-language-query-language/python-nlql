[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > [nlql](.) > **embed**

# embed — Embedder 协议 + 后端 + 缓存

## 模块职责

定义向量化的统一接口与多种后端，以及"永不重复 embed"的缓存层。**核心包不硬依赖任何重后端**——`sentence-transformers / CLIP / onnxruntime` 全部懒加载、按 extras 安装。

## 入口与启动

```python
from nlql import FakeEmbedder, OpenAIEmbedder, EmbeddingCache, CachedEmbedder
from nlql.embed import DoubaoVisionEmbedder, FakeMultimodalEmbedder   # 多模态
from nlql.embed.sentence_transformers import SentenceTransformerEmbedder  # 重，显式导入
```

模块入口 `src/nlql/embed/__init__.py` 故意**不导入** `SentenceTransformerEmbedder`，避免触发重依赖。

## 对外接口

- `Embedder`（Protocol）：`embed(texts: list[str]) -> list[Vector]` + 元信息（`model_id`、`dim`、`modality`）。
- `MultimodalEmbedder`：扩展 `embed` + `embed_images(images) -> list[Vector]`，文本与图像**同空间**；`supports_images(x)` 谓词。
- `EmbeddingCache`：`sha256(model_id + dim + modality + normalized_content)` → vector，可持久化 `.npz`，换模型/截维自动失效；`cache_key(...)`。
- `CachedEmbedder`：包裹任意 embedder，命中缓存则零开销；`Engine` 默认即用 CachedEmbedder。
- `normalize_rows(x)`：单位归一化，使 `matrix @ query` 直接等于 cosine。

## 后端清单

| 类 | 用途 | 依赖 | 备注 |
|---|---|---|---|
| `FakeEmbedder` | 测试/演示 | 无 | 确定性、离线、跨进程稳定；`conftest` 默认 dim=64 |
| `OpenAIEmbedder` | 生产 | httpx | 任何 OpenAI 兼容端点 = `OpenAIEmbedder(base_url=..., api_key=...)`；已真机验证 text-embedding-3-small (512 维) |
| `SentenceTransformerEmbedder` | 本地可选 | `sentence-transformers` (`nlql[local]`) | 懒加载 |
| `DoubaoVisionEmbedder` | 真机多模态 | httpx | 火山方舟 doubao-embedding-vision (2048 维)，纯 HTTP 无 torch |
| `ClipEmbedder` | 本地多模态可选 | CLIP / onnxruntime | 文本+图像同空间 |
| `FakeMultimodalEmbedder` | 多模态测试 | 无 | 离线、确定性 |

## 关键依赖与配置

- 核心：`numpy`、`httpx`（仅 OpenAI/Doubao）。
- 设计哲学：**embedder 即扩展点**——任何厂商 = 一个新 `Embedder` 实现，不给 `Engine` 加 `with_xxx()` 工厂。

## 测试与质量

- `tests/test_embed.py`（Fake/缓存/失效）、`tests/test_doubao.py`（HTTP mock + 真机验证记录）、`tests/test_multimodal.py`（图文互搜）。

## 常见问题 (FAQ)

- **如何接 Cohere/BGE/自托管？** 写一个继承 `Embedder` 的类，传给 `Engine(my_embedder)`。
- **多模态必须本地？** 否——`DoubaoVisionEmbedder` 纯 HTTP，无 torch 也能真机图文互搜。
- **缓存什么时候失效？** key 含 `model_id + dim`，换模型或截维自动失效。

## 相关文件清单

- [`base.py`](base.py) — `Embedder` / `BaseEmbedder` / `normalize_rows`
- [`fake.py`](fake.py) — `FakeEmbedder`
- [`openai.py`](openai.py) — `OpenAIEmbedder`
- [`cache.py`](cache.py) — `EmbeddingCache` / `CachedEmbedder` / `cache_key`
- [`multimodal.py`](multimodal.py) — `MultimodalEmbedder` / `FakeMultimodalEmbedder` / `supports_images`
- [`doubao.py`](doubao.py) / [`clip.py`](clip.py) / [`sentence_transformers.py`](sentence_transformers.py) — 后端实现

## 变更记录

- 2026-07-05：首次生成。
