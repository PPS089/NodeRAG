# ChromaBailianEmbedding — 向量入库节点

## 概述

`ChromaBailianEmbedding` 是 NodeRAG **入库链路的最后一个节点**，负责读取 Hybrid chunks，调用百炼 Embedding API 生成向量，并写入本地 Chroma 向量库。支持多种同步模式（chunk 增量、strict 全量重建、force 强制重写）。

**路径**：`nodes/embeddings/ChromaBailianEmbedding.py`

## 在链路中的位置

```
HybridMarkdownChunk → ChromaBailianEmbedding → ChromaDB/
```

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 百炼 Embedding | 调用百炼 OpenAI 兼容 API 生成向量 |
| 批量入库 | 按 `EMBEDDING_BATCH_SIZE` 分批 embedding + upsert |
| Chunk 增量同步 | `sync_mode=chunk`：逐 chunk 判断是否存在，只处理新增/变更 |
| Strict 全量 | `sync_mode=strict`：先删文档全部向量，再全量写入 |
| 增量兼容 | `sync_mode=incremental`：跳过已存在，仅写入新 chunk |
| Force 强制 | `sync_mode=force` / `force_reembed=True`：强制全量重写 |
| 过期清理 | chunk 模式下自动检测并删除文档中已不存在的旧 chunk |
| 批量请求 | `batched(records, batch_size)` 控制 API 调用频率 |

## 同步模式

| 模式 | 环境变量值 | 行为 |
| --- | --- | --- |
| `chunk` | 默认 | 逐 chunk 比对：已存在→跳过，新增→写入，内容变→覆盖，旧 chunk 已删→从 Chroma 删除 |
| `strict` | `strict` | 先删除文档全部向量，再全量写入（保证一致性） |
| `incremental` | `incremental` | 仅处理新增 chunk，不删除旧数据 |
| `force` | `force` | 强制全量重写（等同于 `CHROMA_FORCE_REEMBED=true`） |

## 输入 / 输出

### 输入

从 `MinerUResult/*/DataCleaned.hybrid_chunks.json` 读取 chunks，仅处理 `should_embed=True` 且 `embedding_text` 非空的 chunk。

### 输出

写入 ChromaDB（本地持久化目录），每个 vector 附带元数据。

## Chroma ID 构造

```python
"<document_id>::<type>::<line_start>-<line_end>::<content_hash_prefix_16>"
```

例如：`HR基础制度库_c4f9::text_chunk::42-68::a1b2c3d4e5f6g7h8`

**为什么不用随机 UUID**：保证同一 chunk 内容不变时 Chroma ID 不变，支持增量同步。

## 写入的 Chroma 元数据

每个 chunk 的 Chroma metadata 包含：

| 字段 | 说明 |
| --- | --- |
| `chunk_id`, `chunk_type` | chunk 标识 |
| `document_id`, `document_name` | 文档标识 |
| `source_file`, `source_path` | 源文件路径 |
| `chunk_file` | 所在 hybrid chunks JSON 路径 |
| `retrieval_role` | `retrieval_child` / `parent_section` |
| `semantic_unit_type` | `text` / `table` / `image` / `section` |
| `title_path` | JSON 序列化的标题路径 |
| `line_range` | JSON 序列化的行号范围 |
| `parent_context` | JSON 序列化的父 section 信息 |
| `neighbor_chunk_ids` | JSON 序列化的邻居 ID |
| `related_ref_chunk_ids` | JSON 序列化的引用关系 |
| `small_to_big_context_ids` | JSON 序列化的 small-to-big IDs |
| `content_hash` | 内容 SHA256 |
| `chunk_strategy` | 分片策略标识 |
| `chunk_schema_version` | 分片 schema 版本 |

**注意**：复杂值（`list`, `dict`）在写入 Chroma 前序列化为 JSON 字符串，因为 Chroma metadata 仅支持标量（`str`, `int`, `float`, `bool`）。

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `EMBEDDING_API_KEY` | ✅ | - | 百炼 API Key（也读取 `BAILIAN_API_KEY` / `LLM_API_KEY`） |
| `EMBEDDING_BASE_URL` | ❌ | `https://dashscope.aliyuncs.com/compatible-mode/v1` | API 地址 |
| `EMBEDDING_MODEL` | ❌ | `text-embedding-v4` | 模型名 |
| `CHROMA_PERSIST_DIR` | ❌ | `ChromaDB` | Chroma 持久化目录 |
| `CHROMA_COLLECTION_NAME` | ❌ | `document_chunks` | Collection 名 |
| `CHROMA_SYNC_MODE` | ❌ | `chunk` | 同步模式 |
| `CHROMA_FORCE_REEMBED` | ❌ | `false` | 强制重写 |
| `EMBEDDING_BATCH_SIZE` | ❌ | `10` | 批量大小 |
| `EMBEDDING_TIMEOUT` | ❌ | `60` | 请求超时（秒） |

## 核心类

### `BailianEmbeddingClient`

```python
class BailianEmbeddingClient:
    embed_texts(texts: list[str]) → list[list[float]]
```

封装百炼 Embedding API，支持 `dimensions` 参数。

### `ChromaBailianIndexer`

```python
class ChromaBailianIndexer:
    get_collection() → Collection
    upsert_chunks(records) → dict   # 主入库方法
    # 内部：
    #   - get_existing_ids()        # 查询已存在 IDs
    #   - delete_documents()        # strict 模式全删
    #   - delete_stale_chunks()     # chunk 模式清理过期
```

## CLI 使用

```powershell
.\.venv\Scripts\python.exe nodes\embeddings\ChromaBailianEmbedding.py
```

## 入库前契约校验

`load_embed_chunks()` 对每个 `should_embed=True` 的 chunk 调用 `validate_hybrid_chunk()`，确保：
- 有 `id`, `type`, `document_id`, `document_name`, `should_embed`, `embedding_text`, `small_to_big_context_ids`
- `should_embed=True` 时 `embedding_text` 非空

## 上下游契约

- **上游**：`HybridMarkdownChunk` → `*.hybrid_chunks.json`
- **下游**：`ChromaRetriever` 查询 ChromaDB
- `ChromaDocumentCleaner` 管理 ChromaDB 中过期的向量数据
