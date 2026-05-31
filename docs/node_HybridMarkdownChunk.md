# HybridMarkdownChunk — Hybrid 增强分片节点

## 概述

`HybridMarkdownChunk` 在 `MarkDownChunk` 的基础分片之上，增加**面向 RAG 检索的丰富元数据**：Parent-Child、Small-to-Big、Table-Aware、邻居窗口和引用关系。它是入库前的**最终分片形态**，输出的 JSON 直接被 `ChromaBailianEmbedding` 消费。

**路径**：`nodes/chunks/HybridMarkdownChunk.py`

## 在链路中的位置

```
DataClean → MarkDownChunk（内部调用）→ HybridMarkdownChunk → ChromaBailianEmbedding
```

## 核心策略

| 策略 | 标识 | 说明 |
| --- | --- | --- |
| **Parent-Child** | `parent_context` | 每个 retrieval child 记录所属 section parent，section 记录子 chunk |
| **Small-to-Big** | `small_to_big_context_ids` | 检索时用小块，回答时扩展上下文——包含 parent + neighbors + refs |
| **Table-Aware** | `table_chunk` | 表格类型独立标记，支持后续加权和 `embedding_text` 中的表描述 |
| **Neighbor Window** | `neighbor_chunk_ids` | 记录每个 chunk 的前后 1 个邻居（NEIGHBOR_WINDOW=1） |
| **Reference Tracking** | `related_ref_chunk_ids` | 跟踪 text_chunk 引用了哪些 table/image，以及 table/image 被哪些 text_chunk 引用 |

## 输入 / 输出

### 输入

- 单个 `DataCleaned.md` 文件路径

### 输出

JSON 文件写入同目录，后缀 `.hybrid_chunks.json`：

```json
[
  {
    "id": "text_chunk_<uuid>",
    "type": "text_chunk",
    "document_id": "HR基础制度库_c4f9a1d879b8",
    "document_name": "HR基础制度库",
    "retrieval_role": "retrieval_child",
    "should_embed": true,
    "semantic_unit_type": "text",
    "parent_context": {
      "parent_section_id": "section_chunk_xxx",
      "parent_section_title": "入职流程",
      "parent_section_title_path": ["HR基础制度库", "入职流程"],
      "parent_section_line_range": [42, 128]
    },
    "neighbor_chunk_ids": {
      "previous": ["text_chunk_prev"],
      "next": ["text_chunk_next"]
    },
    "related_ref_chunk_ids": {
      "outgoing": ["table_chunk_t1"],
      "incoming": []
    },
    "small_to_big_context_ids": [
      "section_chunk_xxx",
      "text_chunk_prev",
      "text_chunk_next",
      "table_chunk_t1"
    ],
    "embedding_text": "文档: HR基础制度库\n章节: HR基础制度库 > 入职流程\n...",
    "chunk_schema_version": "1.0",
    "chunk_strategy": "markdown_header_parent_child_table_aware_small_to_big"
  }
]
```

## 关键字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `document_id` | `str` | 使用 MinerUResult 目录名作为稳定 ID |
| `retrieval_role` | `str` | `parent_section` / `retrieval_child` / `supporting_chunk` |
| `should_embed` | `bool` | `True` 仅对 `text_chunk` / `table_chunk` / `image_chunk` |
| `embedding_text` | `str` | 用于向量化的增强文本（文档名 + 章节路径 + 内容/描述） |
| `small_to_big_context_ids` | `list[str]` | 检索命中后应一起返回的上下文 chunk IDs |
| `parent_context` | `dict` | 所属 section 信息（不含大段正文） |

## `retrieval_role` 分类

| 角色 | chunk_type | should_embed | 说明 |
| --- | --- | --- | --- |
| `parent_section` | `section_chunk` | `false` | 章节标题，仅作为上下文扩展 |
| `retrieval_child` | `text_chunk`, `table_chunk`, `image_chunk` | `true` | 默认入库检索 |
| `supporting_chunk` | 其他 | `false` | 辅助 chunk，不入库 |

## `small_to_big_context_ids` 构成

每个 retrieval child 的 small-to-big 上下文由以下 ID 拼接去重：

- `parent_section_id`（父 section chunk）
- 前后邻居 chunk IDs（NEIGHBOR_WINDOW=1）
- `outgoing` ref target IDs（正文引用的表格/图片）
- `incoming` ref IDs（被哪些正文引用了）

## 核心函数

### `build_hybrid_chunks(input_md, skip_toc=True) → list[dict]`

```python
base_chunks = split_markdown(input_md)           # MarkDownChunk
for chunk in base_chunks:
    enriched = enrich_chunk(chunk, ...)            # 增加元数据
```

`enrich_chunk()` 为每个基础 chunk 计算：
- `find_parent_section_id()` — 最近父 section
- `find_neighbor_chunk_ids()` — 前后邻居
- `collect_ref_target_ids()` — 引用关系（outgoing/incoming）
- `build_search_text()` — 增强 embedding 文本

## CLI 使用

```powershell
# 批量处理 MinerUResult 下所有 DataCleaned.md
.\.venv\Scripts\python.exe nodes\chunks\HybridMarkdownChunk.py

# 处理单个文件
.\.venv\Scripts\python.exe nodes\chunks\HybridMarkdownChunk.py MinerUResult/HR基础制度库_xxx/DataCleaned.md
```

## Embedding 入库契约

`ChromaBailianEmbedding` 依赖以下字段，写入 Chroma 元数据：
- `id`, `type`, `document_id`, `document_name`, `should_embed`, `embedding_text`
- `retrieval_role`, `semantic_unit_type`, `title_path`, `line_range`
- `parent_context`, `neighbor_chunk_ids`, `related_ref_chunk_ids`
- `small_to_big_context_ids`, `content_hash`, `chunk_strategy`, `chunk_schema_version`

契约验证：`nodes/contracts.py` 中的 `validate_hybrid_chunk()` 会在入库前校验 `should_embed=True` 时 `embedding_text` 非空。

## 上下游契约

- **上游**：`MarkDownChunk.split_markdown()` 的基础分片列表
- **下游**：`ChromaBailianEmbedding` 读取 `.hybrid_chunks.json` → 入库
- `HybridChunk` 必须满足 `contracts.py` 中 `HYBRID_CHUNK_REQUIRED_FIELDS` 定义的 7 个必要字段
