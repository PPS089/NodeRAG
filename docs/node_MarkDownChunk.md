# MarkDownChunk — 基础 Markdown 分片节点

## 概述

`MarkDownChunk` 是 NodeRAG 的**基础分片引擎**，将清洗后的 Markdown 文档按语义结构拆分为独立的 chunk。它识别标题层级、表格、图片和正文段落，为后续 Hybrid 增强提供基础数据结构。

**路径**：`nodes/chunks/MarkDownChunk.py`

## 在链路中的位置

```
DataClean → MarkDownChunk（被 HybridMarkdownChunk 内部调用）→ HybridMarkdownChunk → ChromaBailianEmbedding
```

`HybridMarkdownChunk` 内部调用 `MarkDownChunk.split_markdown()` 获取基础分片，然后在此之上增加 RAG 检索元数据。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 标题识别 | 识别 `#`~`######` 标题，支持编号补偿层级（`1.1`→level 2） |
| 目录跳过 | 自动识别并跳过 `# 目录` / `Table of Contents` 区块 |
| HTML 表格分片 | 识别 `<table>...</table>` 结构，解析为 `table_chunk` |
| Markdown 表格分片 | 识别 `| ... |` + `| --- |` 结构，大表格按行切片 |
| 图片识别 | 识别 `![alt](path)` 语法，生成 `image_chunk` |
| 正文段落分片 | 按空行聚合段落，按自然边界切分长文本（支持重叠） |
| 章节层级追踪 | 维护 `section_stack`，每个 chunk 记录所在章节路径 `title_path` |
| 排序索引 | 每个 chunk 有 `order_index`，保持原文顺序 |

## Chunk 类型

| 类型 | 说明 | 生成条件 |
| --- | --- | --- |
| `section_chunk` | 标题 chunk | 每遇到 `#` 标题 |
| `text_chunk` | 正文段落 chunk | 非表格、非图片、非标题的连续文本 |
| `table_chunk` | 表格 chunk | HTML 表格或 Markdown 表格 |
| `image_chunk` | 图片 chunk | `![alt](path)` 语法 |

## 输入 / 输出

### 输入

- 单个 `DataCleaned.md` 文件路径（`str | Path`）

### 输出

JSON 文件写入同目录，后缀 `.chunks.json`：

```json
[
  {
    "id": "section_chunk_<uuid12>",
    "type": "section_chunk",
    "content": "### 入职流程",
    "content_hash": "<sha256>",
    "document_name": "HR基础制度库",
    "source_file": "DataCleaned.md",
    "title_path": ["HR基础制度库", "入职流程"],
    "source_path": "/path/to/DataCleaned.md",
    "line_range": [42, 42],
    "parent_id": "section_chunk_<parent>",
    "child_ids": ["text_chunk_<child1>", "table_chunk_<child2>"],
    "order_index": 7,
    "level": 3,
    "title": "入职流程",
    "heading_line_range": [42, 42],
    "section_line_range": [42, 128],
    "embedding_text": "..."
  }
]
```

## 核心分片参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `MAX_TEXT_CHUNK_CHARS` | 1200 | 正文段落最大字符数 |
| `TEXT_CHUNK_OVERLAP_CHARS` | 120 | 长文本切分时的重叠字符数 |
| `MAX_TABLE_ROWS_PER_CHUNK` | 30 | 大表格每片最大数据行数 |
| `MAX_TABLE_TEXT_CHARS` | 8000 | 表格转文本的最大字符数 |
| `MAX_CELL_TEXT_CHARS` | 500 | 单元格文本截断长度 |

## 表格处理细节

### Markdown 表格

- 大表格（>30 数据行）按行切片，**每片都重复表头**
- 每片记录 `table_part_index` / `table_part_total`
- 通过 `split_large_markdown_table_with_ranges()` 切分，记录原始 line_range

### HTML 表格

- 识别 `<table>...</table>` 结构
- 解析后通过 `html_table_to_text()` 转为结构化文本
- Key-Value 表格（行数≤8，列数≤6 且为偶数）用 `key: value` 格式
- 普通表格用 `Row N: col1: val1; col2: val2` 格式

## 表格/图片引用机制

表格和图片不在正文流中——它们在 `text_chunk` 中留下 REF 占位符：

```text
TABLE_REF:{chunk_id}
IMAGE_REF:{chunk_id}
```

这确保：
- 正文流保持顺序
- `text_chunk.refs` 记录引用了哪些表格/图片
- `table_chunk.referenced_by_text_chunk_ids` 记录被哪些正文引用（反向引用）
- 为 Hybrid 的 `small_to_big_context_ids` 提供关系数据

## CLI 使用

```powershell
# 批量处理 MinerUResult 下所有 DataCleaned.md
.\.venv\Scripts\python.exe nodes\chunks\MarkDownChunk.py

# 处理单个文件
.\.venv\Scripts\python.exe nodes\chunks\MarkDownChunk.py MinerUResult/HR基础制度库_xxx/DataCleaned.md
```

## 上下游契约

- **上游**：`DataClean` 输出的 `DataCleaned.md`
- **下游**：`HybridMarkdownChunk` 调用 `split_markdown()` 获取基础分片列表
- 输出的 `chunks.json` 中 `parent_id` / `child_ids` 形成层级树
- **注意**：通常不直接使用本节点，而是通过 `HybridMarkdownChunk` 调用
