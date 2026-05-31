# ChromaDocumentCleaner — 向量库清理节点

## 概述

`ChromaDocumentCleaner` 负责管理 Chroma 向量库中的过期数据。当用户删除 PDF 或对应的 `MinerUResult` 目录后，Chroma 中的向量不会自动删除——需要运行此节点清理。

**路径**：`nodes/embeddings/ChromaDocumentCleaner.py`

## 三种清理模式

| 模式 | CLI 参数 | 说明 |
| --- | --- | --- |
| **Prune Missing** | 默认（无参数） | 对比 MinerUResult 本地文档和 Chroma 中文档，删除 Chroma 中存在但本地已不存在的文档向量 |
| **Delete by ID** | `--document-id` | 按 `document_id` 精确删除 |
| **Delete by Name** | `--document-name` | 按 `document_name` 删除匹配的文档 |

## 核心逻辑

### Prune Missing 模式

```
1. 扫描 MinerUResult/*/DataCleaned.hybrid_chunks.json → 收集本地 document_ids
2. 查询 Chroma 中所有 metadata.document_id → 收集 Chroma document_ids
3. 计算差集：chroma_document_ids - local_document_ids
4. 删除差集中的所有向量
```

### Delete by Name 模式

```
1. Chroma.get(where={"document_name": name}) → 找到匹配的 document_ids
2. 删除这些 document_id 下的所有向量
```

## CLI 使用

```powershell
# 清理本地已删除文档的向量（最常用）
.\.venv\Scripts\python.exe nodes\embeddings\ChromaDocumentCleaner.py

# 按 document_id 删除
.\.venv\Scripts\python.exe nodes\embeddings\ChromaDocumentCleaner.py --document-id HR基础制度库_c4f9a1d879b8

# 按 document_name 删除
.\.venv\Scripts\python.exe nodes\embeddings\ChromaDocumentCleaner.py --document-name "HR基础制度库"
```

## 输出

```json
{
  "mode": "prune_missing",
  "local_document_count": 7,
  "chroma_document_count": 8,
  "missing_document_ids": ["已删除的文档_c4f9"],
  "deleted_document_count": 1,
  "deleted_vector_count": 45,
  "details": {
    "已删除的文档_c4f9": 45
  }
}
```

## 常见使用场景

| 场景 | 命令 |
| --- | --- |
| 从 `data/` 删除了 PDF，也删了 `MinerUResult/` 目录 | 运行默认清理（prune） |
| 只改了 PDF 内容（保留目录） | 不需要清理；`ChromaBailianEmbedding` 的 chunk 增量模式会自动处理 |
| 指定的某个文档不需要了 | `--document-name "文档名"` |
