# ContextCompressor — 上下文压缩节点

## 概述

`ContextCompressor` 从检索结果中提取和压缩上下文，去重、排序、截断，生成 `context_blocks` 和 `citations`，供 PromptBuilder 组装最终回答 Prompt。

**路径**：`nodes/context/ContextCompressor.py`

## 在链路中的位置

```
ChromaRetriever → Reranker → ContextCompressor → PromptBuilder → LLM 回答
```

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 候选收集 | 从 `hits` + `expanded_context` 收集所有候选上下文块 |
| 去重 | 按 `chunk_identity`（document_id + chunk_id + type + line_range）去重 |
| 排序 | 按综合分数排序（rerank_score + role_bonus + table_bonus - order_penalty） |
| 长度控制 | 三级阈值：`max_context_chars`、`max_block_chars`、`max_blocks` |
| 引用生成 | 为每个选中的 block 分配 `citation_id`（C1, C2, ...） |

## 输入 / 输出

### 输入（RetrievalResult）

```json
{
  "question": "...",
  "hits": [
    {
      "chroma_id": "...",
      "rerank_score": 0.82,
      "document": "...",
      "expanded_context": [
        {"role": "hit", "chunk_id": "...", "content": "..."},
        {"role": "expanded", "chunk_id": "...", "content": "..."}
      ]
    }
  ]
}
```

### 输出（CompressedContext）

```json
{
  "question": "新员工入职当天需要完成哪些事项？",
  "context_blocks": [
    {
      "citation_id": "C1",
      "score": 0.94,
      "role": "hit",
      "chunk_id": "text_chunk_xxx",
      "chunk_type": "text_chunk",
      "document_id": "HR基础制度库_c4f9",
      "document_name": "HR基础制度库",
      "title_path": ["HR基础制度库", "入职流程"],
      "line_range": [42, 68],
      "content": "入职当天需完成...",
      "source_hit": {
        "chroma_id": "...",
        "rerank_score": 0.82
      }
    }
  ],
  "citations": [
    {
      "citation_id": "C1",
      "document_name": "HR基础制度库",
      "title_path": ["HR基础制度库", "入职流程"],
      "line_range": [42, 68],
      "chunk_id": "text_chunk_xxx",
      "chunk_type": "text_chunk"
    }
  ],
  "compression_stats": {
    "input_hit_count": 8,
    "candidate_count": 35,
    "selected_block_count": 12,
    "context_chars": 11420,
    "max_context_chars": 12000,
    "max_block_chars": 1800
  }
}
```

## 默认阈值

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `DEFAULT_MAX_CONTEXT_CHARS` | 12000 | 上下文总字符数上限 |
| `DEFAULT_MAX_BLOCK_CHARS` | 1800 | 单个 block 字符数上限 |
| `DEFAULT_MAX_BLOCKS` | 12 | block 数量上限 |

## 候选收集逻辑

`collect_context_candidates()` 遍历每个 hit 的 `expanded_context`：

1. 如果 `expanded_context` 为空，用 hit 自身构造一个 `role: "hit"` 的候选
2. 遍历 `expanded_context` 中的每个 chunk，计算分数：
   - `base_score` = `rerank_score`（无则用 `score`，再无则 0）
   - `role_bonus` = 0.12（仅 `role == "hit"`）
   - `table_bonus` = 0.08（仅 `chunk_type == "table_chunk"`）
   - `order_penalty` = `order * 0.01`（在扩展上下文中的排序）
3. 用 `chunk_identity`（document_id + chunk_id + type + line_range）去重

## 文本截断策略

`truncate_text(text, max_chars)`：
- 不超限 → 原样返回
- 超限 → 截断 + `...[TRUNCATED N chars]`

## CLI 使用

```powershell
# 从检索结果生成压缩上下文
.\.venv\Scripts\python.exe nodes\context\ContextCompressor.py --input retrieval_result.json --output compressed_context.json

# 自定义阈值
.\.venv\Scripts\python.exe nodes\context\ContextCompressor.py --input retrieval_result.json --max-context-chars 8000 --max-blocks 8
```

## 输入格式兼容

`load_json_input()` 支持多种编码：
1. UTF-8
2. UTF-8 with BOM（`utf-8-sig`）
3. UTF-16（PowerShell `>` 可能产生的格式）
4. UTF-8 with replace（兜底）

## 上下游契约

- **上游**：`ChromaRetriever` + `Reranker` 输出（`RetrievalResult`），需通过 `validate_retrieval_result()` 校验
- **下游**：`PromptBuilder` 消费 `CompressedContext`（`question`, `context_blocks`, `citations`）
- 输出通过 `validate_compressed_context()` 校验
