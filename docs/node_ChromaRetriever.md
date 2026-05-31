# ChromaRetriever — 检索编排节点

## 概述

`ChromaRetriever` 是 NodeRAG **检索阶段的主编排入口**，负责组装向量召回、BM25 关键词召回、多 query 合并去重、检索内初排、MMR 去冗余、上下文扩展等子模块。

**路径**：`nodes/retrieval/ChromaRetriever.py`

## 在链路中的位置

```
... → PermissionGuard（检索前过滤）→ ChromaRetriever → PermissionGuard（检索后过滤）→ Reranker → ContextCompressor → ...
```

## 检索流程

```
1. 向量召回 (chroma_store.vector_recall)
   └── 对每个 query_text 调用百炼 embedding → Chroma.query
2. BM25 召回 (bm25_recall.bm25_recall) [可选]
   └── 从 Chroma 取全量文档，本地计算 BM25
3. 合并去重 (retrieval_utils.merge_hits)
   └── 同 chroma_id 合并 retrieval_sources 和 matched_queries
4. 检索内初排 (rerank_hits)
   └── 融合向量分 + BM25 分 + 词面 + 标题 + 来源 + 表格
5. MMR 去冗余 (mmr_select) [可选]
   └── 避免结果全是同一段相似内容
6. Small-to-Big 上下文扩展 (context_expander)
   └── 按 small_to_big_context_ids 加载关联 chunk
```

## 子模块拆分

| 子模块 | 路径 | 职责 |
| --- | --- | --- |
| `chroma_store` | `nodes/retrieval/chroma_store.py` | Chroma collection 获取 + 向量召回 |
| `bm25_recall` | `nodes/retrieval/bm25_recall.py` | 本地 BM25 关键词召回 |
| `context_expander` | `nodes/retrieval/context_expander.py` | Small-to-Big 上下文扩展 |
| `filters` | `nodes/retrieval/filters.py` | 元数据过滤器（where filter 构造） |
| `retrieval_config` | `nodes/retrieval/retrieval_config.py` | 检索默认参数 |
| `retrieval_utils` | `nodes/retrieval/retrieval_utils.py` | 公共工具（tokenizer, 归一化, 合并等） |

## 输入 / 输出

### 输入参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `question` | `str` | (必传) | 用户问题 |
| `queries` | `list[str]` | `[]` | 额外向量检索 query |
| `keyword_queries` | `list[str]` | `[]` | BM25 关键词 query |
| `pseudo_answer` | `str` | `None` | HyDE 伪答案 |
| `document_ids` | `list[str]` | `[]` | 按 document_id 过滤 |
| `document_names` | `list[str]` | `[]` | 按 document_name 过滤 |
| `chunk_types` | `list[str]` | `[]` | 按 chunk_type 过滤 |
| `per_query_k` | `int` | `12` | 每条 query 向量召回数量 |
| `bm25_k` | `int` | `20` | BM25 召回数量 |
| `rerank_pool_size` | `int` | `40` | 进入 MMR 前的候选池大小 |
| `top_k` | `int` | `8` | 最终返回数量 |
| `use_bm25` | `bool` | `True` | 是否启用 BM25 |
| `use_mmr` | `bool` | `True` | 是否启用 MMR 去冗余 |
| `table_aware` | `bool` | `True` | 是否启用表格意图加权 |
| `mmr_lambda` | `float` | `0.72` | MMR 相关性权重 |

### 输出 (RetrievalResult)

```json
{
  "question": "新员工入职当天需要完成哪些事项？",
  "queries": ["新员工入职当天需要完成哪些事项？", "..."],
  "keyword_queries": ["入职 当天 流程 事项"],
  "where_filter": {"document_name": "HR基础制度库"},
  "raw_vector_hit_count": 24,
  "raw_bm25_hit_count": 20,
  "candidate_count": 35,
  "hit_count": 8,
  "hits": [
    {
      "chroma_id": "HR基础制度库_c4f9::text_chunk::42-68::a1b2",
      "chunk_id": "text_chunk_xxx",
      "document": "...",
      "metadata": {...},
      "distance": 0.23,
      "vector_score": 0.77,
      "bm25_score": 12.5,
      "rerank_score": 0.82,
      "retrieval_sources": ["vector", "bm25"],
      "matched_queries": ["新员工入职..."],
      "expanded_context": [
        {"role": "hit", "chunk_id": "...", "content": "..."},
        {"role": "expanded", "chunk_id": "...", "content": "..."}
      ]
    }
  ],
  "retrieval_pipeline": {
    "vector_recall": "nodes/retrieval/chroma_store.py",
    "bm25_recall": "nodes/retrieval/bm25_recall.py",
    "initial_rerank": "nodes/rerank/Reranker.py",
    "context_expand": "nodes/retrieval/context_expander.py"
  }
}
```

## CLI 使用

```powershell
# 基础检索
.\.venv\Scripts\python.exe nodes\retrieval\ChromaRetriever.py --question "新员工入职当天需要完成哪些事项？" --output retrieval_result.json

# 带文档过滤
.\.venv\Scripts\python.exe nodes\retrieval\ChromaRetriever.py --question "..." --document-name "HR基础制度库"

# 带表格优先
.\.venv\Scripts\python.exe nodes\retrieval\ChromaRetriever.py --question "薪酬等级..." --chunk-type table_chunk

# 完整参数
.\.venv\Scripts\python.exe nodes\retrieval\ChromaRetriever.py --question "..." --query "额外query" --keyword-query "关键词" --per-query-k 20 --top-k 10
```

## 子模块详解

### chroma_store.py — 向量召回

```python
get_collection() → Collection
vector_recall(collection, query_texts, where_filter, per_query_k) → list[hit]
```

- 通过 `BailianEmbeddingClient.embed_texts()` 生成 query 向量
- 调用 `collection.query(query_embeddings=..., n_results=per_query_k, where=where_filter)`
- 返回归一化 hit（`normalize_hit`: distance → score, vector_score）

### bm25_recall.py — BM25 关键词召回

```python
bm25_recall(collection, query_texts, where_filter, bm25_k) → list[hit]
```

- 从 Chroma 取全量文档（`collection.get()`）
- 本地轻量 tokenizer（英文按词 + 中文单字 + bigram）
- 按 BM25 公式 `IDF * TF * (k1+1) / (TF + k1*(1-b+b*len/avg))` 计算分数
- 按分数降序返回 top-k

### context_expander.py — Small-to-Big 上下文扩展

```python
expand_hit_context(hit, context_limit=30) → list[chunk]
```

- 读取 hit 的 `metadata.chunk_file` → 加载完整 hybrid chunks JSON
- 按 `small_to_big_context_ids` 加载关联 chunk
- 按 `line_range` 排序，返回前 `context_limit` 个
- 每个扩展 chunk 标记 `role: "hit"` 或 `role: "expanded"`

### filters.py — 元数据过滤器

```python
build_where_filter(document_ids, document_names, chunk_types) → dict | None
metadata_matches_filter(metadata, where_filter) → bool
```

- 构造 Chroma `where` 过滤条件
- 单值用 `{field: value}`，多值用 `{field: {"$in": [...]}}`
- `$and` 组合多个条件
- `metadata_matches_filter` 用于 BM25 的本地 metadata 过滤

### retrieval_utils.py — 公共工具

| 函数 | 说明 |
| --- | --- |
| `unique_keep_order(values)` | 去重保序 |
| `tokenize_for_search(text)` | 中文分词（单字+bigram）+ 英文按词 |
| `token_set(text)` | token 集合 |
| `normalize_hit(...)` | 标准化 Chroma 召回结果为统一 hit |
| `min_max_normalize_scores(items, field, output_field)` | Min-Max 归一化 |
| `looks_like_table_query(query_texts)` | 判断是否表格类问题 |
| `get_hit_text(hit)` | 获取 hit 的可搜索文本 |
| `merge_hits(hits, top_k)` | 多源合并去重 |
| `jaccard_similarity(left, right)` | Jaccard 相似度 |
| `parse_json_metadata(value, default)` | 容错解析 JSON 序列化的 metadata |

## 上下游契约

- **上游**：`PipelineContext.retrieval_inputs`（含 question/queries/document_names/chunk_types）
- **下游**：`Reranker`（二阶段重排）→ `ContextCompressor`（上下文压缩）
- 输出必须满足 `contracts.py` 中的 `RetrievalResult` 契约（`question`, `hits`）
