# BM25State — BM25 词表持久化节点

## 概述

`BM25State` 为 BM25 关键词检索提供**持久化的词表和文档频率统计**。在 ingestion 阶段记录词汇和 IDF 统计到 JSON 文件，检索时加载使用，保证跨进程/重启的 IDF 一致性。

**路径**：`nodes/retrieval/BM25State.py`

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 词表持久化 | vocab + doc_freq + total_docs + sum_token_len → JSON 文件 |
| 状态加载 | 从 JSON 恢复统计信息 |
| 增量更新 | `increment_add_documents()` / `increment_remove_documents()` |
| BM25 分数计算 | 基于持久化 IDF 统计计算 BM25 分数 |
| 稀疏向量生成 | `get_sparse_vector()` 生成 `{index: score}` 格式的稀疏向量 |
| 并发安全 | `threading.Lock` 保护状态读写 |

## 持久化文件

默认路径：`data/bm25_state.json`

```json
{
  "version": 1,
  "total_docs": 500,
  "sum_token_len": 120000,
  "vocab": {"入职": 0, "员工": 1, "流程": 2, ...},
  "doc_freq": {"入职": 45, "员工": 200, "流程": 120, ...}
}
```

## 核心类：`BM25StateManager`

```python
class BM25StateManager:
    increment_add_documents(texts)        # 新增文档
    increment_remove_documents(texts)     # 删除文档
    increment_add_documents_by_file(path) # 从 MinerUResult 批量添加
    bm25_score(query_tokens, doc_tokens) → float
    score_documents(query_text, documents) → list[dict]
    get_sparse_vector(text) → (dict, bool)
    get_sparse_vector_batch(texts) → (list[dict], bool)
    get_stats() → dict
    reset()
```

### BM25 公式

```
IDF(token) = log((N - df + 0.5) / (df + 0.5) + 1)

BM25(doc) = Σ [IDF(t) * TF(t) * (k1 + 1) / (TF(t) + k1 * (1 - b + b * len / avg))]

k1 = 1.5, b = 0.75
```

## Tokenizer

与 `retrieval_utils.tokenize_for_search()` 一致：
- 英文/数字：按词（`[a-z0-9_]+`）
- 中文：单字 + bigram

## 与 `bm25_recall` 的关系

**注意**：当前 `bm25_recall.py` 使用的是**实时计算** BM25（从 Chroma 全量读取后本地计算），不依赖 `BM25State` 的持久化统计。`BM25State` 是为**未来 Milvus 混合检索**设计的——Milvus 需要预构建稀疏向量索引，此时需要持久化的 IDF 统计。

两者使用相同的 BM25 公式参数（`k1=1.5, b=0.75`）和 tokenizer。

## CLI 使用

```powershell
# 查看状态
.\.venv\Scripts\python.exe nodes\retrieval\BM25State.py --stats

# 从 MinerUResult 批量添加
.\.venv\Scripts\python.exe nodes\retrieval\BM25State.py --add-dir MinerUResult/HR基础制度库_xxx

# 重置状态
.\.venv\Scripts\python.exe nodes\retrieval\BM25State.py --reset
```
