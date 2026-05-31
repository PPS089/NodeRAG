# Reranker — 二阶段规则重排节点

## 概述

`Reranker` 对 `ChromaRetriever` 输出的检索结果进行**二阶段规则重排**：融合向量分、BM25 分、词面覆盖、标题路径匹配、来源加权和表格加权，并通过 MMR 去冗余。支持通过 `config/rerank_config.json` 灵活调节各维度权重。

**路径**：`nodes/rerank/Reranker.py`

## 在链路中的位置

```
ChromaRetriever → PermissionGuard（检索后过滤）→ Reranker → ContextCompressor
```

注意：检索阶段的初排（`stage_retrieve` 内部）和这里的二阶段重排（`stage_rerank`）都使用同一个 `rerank_hits()` 函数，区别仅在于 `stage_name` 和配置上下文。

## 重排分数构成

```
rerank_score =
    weights.vector   * vector_score_norm     # 向量相似度（0.44）
  + weights.bm25     * bm25_score_norm       # BM25 关键词匹配（0.26）
  + weights.lexical  * lexical_overlap_score # 词面重叠（0.16）
  + weights.title_path * title_path_score    # 标题路径匹配（0.14）
  + source_bonus                             # 多源命中加成（0.08）
  + table_bonus                              # 表格问题表格加成（0.12）
```

### 各维度说明

| 维度 | 计算方式 | 说明 |
| --- | --- | --- |
| `vector_score_norm` | Min-Max 归一化 `chroma_distance` | 向量相似度 |
| `bm25_score_norm` | Min-Max 归一化 BM25 分 | 关键词匹配 |
| `lexical_overlap_score` | `|query_tokens ∩ hit_tokens| / |query_tokens|` | 词面覆盖 |
| `title_path_score` | `|query_tokens ∩ title_tokens| / |query_tokens|` | 标题路径匹配 |
| `source_bonus` | 0.08（`retrieval_sources` > 1 时） | 多源命中加成 |
| `table_bonus` | 0.12（表格问题 + `chunk_type=table_chunk` 时） | 表格加权 |

## MMR 去冗余

```
mmr_score = λ * relevance - (1-λ) * max_jaccard(selected_hits)
```

- `λ` (mmr_lambda) = 0.72，越高越偏相关性，越低越偏多样性
- 多样性惩罚 = 当前候选与已选中 hits 的最大 Jaccard 相似度
- 贪心选择：每轮选 MMR 分数最高的 hit 加入 selected

## 重排配置

`config/rerank_config.json`：

```json
{
  "enabled": true,
  "mode": "rule",
  "rerank_top_n": 24,
  "final_top_k": 8,
  "use_mmr": true,
  "table_aware": true,
  "mmr_lambda": 0.72,
  "weights": {
    "vector": 0.44,
    "bm25": 0.26,
    "lexical": 0.16,
    "title_path": 0.14
  },
  "bonuses": {
    "multi_source": 0.08,
    "table_chunk": 0.12
  }
}
```

每一项均可通过环境变量或参数覆盖。

## 每个 hit 追加的字段

| 字段 | 说明 |
| --- | --- |
| `rerank_score` | 综合重排分数 |
| `rerank_stage` | 重排阶段标识（`retrieval_initial_rerank` / `second_stage_rule_rerank`） |
| `rerank_features` | 各维度分数明细 |
| `rerank_weights` | 使用的权重配置 |
| `rerank_reason` | Top 3 加分维度（如 `vector_score_norm=0.820；bm25_score_norm=0.640；lexical_overlap_score=0.375`） |
| `mmr_score` | MMR 最终分数（仅在 MMR 选中后设置） |

## 核心函数

```python
rerank_hits(hits, query_texts, table_aware, stage_name, config) → list[hit]
mmr_select(hits, top_k, lambda_mult) → list[hit]
rerank_retrieval_result(retrieval_result, query_texts, ...) → dict
```

## CLI 使用

```powershell
# 独立对检索结果做二阶段重排
.\.venv\Scripts\python.exe nodes\rerank\Reranker.py --input retrieval_result.json --output retrieval_reranked.json

# 覆盖参数
.\.venv\Scripts\python.exe nodes\rerank\Reranker.py --input retrieval_result.json --final-top-k 8 --mmr-lambda 0.8
```

## 表格意图检测

`looks_like_table_query(query_texts)` 检测 query 是否包含表相关关键词：

```
表, 表格, 价格, 金额, 预算, 薪酬, 等级, 比例, 字段, 清单, 模板
```

如果匹配且 `table_aware=True`，`table_chunk` 类型 hit 获得 +0.12 bonus。

## 上下游契约

- **上游**：`ChromaRetriever` 输出（含 `hits`, `queries`, `keyword_queries`）
- **下游**：`ContextCompressor` 消费 `hits`（按 `rerank_score` 排序）
- 输出仍满足 `RetrievalResult` 基础契约（`question`, `hits`）
- 重排只排序和截断，不做权限判断（权限过滤在重排之前完成）
