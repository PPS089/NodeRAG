# NodeRAG 节点输入输出契约

本文档用于说明 `nodes/` 中节点之间的主要数据契约，以及在 RAG 链路中插入新节点时需要遵守的边界。

## 1. 总体链路

当前在线问答链路由 `rags/RAGPipeline.py` 编排：

```text
用户问题
  -> 意图路由 stage_route
  -> 问题改写 stage_rewrite
  -> 伪答案 stage_pseudo_answer
  -> 检索参数准备 stage_prepare_retrieval
  -> 权限检索约束 stage_apply_permission
  -> Chroma 检索 stage_retrieve
  -> 权限结果过滤 stage_filter_retrieval_permission
  -> 二阶段重排 stage_rerank
  -> 上下文压缩 stage_compress
  -> Prompt 组装 stage_prompt
  -> LLM 回答 stage_answer
```

每个 stage 接收并返回同一个 `PipelineContext` 字典。新增节点优先作为一个新的 stage 插入 `RAGPipeline.build_stages()`，不要直接改其他节点内部逻辑。

## 2. PipelineContext 关键字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `question` | `str` | 原始用户问题 |
| `trace_id` | `str` | 当前请求链路 ID，用于日志和 trace 关联 |
| `input_document_names` | `list[str]` | CLI 或调用方传入的文档过滤条件 |
| `input_chunk_types` | `list[str]` | CLI 或调用方传入的 chunk 类型过滤条件 |
| `route_result` | `dict | None` | 意图路由输出 |
| `rewrite_result` | `dict | None` | 问题改写输出 |
| `pseudo_answer_result` | `dict | None` | 伪答案输出 |
| `retrieval_inputs` | `dict` | 检索节点入参汇总 |
| `retrieval_result` | `dict | None` | 检索结果 |
| `rerank_result` | `dict | None` | 二阶段重排统计信息 |
| `context_result` | `dict | None` | 压缩后的上下文 |
| `prompt_result` | `dict | None` | Prompt 组装结果 |
| `answer` | `str | None` | 最终回答 |
| `permission_context` | `dict` | 当前模拟权限等级、可访问知识库、权限码等 |
| `permission_denied` | `bool` | 是否因为显式请求无权限文档而提前结束 |
| `needs_retrieval` | `bool` | 是否需要知识库检索 |
| `stop_pipeline` | `bool` | 是否提前结束后续 stage |

## 3. 已显式校验的节点契约

契约校验集中在 `nodes/contracts.py`。

### 3.1 HybridChunk

用于 embedding 入库，必要字段：

```text
id
type
document_id
document_name
should_embed
embedding_text
small_to_big_context_ids
```

约束：

- `should_embed=True` 时，`embedding_text` 不能为空。
- 新增 chunker 必须保持这些字段，否则 `ChromaBailianEmbedding.py` 会拒绝入库。

### 3.2 RetrievalResult

用于检索到上下文压缩，必要字段：

```text
question
hits
```

其中每个 `hit` 必要字段：

```text
chroma_id
chunk_id
metadata
```

约束：

- `hits` 必须是 list。
- 新增检索节点如果替换 `ChromaRetriever.py`，必须输出兼容结构。
- 重排节点可以更新 `hits` 顺序，并可追加 `rerank_score`、`rerank_stage`、`rerank_reason`、`rerank_features`。

### 3.3 CompressedContext

用于上下文压缩到 Prompt 组装，必要字段：

```text
question
context_blocks
citations
```

约束：

- `context_blocks` 必须是 list。
- `citations` 必须是 list。

### 3.4 PromptPayload

用于 Prompt 组装到 LLM 回答，必要字段：

```text
question
messages
citations
```

约束：

- `messages` 必须是 OpenAI-compatible chat messages list。
- `citations` 必须是 list。

## 4. 推荐插入点

| 插入位置 | 适合新增的节点 | 需要读写的字段 |
| --- | --- | --- |
| `stage_route` 后 | 权限判断、租户判断、问题安全分类 | 读 `question`，写自定义字段或 `stop_pipeline` |
| `stage_rewrite` 后 | 多查询扩展、同义词扩展、术语标准化 | 读写 `rewrite_result` |
| `stage_pseudo_answer` 后 | HyDE 优化、伪答案质量过滤 | 读写 `pseudo_answer_result` |
| `stage_prepare_retrieval` 后 | 元数据过滤增强、召回策略选择、权限检索约束 | 读写 `retrieval_inputs` |
| `stage_retrieve` 后 | 权限结果过滤、结果重排、去重、分数归一化 | 读写 `retrieval_result.hits` |
| `stage_compress` 后 | 上下文二次压缩、引用修正 | 读写 `context_result` |
| `stage_prompt` 后 | Prompt 审计、模板切换、回答风格控制 | 读写 `prompt_result.messages` |

## 5. 新节点编写约定

新增 stage 建议使用以下形式：

```python
def stage_custom(self, context: PipelineContext) -> PipelineContext:
    if context.get("stop_pipeline"):
        return context

    # 读取已有字段，写入自定义字段或更新标准字段。
    return context
```

然后在 `RAGPipeline.build_stages()` 中插入：

```python
return [
    self.stage_route,
    self.stage_rewrite,
    self.stage_custom,
    self.stage_pseudo_answer,
    ...
]
```

## 6. 常见耦合风险

- 字段名隐式依赖：节点之间通过 dict 字段传递数据，字段改名会影响后续节点。
- JSON 编码问题：PowerShell 的 `>` 可能写出 UTF-16，节点间传 JSON 文件时优先使用脚本自带 `--output`。
- Chroma 元数据依赖：检索、清理、增量入库依赖 `document_id`、`document_name`、`chunk_hash` 等元数据。
- 权限配置依赖：权限节点依赖 `metadata.document_name` 能匹配 `config/knowledge_permissions.json` 中的 `document_names` 或 `document_patterns`。
- 文档删除与向量删除分离：删除 PDF 不会自动删除向量库数据，需要运行 `ChromaDocumentCleaner.py`。
- LLM 节点外部依赖：路由、改写、伪答案、最终回答都依赖 `.env` 中的模型服务配置。

## 7. 权限节点契约

权限节点位于 `nodes/auth/PermissionGuard.py`，配置文件为 `config/knowledge_permissions.json`。

当前权限等级：

```text
L1 < L2 < L3 < L4 < L5
```

等级继承规则：

- L1 只能访问 L1。
- L3 可以访问 L1、L2、L3。
- L5 可以访问全部已配置知识库。

权限节点做两层控制：

1. `stage_apply_permission`：检索前改写 `retrieval_inputs.document_names`，只保留当前权限可访问的文档。
2. `stage_filter_retrieval_permission`：检索后过滤 `retrieval_result.hits` 和 `expanded_context`，防止未授权内容进入上下文压缩和 Prompt。

新增知识库时只改配置：

```json
{
  "knowledge_base": "新知识库",
  "required_level": "L3",
  "permission_code": "KB.NEW.READ",
  "document_names": ["新知识库", "新知识库.pdf"],
  "document_patterns": ["*新知识库*"]
}
```

注意：Chroma 检索前过滤只能使用精确 `document_name`，因此真实 PDF 名称最好写入 `document_names`；`document_patterns` 主要用于检索后兜底过滤。

## 8. 重排节点契约

重排节点位于 `nodes/rerank/Reranker.py`。

默认配置文件：

```text
config/rerank_config.json
```

输入：

```text
RetrievalResult
query_texts
rerank_top_n
final_top_k
```

输出：

```text
RetrievalResult
  hits # 已按二阶段重排后的顺序更新
  rerank # 重排统计信息
```

每个 hit 可追加字段：

```text
rerank_score
rerank_stage
rerank_reason
rerank_features
mmr_score
```

Pipeline 中的位置：

```text
stage_filter_retrieval_permission
-> stage_rerank
-> stage_compress
```

约束：

- 重排节点只排序和截断 `hits`，不做权限判断。
- 如果未来接入外部模型重排，必须保持在权限过滤之后。
- 输出仍必须满足 RetrievalResult 基础契约。

## 9. 日志契约

日志工具位于：

```text
utils/RAGLogger.py
```

默认日志文件：

```text
logs/YYYY/MM/YYYY-MM-DD.jsonl
```

每行是一条 JSON，主要字段：

```text
ts
trace_id
event
stage
status
elapsed_ms
metrics
error_type
error_message
```

当前事件类型：

```text
pipeline_start
pipeline_end
pipeline_error
stage_start
stage_end
stage_error
```

约束：

- 日志不记录完整文档内容。
- Pipeline 返回结果包含 `trace_id`，方便和日志关联。
- 可以通过 `--no-log` 关闭日志，通过 `--log-file` 指定日志路径。

## 10. 插入新节点的最低要求

1. 不破坏已有标准字段。
2. 如果替换标准节点，输出必须满足 `nodes/contracts.py` 中的契约。
3. 对外部 API、向量库、文件路径等副作用保持显式配置，不写死路径和密钥。
4. 新节点失败时尽量抛出清晰异常，避免静默返回不完整字段。
