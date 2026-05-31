# NodeRAG Nodes 节点文档索引

本文档索引 `nodes/` 目录下所有节点的详细文档。每个节点文档包含：概述、在链路中的位置、核心能力、输入/输出契约、关键参数、CLI 使用和上下游依赖。

## 入库链路（Ingestion Pipeline）

```
data/*.pdf → MinerUStandardReader → DataClean → HybridMarkdownChunk → ChromaBailianEmbedding → ChromaDB/
```

| 节点 | 文档 | 职责 |
| --- | --- | --- |
| MinerUStandardReader | [node_MinerUStandardReader.md](node_MinerUStandardReader.md) | PDF → MinerU API → Markdown |
| DataClean | [node_DataClean.md](node_DataClean.md) | 清洗 OCR 噪声、规范化格式 |
| HybridMarkdownChunk | [node_HybridMarkdownChunk.md](node_HybridMarkdownChunk.md) | Parent-Child + Small-to-Big + Table-Aware 分片 |
| ChromaBailianEmbedding | [node_ChromaBailianEmbedding.md](node_ChromaBailianEmbedding.md) | 百炼 Embedding + Chroma 向量入库 |
| MarkDownChunk | [node_MarkDownChunk.md](node_MarkDownChunk.md) | 基础 Markdown 分片引擎（被 Hybrid 内部调用） |
| ChromaDocumentCleaner | [node_ChromaDocumentCleaner.md](node_ChromaDocumentCleaner.md) | 清理 Chroma 中过期向量 |

## 查询链路（Query Pipeline）— standard 模式

```
用户问题
→ IntentRouter
→ QuestionRewriter
→ PseudoAnswer
→ PermissionGuard（检索前过滤）
→ ChromaRetriever
→ PermissionGuard（检索后过滤）
→ Reranker
→ ContextCompressor
→ PromptBuilder
→ LLM 最终回答
```

| 阶段 | 节点 | 文档 | 职责 |
| --- | --- | --- | --- |
| 意图路由 | IntentRouter | [node_IntentRouter.md](node_IntentRouter.md) | LLM 判断意图、是否需要检索、目标文档 |
| 问题改写 | QuestionRewriter | [node_QuestionRewriter.md](node_QuestionRewriter.md) | LLM 改写问题、生成多 query、提取关键词 |
| 伪答案 | PseudoAnswer | [node_PseudoAnswer.md](node_PseudoAnswer.md) | HyDE 策略：假答案辅助向量召回 |
| 权限 | PermissionGuard | [node_PermissionGuard.md](node_PermissionGuard.md) | L1~L5 等级权限：检索前+检索后过滤 |
| 检索 | ChromaRetriever | [node_ChromaRetriever.md](node_ChromaRetriever.md) | 向量召回 + BM25 + 初排 + MMR + 上下文扩展 |
| 重排 | Reranker | [node_Reranker.md](node_Reranker.md) | 二阶段规则重排 + MMR 去冗余 |
| 上下文压缩 | ContextCompressor | [node_ContextCompressor.md](node_ContextCompressor.md) | 去重、排序、截断、生成引用编号 |
| Prompt 组装 | PromptBuilder | [node_PromptBuilder.md](node_PromptBuilder.md) | 组装最终回答 messages + citations |

## 基础组件

| 节点 | 文档 | 职责 |
| --- | --- | --- |
| LLMClient | [node_LLMClient.md](node_LLMClient.md) | 百炼 OpenAI 兼容 Chat 客户端（所有 LLM 节点的底层） |

## 可选增强节点

| 节点 | 文档 | 职责 |
| --- | --- | --- |
| AutoMerger | [node_AutoMerger.md](node_AutoMerger.md) | 父子层级自动合并（多子块命中→替换为父块） |
| GradeAndRewrite | [node_GradeAndRewrite.md](node_GradeAndRewrite.md) | 相关性评估 + 回流重写（Grade → Rewrite Loop） |
| BM25State | [node_BM25State.md](node_BM25State.md) | BM25 词表持久化（为 Milvus 混合检索做准备） |

## 数据契约

所有节点间的数据契约定义在 `nodes/contracts.py`，详见 [node_io_contract.md](node_io_contract.md)：

| 契约 | 必要字段 | 使用节点 |
| --- | --- | --- |
| HybridChunk | `id`, `type`, `document_id`, `document_name`, `should_embed`, `embedding_text`, `small_to_big_context_ids` | ChromaBailianEmbedding |
| RetrievalResult | `question`, `hits[].chroma_id`, `hits[].chunk_id`, `hits[].metadata` | ContextCompressor, Reranker |
| CompressedContext | `question`, `context_blocks`, `citations` | PromptBuilder |
| PromptPayload | `question`, `messages`, `citations` | LLM 最终回答 |

## 配置文件

| 文件 | 用途 |
| --- | --- |
| `.env` | API Key、Chroma 路径、模型配置 |
| `config/knowledge_permissions.json` | L1~L5 权限等级与知识库映射 |
| `config/rerank_config.json` | 重排权重与 MMR 参数 |
| `config/auto_merge_config.json` | Auto-Merging 合并阈值与层级 |

## 其他参考文档

- [nodes节点使用说明.md](nodes节点使用说明.md) — CLI 使用手册（所有节点的命令示例）
- [node_io_contract.md](node_io_contract.md) — 详细的节点输入输出契约与 Pipeline 架构
- [RAG运行说明.md](RAG运行说明.md) — RAG 完整运行说明
