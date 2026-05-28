# nodes 节点使用说明

本文档说明 `nodes` 目录下各处理节点的职责、输入输出和常用命令。所有命令默认在项目根目录执行：

```powershell
cd 自己的项目根目录
```

## 0. 环境配置

节点脚本默认在项目根目录执行，并从项目根目录查找 `.env`、`data/`、`MinerUResult/`、`ChromaDB/` 等路径。

推荐先准备：

```text
data/                         # 放入待处理 PDF
.env                          # 本地模型、MinerU、Chroma 配置
config/knowledge_permissions.json
config/rerank_config.json
```

`.env` 至少需要包含：

```env
MINERU_API_TOKEN=你的MinerUToken

LLM_API_KEY=你的百炼或OpenAI兼容Key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus

EMBEDDING_API_KEY=你的百炼EmbeddingKey
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4

CHROMA_PERSIST_DIR=ChromaDB
CHROMA_COLLECTION_NAME=document_chunks
CHROMA_SYNC_MODE=chunk
```

如果只验证非 LLM 节点，比如分片或权限配置检查，对应外部 API 配置可以暂时不填；一旦运行 MinerU、embedding、路由、改写、伪答案或最终回答节点，就需要相应密钥可用。

## 1. 文档处理节点

### 1.1 MinerUStandardReader.py

路径：

```text
nodes/documents/MinerUStandardReader.py
```

职责：

- 从项目根目录 `data` 读取 PDF。
- 调用 MinerU 解析 PDF。
- 解析结果保存到 `MinerUResult/<文档名>_<hash>/`。
- 每个 PDF 的结果相互隔离。

使用：

```powershell
.\.venv\Scripts\python.exe nodes\documents\MinerUStandardReader.py
```

运行后会要求输入：

```text
确认
```

输出：

```text
MinerUResult/<文档目录>/full.md
MinerUResult/<文档目录>/mineru_meta.json
MinerUResult/<文档目录>/mineru_result.zip
```

### 1.2 DataClean.py

路径：

```text
nodes/documents/DataClean.py
```

职责：

- 从 `MinerUResult/*/full.md` 读取 MinerU Markdown。
- 清洗 OCR/PDF 噪声。
- 保护 HTML 表格。
- 输出清洗后的 Markdown。

使用：

```powershell
.\.venv\Scripts\python.exe nodes\documents\DataClean.py
```

输出：

```text
MinerUResult/<文档目录>/DataCleaned.md
```

## 2. 分片节点

### 2.1 MarkDownChunk.py

路径：

```text
nodes/chunks/MarkDownChunk.py
```

职责：

- 基础 Markdown 分片。
- 识别标题、正文、HTML 表格、Markdown 表格、图片。
- 输出基础 chunks。

使用：

```powershell
.\.venv\Scripts\python.exe nodes\chunks\MarkDownChunk.py
```

输出：

```text
MinerUResult/<文档目录>/DataCleaned.chunks.json
```

### 2.2 HybridMarkdownChunk.py

路径：

```text
nodes/chunks/HybridMarkdownChunk.py
```

职责：

- 在基础分片上增加 RAG 检索元数据。
- 支持 Parent-Child。
- 支持 Small-to-Big。
- 支持 Table-Aware。
- 标记 `should_embed`。

使用：

```powershell
.\.venv\Scripts\python.exe nodes\chunks\HybridMarkdownChunk.py
```

输出：

```text
MinerUResult/<文档目录>/DataCleaned.hybrid_chunks.json
```

核心字段：

```text
document_id
document_name
retrieval_role
should_embed
parent_context
neighbor_chunk_ids
related_ref_chunk_ids
small_to_big_context_ids
embedding_text
```

## 3. 向量入库节点

### 3.1 ChromaBailianEmbedding.py

路径：

```text
nodes/embeddings/ChromaBailianEmbedding.py
```

职责：

- 读取 `DataCleaned.hybrid_chunks.json`。
- 只处理 `should_embed=True` 的 chunk。
- 调用百炼 embedding。
- 写入 Chroma。
- 支持 chunk 级增量同步。

使用：

```powershell
.\.venv\Scripts\python.exe nodes\embeddings\ChromaBailianEmbedding.py
```

输出：

```text
ChromaDB/
```

同步模式：

```env
CHROMA_SYNC_MODE=chunk
```

该模式下：

- 当前 chunk 已存在：跳过。
- 当前 chunk 新增：embedding 后写入。
- 当前 chunk 内容变化：生成新向量。
- 当前文档中旧 chunk 已不存在：从 Chroma 删除。

### 3.2 ChromaDocumentCleaner.py

路径：

```text
nodes/embeddings/ChromaDocumentCleaner.py
```

职责：

- 删除 Chroma 中指定文档的向量。
- 清理本地 `MinerUResult` 已不存在的文档向量。

清理本地已删除文档：

```powershell
.\.venv\Scripts\python.exe nodes\embeddings\ChromaDocumentCleaner.py
```

按 `document_id` 删除：

```powershell
.\.venv\Scripts\python.exe nodes\embeddings\ChromaDocumentCleaner.py `
  --document-id HR基础制度库_c4f9a1d879b8
```

按 `document_name` 删除：

```powershell
.\.venv\Scripts\python.exe nodes\embeddings\ChromaDocumentCleaner.py `
  --document-name HR基础制度库
```

## 4. 查询前处理节点

### 4.1 IntentRouter.py

路径：

```text
nodes/query/IntentRouter.py
```

职责：

- 判断问题是否需要检索。
- 判断意图类型。
- 判断目标文档。
- 生成 metadata filter 建议。

使用：

```powershell
.\.venv\Scripts\python.exe nodes\query\IntentRouter.py `
  --question "新员工入职当天需要完成哪些事项？"
```

### 4.2 QuestionRewriter.py

路径：

```text
nodes/query/QuestionRewriter.py
```

职责：

- 改写用户问题。
- 生成向量检索 query。
- 生成关键词检索 query。

使用：

```powershell
.\.venv\Scripts\python.exe nodes\query\QuestionRewriter.py `
  --question "忘记打卡找谁处理？"
```

### 4.3 PseudoAnswer.py

路径：

```text
nodes/query/PseudoAnswer.py
```

职责：

- 生成 HyDE 伪答案。
- 辅助向量召回。
- 伪答案不作为事实依据。

使用：

```powershell
.\.venv\Scripts\python.exe nodes\query\PseudoAnswer.py `
  --question "试用期绩效目标什么时候确认？"
```

### 4.4 LLMClient.py

路径：

```text
nodes/query/LLMClient.py
```

职责：

- 百炼 OpenAI 兼容 Chat 客户端。
- 支持 JSON 输出。
- 支持普通文本回答。

通常由其他节点调用，不需要单独运行。

## 5. 检索节点

### 5.1 ChromaRetriever.py

路径：

```text
nodes/retrieval/ChromaRetriever.py
```

职责：

- 检索编排入口，负责组装各检索子模块。
- 调用 `chroma_store.py` 做 Chroma 向量召回。
- 调用 `bm25_recall.py` 做 BM25 关键词召回。
- 多 query 合并去重。
- 调用 `nodes/rerank/Reranker.py` 做检索内初排和 MMR 去冗余。
- Table-Aware 加权。
- 调用 `context_expander.py` 做 Small-to-Big 上下文扩展。

拆分后的子模块：

```text
nodes/retrieval/chroma_store.py       # Chroma collection 获取和向量召回
nodes/retrieval/bm25_recall.py        # BM25 关键词召回
nodes/retrieval/context_expander.py   # Small-to-Big 上下文扩展
nodes/retrieval/filters.py            # 元数据过滤器
nodes/retrieval/retrieval_config.py   # 检索默认参数
nodes/retrieval/retrieval_utils.py    # 公共工具函数
```

基础使用：

```powershell
.\.venv\Scripts\python.exe nodes\retrieval\ChromaRetriever.py `
  --question "新员工入职当天需要完成哪些事项？" `
  --output retrieval_result.json
```

带文档过滤：

```powershell
.\.venv\Scripts\python.exe nodes\retrieval\ChromaRetriever.py `
  --question "新员工入职当天需要完成哪些事项？" `
  --document-name "HR基础制度库" `
  --output retrieval_result.json
```

带多 query：

```powershell
.\.venv\Scripts\python.exe nodes\retrieval\ChromaRetriever.py `
  --question "忘记打卡找谁处理？" `
  --query "考勤异常处理流程" `
  --keyword-query "忘记打卡 审批 处理人 考勤异常" `
  --output retrieval_result.json
```

表格优先：

```powershell
.\.venv\Scripts\python.exe nodes\retrieval\ChromaRetriever.py `
  --question "薪酬等级对应规则是什么？" `
  --chunk-type table_chunk `
  --output retrieval_result.json
```

## 6. 重排节点

### 6.1 Reranker.py

路径：

```text
nodes/rerank/Reranker.py
```

职责：

- 对 `ChromaRetriever.py` 输出的 `RetrievalResult` 做二阶段规则重排。
- 融合向量分、BM25 分、词面覆盖、标题路径匹配、来源加权和表格加权。
- 支持 MMR 去冗余。
- 输出 `rerank_score`、`rerank_features`、`rerank_reason`，方便调试。
- 默认读取 `config/rerank_config.json`，可通过参数覆盖。

单独使用：

```powershell
.\.venv\Scripts\python.exe nodes\rerank\Reranker.py `
  --input retrieval_result.json `
  --config config\rerank_config.json `
  --final-top-k 8 `
  --output retrieval_reranked.json
```

在完整 RAG 中，重排位置是：

```text
PermissionGuard 检索后过滤
→ Reranker 二阶段重排
→ ContextCompressor 上下文压缩
```

## 7. 权限校验节点

### 7.1 PermissionGuard.py

路径：

```text
nodes/auth/PermissionGuard.py
```

职责：

- 根据模拟权限等级 L1-L5 判断可访问知识库。
- 从 `config/knowledge_permissions.json` 读取权限配置。
- 支持按 `document_names` 精确匹配和 `document_patterns` 通配符匹配。
- 在完整 RAG 中用于检索前过滤和检索后兜底过滤。

查看某个权限等级可访问内容：

```powershell
.\.venv\Scripts\python.exe nodes\auth\PermissionGuard.py `
  --permission-level L3
```

扩展权限：

```text
修改 config/knowledge_permissions.json，新增 knowledge_bases 条目。
```

## 8. 上下文压缩节点

### 8.1 ContextCompressor.py

路径：

```text
nodes/context/ContextCompressor.py
```

职责：

- 读取 `ChromaRetriever.py` 输出。
- 合并去重上下文。
- 控制上下文长度。
- 保留引用编号。
- 输出 `context_blocks` 和 `citations`。

使用：

```powershell
.\.venv\Scripts\python.exe nodes\context\ContextCompressor.py `
  --input retrieval_result.json `
  --output compressed_context.json
```

## 9. Prompt 组装节点

### 9.1 PromptBuilder.py

路径：

```text
nodes/prompt/PromptBuilder.py
```

职责：

- 读取压缩上下文。
- 组装最终回答 Prompt。
- 输出 OpenAI/百炼兼容 `messages`。

使用：

```powershell
.\.venv\Scripts\python.exe nodes\prompt\PromptBuilder.py `
  --question "新员工入职当天需要完成哪些事项？" `
  --context compressed_context.json `
  --output final_prompt.json
```

带前处理结果：

```powershell
.\.venv\Scripts\python.exe nodes\prompt\PromptBuilder.py `
  --question "新员工入职当天需要完成哪些事项？" `
  --context compressed_context.json `
  --route route.json `
  --rewrite rewrite.json `
  --pseudo-answer pseudo_answer.json `
  --output final_prompt.json
```

## 10. 推荐完整执行顺序

首次构建知识库：

```powershell
.\.venv\Scripts\python.exe rags\IngestPipeline.py
```

只查看将处理哪些 PDF，不上传、不入库：

```powershell
.\.venv\Scripts\python.exe rags\IngestPipeline.py --dry-run
```

一键脚本会执行：

```text
MinerUStandardReader -> DataClean -> HybridMarkdownChunk -> ChromaBailianEmbedding
```

增量说明：

- MinerU 使用文件内容 hash + 解析参数缓存，相同内容默认不重复上传。
- Chroma embedding 使用 chunk 级增量，已存在 chunk 默认跳过，更新文档时覆盖变化的 chunk。

分步构建知识库：

```powershell
.\.venv\Scripts\python.exe nodes\documents\MinerUStandardReader.py
.\.venv\Scripts\python.exe nodes\documents\DataClean.py
.\.venv\Scripts\python.exe nodes\chunks\HybridMarkdownChunk.py
.\.venv\Scripts\python.exe nodes\embeddings\ChromaBailianEmbedding.py
```

单次检索和 Prompt 组装：

```powershell
.\.venv\Scripts\python.exe nodes\retrieval\ChromaRetriever.py `
  --question "新员工入职当天需要完成哪些事项？" `
  --output retrieval_result.json

.\.venv\Scripts\python.exe nodes\context\ContextCompressor.py `
  --input retrieval_result.json `
  --output compressed_context.json

.\.venv\Scripts\python.exe nodes\prompt\PromptBuilder.py `
  --question "新员工入职当天需要完成哪些事项？" `
  --context compressed_context.json `
  --output final_prompt.json
```

## 11. 完整 RAG 编排

虽然本文档聚焦 `nodes`，但项目已提供完整编排：

```text
rags/RAGPipeline.py
```

单次运行：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --permission-level L2 `
  --rerank-config config\rerank_config.json `
  --question "新员工入职当天需要完成哪些事项？"
```

运行模式：

```text
standard # 默认模式：IntentRouter -> QuestionRewriter -> PseudoAnswer -> 检索 -> 回答
fast     # 快速模式：跳过路由/改写/伪答案，只用原问题检索
```

快速模式：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --fast `
  --permission-level L2 `
  --question "新员工入职当天需要完成哪些事项？"
```

也可以显式指定：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --mode fast `
  --permission-level L2 `
  --question "新员工入职当天需要完成哪些事项？"
```

指定日志文件：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --permission-level L2 `
  --log-file logs `
  --question "新员工入职当天需要完成哪些事项？"
```

默认会写入类似：

```text
logs/2026/05/2026-05-28.jsonl
```

关闭日志：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --no-log `
  --question "新员工入职当天需要完成哪些事项？"
```

循环模式：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py
```

未传 `--permission-level` 进入循环模式时，会先提示选择模拟权限等级。单次运行未传时默认 L1。

退出循环：

```text
q
```

完整链路：

```text
standard:
IntentRouter
→ QuestionRewriter
→ PseudoAnswer
→ PermissionGuard
→ ChromaRetriever
→ PermissionGuard
→ Reranker
→ ContextCompressor
→ PromptBuilder
→ LLM 最终回答

fast:
PermissionGuard
→ ChromaRetriever
→ PermissionGuard
→ Reranker
→ ContextCompressor
→ PromptBuilder
→ LLM 最终回答
```

## 12. 注意事项

- 不建议使用 PowerShell 的 `>` 保存 JSON，可能写成 UTF-16。
- 优先使用脚本自带 `--output` 参数。
- `PseudoAnswer` 只用于检索增强，不能作为事实依据。
- `fast` 模式返回和日志中都会标记 `pipeline_mode=fast`，便于和 `standard` 模式区分。
- 最终回答必须基于 `ContextCompressor` 产生的上下文。
- 权限配置依赖 Chroma metadata 中的 `document_name`，真实 PDF 名称建议写入 `config/knowledge_permissions.json` 的 `document_names`。
- 默认日志按上海时区年月日分文件，形如 `logs/YYYY/MM/YYYY-MM-DD.jsonl`，每行一条 JSON，包含 `ts`、`timezone=Asia/Shanghai`、`trace_id`、`stage`、`elapsed_ms`、`status` 和关键计数。
- 删除 PDF 后，如果也删除了对应 `MinerUResult` 目录，需要运行 `ChromaDocumentCleaner.py` 清理向量库。
