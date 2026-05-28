# NodeRAG

NodeRAG 是一个本地脚本式 RAG 项目，用于把 `data/` 目录下的 PDF 文档解析、清洗、分片、向量化后写入 Chroma，并通过权限控制、检索、重排、上下文压缩和 Prompt 组装完成知识库问答。

## 核心能力

- PDF 批量解析：使用 MinerU 解析 `data/*.pdf`。
- 增量入库：相同 PDF 内容不重复上传 MinerU，已存在 chunk 不重复 embedding。
- Markdown 清洗：清理 OCR/PDF 噪声，保留表格结构。
- Hybrid 分片：支持 Parent-Child、Small-to-Big、Table-Aware。
- Chroma 向量库：使用百炼 embedding 模型写入 Chroma。
- 检索增强：向量召回、BM25、初排、MMR、二阶段重排。
- 权限控制：按 L1-L5 模拟知识库访问权限。
- 两种问答模式：`standard` 完整前处理，`fast` 只用原问题检索。
- 结构化日志：按上海时区写入 `logs/YYYY/MM/YYYY-MM-DD.jsonl`。

## 项目结构

```text
NodeRAG
|--data                         # 原始 PDF
|--config                       # 权限、重排等配置
|--docs                         # 项目说明文档
|--MinerUResult                 # MinerU/清洗/分片结果，运行生成
|--nodes                        # 单能力节点
|--rags                         # 完整 Pipeline 编排
|--utils                        # 工具函数
|--ChromaDB                     # Chroma 持久化目录，运行生成
|--logs                         # JSONL 日志，运行生成
```

更详细结构见：

```text
docs/项目结构说明1.txt
```

## 环境要求

- Python 3.11+
- 可访问 MinerU API
- 可访问百炼/OpenAI-compatible LLM 和 embedding API
- ChromaDB 本地持久化

安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

如果使用 `uv`：

```powershell
uv sync
```

## 环境变量

在项目根目录创建 `.env`，至少配置：

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
EMBEDDING_BATCH_SIZE=16
```

实际变量名以 `nodes/query/LLMClient.py` 和 `nodes/embeddings/ChromaBailianEmbedding.py` 为准。

## 一键构建知识库

把 PDF 放入：

```text
data/
```

先预览将处理的 PDF：

```powershell
.\.venv\Scripts\python.exe rags\IngestPipeline.py --dry-run
```

执行完整入库：

```powershell
.\.venv\Scripts\python.exe rags\IngestPipeline.py
```

完整流程：

```text
PDF
-> MinerUStandardReader
-> DataClean
-> HybridMarkdownChunk
-> ChromaBailianEmbedding
-> ChromaDB
```

增量规则：

- MinerU：按文件内容 hash + 解析参数缓存，相同内容默认不重复上传。
- Cleaning/Chunking：会基于现有 `MinerUResult` 重新生成清洗和分片结果。
- Embedding：按 chunk id 增量入库，已存在 chunk 跳过；`CHROMA_SYNC_MODE=chunk` 会删除同文档过期 chunk。

常用参数：

```powershell
# 跳过 MinerU，复用已有 MinerUResult
.\.venv\Scripts\python.exe rags\IngestPipeline.py --skip-mineru

# 强制重新解析 PDF
.\.venv\Scripts\python.exe rags\IngestPipeline.py --force-reparse

# 只跑到分片，不入向量库
.\.venv\Scripts\python.exe rags\IngestPipeline.py --skip-embedding
```

## 运行 RAG 问答

默认完整模式：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --permission-level L2 `
  --question "新员工入职当天需要完成哪些事项？"
```

快速模式：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --fast `
  --permission-level L2 `
  --question "新员工入职当天需要完成哪些事项？"
```

循环模式：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py
```

退出循环：

```text
q
```

## RAG 模式

`standard` 是默认模式：

```text
IntentRouter
-> QuestionRewriter
-> PseudoAnswer
-> PermissionGuard
-> ChromaRetriever
-> PermissionGuard
-> Reranker
-> ContextCompressor
-> PromptBuilder
-> LLM
```

`fast` 模式只用原问题检索：

```text
PermissionGuard
-> ChromaRetriever
-> PermissionGuard
-> Reranker
-> ContextCompressor
-> PromptBuilder
-> LLM
```

切换方式：

```powershell
--mode standard
--mode fast
--fast
```

`--skip-preprocess` 保留为兼容旧参数，等价于 `--fast`。

## 权限配置

权限配置文件：

```text
config/knowledge_permissions.json
```

权限等级：

```text
L1 < L2 < L3 < L4 < L5
```

等级继承：

- L1 只能访问 L1。
- L3 可访问 L1-L3。
- L5 可访问全部已配置知识库。

查看某个权限等级可访问内容：

```powershell
.\.venv\Scripts\python.exe nodes\auth\PermissionGuard.py --permission-level L3
```

## 重排配置

重排配置文件：

```text
config/rerank_config.json
```

可配置内容：

- `rerank_top_n`
- `final_top_k`
- `use_mmr`
- `table_aware`
- `mmr_lambda`
- `weights.vector`
- `weights.bm25`
- `weights.lexical`
- `weights.title_path`
- `bonuses.multi_source`
- `bonuses.table_chunk`

单独运行重排：

```powershell
.\.venv\Scripts\python.exe nodes\rerank\Reranker.py `
  --input retrieval_result.json `
  --config config\rerank_config.json `
  --output retrieval_reranked.json
```

## 日志

默认日志位置：

```text
logs/YYYY/MM/YYYY-MM-DD.jsonl
```

日志使用上海时区：

```json
{
  "ts": "2026-05-28T13:03:10.240129+08:00",
  "timezone": "Asia/Shanghai",
  "trace_id": "...",
  "event": "stage_end"
}
```

指定日志根目录或文件：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --log-file logs `
  --question "新员工入职当天需要完成哪些事项？"
```

关闭日志：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py --no-log
```

## 常用节点命令

```powershell
# MinerU 解析
.\.venv\Scripts\python.exe nodes\documents\MinerUStandardReader.py

# 数据清洗
.\.venv\Scripts\python.exe nodes\documents\DataClean.py

# Hybrid 分片
.\.venv\Scripts\python.exe nodes\chunks\HybridMarkdownChunk.py

# Chroma 入库
.\.venv\Scripts\python.exe nodes\embeddings\ChromaBailianEmbedding.py

# 检索
.\.venv\Scripts\python.exe nodes\retrieval\ChromaRetriever.py `
  --question "新员工入职当天需要完成哪些事项？" `
  --output retrieval_result.json

# 上下文压缩
.\.venv\Scripts\python.exe nodes\context\ContextCompressor.py `
  --input retrieval_result.json `
  --output compressed_context.json

# Prompt 组装
.\.venv\Scripts\python.exe nodes\prompt\PromptBuilder.py `
  --question "新员工入职当天需要完成哪些事项？" `
  --context compressed_context.json `
  --output final_prompt.json
```

## 文档索引

```text
docs/项目结构说明1.txt       # 项目结构
docs/nodes节点使用说明.md    # 节点说明和常用命令
docs/node_io_contract.md     # 节点输入输出契约
docs/知识库入库流程.md       # PDF 到 Chroma 的一键入库流程
docs/RAG运行说明.md          # RAG 单次、循环、standard/fast 模式
docs/权限配置说明.md         # 权限等级、权限码、扩展方式
docs/日志说明.md             # JSONL 日志格式、上海时区、排查方法
docs/分片策略.txt            # 分片策略说明
docs/重排策略.txt            # 重排策略说明
docs/删除向量库数据.txt       # 删除 PDF 后清理向量库
```

## 注意事项

- 不建议用 PowerShell 的 `>` 保存 JSON，可能写成 UTF-16；优先使用脚本自带 `--output`。
- `.env`、`ChromaDB/`、`MinerUResult/`、`logs/` 都是本地运行产物或敏感配置，不应提交。
- 删除 PDF 不会自动删除 Chroma 中历史向量，需要运行 `ChromaDocumentCleaner.py`。
- 如果问答慢，优先使用 `--fast` 定位和提速。
