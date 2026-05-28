# RAG 运行说明

本文档说明如何运行完整问答 Pipeline，以及 `standard` 和 `fast` 两种模式的区别。

## 1. 前置条件

运行 RAG 前需要先完成知识库入库：

```powershell
.\.venv\Scripts\python.exe rags\IngestPipeline.py
```

同时确保 `.env` 中的 LLM、embedding、Chroma 配置可用。

## 2. 单次问答

默认完整模式：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --permission-level L2 `
  --question "新员工入职当天需要完成哪些事项？"
```

输出完整 trace 到文件：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --permission-level L2 `
  --question "新员工入职当天需要完成哪些事项？" `
  --output rag_result.json
```

## 3. 循环问答

不传 `--question` 时进入循环模式：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py
```

循环模式会提示选择模拟权限等级。输入问题后连续问答，输入以下内容退出：

```text
q
```

## 4. standard 模式

`standard` 是默认模式，适合需要更高召回质量的问题。

链路：

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

显式指定：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --mode standard `
  --permission-level L2 `
  --question "新员工入职当天需要完成哪些事项？"
```

特点：

- 会调用 LLM 做意图路由。
- 会调用 LLM 做问题改写。
- 会调用 LLM 生成 HyDE 伪答案。
- 检索输入更丰富，但延迟更高。

## 5. fast 模式

`fast` 模式跳过路由、改写和伪答案，只用原问题检索，适合日常快速问答或定位性能问题。

链路：

```text
PermissionGuard
-> ChromaRetriever
-> PermissionGuard
-> Reranker
-> ContextCompressor
-> PromptBuilder
-> LLM
```

快捷启用：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --fast `
  --permission-level L2 `
  --question "新员工入职当天需要完成哪些事项？"
```

显式指定：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --mode fast `
  --permission-level L2 `
  --question "新员工入职当天需要完成哪些事项？"
```

兼容旧参数：

```text
--skip-preprocess
```

该参数等价于 `--fast`，新命令优先使用 `--fast` 或 `--mode fast`。

## 6. 权限等级

可选权限等级：

```text
L1 < L2 < L3 < L4 < L5
```

示例：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --permission-level L4 `
  --question "薪酬等级对应规则是什么？"
```

如果问题显式指定了当前权限不可访问的文档，Pipeline 会提前返回权限不足结果；检索后还会再次过滤，防止未授权 chunk 进入上下文。

## 7. 检索控制参数

限制文档：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --permission-level L2 `
  --document-name "HR基础制度库_正式版" `
  --question "新员工入职当天需要完成哪些事项？"
```

限制 chunk 类型：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --permission-level L4 `
  --chunk-type table_chunk `
  --question "薪酬等级对应规则是什么？"
```

调整召回数量：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --top-k 8 `
  --per-query-k 12 `
  --question "销售价格审批规则是什么？"
```

## 8. 重排控制参数

默认使用 `config/rerank_config.json`：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --rerank-config config\rerank_config.json `
  --question "新员工入职当天需要完成哪些事项？"
```

覆盖参与重排候选数量：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --rerank-top-n 16 `
  --question "新员工入职当天需要完成哪些事项？"
```

关闭二阶段重排：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py `
  --no-second-stage-rerank `
  --question "新员工入职当天需要完成哪些事项？"
```

## 9. 日志和 trace

默认日志：

```text
logs/YYYY/MM/YYYY-MM-DD.jsonl
```

日志使用上海时区，字段包含：

```text
ts
timezone=Asia/Shanghai
trace_id
pipeline_mode
stage
elapsed_ms
status
metrics
```

关闭日志：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py --no-log --question "测试问题"
```

打印完整 trace：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py --show-trace --question "测试问题"
```

保存每轮 trace：

```powershell
.\.venv\Scripts\python.exe rags\RAGPipeline.py --trace-dir traces
```

## 10. 性能建议

回答慢时优先判断慢在哪个阶段：

- 使用 `--fast` 跳过路由、改写、伪答案，确认是否是前处理 LLM 调用慢。
- 查看日志中各 stage 的 `elapsed_ms`。
- 如果 `stage_retrieve` 慢，检查 Chroma 规模和召回参数。
- 如果 `stage_answer` 慢，通常是最终 LLM 生成慢。
- 如果首次问答慢，可能包含模型服务冷启动或 Chroma 初始化开销。
