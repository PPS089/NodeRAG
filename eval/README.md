# RAG 检索召回评测

本目录用于评估 `nodes/retrieval/ChromaRetriever.py` 的检索质量。

## 评测集

```text
eval/rag_retrieval_eval.jsonl
```

每行一个问题，字段说明：

- `id`：用例编号。
- `question`：用户问题。
- `expected_documents`：期望命中的 `document_name`。
- `expected_keywords`：期望上下文中出现的关键事实。
- `category`：问题类型。
- `required_permission_level`：完整回答所需最低权限。

## 运行

只校验评测集格式：

```powershell
.\.venv\Scripts\python.exe eval\EvaluateRetrieval.py --dry-run
```

快速评估前 3 条：

```powershell
.\.venv\Scripts\python.exe eval\EvaluateRetrieval.py --limit 3
```

完整评估：

```powershell
.\.venv\Scripts\python.exe eval\EvaluateRetrieval.py `
  --output eval\retrieval_eval_result.json
```

## 指标

- `doc_hit@K`：TopK 中是否命中任一期望文档。
- `doc_recall@K`：TopK 命中的期望文档比例，适合多文档问题。
- `keyword_recall@K`：TopK 上下文命中的期望关键词比例。
- `mrr`：第一个期望文档排名的倒数。

当前评测重点是召回，不评估最终 LLM 答案质量。
