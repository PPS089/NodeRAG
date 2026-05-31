# IntentRouter — 意图路由节点

## 概述

`IntentRouter` 是 RAG 查询链路的**第一个预处理节点**，通过 LLM 判断用户问题的意图类型、是否需要知识库检索、应该查询哪些文档，并生成 metadata filter 建议。

**路径**：`nodes/query/IntentRouter.py`

## 在链路中的位置

```
用户问题 → IntentRouter → QuestionRewriter → PseudoAnswer → ... → 检索 → 回答
```

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 检索判断 | 判断问题是否需要检索知识库（`needs_retrieval`） |
| 意图分类 | 将问题归类到 9 种预定义意图 |
| 文档定位 | 从可用文档列表中推荐目标文档 |
| Filter 生成 | 生成 `document_name` 和 `chunk_type` 的 metadata filter |

## 意图类型

| 意图 | 说明 |
| --- | --- |
| `policy_lookup` | 制度查询 |
| `contract_lookup` | 合同查询 |
| `price_lookup` | 价格查询 |
| `budget_lookup` | 预算查询 |
| `product_lookup` | 产品查询 |
| `hr_lookup` | HR 相关查询 |
| `general_qa` | 通用问答 |
| `chitchat` | 闲聊 |
| `out_of_scope` | 超出范围 |

## 输入 / 输出

### 输入

- `question: str` — 用户原始问题
- `temperature: float`（默认 0.1）— LLM 采样温度

### 输出

```json
{
  "question": "新员工入职当天需要完成哪些事项？",
  "needs_retrieval": true,
  "intent": "hr_lookup",
  "target_documents": ["HR基础制度库"],
  "metadata_filters": {
    "document_name": ["HR基础制度库"],
    "chunk_type": ["text_chunk", "table_chunk"]
  },
  "query_type": "procedure",
  "confidence": 0.92,
  "reason": "问题询问HR入职流程，需要检索HR制度库"
}
```

### 输出字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `needs_retrieval` | `bool` | 是否需要检索；闲聊和纯通用问题为 `false` |
| `intent` | `str` | 意图分类（9 种之一） |
| `target_documents` | `list[str]` | 推荐检索的文档名；不确定时为空数组 |
| `metadata_filters` | `dict` | 包含 `document_name` 和 `chunk_type` 的过滤建议 |
| `query_type` | `str` | `fact` / `procedure` / `comparison` / `summary` / `table_lookup` / `definition` / `other` |
| `confidence` | `float` | 0~1 置信度 |
| `reason` | `str` | 一句话路由依据 |

## 实现细节

- 使用 `OpenAICompatibleChatClient.chat_json()` 调用 LLM
- System Prompt 包含完整的输出 JSON schema 和规则
- User Prompt 包含当前 `MinerUResult` 下的可用文档列表（`find_available_documents()`）
- 输出兜底：`result.setdefault("question", question)` 保证原始问题不丢失

## CLI 使用

```powershell
.\.venv\Scripts\python.exe nodes\query\IntentRouter.py --question "新员工入职当天需要完成哪些事项？"
```

## 上下游契约

- **上游**：用户原始问题（`PipelineContext.question`）
- **下游**：写入 `PipelineContext.route_result`；`PromptBuilder` 可选接收路由结果
- `needs_retrieval=false` 时 Pipeline 会提前结束（`stop_pipeline`）
