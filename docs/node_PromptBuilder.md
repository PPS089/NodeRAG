# PromptBuilder — Prompt 组装节点

## 概述

`PromptBuilder` 是 RAG 查询链路的**最后一个预处理节点**，将压缩上下文组装为 OpenAI/百炼兼容的 `messages` 格式，并整合路由、改写、伪答案等前处理结果。

**路径**：`nodes/prompt/PromptBuilder.py`

## 在链路中的位置

```
... → ContextCompressor → PromptBuilder → LLM 最终回答
```

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 上下文格式化 | 将 `context_blocks` 格式化为 `[C1] 文档: xxx; 章节: xxx` 带引用编号的文本 |
| 前处理结果整合 | 将路由、改写、伪答案作为辅助信息嵌入 Prompt |
| 引用管理 | 保持 citations 列表，供下游最终回答引用 |
| 契约校验 | 输入校验 `CompressedContext`，输出校验 `PromptPayload` |

## 输入 / 输出

### 输入

```python
build_prompt(
    question,           # str — 用户原始问题
    context_result,     # dict — ContextCompressor 输出
    route_result,       # dict | None — IntentRouter 输出
    rewrite_result,     # dict | None — QuestionRewriter 输出
    pseudo_answer_result, # dict | None — PseudoAnswer 输出
) → dict
```

### 输出（PromptPayload）

```json
{
  "question": "新员工入职当天需要完成哪些事项？",
  "messages": [
    {
      "role": "system",
      "content": "你是企业知识库 RAG 问答助手。\n\n回答规则：\n1. 只能基于提供的【检索上下文】回答。\n..."
    },
    {
      "role": "user",
      "content": "【用户问题】\n新员工入职当天需要完成哪些事项？\n\n【检索上下文】\n[C1] 文档: HR基础制度库; ...\n---\n[C2] 文档: ...\n\n【输出要求】\n- 先给出直接答案。\n- 再列出依据，引用 [C1] 这样的证据编号。\n- 如果依据不足，说明缺失哪些资料。"
    }
  ],
  "citations": [
    {"citation_id": "C1", "document_name": "HR基础制度库", "title_path": [...], ...}
  ],
  "context_stats": {
    "input_hit_count": 8,
    "selected_block_count": 5,
    "context_chars": 4520
  }
}
```

## System Prompt 规则

| 规则 | 说明 |
| --- | --- |
| 仅基于上下文 | 只能基于提供的【检索上下文】回答 |
| 不知即答不知 | 上下文不足时明确说明"当前知识库资料不足以确认" |
| 不编造 | 不能编造制度、价格、预算、合同条款、日期或比例 |
| 引用证据 | 回答中必须引用 `[C1]`、`[C2]` 格式的证据编号 |
| 表格处理 | 表格内容按字段或行解释，必要时用列表呈现 |
| 测试数据声明 | 上下文声明为测试数据时，回答要说明这是测试资料 |

## 辅助信息整合

`build_auxiliary_text()` 将前处理结果作为 Prompt 的一部分：

```
【路由结果】
{route_result JSON}

【问题改写】
{rewrite_result JSON}

【伪答案检索辅助】
{pseudo_answer JSON + "伪答案仅用于检索辅助，不可作为事实依据。"}
```

**注意**：伪答案在嵌入时会追加 `pseudo_answer_note` 免责声明。

## CLI 使用

```powershell
# 基础使用
.\.venv\Scripts\python.exe nodes\prompt\PromptBuilder.py --question "新员工入职当天需要完成哪些事项？" --context compressed_context.json --output final_prompt.json

# 带前处理结果
.\.venv\Scripts\python.exe nodes\prompt\PromptBuilder.py --question "..." --context compressed_context.json --route route.json --rewrite rewrite.json --pseudo-answer pseudo_answer.json --output final_prompt.json
```

## 输出使用

PromptPayload 的 `messages` 可以直接传给 `OpenAICompatibleChatClient.chat_messages()` 生成最终回答。

## 上下游契约

- **上游**：`ContextCompressor` 输出（`CompressedContext`），需通过 `validate_compressed_context()` 校验
- **下游**：LLM 最终回答（`chat_messages(messages)`）
- 输出通过 `validate_prompt_payload()` 校验（`question`, `messages`, `citations` 必要字段）
