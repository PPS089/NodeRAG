# LLMClient — 百炼 OpenAI 兼容 Chat 客户端

## 概述

`LLMClient` 是 NodeRAG 中**所有 LLM 调用节点的底层客户端**，封装百炼（阿里云 DashScope）的 OpenAI 兼容 Chat Completions API。IntentRouter、QuestionRewriter、PseudoAnswer、PromptBuilder 的回答生成都依赖此模块。

**路径**：`nodes/query/LLMClient.py`

## 核心能力

| 能力 | 说明 |
| --- | --- |
| JSON 输出 | `chat_json()` 调用 LLM 并解析 JSON 响应（支持 `response_format: json_object`） |
| 文本输出 | `chat_text()` 调用 LLM 返回纯文本 |
| 消息列表 | `chat_messages()` 支持自定义 messages 数组 |
| JSON 容错 | `parse_json_object()` 先尝试 `json.loads`，失败后用正则提取 `{...}` |
| API Key 校验 | `validate_ascii_secret()` 确保 Key 不含中文等非 ASCII 字符 |
| 可用文档发现 | `find_available_documents()` 扫描 MinerUResult 目录 |

## 核心类

### `OpenAICompatibleChatClient`

```python
class OpenAICompatibleChatClient:
    def __init__(api_key, base_url, model, timeout)
    def chat_json(system_prompt, user_prompt, temperature) → dict
    def chat_text(system_prompt, user_prompt, temperature) → str
    def chat_messages(messages, temperature) → str
```

### 环境变量读取优先级

| 配置项 | 环境变量（优先级从高到低） | 默认值 |
| --- | --- | --- |
| API Key | `LLM_API_KEY` | —（必填） |
| Base URL | `LLM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Model | `LLM_MODEL_ID` | `qwen3-max` |
| Timeout | `LLM_TIMEOUT` | `60` |

## `chat_json()` 的 JSON 解析流程

1. 发送请求时设置 `response_format: {"type": "json_object"}`
2. 尝试 `json.loads(content)` 直接解析
3. 如果失败，用正则 `re.search(r"\{.*\}", text, re.S)` 提取 JSON 对象
4. 如果仍然失败，抛出 `ValueError`

## 辅助函数

| 函数 | 说明 |
| --- | --- |
| `parse_json_object(text)` | 容错 JSON 解析 |
| `find_available_documents()` | 扫描 MinerUResult 目录，返回文档名列表 |
| `documents_prompt_text(documents)` | 格式化文档列表为 Prompt 文本 |
| `build_question_parser(description)` | 构建标准的 CLI 参数解析器（`--question`, `--temperature`） |
| `get_question(args)` | 从 CLI args 提取 question（支持 `--question` 和位置参数） |
| `print_json(data)` | 美化打印 JSON |

## 调用方

| 节点 | 调用方式 |
| --- | --- |
| `IntentRouter` | `OpenAICompatibleChatClient().chat_json(system_prompt, user_prompt)` |
| `QuestionRewriter` | 同上 |
| `PseudoAnswer` | 同上（temperature=0.2） |
| `PromptBuilder` (最终回答) | `chat_messages(messages)` |
| `GradeAndRewrite` | `chat_json()` + `chat_text()` |

## 注意事项

- API Key 必须是纯 ASCII，不能包含中文或特殊字符（`validate_ascii_secret` 检查）
- JSON 模式下 LLM 仍可能返回包裹在 markdown 代码块中的 JSON，`parse_json_object` 做了容错
- `chat_json` 不保证返回字段完整性——调用方需要做兜底（如 `result.setdefault("question", question)`）
