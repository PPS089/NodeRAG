# QuestionRewriter — 问题改写节点

## 概述

`QuestionRewriter` 通过 LLM 将用户原始问题改写为**更适合向量检索和关键词检索**的表达形式，同时保留原意。生成多组检索 query 和关键词组合。

**路径**：`nodes/query/QuestionRewriter.py`

## 在链路中的位置

```
IntentRouter → QuestionRewriter → PseudoAnswer → 检索 → ...
```

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 问题改写 | 补全简称、口语、省略主语，生成完整清晰的检索问句 |
| 多 Query 生成 | 生成 1~3 条向量检索 query（`search_queries`） |
| 关键词提取 | 生成 1~5 组关键词组合（`keyword_queries`） |
| 实体识别 | 识别制度名、产品名、角色名等实体 |
| 时间约束 | 提取时间条件（如"2024年"、"上季度"） |
| 改写约束 | `must_not_change` 标记不能改写或推断的关键约束 |

## 输入 / 输出

### 输入

- `question: str` — 用户原始问题

### 输出

```json
{
  "original_question": "忘记打卡找谁处理？",
  "rewritten_question": "员工忘记打卡后应该联系谁处理考勤异常？补打卡的审批流程是什么？",
  "search_queries": [
    "忘记打卡 考勤异常 处理流程",
    "补打卡 审批 联系人"
  ],
  "keyword_queries": [
    "忘记打卡 审批 处理人 考勤异常",
    "补打卡 流程 考勤"
  ],
  "entities": ["考勤制度", "HR"],
  "time_constraints": [],
  "must_not_change": ["具体审批人姓名"]
}
```

## 规则约束

| 规则 | 说明 |
| --- | --- |
| 不编造事实 | 不能添加原始问题中不存在的信息 |
| 不改变问题 | 不能把问题改成另一个问题 |
| 多 query 上限 | `search_queries` ≤ 3；`keyword_queries` ≤ 5 |
| 补全口语 | 简称、口语、省略主语要补全为适合检索的表达 |

## CLI 使用

```powershell
.\.venv\Scripts\python.exe nodes\query\QuestionRewriter.py --question "忘记打卡找谁处理？"
```

## 上下游契约

- **上游**：用户原始问题 + 可选 `IntentRouter` 的路由结果
- **下游**：写入 `PipelineContext.rewrite_result`；`ChromaRetriever` 使用 `search_queries` 和 `keyword_queries`
