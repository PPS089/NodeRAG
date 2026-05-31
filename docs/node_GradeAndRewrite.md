# GradeAndRewrite — 相关性评估与回流重写节点

## 概述

`GradeAndRewrite` 实现 **Grade → Rewrite 回流循环**：评估检索结果与问题的相关性，如果不通过则触发问题改写 + 扩展查询，支持重新检索。

**路径**：`nodes/retrieval/GradeAndRewrite.py`

## 核心流程

```
1. 评估检索结果相关性（LLM grade）
2. 如果通过 → 返回
3. 如果不通过 → 选择改写策略（step_back / hyde / simple）
4. 生成扩展查询
5. 由调用方（RAGPipeline）执行重新检索
```

## 三种改写策略

| 策略 | 适用场景 | 生成内容 |
| --- | --- | --- |
| `step_back` | 问题包含具体名称/日期/数值，需要先理解通用概念 | 退步问题 + 退步答案 |
| `hyde` | 问题模糊、概念性、需要解释或定义 | 假设性文档 |
| `simple` | 问题已足够清晰，无需扩展 | 仅原问题 |

### Step-back 示例

```
用户问题: "2023年Q3的销售收入是多少"
退步问题: "公司季度销售收入如何计算和汇报"
退步答案: "季度销售收入通常按合同签订金额或回款金额统计..."
```

### HyDE 示例

```
用户问题: "什么是 Scrum？"
假设性文档: "Scrum 是一种敏捷开发框架，通过固定时长的 Sprint 迭代交付产品增量..."
```

## 相关性评估

```
LLM 二分类: yes（相关）/ no（不相关）
评估 5 个 top hits 的前 500 字符
```

输出：
```json
{
  "grade_score": "yes",
  "passed": true,
  "relevant_count": 5,
  "total_count": 8
}
```

## `grade_and_rewrite_loop()` 回流循环

```
attempt 1: grade → 不通过 → rewrite
attempt 2: grade → 不通过 → rewrite
attempt 3: grade → 不通过 → 返回 failed
```

默认 `max_retry = 2`（最多一次重试）。

**注意**：此函数仅负责评估和改写，不执行重新检索。重新检索由 `RAGPipeline` 在检测到 grade 不通过后调用 `stage_retrieve` 重新执行。

## 核心函数

```python
grade_documents(question, hits) → dict
grade_retrieval_result(retrieval_result, question) → dict
choose_rewrite_strategy(question) → str
generate_step_back_question(question) → str
answer_step_back_question(step_back_question) → str
generate_hypothetical_document(question) → str
rewrite_question(question, strategy) → dict
grade_and_rewrite_loop(question, retrieval_result, max_retry) → dict
```

## CLI 使用

```powershell
# 仅评估
.\.venv\Scripts\python.exe nodes\retrieval\GradeAndRewrite.py retrieval_result.json --mode grade

# 仅改写
.\.venv\Scripts\python.exe nodes\retrieval\GradeAndRewrite.py retrieval_result.json --mode rewrite --question "用户问题"

# 回流循环
.\.venv\Scripts\python.exe nodes\retrieval\GradeAndRewrite.py retrieval_result.json --mode loop --max-retry 2
```

## 注意

- 当前 `RAGPipeline` 默认不启用 Grade → Rewrite 回流循环；这是**可选的增强节点**
- 启用会显著增加 LLM 调用次数（每次评估 + 改写各一次 API 调用）
- 适合对召回质量要求极高的场景
