# PseudoAnswer — HyDE 伪答案生成节点

## 概述

`PseudoAnswer` 实现 **HyDE（Hypothetical Document Embeddings）** 策略：通过 LLM 生成一个"可能答案"的检索辅助文本，用于**提升向量召回质量**。伪答案不作为最终回答，也不作为事实依据。

**路径**：`nodes/query/PseudoAnswer.py`

## 在链路中的位置

```
QuestionRewriter → PseudoAnswer → 检索（作为额外 query）→ ...
```

## 核心原理

向量检索的问题是：用户问题通常很短，而知识库中的 chunk 可能用不同的措辞表达相同内容。HyDE 策略让 LLM 先生成一个"如果是答案大概长什么样"的文本，用这个伪答案作为额外的检索 query，因为伪答案的语义向量更接近知识库中实际相关 chunk 的向量。

## 输入 / 输出

### 输入

- `question: str` — 用户原始问题
- `temperature: float`（默认 0.2）— 比路由/改写略高，允许一定的"合理推测"

### 输出

```json
{
  "question": "试用期绩效目标什么时候确认？",
  "pseudo_answer": "根据公司制度，新员工试用期绩效目标通常在入职后1-2周内由直属上级与员工共同确认，并以书面形式记录。具体时间节点可能因岗位和部门有所不同。",
  "retrieval_terms": ["试用期绩效目标", "确认时间", "绩效考核", "新员工"],
  "expected_evidence": ["制度条款", "流程说明", "表格模板"],
  "risk_notes": ["具体时间以公司制度原文为准", "不同岗位可能有差异"]
}
```

## 规则约束

| 规则 | 说明 |
| --- | --- |
| `pseudo_answer` ≤ 300 字 | 控制生成长度 |
| 不能编造具体数据 | 不能编造具体制度条款、金额、日期、比例 |
| `risk_notes` 提示 | 涉及具体数值/政策/合同条款时必须提示以原文为准 |
| 目标是辅助检索 | 不是最终回答 |

## 安全性

- Pipeline 中伪答案仅传递给检索节点（作为额外 query），不直接展示给用户
- `PromptBuilder` 接收伪答案时会追加 `pseudo_answer_note = "伪答案仅用于检索辅助，不可作为事实依据"`
- 最终 LLM 回答仅基于检索到的真实上下文，不依赖伪答案

## CLI 使用

```powershell
.\.venv\Scripts\python.exe nodes\query\PseudoAnswer.py --question "试用期绩效目标什么时候确认？"
```

## 上下游契约

- **上游**：用户原始问题
- **下游**：写入 `PipelineContext.pseudo_answer_result`；`ChromaRetriever` 将 `pseudo_answer` 作为额外 vector query
