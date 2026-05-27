from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.query.LLMClient import (  # noqa: E402
    OpenAICompatibleChatClient,
    build_question_parser,
    documents_prompt_text,
    find_available_documents,
    get_question,
    print_json,
)


SYSTEM_PROMPT = """
你是 RAG HyDE 伪答案生成器。你只输出 JSON，不输出解释性自然语言。

目标：
生成一个“可能答案”的检索辅助文本，用于提升向量召回。该伪答案不是最终答案，不能当作事实依据。

输出 JSON 字段：
{
  "question": "原始问题",
  "pseudo_answer": "用于检索扩展的伪答案文本",
  "retrieval_terms": ["应该被召回的关键词或短语"],
  "expected_evidence": ["希望检索到的证据类型，如制度条款、表格行、流程说明"],
  "risk_notes": ["不确定点或需要原文验证的点"]
}

规则：
- 可以基于问题生成合理的答案形态，但必须避免具体编造制度条款、金额、日期、比例。
- 伪答案应帮助检索相关 chunk，不是最终用户回答。
- 如果问题要求具体数值、政策、合同条款，risk_notes 必须提示需要以知识库原文为准。
- pseudo_answer 控制在 300 字以内。
""".strip()


def generate_pseudo_answer(question: str, temperature: float = 0.2) -> Dict[str, Any]:
    documents = find_available_documents()
    user_prompt = f"""
可用文档：
{documents_prompt_text(documents)}

用户问题：
{question}
""".strip()

    result = OpenAICompatibleChatClient().chat_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
    )
    result.setdefault("question", question)
    return result


def main() -> Dict[str, Any]:
    parser = build_question_parser("RAG HyDE 伪答案生成")
    args = parser.parse_args()
    question = get_question(args)
    result = generate_pseudo_answer(question=question, temperature=args.temperature)
    print_json(result)
    return result


if __name__ == "__main__":
    main()
