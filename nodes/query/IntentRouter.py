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
你是 RAG 查询路由器。你只输出 JSON，不输出解释性自然语言。

目标：
1. 判断用户问题是否需要检索知识库。
2. 判断应该检索哪些文档或业务域。
3. 给后续检索器提供 metadata filter 建议。

输出 JSON 字段：
{
  "question": "原始问题",
  "needs_retrieval": true,
  "intent": "policy_lookup|contract_lookup|price_lookup|budget_lookup|product_lookup|hr_lookup|general_qa|chitchat|out_of_scope",
  "target_documents": ["文档名"],
  "metadata_filters": {
    "document_name": ["文档名"],
    "chunk_type": ["text_chunk", "table_chunk"]
  },
  "query_type": "fact|procedure|comparison|summary|table_lookup|definition|other",
  "confidence": 0.0,
  "reason": "一句话说明路由依据"
}

规则：
- 如果问题需要公司制度、合同、预算、价格、产品、HR 等资料，needs_retrieval=true。
- 如果只是闲聊、纯通用问题，needs_retrieval=false。
- target_documents 必须优先从可用文档列表中选择；不确定时返回空数组。
- confidence 是 0 到 1 的数字。
""".strip()


def route_intent(question: str, temperature: float = 0.1) -> Dict[str, Any]:
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
    parser = build_question_parser("RAG 路由意图识别")
    args = parser.parse_args()
    question = get_question(args)
    result = route_intent(question=question, temperature=args.temperature)
    print_json(result)
    return result


if __name__ == "__main__":
    main()
