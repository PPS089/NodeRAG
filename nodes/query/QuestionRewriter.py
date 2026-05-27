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
你是 RAG 问题改写器。你只输出 JSON，不输出解释性自然语言。

目标：
把用户问题改写成更适合向量检索和关键词检索的表达，同时保留原意。

输出 JSON 字段：
{
  "original_question": "原始问题",
  "rewritten_question": "更清晰、完整、适合检索的问题",
  "search_queries": ["用于向量检索的查询1", "用于向量检索的查询2"],
  "keyword_queries": ["关键词组合1", "关键词组合2"],
  "entities": ["实体、制度名、产品名、角色名等"],
  "time_constraints": ["时间条件，没有则空数组"],
  "must_not_change": ["不能改写或推断的关键约束"]
}

规则：
- 不要编造事实。
- 不要把问题改成另一个问题。
- search_queries 控制在 3 条以内。
- keyword_queries 控制在 5 条以内。
- 如果问题里有简称、口语、省略主语，要补全为适合检索的表达。
""".strip()


def rewrite_question(question: str, temperature: float = 0.1) -> Dict[str, Any]:
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
    result.setdefault("original_question", question)
    return result


def main() -> Dict[str, Any]:
    parser = build_question_parser("RAG 用户问题改写")
    args = parser.parse_args()
    question = get_question(args)
    result = rewrite_question(question=question, temperature=args.temperature)
    print_json(result)
    return result


if __name__ == "__main__":
    main()
