"""
文档相关性评估 + 回流重写（Grade → Rewrite Loop）。

核心流程：
1. stage_grade：评估召回文档与问题的相关性（LLM 二分类：yes/no）
2. grade_retrieval_result：对检索结果做相关性评分
3. 如果相关性不足，触发 rewrite_question + 重新检索

参考 SuperMew rag_pipeline.py 的 grade_documents_node + rewrite_question_node 设计，
适配 NodeRAG 的 Pipeline Stage 架构。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.query.LLMClient import OpenAICompatibleChatClient  # noqa: E402


# ------------------
# LLM 评分模型
# ------------------

GRADE_PROMPT_TEMPLATE = """你是一个评估者，判断检索到的文档片段是否与用户问题相关。

用户问题：{question}

检索到的文档片段：
{context}

评估标准：
- 如果文档包含与问题相关的关键词、语义或具体信息，返回 'yes'
- 如果文档与问题无关或明显不匹配，返回 'no'

只输出 'yes' 或 'no'，不要输出其他内容。"""


class GradeDocuments(BaseModel):
    """LLM 相关性评分结果。"""

    binary_score: str = Field(
        description="相关性评分：'yes' 表示相关，'no' 表示不相关"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="简要说明评分理由（可选）"
    )


class RewriteStrategy(BaseModel):
    """查询扩展策略选择。"""

    strategy: str = Field(
        description="查询扩展策略：step_back（退步问题）、hyde（假设性文档）、simple（仅原问题）"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="选择该策略的原因（可选）"
    )


# ------------------
# LLM 客户端（评分用）
# ------------------

_grader_client: Optional[OpenAICompatibleChatClient] = None


def get_grader_client() -> OpenAICompatibleChatClient:
    """获取评分用 LLM 客户端（全局单例）。"""
    global _grader_client
    if _grader_client is None:
        _grader_client = OpenAICompatibleChatClient()
    return _grader_client


# ------------------
# 核心评估函数
# ------------------

def grade_documents(
    question: str,
    hits: Sequence[Dict[str, Any]],
    client: Optional[OpenAICompatibleChatClient] = None,
    use_structured_output: bool = True,
) -> Dict[str, Any]:
    """
    评估检索结果的相关性。

    :param question: 用户问题
    :param hits: 检索结果列表
    :param client: LLM 客户端（可选，默认使用 get_grader_client）
    :param use_structured_output: 是否使用结构化输出（依赖模型支持）
    :return: {
        "grade_score": "yes" | "no",
        "grade_reasoning": str,
        "grade_model": str,
        "passed": bool,
        "relevant_count": int,
        "total_count": int,
    }
    """
    if not hits:
        return {
            "grade_score": "no",
            "grade_reasoning": "检索结果为空",
            "grade_model": None,
            "passed": False,
            "relevant_count": 0,
            "total_count": 0,
        }

    # 构建上下文摘要（最多 5 个片段，每个截取前 500 字符）
    context_parts = []
    for i, hit in enumerate(hits[:5], 1):
        text = hit.get("document", "") or ""
        text = text[:500] + "..." if len(text) > 500 else text
        metadata = hit.get("metadata") or {}
        doc_name = metadata.get("document_name", "未知文档")
        context_parts.append(f"[{i}] {doc_name}:\n{text}")

    context_text = "\n\n---\n\n".join(context_parts)
    prompt = GRADE_PROMPT_TEMPLATE.format(question=question, context=context_text)

    try:
        if client is None:
            client = get_grader_client()

        if use_structured_output:
            # 尝试使用结构化输出（依赖模型支持 function calling / json_object）
            result = client.chat_json(
                system_prompt="你是一个评估者。严格按照要求输出 JSON 对象。",
                user_prompt=prompt,
                temperature=0.0,
            )
            grade_score = str(result.get("binary_score", "no")).strip().lower()
            grade_reasoning = result.get("reasoning", "")
        else:
            # Fallback：纯文本解析
            response = client.chat_text(
                system_prompt="你是一个评估者。只输出 yes 或 no。",
                user_prompt=prompt,
                temperature=0.0,
            )
            grade_score = response.strip().lower()
            if "yes" in grade_score:
                grade_score = "yes"
            elif "no" in grade_score:
                grade_score = "no"
            else:
                grade_score = "no"
            grade_reasoning = ""

        passed = grade_score == "yes"
        relevant_count = sum(
            1 for hit in hits
            if hit.get("rerank_score", 0.0) > 0.3 or hit.get("bm25_score", 0.0) > 0
        )

        return {
            "grade_score": grade_score,
            "grade_reasoning": grade_reasoning,
            "grade_model": client.model,
            "passed": passed,
            "relevant_count": relevant_count,
            "total_count": len(hits),
        }

    except Exception as e:
        # LLM 调用失败时，保守返回不通过
        return {
            "grade_score": "unknown",
            "grade_reasoning": f"LLM 调用失败: {str(e)}",
            "grade_model": None,
            "passed": False,
            "relevant_count": 0,
            "total_count": len(hits),
            "error": str(e),
        }


def grade_retrieval_result(
    retrieval_result: Dict[str, Any],
    question: Optional[str] = None,
    client: Optional[OpenAICompatibleChatClient] = None,
    use_structured_output: bool = True,
) -> Dict[str, Any]:
    """
    对检索结果进行相关性评估，增加 grade 字段。

    :param retrieval_result: ChromaRetriever 返回的检索结果
    :param question: 用户问题（从 retrieval_result.question 读取，如无则必传）
    :param client: LLM 客户端
    :param use_structured_output: 是否使用结构化输出
    :return: 增加 grade 字段的检索结果
    """
    question = question or retrieval_result.get("question", "")
    hits = retrieval_result.get("hits") or []

    grade_result = grade_documents(
        question=question,
        hits=hits,
        client=client,
        use_structured_output=use_structured_output,
    )

    retrieval_result = dict(retrieval_result)
    retrieval_result["grade"] = grade_result

    return retrieval_result


# ------------------
# 查询扩展策略
# ------------------

REWRITE_PROMPT_TEMPLATE = """根据用户问题，选择最合适的查询扩展策略。

用户问题：{question}

策略说明：
- step_back（退步问题）：问题包含具体名称、日期、代码、数值等细节，需要先理解通用概念。
  例如："2023年Q3的销售收入是多少" → "公司季度销售收入如何计算和汇报"
- hyde（假设性文档）：问题模糊、概念性、需要解释或定义。
  例如："什么是 Scrum？" → 生成一段 Scrum 方法论的假设性说明文档
- simple（仅原问题）：问题已足够清晰，无需扩展。

只输出策略名（step_back / hyde / simple），不要输出其他内容。"""


STEPBACK_QUESTION_PROMPT = """请将用户的具体问题抽象成更高层次、更概括的"退步问题"，
用于探寻背后的通用原理或核心概念。

只输出退步问题一句话，不要解释，不要加引号。
用户问题：{question}"""


STEPBACK_ANSWER_PROMPT = """请简要回答以下退步问题，提供通用原理/背景知识，控制在120字以内。
只输出答案，不要列出推理过程。
退步问题：{step_back_question}"""


HYDE_DOCUMENT_PROMPT = """请基于用户问题生成一段"假设性文档"，内容应像真实资料片段，
用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。
只输出文档正文，不要标题或解释。
用户问题：{question}"""


def choose_rewrite_strategy(
    question: str,
    client: Optional[OpenAICompatibleChatClient] = None,
) -> str:
    """选择查询扩展策略。"""
    if client is None:
        client = get_grader_client()

    prompt = REWRITE_PROMPT_TEMPLATE.format(question=question)
    try:
        response = client.chat_text(
            system_prompt="你是一个查询策略选择器。只输出策略名，不要输出其他内容。",
            user_prompt=prompt,
            temperature=0.0,
        )
        response = response.strip().lower()
        if "step_back" in response:
            return "step_back"
        elif "hyde" in response:
            return "hyde"
        else:
            return "simple"
    except Exception:
        return "simple"


def generate_step_back_question(
    question: str,
    client: Optional[OpenAICompatibleChatClient] = None,
) -> str:
    """生成退步问题。"""
    if client is None:
        client = get_grader_client()

    prompt = STEPBACK_QUESTION_PROMPT.format(question=question)
    try:
        return client.chat_text(
            system_prompt="你是一个问题改写专家。只输出一句话问题，不要解释。",
            user_prompt=prompt,
            temperature=0.2,
        ).strip()
    except Exception:
        return ""


def answer_step_back_question(
    step_back_question: str,
    client: Optional[OpenAICompatibleChatClient] = None,
) -> str:
    """回答退步问题。"""
    if client is None:
        client = get_grader_client()

    if not step_back_question:
        return ""

    prompt = STEPBACK_ANSWER_PROMPT.format(step_back_question=step_back_question)
    try:
        return client.chat_text(
            system_prompt="你是一个知识问答专家。只输出答案，控制在120字以内。",
            user_prompt=prompt,
            temperature=0.2,
        ).strip()
    except Exception:
        return ""


def generate_hypothetical_document(
    question: str,
    client: Optional[OpenAICompatibleChatClient] = None,
) -> str:
    """生成假设性文档（HyDE）。"""
    if client is None:
        client = get_grader_client()

    prompt = HYDE_DOCUMENT_PROMPT.format(question=question)
    try:
        return client.chat_text(
            system_prompt="你是一个文档编写专家。只输出文档正文，模拟真实资料片段。",
            user_prompt=prompt,
            temperature=0.3,
        ).strip()
    except Exception:
        return ""


def rewrite_question(
    question: str,
    strategy: Optional[str] = None,
    client: Optional[OpenAICompatibleChatClient] = None,
) -> Dict[str, Any]:
    """
    查询改写主函数。

    :param question: 用户问题
    :param strategy: 扩展策略（可选，默认自动选择）
    :param client: LLM 客户端
    :return: {
        "original_question": str,
        "strategy": str,
        "step_back_question": str,
        "step_back_answer": str,
        "hypothetical_document": str,
        "expanded_queries": [str],
    }
    """
    if client is None:
        client = get_grader_client()

    if strategy is None:
        strategy = choose_rewrite_strategy(question, client=client)

    step_back_question = ""
    step_back_answer = ""
    hypothetical_doc = ""
    expanded_queries = [question]

    if strategy in ("step_back",):
        step_back_question = generate_step_back_question(question, client=client)
        step_back_answer = answer_step_back_question(step_back_question, client=client)
        if step_back_question:
            expanded_queries.append(step_back_question)
        if step_back_answer:
            expanded_queries.append(step_back_answer)

    if strategy in ("hyde",):
        hypothetical_doc = generate_hypothetical_document(question, client=client)
        if hypothetical_doc:
            expanded_queries.append(hypothetical_doc)

    return {
        "original_question": question,
        "strategy": strategy,
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "hypothetical_document": hypothetical_doc,
        "expanded_queries": expanded_queries,
    }


# ------------------
# 回流循环（Grade → Rewrite Loop）
# ------------------

DEFAULT_MAX_RETRY = 2


def grade_and_rewrite_loop(
    question: str,
    initial_retrieval_result: Dict[str, Any],
    client: Optional[OpenAICompatibleChatClient] = None,
    max_retry: int = DEFAULT_MAX_RETRY,
    use_structured_output: bool = True,
) -> Dict[str, Any]:
    """
    Grade → Rewrite 回流循环主函数。

    流程：
    1. 评估初次检索结果
    2. 如果通过，直接返回
    3. 如果不通过，触发 rewrite + 重新检索（由调用方执行）
    4. 记录所有尝试的 grade 和 rewrite 结果

    注意：此函数仅负责评估和改写，不执行重新检索。
    重新检索由 RAGPipeline.stage_* 在检测到 grade 不通过后调用。

    :param question: 用户问题
    :param initial_retrieval_result: 初次检索结果
    :param client: LLM 客户端
    :param max_retry: 最大重试次数（当前版本仅支持 1 次重试）
    :param use_structured_output: 是否使用结构化输出
    :return: {
        "passed": bool,
        "final_grade": Dict,
        "rewrite_result": Dict | None,
        "attempt_count": int,
        "grade_history": [Dict],
        "rewrite_history": [Dict],
    }
    """
    if client is None:
        client = get_grader_client()

    grade_history = []
    rewrite_history = []
    current_retrieval = dict(initial_retrieval_result)

    for attempt in range(max_retry + 1):
        # 评估当前检索结果
        grade_result = grade_documents(
            question=question,
            hits=current_retrieval.get("hits") or [],
            client=client,
            use_structured_output=use_structured_output,
        )
        grade_result["attempt"] = attempt + 1
        grade_history.append(grade_result)

        if grade_result.get("passed", False):
            # 评估通过，返回结果
            return {
                "passed": True,
                "final_grade": grade_result,
                "rewrite_result": None,
                "attempt_count": attempt + 1,
                "grade_history": grade_history,
                "rewrite_history": rewrite_history,
            }

        # 评估不通过，触发改写（仅在重试次数内）
        if attempt < max_retry:
            rewrite_result = rewrite_question(question, client=client)
            rewrite_result["attempt"] = attempt + 1
            rewrite_history.append(rewrite_result)

    # 所有尝试均未通过
    return {
        "passed": False,
        "final_grade": grade_history[-1] if grade_history else {},
        "rewrite_result": rewrite_history[-1] if rewrite_history else None,
        "attempt_count": len(grade_history),
        "grade_history": grade_history,
        "rewrite_history": rewrite_history,
    }


# ------------------
# CLI 入口
# ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="文档相关性评估工具：从 JSON 文件读取检索结果并评估相关性"
    )
    parser.add_argument("input", nargs="?", help="检索结果 JSON 文件路径")
    parser.add_argument("--question", "-q", help="用户问题（优先使用，忽略检索结果中的 question）")
    parser.add_argument("--no-structured", action="store_true", help="不使用结构化输出，改用纯文本解析")
    parser.add_argument("--max-retry", type=int, default=DEFAULT_MAX_RETRY, help="最大重试次数")
    parser.add_argument("--mode", choices=["grade", "rewrite", "loop"], default="grade", help="运行模式")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    args = parser.parse_args()

    # 加载检索结果
    if args.input:
        retrieval_result = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        retrieval_result = json.loads(sys.stdin.read())

    question = args.question or retrieval_result.get("question", "")
    use_structured = not args.no_structured

    if args.mode == "grade":
        result = grade_retrieval_result(
            retrieval_result=retrieval_result,
            question=question,
            use_structured_output=use_structured,
        )
    elif args.mode == "rewrite":
        result = rewrite_question(
            question=question,
            client=get_grader_client(),
        )
    else:  # loop
        result = grade_and_rewrite_loop(
            question=question,
            initial_retrieval_result=retrieval_result,
            max_retry=args.max_retry,
            use_structured_output=use_structured,
        )

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)