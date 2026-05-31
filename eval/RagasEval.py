"""
RAGAS 风格评估模块 — 使用项目自有 LLM 客户端实现核心指标。

不引入 ragas 库依赖，完全复用项目现有的：
- OpenAICompatibleChatClient  (LLM 调用)
- BailianEmbeddingClient     (embedding 生成)

覆盖指标：
- Faithfulness       (忠实度)：答案中的声明是否被检索上下文支持
- Answer Relevancy   (答案相关性)：答案与问题的语义匹配度

用法：
    from eval.RagasEval import evaluate_single, evaluate_batch

    result = evaluate_single(question="...", answer="...", contexts=["...", "..."])
    print(result["faithfulness"]["score"])     # 0.0 ~ 1.0
    print(result["answer_relevancy"]["score"]) # 0.0 ~ 1.0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.query.LLMClient import OpenAICompatibleChatClient  # noqa: E402
from nodes.embeddings.ChromaBailianEmbedding import BailianEmbeddingClient  # noqa: E402


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# Answer Relevancy：从答案反向生成问题的数量
DEFAULT_N_QUESTIONS = 3

# 提取上下文文本时，每个 hit 的最大字符数
MAX_CONTEXT_CHARS_PER_HIT = 800


# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

CLAIMS_EXTRACTION_PROMPT = """将以下回答拆解为一系列独立的原子声明（atomic claims）。
每条声明应为一个简单、可独立验证的陈述句。每行一条声明，以 "- " 开头。
如果回答中没有可验证的声明，输出 "无声明"。

回答：
{answer}

声明："""


FAITHFULNESS_VERDICT_PROMPT = """根据提供的上下文信息，判断以下声明是否被上下文支持。

上下文：
{context}

声明：{claim}

判断标准：
- 如果声明中的信息在上下文中明确出现，或被上下文合理支持，判 "yes"
- 如果声明在上下文中找不到依据，或与上下文相矛盾，判 "no"

只输出一个 JSON 对象：
{{"verdict": "yes" 或 "no", "reason": "一句话理由"}}"""


REVERSE_QUESTIONS_PROMPT = """根据以下回答，生成 {n} 个可能导致这个回答的问题。
每个问题应独立、具体。每行一个问题，以 "- " 开头。

回答：
{answer}

问题："""


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _extract_context_texts(retrieval_result: Dict[str, Any]) -> List[str]:
    """从检索结果中提取上下文文本列表。"""
    hits = retrieval_result.get("hits") or []
    texts = []
    for hit in hits:
        text = (
            hit.get("expanded_context")
            or hit.get("document")
            or hit.get("content")
            or ""
        )
        if text:
            texts.append(text[:MAX_CONTEXT_CHARS_PER_HIT])
    return texts


def _extract_context_texts_from_list(contexts: Sequence[str]) -> List[str]:
    """从字符串列表中提取上下文文本（截断过长片段）。"""
    return [str(c)[:MAX_CONTEXT_CHARS_PER_HIT] for c in contexts if c]


def _parse_bullet_lines(text: str) -> List[str]:
    """解析 LLM 返回的 - 开头的列表。"""
    lines = []
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if item and item != "无声明":
                lines.append(item)
    return lines


def _parse_verdict_json(text: str) -> Dict[str, str]:
    """解析 Faithfulness verdict JSON。"""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {"verdict": str(data.get("verdict", "no")), "reason": str(data.get("reason", ""))}
    except json.JSONDecodeError:
        pass

    # 宽松解析
    import re
    text_lower = text.lower()
    verdict = "no"
    if '"yes"' in text_lower or "'yes'" in text_lower or "verdict" in text_lower and "yes" in text_lower:
        verdict = "yes"

    return {"verdict": verdict, "reason": text[:120]}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度。"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Faithfulness 评估
# ---------------------------------------------------------------------------

def evaluate_faithfulness(
    question: str,
    answer: str,
    contexts: Sequence[str],
    client: Optional[OpenAICompatibleChatClient] = None,
) -> Dict[str, Any]:
    """
    评估答案忠实度：答案中的声明是否被上下文支持。

    :param question: 用户问题（保留接口一致性，实际不使用）
    :param answer: RAG 生成的答案
    :param contexts: 检索到的上下文文本列表
    :param client: LLM 客户端
    :return: {
        "score": float,          # 0.0 ~ 1.0
        "claims": [str],         # 提取的原子声明
        "verdicts": [dict],      # 每条声明的判定
        "supported": int,        # 被支持的声明数
        "total": int,            # 总声明数
    }
    """
    if client is None:
        client = OpenAICompatibleChatClient()

    # Step 1: 提取原子声明
    claims_prompt = CLAIMS_EXTRACTION_PROMPT.format(answer=answer)
    claims_text = client.chat_text(
        system_prompt="你是一个文本分析专家。只输出声明列表，不要解释。",
        user_prompt=claims_prompt,
        temperature=0.0,
    )
    claims = _parse_bullet_lines(claims_text)

    if not claims:
        return {
            "score": 0.0,
            "claims": [],
            "verdicts": [],
            "supported": 0,
            "total": 0,
            "note": "未能从回答中提取声明",
        }

    # Step 2: 构建上下文
    context_text = "\n\n---\n\n".join(
        f"[{i + 1}] {c}" for i, c in enumerate(contexts[:5])
    )

    # Step 3: 逐条验证
    verdicts = []
    supported = 0

    for claim in claims:
        verdict_prompt = FAITHFULNESS_VERDICT_PROMPT.format(
            context=context_text,
            claim=claim,
        )
        try:
            verdict_text = client.chat_text(
                system_prompt="你是一个事实核查专家。严格按照要求输出 JSON。",
                user_prompt=verdict_prompt,
                temperature=0.0,
            )
            verdict = _parse_verdict_json(verdict_text)
        except Exception:
            verdict = {"verdict": "unknown", "reason": "LLM 调用失败"}

        verdict["claim"] = claim
        verdicts.append(verdict)
        if verdict["verdict"] == "yes":
            supported += 1

    score = supported / len(claims) if claims else 0.0

    return {
        "score": score,
        "claims": claims,
        "verdicts": verdicts,
        "supported": supported,
        "total": len(claims),
    }


# ---------------------------------------------------------------------------
# Answer Relevancy 评估
# ---------------------------------------------------------------------------

def evaluate_answer_relevancy(
    question: str,
    answer: str,
    embedding_client: Optional[BailianEmbeddingClient] = None,
    llm_client: Optional[OpenAICompatibleChatClient] = None,
    n_questions: int = DEFAULT_N_QUESTIONS,
) -> Dict[str, Any]:
    """
    评估答案相关性：答案是否真正回答了问题。

    算法：
    1. LLM 从答案反向生成 N 个问题
    2. 计算每个生成问题与原问题的 embedding 余弦相似度
    3. score = mean(similarities)

    :param question: 用户原问题
    :param answer: RAG 生成的答案
    :param embedding_client: Embedding 客户端
    :param llm_client: LLM 客户端
    :param n_questions: 生成的问题数量
    :return: {
        "score": float,                  # 0.0 ~ 1.0
        "generated_questions": [str],    # 生成的反向问题
        "similarities": [float],         # 每个问题的相似度
    }
    """
    if llm_client is None:
        llm_client = OpenAICompatibleChatClient()

    # Step 1: 反向生成问题
    reverse_prompt = REVERSE_QUESTIONS_PROMPT.format(n=n_questions, answer=answer)
    reverse_text = llm_client.chat_text(
        system_prompt="你是一个问题生成专家。只输出问题列表，不要解释。",
        user_prompt=reverse_prompt,
        temperature=0.3,  # 略微提高温度以增加多样性
    )
    generated_questions = _parse_bullet_lines(reverse_text)

    if not generated_questions:
        # 如果解析失败，尝试按行分割
        generated_questions = [
            line.strip()
            for line in reverse_text.strip().split("\n")
            if line.strip() and "?" in line
        ]

    if not generated_questions:
        return {
            "score": 0.0,
            "generated_questions": [],
            "similarities": [],
            "note": "未能生成反向问题",
        }

    # Step 2: 生成 embeddings 并计算相似度
    if embedding_client is None:
        embedding_client = BailianEmbeddingClient()

    all_texts = [question] + generated_questions
    embeddings = embedding_client.embed_texts(all_texts)

    question_embedding = np.array(embeddings[0])
    similarities = []

    for i, gen_emb in enumerate(embeddings[1:], start=1):
        sim = _cosine_similarity(question_embedding, np.array(gen_emb))
        similarities.append(round(sim, 4))

    score = float(np.mean(similarities)) if similarities else 0.0

    return {
        "score": round(score, 4),
        "generated_questions": generated_questions,
        "similarities": similarities,
    }


# ---------------------------------------------------------------------------
# 综合评估
# ---------------------------------------------------------------------------

def evaluate_single(
    question: str,
    answer: str,
    contexts: Optional[Sequence[str]] = None,
    retrieval_result: Optional[Dict[str, Any]] = None,
    llm_client: Optional[OpenAICompatibleChatClient] = None,
    embedding_client: Optional[BailianEmbeddingClient] = None,
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    对单次 RAG 结果运行 RAGAS 风格评估。

    :param question: 用户问题
    :param answer: RAG 生成的答案
    :param contexts: 上下文文本列表（与 retrieval_result 二选一）
    :param retrieval_result: ChromaRetriever 返回的检索结果（与 contexts 二选一）
    :param llm_client: LLM 客户端
    :param embedding_client: Embedding 客户端
    :param metrics: 要计算的指标列表，默认全部 ["faithfulness", "answer_relevancy"]
    :return: {"faithfulness": {...}, "answer_relevancy": {...}}
    """
    if metrics is None:
        metrics = ["faithfulness", "answer_relevancy"]

    if llm_client is None:
        llm_client = OpenAICompatibleChatClient()

    # 解析上下文
    if contexts is not None:
        context_texts = _extract_context_texts_from_list(contexts)
    elif retrieval_result is not None:
        context_texts = _extract_context_texts(retrieval_result)
    else:
        context_texts = []

    result: Dict[str, Any] = {}

    if "faithfulness" in metrics:
        result["faithfulness"] = evaluate_faithfulness(
            question=question,
            answer=answer,
            contexts=context_texts,
            client=llm_client,
        )

    if "answer_relevancy" in metrics:
        result["answer_relevancy"] = evaluate_answer_relevancy(
            question=question,
            answer=answer,
            embedding_client=embedding_client,
            llm_client=llm_client,
        )

    return result


def evaluate_batch(
    cases: Sequence[Dict[str, Any]],
    llm_client: Optional[OpenAICompatibleChatClient] = None,
    embedding_client: Optional[BailianEmbeddingClient] = None,
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    批量评估多个 RAG 结果。

    :param cases: 列表，每个元素包含 question, answer, contexts(或 retrieval_result)
    :param llm_client: LLM 客户端
    :param embedding_client: Embedding 客户端
    :param metrics: 指标列表
    :return: {"summary": {...}, "details": [...]}
    """
    if llm_client is None:
        llm_client = OpenAICompatibleChatClient()

    details = []
    for i, case in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] 评估: {case.get('question', '')[:60]}...", file=sys.stderr)
        try:
            eval_result = evaluate_single(
                question=str(case.get("question", "")),
                answer=str(case.get("answer", "")),
                contexts=case.get("contexts"),
                retrieval_result=case.get("retrieval_result"),
                llm_client=llm_client,
                embedding_client=embedding_client,
                metrics=metrics,
            )
            details.append({**case, "ragas_eval": eval_result})
        except Exception as exc:
            details.append({**case, "ragas_eval": {"error": str(exc)}})

    # 汇总
    faithfulness_scores = []
    relevancy_scores = []
    errors = 0

    for d in details:
        ev = d.get("ragas_eval", {})
        if "error" in ev:
            errors += 1
            continue
        if ev.get("faithfulness", {}).get("score") is not None:
            faithfulness_scores.append(ev["faithfulness"]["score"])
        if ev.get("answer_relevancy", {}).get("score") is not None:
            relevancy_scores.append(ev["answer_relevancy"]["score"])

    summary = {
        "case_count": len(cases),
        "error_count": errors,
    }

    if faithfulness_scores:
        summary["faithfulness"] = {
            "mean": round(float(np.mean(faithfulness_scores)), 4),
            "min": round(float(np.min(faithfulness_scores)), 4),
            "max": round(float(np.max(faithfulness_scores)), 4),
            "count": len(faithfulness_scores),
        }

    if relevancy_scores:
        summary["answer_relevancy"] = {
            "mean": round(float(np.mean(relevancy_scores)), 4),
            "min": round(float(np.min(relevancy_scores)), 4),
            "max": round(float(np.max(relevancy_scores)), 4),
            "count": len(relevancy_scores),
        }

    return {"summary": summary, "details": details}


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAGAS 风格评估工具：对 RAG 回答运行 Faithfulness 和 Answer Relevancy 评估"
    )
    parser.add_argument(
        "input", nargs="?",
        help="RAG 结果 JSON 文件路径（需包含 question, answer, retrieval_result 字段）"
    )
    parser.add_argument("--question", "-q", help="用户问题（覆盖文件中的 question）")
    parser.add_argument("--answer", "-a", help="RAG 答案（覆盖文件中的 answer）")
    parser.add_argument("--context-file", help="检索结果 JSON 文件路径（contexts 来源）")
    parser.add_argument(
        "--metrics", default="faithfulness,answer_relevancy",
        help="要计算的指标，逗号分隔，默认 faithfulness,answer_relevancy"
    )
    parser.add_argument("--no-faithfulness", action="store_true", help="跳过 Faithfulness")
    parser.add_argument("--no-answer-relevancy", action="store_true", help="跳过 Answer Relevancy")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    args = parser.parse_args()

    # 确定指标
    metrics = []
    if not args.no_faithfulness:
        metrics.append("faithfulness")
    if not args.no_answer_relevancy:
        metrics.append("answer_relevancy")

    # 加载数据
    if args.input:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        data = json.loads(sys.stdin.read())

    question = args.question or str(data.get("question", ""))
    answer = args.answer or str(data.get("answer", ""))

    # 加载 contexts
    contexts = None
    retrieval_result = None
    if args.context_file:
        retrieval_result = json.loads(Path(args.context_file).read_text(encoding="utf-8"))
    elif "retrieval_result" in data:
        retrieval_result = data["retrieval_result"]
    elif "contexts" in data:
        contexts = data["contexts"]

    result = evaluate_single(
        question=question,
        answer=answer,
        contexts=contexts,
        retrieval_result=retrieval_result,
        metrics=metrics,
    )

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)
