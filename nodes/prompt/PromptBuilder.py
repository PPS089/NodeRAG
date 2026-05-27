from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = """
你是企业知识库 RAG 问答助手。

回答规则：
1. 只能基于提供的【检索上下文】回答。
2. 如果上下文不足以回答，明确说明“当前知识库资料不足以确认”。
3. 不要编造制度、价格、预算、合同条款、日期或比例。
4. 回答中必须引用证据编号，例如 [C1]、[C2]。
5. 表格内容要按字段或行解释，必要时用列表呈现。
6. 如果上下文声明为测试数据，回答时要说明这是测试资料。
""".strip()


def load_json_file(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_context(input_path: Optional[str]) -> Dict[str, Any]:
    if input_path:
        return json.loads(Path(input_path).read_text(encoding="utf-8"))

    raw = sys.stdin.read().strip()
    if not raw:
        raise ValueError("请通过 --context 传入压缩上下文 JSON 文件，或通过 stdin 输入 JSON")
    return json.loads(raw)


def format_title_path(title_path: Any) -> str:
    if isinstance(title_path, list):
        return " > ".join(str(item) for item in title_path if item)
    return str(title_path or "")


def build_context_text(context_result: Dict[str, Any]) -> str:
    blocks = context_result.get("context_blocks", [])
    if not blocks:
        return "未检索到可用上下文。"

    parts = []
    for block in blocks:
        title_path = format_title_path(block.get("title_path"))
        line_range = block.get("line_range") or []
        line_text = f" 行号: {line_range}" if line_range else ""
        header = (
            f"[{block.get('citation_id')}] "
            f"文档: {block.get('document_name', '')}; "
            f"章节: {title_path}; "
            f"类型: {block.get('chunk_type', '')};"
            f"{line_text}"
        )
        parts.append(f"{header}\n{block.get('content', '')}")

    return "\n\n---\n\n".join(parts)


def build_auxiliary_text(
    route_result: Optional[Dict[str, Any]],
    rewrite_result: Optional[Dict[str, Any]],
    pseudo_answer_result: Optional[Dict[str, Any]],
) -> str:
    parts = []

    if route_result:
        parts.append("【路由结果】\n" + json.dumps(route_result, ensure_ascii=False, indent=2))

    if rewrite_result:
        parts.append("【问题改写】\n" + json.dumps(rewrite_result, ensure_ascii=False, indent=2))

    if pseudo_answer_result:
        safe_pseudo = dict(pseudo_answer_result)
        if safe_pseudo.get("pseudo_answer"):
            safe_pseudo["pseudo_answer_note"] = "伪答案仅用于检索辅助，不可作为事实依据。"
        parts.append("【伪答案检索辅助】\n" + json.dumps(safe_pseudo, ensure_ascii=False, indent=2))

    return "\n\n".join(parts)


def build_prompt(
    question: str,
    context_result: Dict[str, Any],
    route_result: Optional[Dict[str, Any]] = None,
    rewrite_result: Optional[Dict[str, Any]] = None,
    pseudo_answer_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context_text = build_context_text(context_result)
    auxiliary_text = build_auxiliary_text(route_result, rewrite_result, pseudo_answer_result)

    user_parts = [
        f"【用户问题】\n{question}",
        f"【检索上下文】\n{context_text}",
    ]
    if auxiliary_text:
        user_parts.append(auxiliary_text)

    user_parts.append(
        "【输出要求】\n"
        "- 先给出直接答案。\n"
        "- 再列出依据，引用 [C1] 这样的证据编号。\n"
        "- 如果依据不足，说明缺失哪些资料。"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]

    return {
        "question": question,
        "messages": messages,
        "citations": context_result.get("citations", []),
        "context_stats": context_result.get("compression_stats", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="组装 RAG 最终回答 Prompt。")
    parser.add_argument("--question", "-q", required=True, help="用户原始问题。")
    parser.add_argument("--context", "-c", help="ContextCompressor 输出 JSON 文件；不传则从 stdin 读取。")
    parser.add_argument("--route", help="IntentRouter 输出 JSON 文件。")
    parser.add_argument("--rewrite", help="QuestionRewriter 输出 JSON 文件。")
    parser.add_argument("--pseudo-answer", help="PseudoAnswer 输出 JSON 文件。")
    parser.add_argument("--output", "-o", help="输出 Prompt JSON 文件；不传则打印到 stdout。")
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    context_result = load_context(args.context)
    result = build_prompt(
        question=args.question,
        context_result=context_result,
        route_result=load_json_file(args.route),
        rewrite_result=load_json_file(args.rewrite),
        pseudo_answer_result=load_json_file(args.pseudo_answer),
    )

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)

    return result


if __name__ == "__main__":
    main()
