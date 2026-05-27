from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.context.ContextCompressor import compress_context  # noqa: E402
from nodes.prompt.PromptBuilder import build_prompt  # noqa: E402
from nodes.query.IntentRouter import route_intent  # noqa: E402
from nodes.query.LLMClient import OpenAICompatibleChatClient  # noqa: E402
from nodes.query.PseudoAnswer import generate_pseudo_answer  # noqa: E402
from nodes.query.QuestionRewriter import rewrite_question  # noqa: E402
from nodes.retrieval.ChromaRetriever import retrieve  # noqa: E402
from nodes.auth.PermissionGuard import (  # noqa: E402
    DEFAULT_CONFIG_PATH as DEFAULT_PERMISSION_CONFIG_PATH,
    DEFAULT_PERMISSION_LEVEL,
    apply_permission_to_retrieval_inputs,
    build_permission_context,
    filter_retrieval_result_by_permission,
)


DEFAULT_TOP_K = 8
DEFAULT_PER_QUERY_K = 12
DEFAULT_MAX_CONTEXT_CHARS = 12000
DEFAULT_MAX_BLOCKS = 12
EXIT_COMMANDS = {"q", "quit", "exit", "退出", "结束"}
PipelineContext = Dict[str, Any]
PipelineStage = Callable[[PipelineContext], PipelineContext]


def unique_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def extract_route_document_names(route_result: Optional[Dict[str, Any]]) -> List[str]:
    if not route_result:
        return []

    names = normalize_list(route_result.get("target_documents"))
    metadata_filters = route_result.get("metadata_filters") or {}
    names.extend(normalize_list(metadata_filters.get("document_name")))
    return unique_keep_order(names)


def extract_route_chunk_types(route_result: Optional[Dict[str, Any]]) -> List[str]:
    if not route_result:
        return []

    metadata_filters = route_result.get("metadata_filters") or {}
    return unique_keep_order(normalize_list(metadata_filters.get("chunk_type")))


def build_direct_answer(question: str, temperature: float = 0.2) -> str:
    return OpenAICompatibleChatClient().chat_text(
        system_prompt=(
            "你是企业知识库助手。当前问题不需要检索知识库时，可以直接回答；"
            "如果问题涉及企业制度、合同、价格、预算或内部资料，说明需要检索知识库。"
        ),
        user_prompt=question,
        temperature=temperature,
    )


class RAGPipeline:
    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        per_query_k: int = DEFAULT_PER_QUERY_K,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_blocks: int = DEFAULT_MAX_BLOCKS,
        temperature: float = 0.2,
        skip_preprocess: bool = False,
        permission_level: str = DEFAULT_PERMISSION_LEVEL,
        permission_config: str | Path = DEFAULT_PERMISSION_CONFIG_PATH,
    ) -> None:
        self.top_k = top_k
        self.per_query_k = per_query_k
        self.max_context_chars = max_context_chars
        self.max_blocks = max_blocks
        self.temperature = temperature
        self.skip_preprocess = skip_preprocess
        self.permission_context = build_permission_context(permission_level, permission_config)
        self.chat_client = OpenAICompatibleChatClient()

    def build_stages(self) -> List[PipelineStage]:
        return [
            self.stage_route,
            self.stage_rewrite,
            self.stage_pseudo_answer,
            self.stage_prepare_retrieval,
            self.stage_apply_permission,
            self.stage_retrieve,
            self.stage_filter_retrieval_permission,
            self.stage_compress,
            self.stage_prompt,
            self.stage_answer,
        ]

    def create_context(
        self,
        question: str,
        document_names: Optional[Sequence[str]] = None,
        chunk_types: Optional[Sequence[str]] = None,
    ) -> PipelineContext:
        return {
            "question": question,
            "input_document_names": list(document_names or []),
            "input_chunk_types": list(chunk_types or []),
            "started_at": time.time(),
            "needs_retrieval": True,
            "stop_pipeline": False,
            "route_result": None,
            "rewrite_result": None,
            "pseudo_answer_result": None,
            "retrieval_inputs": {},
            "retrieval_result": None,
            "context_result": None,
            "prompt_result": None,
            "answer": None,
            "permission_context": self.permission_context,
            "permission_denied": False,
        }

    def stage_route(self, context: PipelineContext) -> PipelineContext:
        if self.skip_preprocess:
            return context

        question = context["question"]
        route_result = route_intent(question)
        context["route_result"] = route_result

        if route_result.get("needs_retrieval") is False:
            context["answer"] = build_direct_answer(question, temperature=self.temperature)
            context["needs_retrieval"] = False
            context["stop_pipeline"] = True

        return context

    def stage_rewrite(self, context: PipelineContext) -> PipelineContext:
        if self.skip_preprocess or context.get("stop_pipeline"):
            return context

        context["rewrite_result"] = rewrite_question(context["question"])
        return context

    def stage_pseudo_answer(self, context: PipelineContext) -> PipelineContext:
        if self.skip_preprocess or context.get("stop_pipeline"):
            return context

        context["pseudo_answer_result"] = generate_pseudo_answer(context["question"])
        return context

    def stage_prepare_retrieval(self, context: PipelineContext) -> PipelineContext:
        if context.get("stop_pipeline"):
            return context

        route_result = context.get("route_result")
        rewrite_result = context.get("rewrite_result")
        pseudo_answer_result = context.get("pseudo_answer_result")
        search_queries = []
        keyword_queries = []
        pseudo_answer_text = None

        if rewrite_result:
            search_queries = normalize_list(rewrite_result.get("search_queries"))
            if rewrite_result.get("rewritten_question"):
                search_queries.insert(0, str(rewrite_result["rewritten_question"]))
            keyword_queries = normalize_list(rewrite_result.get("keyword_queries"))

        if pseudo_answer_result:
            pseudo_answer_text = str(pseudo_answer_result.get("pseudo_answer") or "").strip() or None

        final_document_names = unique_keep_order(
            list(context.get("input_document_names") or []) + extract_route_document_names(route_result)
        )
        final_chunk_types = unique_keep_order(
            list(context.get("input_chunk_types") or []) + extract_route_chunk_types(route_result)
        )

        context["retrieval_inputs"] = {
            "search_queries": search_queries,
            "keyword_queries": keyword_queries,
            "pseudo_answer_text": pseudo_answer_text,
            "document_names": final_document_names,
            "chunk_types": final_chunk_types,
        }
        return context

    def stage_apply_permission(self, context: PipelineContext) -> PipelineContext:
        if context.get("stop_pipeline"):
            return context

        retrieval_inputs = apply_permission_to_retrieval_inputs(
            context.get("retrieval_inputs") or {},
            self.permission_context,
        )
        context["retrieval_inputs"] = retrieval_inputs

        if retrieval_inputs.get("permission_denied"):
            denied_names = retrieval_inputs.get("denied_document_names") or []
            context["permission_denied"] = True
            context["answer"] = (
                f"当前模拟权限等级 {self.permission_context['permission_level']} "
                f"无权访问请求的知识库或文档：{', '.join(denied_names)}。"
            )
            context["stop_pipeline"] = True

        return context

    def stage_retrieve(self, context: PipelineContext) -> PipelineContext:
        if context.get("stop_pipeline"):
            return context

        retrieval_inputs = context.get("retrieval_inputs") or {}
        context["retrieval_result"] = retrieve(
            question=context["question"],
            queries=retrieval_inputs.get("search_queries") or [],
            keyword_queries=retrieval_inputs.get("keyword_queries") or [],
            pseudo_answer=retrieval_inputs.get("pseudo_answer_text"),
            document_names=retrieval_inputs.get("document_names") or [],
            chunk_types=retrieval_inputs.get("chunk_types") or [],
            per_query_k=self.per_query_k,
            top_k=self.top_k,
            expand_context=True,
        )
        return context

    def stage_filter_retrieval_permission(self, context: PipelineContext) -> PipelineContext:
        if context.get("stop_pipeline"):
            return context

        context["retrieval_result"] = filter_retrieval_result_by_permission(
            context["retrieval_result"],
            self.permission_context,
        )
        return context

    def stage_compress(self, context: PipelineContext) -> PipelineContext:
        if context.get("stop_pipeline"):
            return context

        context["context_result"] = compress_context(
            retrieval_result=context["retrieval_result"],
            max_context_chars=self.max_context_chars,
            max_blocks=self.max_blocks,
        )
        return context

    def stage_prompt(self, context: PipelineContext) -> PipelineContext:
        if context.get("stop_pipeline"):
            return context

        context["prompt_result"] = build_prompt(
            question=context["question"],
            context_result=context["context_result"],
            route_result=context.get("route_result"),
            rewrite_result=context.get("rewrite_result"),
            pseudo_answer_result=context.get("pseudo_answer_result"),
        )
        return context

    def stage_answer(self, context: PipelineContext) -> PipelineContext:
        if context.get("stop_pipeline"):
            return context

        context["answer"] = self.chat_client.chat_messages(
            messages=context["prompt_result"]["messages"],
            temperature=self.temperature,
        )
        return context

    def build_result(self, context: PipelineContext) -> Dict[str, Any]:
        elapsed_seconds = round(time.time() - context["started_at"], 3)
        permission_summary = {
            "permission_level": self.permission_context.get("permission_level"),
            "allowed_permission_codes": self.permission_context.get("allowed_permission_codes", []),
            "allowed_knowledge_bases": self.permission_context.get("allowed_knowledge_bases", []),
            "denied_knowledge_bases": self.permission_context.get("denied_knowledge_bases", []),
        }

        if context.get("permission_denied"):
            return {
                "question": context["question"],
                "answer": context.get("answer", ""),
                "route": context.get("route_result"),
                "permission": permission_summary,
                "permission_denied": True,
                "needs_retrieval": True,
                "elapsed_seconds": elapsed_seconds,
            }

        if context.get("needs_retrieval") is False:
            return {
                "question": context["question"],
                "answer": context.get("answer", ""),
                "route": context.get("route_result"),
                "permission": permission_summary,
                "needs_retrieval": False,
                "elapsed_seconds": elapsed_seconds,
            }

        retrieval_result = context.get("retrieval_result") or {}
        prompt_result = context.get("prompt_result") or {}

        return {
            "question": context["question"],
            "answer": context.get("answer", ""),
            "citations": prompt_result.get("citations", []),
            "route": context.get("route_result"),
            "rewrite": context.get("rewrite_result"),
            "pseudo_answer": context.get("pseudo_answer_result"),
            "permission": permission_summary,
            "retrieval": {
                "hit_count": retrieval_result.get("hit_count"),
                "candidate_count": retrieval_result.get("candidate_count"),
                "where_filter": retrieval_result.get("where_filter"),
                "queries": retrieval_result.get("queries"),
                "keyword_queries": retrieval_result.get("keyword_queries"),
                "permission": retrieval_result.get("permission", {}),
            },
            "context_stats": prompt_result.get("context_stats", {}),
            "needs_retrieval": True,
            "elapsed_seconds": elapsed_seconds,
        }

    def run(
        self,
        question: str,
        document_names: Optional[Sequence[str]] = None,
        chunk_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        context = self.create_context(
            question=question,
            document_names=document_names,
            chunk_types=chunk_types,
        )

        for stage in self.build_stages():
            context = stage(context)
            if context.get("stop_pipeline"):
                break

        return self.build_result(context)


def print_answer(result: Dict[str, Any], show_trace: bool = False) -> None:
    print("\n回答：")
    print(result.get("answer", ""))

    citations = result.get("citations") or []
    if citations:
        print("\n引用：")
        for citation in citations:
            title_path = " > ".join(citation.get("title_path") or [])
            print(
                f"- [{citation.get('citation_id')}] "
                f"{citation.get('document_name')} | {title_path} | 行 {citation.get('line_range')}"
            )

    if show_trace:
        print("\nTrace：")
        print(json.dumps(result, ensure_ascii=False, indent=2))


def save_trace(result: Dict[str, Any], trace_dir: str | Path) -> Path:
    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)
    file_path = trace_path / f"rag_trace_{int(time.time())}.json"
    file_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def resolve_permission_level(args: argparse.Namespace, interactive: bool = False) -> str:
    if args.permission_level:
        return args.permission_level

    if not interactive:
        return DEFAULT_PERMISSION_LEVEL

    selected = input("请选择模拟权限等级 L1/L2/L3/L4/L5，直接回车默认 L1> ").strip().upper()
    return selected or DEFAULT_PERMISSION_LEVEL


def run_interactive(args: argparse.Namespace) -> None:
    permission_level = resolve_permission_level(args, interactive=True)
    pipeline = RAGPipeline(
        top_k=args.top_k,
        per_query_k=args.per_query_k,
        max_context_chars=args.max_context_chars,
        max_blocks=args.max_blocks,
        temperature=args.temperature,
        skip_preprocess=args.skip_preprocess,
        permission_level=permission_level,
        permission_config=args.permission_config,
    )

    allowed_names = "、".join(pipeline.permission_context.get("allowed_knowledge_bases", []))
    print(f"当前模拟权限等级: {pipeline.permission_context['permission_level']}")
    print(f"可访问知识库: {allowed_names or '无'}")
    print("进入 RAG 循环模式。输入 q / quit / exit / 退出 结束。")
    while True:
        question = input("\n问题> ").strip()
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break

        result = pipeline.run(
            question=question,
            document_names=args.document_name,
            chunk_types=args.chunk_type,
        )
        print_answer(result, show_trace=args.show_trace)
        if args.trace_dir:
            trace_path = save_trace(result, args.trace_dir)
            print(f"\nTrace 已保存: {trace_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="循环式 RAG Pipeline：路由、改写、伪答案、检索、压缩、Prompt、最终回答。")
    parser.add_argument("question_arg", nargs="*", help="单次问题；不传则进入循环模式。")
    parser.add_argument("--question", "-q", help="单次问题。")
    parser.add_argument("--document-name", action="append", default=[], help="限制检索文档名，可重复传入。")
    parser.add_argument("--chunk-type", action="append", default=[], help="限制 chunk_type，可重复传入。")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--per-query-k", type=int, default=DEFAULT_PER_QUERY_K)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过路由/改写/伪答案，直接检索原问题。")
    parser.add_argument("--permission-level", choices=["L1", "L2", "L3", "L4", "L5"], help="模拟用户权限等级。")
    parser.add_argument("--permission-config", default=str(DEFAULT_PERMISSION_CONFIG_PATH), help="知识库权限配置 JSON 路径。")
    parser.add_argument("--show-trace", action="store_true", help="打印完整链路 JSON。")
    parser.add_argument("--trace-dir", help="保存每轮完整链路 JSON 的目录。")
    parser.add_argument("--output", "-o", help="单次问题时输出 JSON 文件。")
    return parser.parse_args()


def get_single_question(args: argparse.Namespace) -> str:
    return (args.question or " ".join(args.question_arg)).strip()


def main() -> Optional[Dict[str, Any]]:
    args = parse_args()
    question = get_single_question(args)

    if not question:
        run_interactive(args)
        return None

    pipeline = RAGPipeline(
        top_k=args.top_k,
        per_query_k=args.per_query_k,
        max_context_chars=args.max_context_chars,
        max_blocks=args.max_blocks,
        temperature=args.temperature,
        skip_preprocess=args.skip_preprocess,
        permission_level=resolve_permission_level(args),
        permission_config=args.permission_config,
    )
    result = pipeline.run(
        question=question,
        document_names=args.document_name,
        chunk_types=args.chunk_type,
    )

    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print_answer(result, show_trace=args.show_trace)

    if args.trace_dir:
        trace_path = save_trace(result, args.trace_dir)
        print(f"\nTrace 已保存: {trace_path}")

    return result


if __name__ == "__main__":
    main()
