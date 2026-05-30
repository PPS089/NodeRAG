"""
EnhancedRAGPipeline — 增强型 RAG 管线（独立实现，不继承 RAGPipeline）。

集成三项增强能力：

1. BM25State  — BM25 词表持久化，用于检索后 BM25 重打分（跨进程 IDF 一致）
2. AutoMerger — 父子层级自动合并（多子块命中同一父块时，替换为父块扩大上下文）
3. GradeAndRewrite — 文档相关性评估 + 不通过时查询改写回流重检

预处理优化：
  Rewrite + PseudoAnswer 两个独立 LLM 调用使用 ThreadPoolExecutor 并行执行。

新增 Stage 顺序：
  stage_route → stage_preprocess_parallel → stage_prepare_retrieval
  → stage_apply_permission → stage_retrieve → stage_filter_retrieval_permission
  → stage_bm25_rescore   ← NEW  (BM25 持久化重打分)
  → stage_auto_merge     ← NEW  (父子层级合并)
  → stage_grade          ← NEW  (LLM 相关性评估)
      ↓ 不通过 → rewrite → 回流到 stage_retrieve（最多 retry 次）
  → stage_rerank → stage_compress → stage_prompt → stage_answer

用法：
  pipeline = EnhancedRAGPipeline(
      enable_bm25_rescore=True,
      enable_auto_merge=True,
      enable_grade_rewrite=True,
      grade_max_retry=2,
      ...
  )
  result = pipeline.run("你的问题")
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- 常量 & 工具函数（从 RAGPipeline 模块导入） ----
from rags.RAGPipeline import (  # noqa: E402
    DEFAULT_MAX_BLOCKS,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_PER_QUERY_K,
    DEFAULT_TOP_K,
    PipelineContext,
    build_direct_answer,
    extract_route_chunk_types,
    extract_route_document_names,
    normalize_list,
    unique_keep_order,
)

# ---- 预处理节点 ----
from nodes.query.LLMClient import OpenAICompatibleChatClient  # noqa: E402
from nodes.query.IntentRouter import route_intent  # noqa: E402
from nodes.query.QuestionRewriter import rewrite_question  # noqa: E402
from nodes.query.PseudoAnswer import generate_pseudo_answer  # noqa: E402

# ---- 检索 & 后处理节点 ----
from nodes.retrieval.ChromaRetriever import retrieve  # noqa: E402
from nodes.context.ContextCompressor import compress_context  # noqa: E402
from nodes.prompt.PromptBuilder import build_prompt  # noqa: E402
from nodes.rerank.Reranker import (  # noqa: E402
    DEFAULT_CONFIG_PATH as DEFAULT_RERANK_CONFIG_PATH,
    DEFAULT_RERANK_TOP_N,
    load_rerank_config,
    rerank_retrieval_result,
)

# ---- 权限 ----
from nodes.auth.PermissionGuard import (  # noqa: E402
    DEFAULT_CONFIG_PATH as DEFAULT_PERMISSION_CONFIG_PATH,
    DEFAULT_PERMISSION_LEVEL,
    apply_permission_to_retrieval_inputs,
    build_permission_context,
    filter_retrieval_result_by_permission,
)

# ---- 日志 ----
from utils.RAGLogger import DEFAULT_LOG_FILE, RAGLogger, new_trace_id  # noqa: E402

# ---- 增强模块 ----
from nodes.retrieval.AutoMerger import (  # noqa: E402
    integrate_auto_merge_to_retrieval,
    load_auto_merge_config,
)
from nodes.retrieval.BM25State import (  # noqa: E402
    BM25StateManager,
    get_bm25_state_manager,
)
from nodes.retrieval.GradeAndRewrite import (  # noqa: E402
    DEFAULT_MAX_RETRY,
    grade_retrieval_result,
    rewrite_question as grade_rewrite_question,
)

# ---------------------------------------------------------------------------
# 本地常量
# ---------------------------------------------------------------------------

DEFAULT_AUTO_MERGE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "auto_merge_config.json"
PIPELINE_MODES = {"standard", "fast"}


# ---------------------------------------------------------------------------
# EnhancedRAGPipeline（独立实现）
# ---------------------------------------------------------------------------

class EnhancedRAGPipeline:
    """增强型 RAG Pipeline：BM25 持久化 + AutoMerge + Grade&Rewrite 回流 + 并行预处理。"""

    def __init__(
        self,
        # ---- 管线参数 ----
        top_k: int = DEFAULT_TOP_K,
        per_query_k: int = DEFAULT_PER_QUERY_K,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_blocks: int = DEFAULT_MAX_BLOCKS,
        temperature: float = 0.2,
        skip_preprocess: bool = False,
        pipeline_mode: str = "standard",
        permission_level: str = DEFAULT_PERMISSION_LEVEL,
        permission_config: str | Path | None = None,
        rerank_config: str | Path | None = None,
        rerank_top_n: Optional[int] = None,
        disable_second_stage_rerank: bool = False,
        log_enabled: bool = True,
        log_file: str | Path | None = None,
        # ---- 增强功能开关 ----
        enable_bm25_rescore: bool = True,
        enable_auto_merge: bool = True,
        enable_grade_rewrite: bool = True,
        # ---- AutoMerger 参数 ----
        auto_merge_config: str | Path | None = None,
        auto_merge_threshold: int = 2,
        auto_merge_max_steps: int = 2,
        # ---- GradeAndRewrite 参数 ----
        grade_max_retry: int = DEFAULT_MAX_RETRY,
        grade_use_structured_output: bool = True,
        # ---- BM25State 参数 ----
        bm25_state_path: str | Path | None = None,
    ) -> None:
        # ---- 管线核心属性 ----
        self.top_k = top_k
        self.per_query_k = per_query_k
        self.max_context_chars = max_context_chars
        self.max_blocks = max_blocks
        self.temperature = temperature
        self.pipeline_mode = self.resolve_pipeline_mode(pipeline_mode, skip_preprocess)
        self.skip_preprocess = self.pipeline_mode == "fast"
        self.rerank_config_path = rerank_config or ""
        self.rerank_config = load_rerank_config(rerank_config or "")
        self.rerank_top_n = int(
            rerank_top_n
            if rerank_top_n is not None
            else self.rerank_config.get("rerank_top_n", DEFAULT_RERANK_TOP_N)
        )
        self.disable_second_stage_rerank = disable_second_stage_rerank
        self.permission_context = build_permission_context(permission_level, permission_config or "")
        self.logger = RAGLogger(log_file=log_file or "", enabled=log_enabled)
        self.chat_client = OpenAICompatibleChatClient()

        # ---- 增强功能开关 ----
        self.enable_bm25_rescore = enable_bm25_rescore
        self.enable_auto_merge = enable_auto_merge
        self.enable_grade_rewrite = enable_grade_rewrite

        # ---- AutoMerger 参数 ----
        self.auto_merge_config_path = auto_merge_config or DEFAULT_AUTO_MERGE_CONFIG
        self.auto_merge_config = load_auto_merge_config(self.auto_merge_config_path)
        self.auto_merge_threshold = auto_merge_threshold
        self.auto_merge_max_steps = auto_merge_max_steps

        # ---- GradeAndRewrite 参数 ----
        self.grade_max_retry = grade_max_retry
        self.grade_use_structured_output = grade_use_structured_output

        # ---- BM25State ----
        self.bm25_manager: Optional[BM25StateManager] = None
        if enable_bm25_rescore:
            self.bm25_manager = (
                get_bm25_state_manager()
                if bm25_state_path is None
                else BM25StateManager(state_path=bm25_state_path)
            )

    @staticmethod
    def resolve_pipeline_mode(pipeline_mode: str, skip_preprocess: bool = False) -> str:
        if skip_preprocess:
            return "fast"
        mode = str(pipeline_mode or "standard").strip().lower()
        if mode not in PIPELINE_MODES:
            raise ValueError(f"pipeline_mode 仅支持 {sorted(PIPELINE_MODES)}，当前值: {pipeline_mode}")
        return mode

    # ------------------------------------------------------------------
    # Context 管理
    # ------------------------------------------------------------------

    def create_context(
        self,
        question: str,
        document_names: Optional[Sequence[str]] = None,
        chunk_types: Optional[Sequence[str]] = None,
    ) -> PipelineContext:
        return {
            "trace_id": new_trace_id(),
            "pipeline_mode": self.pipeline_mode,
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
            "rerank_result": None,
            "context_result": None,
            "prompt_result": None,
            "answer": None,
            "permission_context": self.permission_context,
            "permission_denied": False,
        }

    def collect_stage_metrics(self, context: PipelineContext) -> Dict[str, Any]:
        retrieval_inputs = context.get("retrieval_inputs") or {}
        retrieval_result = context.get("retrieval_result") or {}
        context_result = context.get("context_result") or {}
        prompt_result = context.get("prompt_result") or {}

        metrics = {
            "pipeline_mode": self.pipeline_mode,
            "permission_level": self.permission_context.get("permission_level"),
            "stop_pipeline": bool(context.get("stop_pipeline")),
            "needs_retrieval": bool(context.get("needs_retrieval")),
            "permission_denied": bool(context.get("permission_denied")),
            "document_filter_count": len(retrieval_inputs.get("document_names") or []),
            "chunk_type_filter_count": len(retrieval_inputs.get("chunk_types") or []),
            "search_query_count": len(retrieval_inputs.get("search_queries") or []),
            "keyword_query_count": len(retrieval_inputs.get("keyword_queries") or []),
            "candidate_count": retrieval_result.get("candidate_count"),
            "hit_count": retrieval_result.get("hit_count"),
            "raw_vector_hit_count": retrieval_result.get("raw_vector_hit_count"),
            "raw_bm25_hit_count": retrieval_result.get("raw_bm25_hit_count"),
            "rerank": retrieval_result.get("rerank", {}),
            "context_block_count": len(context_result.get("context_blocks") or []),
            "citation_count": len(context_result.get("citations") or []),
            "message_count": len(prompt_result.get("messages") or []),
        }

        # AutoMerger 指标
        auto_merge_meta = retrieval_result.get("auto_merge") or {}
        metrics["auto_merge_enabled"] = auto_merge_meta.get("enabled", False)
        metrics["auto_merge_total_merged"] = auto_merge_meta.get("total_merged_count", 0)

        # Grade 指标
        grade_result = context.get("grade_result") or {}
        metrics["grade_score"] = grade_result.get("grade_score")
        metrics["grade_passed"] = grade_result.get("passed")
        metrics["grade_attempts"] = len(context.get("grade_rewrite_history") or []) + 1

        # BM25 重打分指标
        bm25_rescore = retrieval_result.get("bm25_rescore") or {}
        metrics["bm25_rescore_enabled"] = bm25_rescore.get("enabled", False)
        metrics["bm25_rescore_skipped"] = bm25_rescore.get("skipped", False)

        return metrics

    def run_stage(self, stage, context: PipelineContext) -> PipelineContext:
        stage_name = getattr(stage, "__name__", str(stage))
        started_at = self.logger.stage_start(
            trace_id=context["trace_id"],
            stage=stage_name,
            metrics=self.collect_stage_metrics(context),
        )
        try:
            next_context = stage(context)
        except Exception as error:
            self.logger.stage_error(
                trace_id=context["trace_id"],
                stage=stage_name,
                started_at=started_at,
                error=error,
            )
            raise

        self.logger.stage_end(
            trace_id=next_context["trace_id"],
            stage=stage_name,
            started_at=started_at,
            metrics=self.collect_stage_metrics(next_context),
        )
        return next_context

    # ------------------------------------------------------------------
    # Stage 列表
    # ------------------------------------------------------------------

    def build_stages(self) -> list:
        """返回增强后的 stage 列表，预处理阶段已合并为并行调用。"""
        return [
            self.stage_route,
            self.stage_preprocess_parallel,  # ← 并行 Rewrite + PseudoAnswer
            self.stage_prepare_retrieval,
            self.stage_apply_permission,
            self.stage_retrieve,
            self.stage_filter_retrieval_permission,
            self.stage_bm25_rescore,         # ← 新增 1
            self.stage_auto_merge,           # ← 新增 2
            self.stage_grade,                # ← 新增 3
            self.stage_rerank,
            self.stage_compress,
            self.stage_prompt,
            self.stage_answer,
        ]

    # ------------------------------------------------------------------
    # Stage: 路由
    # ------------------------------------------------------------------

    def stage_route(self, context: PipelineContext) -> PipelineContext:
        if self.pipeline_mode == "fast":
            return context

        question = context["question"]
        route_result = route_intent(question)
        context["route_result"] = route_result

        if route_result.get("needs_retrieval") is False:
            context["answer"] = build_direct_answer(question, temperature=self.temperature)
            context["needs_retrieval"] = False
            context["stop_pipeline"] = True

        return context

    # ------------------------------------------------------------------
    # Stage: 并行预处理（Rewrite + PseudoAnswer） ← 优化点
    # ------------------------------------------------------------------

    def stage_preprocess_parallel(self, context: PipelineContext) -> PipelineContext:
        """并行执行 Rewrite 和 PseudoAnswer 两个独立 LLM 调用。

        两者都只依赖 context["question"]，互不依赖，使用 ThreadPoolExecutor
        并行发起 HTTP 请求，将串行延迟从 (T_rewrite + T_pseudo) 降至
        max(T_rewrite, T_pseudo)。
        """
        if self.pipeline_mode == "fast" or context.get("stop_pipeline"):
            return context

        question = context["question"]

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_rewrite = executor.submit(rewrite_question, question)
            future_pseudo = executor.submit(generate_pseudo_answer, question)

            # result() 会 propagate 异常，行为与串行一致
            context["rewrite_result"] = future_rewrite.result()
            context["pseudo_answer_result"] = future_pseudo.result()

        return context

    # ------------------------------------------------------------------
    # Stage: 准备检索参数
    # ------------------------------------------------------------------

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
            list(context.get("input_document_names") or [])
            + extract_route_document_names(route_result)
        )
        final_chunk_types = unique_keep_order(
            list(context.get("input_chunk_types") or [])
            + extract_route_chunk_types(route_result)
        )

        context["retrieval_inputs"] = {
            "search_queries": search_queries,
            "keyword_queries": keyword_queries,
            "pseudo_answer_text": pseudo_answer_text,
            "document_names": final_document_names,
            "chunk_types": final_chunk_types,
        }
        return context

    # ------------------------------------------------------------------
    # Stage: 权限过滤
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Stage: 检索
    # ------------------------------------------------------------------

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
            rerank_pool_size=max(self.rerank_top_n, self.top_k),
            top_k=max(self.rerank_top_n, self.top_k),
            expand_context=True,
        )
        return context

    # ------------------------------------------------------------------
    # Stage: 检索结果权限过滤
    # ------------------------------------------------------------------

    def stage_filter_retrieval_permission(self, context: PipelineContext) -> PipelineContext:
        if context.get("stop_pipeline"):
            return context

        context["retrieval_result"] = filter_retrieval_result_by_permission(
            context["retrieval_result"],
            self.permission_context,
        )
        return context

    # ------------------------------------------------------------------
    # Stage: 二阶段重排
    # ------------------------------------------------------------------

    def stage_rerank(self, context: PipelineContext) -> PipelineContext:
        if context.get("stop_pipeline") or self.disable_second_stage_rerank:
            return context

        retrieval_inputs = context.get("retrieval_inputs") or {}
        query_texts = unique_keep_order(
            [context["question"]]
            + list(retrieval_inputs.get("search_queries") or [])
            + list(retrieval_inputs.get("keyword_queries") or [])
            + (
                [retrieval_inputs.get("pseudo_answer_text")]
                if retrieval_inputs.get("pseudo_answer_text")
                else []
            )
        )
        context["retrieval_result"] = rerank_retrieval_result(
            retrieval_result=context["retrieval_result"],
            query_texts=query_texts,
            rerank_top_n=self.rerank_top_n,
            final_top_k=self.top_k,
            use_mmr=True,
            table_aware=True,
            config_path=self.rerank_config_path,
        )
        context["rerank_result"] = context["retrieval_result"].get("rerank", {})
        return context

    # ------------------------------------------------------------------
    # Stage: 上下文压缩
    # ------------------------------------------------------------------

    def stage_compress(self, context: PipelineContext) -> PipelineContext:
        if context.get("stop_pipeline"):
            return context

        context["context_result"] = compress_context(
            retrieval_result=context["retrieval_result"],
            max_context_chars=self.max_context_chars,
            max_blocks=self.max_blocks,
        )
        return context

    # ------------------------------------------------------------------
    # Stage: Prompt 构建
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Stage: 最终回答
    # ------------------------------------------------------------------

    def stage_answer(self, context: PipelineContext) -> PipelineContext:
        if context.get("stop_pipeline"):
            return context

        context["answer"] = self.chat_client.chat_messages(
            messages=context["prompt_result"]["messages"],
            temperature=self.temperature,
        )
        return context

    # ------------------------------------------------------------------
    # 增强 Stage 1：BM25 持久化重打分
    # ------------------------------------------------------------------

    def stage_bm25_rescore(self, context: PipelineContext) -> PipelineContext:
        """使用 BM25StateManager 的持久化 IDF 对召回结果重打分。

        仅在 enable_bm25_rescore=True 且 BM25StateManager 可用时生效。
        如果词表为空（未做过 ingestion），则跳过。
        """
        if context.get("stop_pipeline"):
            return context

        if not self.enable_bm25_rescore or self.bm25_manager is None:
            return context

        retrieval_result = context.get("retrieval_result") or {}
        hits = retrieval_result.get("hits") or []
        if not hits:
            return context

        stats = self.bm25_manager.get_stats()
        if stats.get("total_docs", 0) == 0:
            retrieval_result["bm25_rescore"] = {
                "enabled": True,
                "skipped": True,
                "reason": "bm25_vocab_empty",
            }
            context["retrieval_result"] = retrieval_result
            return context

        retrieval_inputs = context.get("retrieval_inputs") or {}
        query_texts = unique_keep_order(
            [context["question"]]
            + list(retrieval_inputs.get("search_queries") or [])
            + list(retrieval_inputs.get("keyword_queries") or [])
        )
        combined_query = " ".join(query_texts)

        rescored = self.bm25_manager.score_documents(combined_query, hits)

        retrieval_result["hits"] = rescored
        retrieval_result["bm25_rescore"] = {
            "enabled": True,
            "skipped": False,
            "total_docs": stats["total_docs"],
            "vocab_size": stats["vocab_size"],
            "avg_doc_len": stats["avg_doc_len"],
        }
        context["retrieval_result"] = retrieval_result
        return context

    # ------------------------------------------------------------------
    # 增强 Stage 2：父子层级自动合并
    # ------------------------------------------------------------------

    def stage_auto_merge(self, context: PipelineContext) -> PipelineContext:
        """将命中同一父块的多个子块合并为父块，扩大上下文。

        仅在 enable_auto_merge=True 时生效。
        """
        if context.get("stop_pipeline"):
            return context

        if not self.enable_auto_merge:
            return context

        retrieval_result = context.get("retrieval_result") or {}
        hits = retrieval_result.get("hits") or []
        if not hits:
            return context

        retrieval_result = integrate_auto_merge_to_retrieval(
            retrieval_result=retrieval_result,
            top_k=max(self.rerank_top_n, self.top_k),
            merge_threshold=self.auto_merge_threshold,
            max_steps=self.auto_merge_max_steps,
            config_path=self.auto_merge_config_path,
        )
        context["retrieval_result"] = retrieval_result
        return context

    # ------------------------------------------------------------------
    # 增强 Stage 3：文档相关性评估
    # ------------------------------------------------------------------

    def stage_grade(self, context: PipelineContext) -> PipelineContext:
        """LLM 评估检索结果与问题的相关性。

        仅在 enable_grade_rewrite=True 时生效。
        设置 context["grade_passed"] 供外层回流循环判断。
        """
        if context.get("stop_pipeline"):
            return context

        if not self.enable_grade_rewrite:
            context["grade_passed"] = True
            context["grade_result"] = {"skipped": True, "reason": "grade_rewrite_disabled"}
            return context

        retrieval_result = context.get("retrieval_result") or {}
        hits = retrieval_result.get("hits") or []

        if not hits:
            context["grade_passed"] = False
            context["grade_result"] = {
                "grade_score": "no",
                "grade_reasoning": "检索结果为空",
                "passed": False,
                "relevant_count": 0,
                "total_count": 0,
            }
            return context

        graded = grade_retrieval_result(
            retrieval_result=retrieval_result,
            question=context["question"],
            use_structured_output=self.grade_use_structured_output,
        )

        grade_info = graded.get("grade", {})
        context["grade_passed"] = grade_info.get("passed", False)
        context["grade_result"] = grade_info
        context["retrieval_result"] = graded
        return context

    # ------------------------------------------------------------------
    # 查询改写（供回流使用）
    # ------------------------------------------------------------------

    def _rewrite_and_prepare_retry(self, context: PipelineContext) -> PipelineContext:
        """触发查询改写，更新 retrieval_inputs 中的 search_queries 供重新检索。

        调用 GradeAndRewrite.rewrite_question 获取扩展查询，
        将结果注入 context["retrieval_inputs"]["search_queries"]。
        """
        question = context["question"]
        rewrite_result = grade_rewrite_question(question)

        rewrite_history: list = context.setdefault("grade_rewrite_history", [])
        rewrite_history.append(rewrite_result)

        expanded_queries: list = rewrite_result.get("expanded_queries") or [question]
        retrieval_inputs = context.get("retrieval_inputs") or {}

        existing = list(retrieval_inputs.get("search_queries") or [])
        new_queries = unique_keep_order(existing + expanded_queries)

        retrieval_inputs["search_queries"] = new_queries
        retrieval_inputs["grade_rewrite_strategy"] = rewrite_result.get("strategy", "simple")
        retrieval_inputs["grade_rewrite_step_back_question"] = rewrite_result.get("step_back_question", "")
        retrieval_inputs["grade_rewrite_step_back_answer"] = rewrite_result.get("step_back_answer", "")
        retrieval_inputs["grade_rewrite_hypothetical_doc"] = rewrite_result.get("hypothetical_document", "")

        context["retrieval_inputs"] = retrieval_inputs
        context["retrieval_result"] = None

        self.logger.log(
            event="grade_rewrite_retry",
            trace_id=context["trace_id"],
            strategy=rewrite_result.get("strategy"),
            expanded_query_count=len(expanded_queries),
            attempt=len(rewrite_history),
        )

        return context

    # ------------------------------------------------------------------
    # 结果构建
    # ------------------------------------------------------------------

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
                "trace_id": context["trace_id"],
                "pipeline_mode": context.get("pipeline_mode", self.pipeline_mode),
                "question": context["question"],
                "answer": context.get("answer", ""),
                "route": context.get("route_result"),
                "permission": permission_summary,
                "permission_denied": True,
                "needs_retrieval": True,
                "elapsed_seconds": elapsed_seconds,
                "enhanced": {
                    "bm25_rescore_enabled": self.enable_bm25_rescore,
                    "auto_merge_enabled": self.enable_auto_merge,
                    "grade_rewrite_enabled": self.enable_grade_rewrite,
                },
            }

        if context.get("needs_retrieval") is False:
            return {
                "trace_id": context["trace_id"],
                "pipeline_mode": context.get("pipeline_mode", self.pipeline_mode),
                "question": context["question"],
                "answer": context.get("answer", ""),
                "route": context.get("route_result"),
                "permission": permission_summary,
                "needs_retrieval": False,
                "elapsed_seconds": elapsed_seconds,
                "enhanced": {
                    "bm25_rescore_enabled": self.enable_bm25_rescore,
                    "auto_merge_enabled": self.enable_auto_merge,
                    "grade_rewrite_enabled": self.enable_grade_rewrite,
                },
            }

        retrieval_result = context.get("retrieval_result") or {}
        prompt_result = context.get("prompt_result") or {}

        result = {
            "trace_id": context["trace_id"],
            "pipeline_mode": context.get("pipeline_mode", self.pipeline_mode),
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
                "rerank": retrieval_result.get("rerank", {}),
            },
            "context_stats": prompt_result.get("context_stats", {}),
            "needs_retrieval": True,
            "elapsed_seconds": elapsed_seconds,
            # 增强字段
            "auto_merge": retrieval_result.get("auto_merge") or {},
            "grade": {
                "result": context.get("grade_result"),
                "rewrite_history": context.get("grade_rewrite_history") or [],
            },
            "bm25_rescore": retrieval_result.get("bm25_rescore") or {},
            "enhanced": {
                "bm25_rescore_enabled": self.enable_bm25_rescore,
                "auto_merge_enabled": self.enable_auto_merge,
                "grade_rewrite_enabled": self.enable_grade_rewrite,
                "grade_max_retry": self.grade_max_retry,
            },
        }

        return result

    # ------------------------------------------------------------------
    # 核心：run() 实现 Grade → Rewrite 回流循环
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        document_names: Optional[Sequence[str]] = None,
        chunk_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """执行增强版 RAG Pipeline。

        预处理阶段：Route → 并行(Rewrite + PseudoAnswer) → Prepare → Permission
        检索阶段：Retrieve → BM25 → AutoMerge → Grade（不通过则 Rewrite 回流）
        后处理阶段：Rerank → Compress → Prompt → Answer
        """
        context = self.create_context(
            question=question,
            document_names=document_names,
            chunk_types=chunk_types,
        )
        self.logger.log(
            event="pipeline_start",
            trace_id=context["trace_id"],
            pipeline_mode=self.pipeline_mode,
            question_length=len(question),
            permission_level=self.permission_context.get("permission_level"),
            enhanced={
                "bm25_rescore": self.enable_bm25_rescore,
                "auto_merge": self.enable_auto_merge,
                "grade_rewrite": self.enable_grade_rewrite,
                "grade_max_retry": self.grade_max_retry,
            },
        )

        try:
            # ---- Phase 1: 预处理阶段 ----
            pre_retrieval_stages = [
                self.stage_route,
                self.stage_preprocess_parallel,  # ← 并行
                self.stage_prepare_retrieval,
                self.stage_apply_permission,
            ]
            for stage in pre_retrieval_stages:
                context = self.run_stage(stage, context)
                if context.get("stop_pipeline"):
                    return self._finish(context)

            # ---- Phase 2: 检索 → 评估 回流循环 ----
            retrieval_stages = [
                self.stage_retrieve,
                self.stage_filter_retrieval_permission,
                self.stage_bm25_rescore,
                self.stage_auto_merge,
                self.stage_grade,
            ]

            max_attempts = self.grade_max_retry + 1 if self.enable_grade_rewrite else 1
            for attempt in range(max_attempts):
                for stage in retrieval_stages:
                    context = self.run_stage(stage, context)
                    if context.get("stop_pipeline"):
                        return self._finish(context)

                if context.get("grade_passed", True):
                    self.logger.log(
                        event="grade_passed",
                        trace_id=context["trace_id"],
                        attempt=attempt + 1,
                    )
                    break

                if attempt < max_attempts - 1:
                    self.logger.log(
                        event="grade_retry",
                        trace_id=context["trace_id"],
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        grade_score=context.get("grade_result", {}).get("grade_score"),
                    )
                    context = self._rewrite_and_prepare_retry(context)
                else:
                    self.logger.log(
                        event="grade_exhausted",
                        trace_id=context["trace_id"],
                        total_attempts=max_attempts,
                        final_grade=context.get("grade_result", {}).get("grade_score"),
                    )

            # ---- Phase 3: 后处理阶段 ----
            post_retrieval_stages = [
                self.stage_rerank,
                self.stage_compress,
                self.stage_prompt,
                self.stage_answer,
            ]
            for stage in post_retrieval_stages:
                context = self.run_stage(stage, context)
                if context.get("stop_pipeline"):
                    break

            return self._finish(context)

        except Exception as error:
            self.logger.log(
                event="pipeline_error",
                trace_id=context["trace_id"],
                status="error",
                elapsed_ms=round((time.time() - context["started_at"]) * 1000, 3),
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise

    def _finish(self, context: PipelineContext) -> Dict[str, Any]:
        """统一收尾：构建结果 + 打日志。"""
        result = self.build_result(context)
        self.logger.log(
            event="pipeline_end",
            trace_id=context["trace_id"],
            status="ok",
            elapsed_ms=round((time.time() - context["started_at"]) * 1000, 3),
            metrics=self.collect_stage_metrics(context),
        )
        return result


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    from rags.RAGPipeline import (
        EXIT_COMMANDS,
        print_answer,
        resolve_permission_level,
        save_trace,
    )

    parser = argparse.ArgumentParser(
        description="增强型 RAG Pipeline：BM25 持久化 + AutoMerge + Grade&Rewrite 回流 + 并行预处理"
    )
    parser.add_argument("question_arg", nargs="*", help="单次问题；不传则进入循环模式。")
    parser.add_argument("--question", "-q", help="单次问题。")
    parser.add_argument("--document-name", action="append", default=[], help="限制检索文档名。")
    parser.add_argument("--chunk-type", action="append", default=[], help="限制 chunk_type。")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--per-query-k", type=int, default=DEFAULT_PER_QUERY_K)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mode", choices=["standard", "fast"], default="standard")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--permission-level", choices=["L1", "L2", "L3", "L4", "L5"])
    parser.add_argument("--permission-config", default=str(DEFAULT_PERMISSION_CONFIG_PATH))
    parser.add_argument("--rerank-config", default=str(DEFAULT_RERANK_CONFIG_PATH))
    parser.add_argument("--rerank-top-n", type=int)
    parser.add_argument("--no-second-stage-rerank", action="store_true")
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE))
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument("--trace-dir")
    parser.add_argument("--output", "-o")

    # ---- 增强功能参数 ----
    parser.add_argument("--no-bm25-rescore", action="store_true", help="关闭 BM25 持久化重打分")
    parser.add_argument("--no-auto-merge", action="store_true", help="关闭父子层级自动合并")
    parser.add_argument("--no-grade-rewrite", action="store_true", help="关闭相关性评估与回流")
    parser.add_argument("--auto-merge-config", help="AutoMerge 配置文件路径")
    parser.add_argument("--auto-merge-threshold", type=int, default=2)
    parser.add_argument("--auto-merge-max-steps", type=int, default=2)
    parser.add_argument("--grade-max-retry", type=int, default=DEFAULT_MAX_RETRY)
    parser.add_argument("--grade-no-structured", action="store_true", help="Grade 阶段不使用结构化输出")
    parser.add_argument("--bm25-state-path", help="BM25 状态文件路径")

    args = parser.parse_args()

    def _resolve_mode() -> str:
        if args.fast or args.skip_preprocess:
            return "fast"
        return args.mode

    def _get_question() -> str:
        return (args.question or " ".join(args.question_arg)).strip()

    question = _get_question()

    if not question:
        perm_level = resolve_permission_level(args, interactive=True)
        pipeline = EnhancedRAGPipeline(
            top_k=args.top_k,
            per_query_k=args.per_query_k,
            max_context_chars=args.max_context_chars,
            max_blocks=args.max_blocks,
            temperature=args.temperature,
            pipeline_mode=_resolve_mode(),
            permission_level=perm_level,
            permission_config=args.permission_config,
            rerank_config=args.rerank_config,
            rerank_top_n=args.rerank_top_n,
            disable_second_stage_rerank=args.no_second_stage_rerank,
            log_enabled=not args.no_log,
            log_file=args.log_file,
            enable_bm25_rescore=not args.no_bm25_rescore,
            enable_auto_merge=not args.no_auto_merge,
            enable_grade_rewrite=not args.no_grade_rewrite,
            auto_merge_config=args.auto_merge_config,
            auto_merge_threshold=args.auto_merge_threshold,
            auto_merge_max_steps=args.auto_merge_max_steps,
            grade_max_retry=args.grade_max_retry,
            grade_use_structured_output=not args.grade_no_structured,
            bm25_state_path=args.bm25_state_path,
        )

        allowed_names = "、".join(pipeline.permission_context.get("allowed_knowledge_bases", []))
        print(f"当前模拟权限等级: {pipeline.permission_context['permission_level']}")
        print(f"当前 RAG 模式: {pipeline.pipeline_mode}")
        print(
            f"增强功能: BM25重打分={pipeline.enable_bm25_rescore}, "
            f"AutoMerge={pipeline.enable_auto_merge}, "
            f"Grade&Rewrite={pipeline.enable_grade_rewrite}"
        )
        print(f"预处理: Rewrite + PseudoAnswer 并行")
        print(f"可访问知识库: {allowed_names or '无'}")
        print("进入 RAG 循环模式。输入 q / quit / exit / 退出 结束。")

        while True:
            q = input("\n问题> ").strip()
            if not q:
                continue
            if q.lower() in EXIT_COMMANDS:
                break

            result = pipeline.run(
                question=q,
                document_names=args.document_name,
                chunk_types=args.chunk_type,
            )
            print_answer(result, show_trace=args.show_trace)
            if args.trace_dir:
                trace_path = save_trace(result, args.trace_dir)
                print(f"\nTrace 已保存: {trace_path}")

    else:
        pipeline = EnhancedRAGPipeline(
            top_k=args.top_k,
            per_query_k=args.per_query_k,
            max_context_chars=args.max_context_chars,
            max_blocks=args.max_blocks,
            temperature=args.temperature,
            pipeline_mode=_resolve_mode(),
            permission_level=resolve_permission_level(args),
            permission_config=args.permission_config,
            rerank_config=args.rerank_config,
            rerank_top_n=args.rerank_top_n,
            disable_second_stage_rerank=args.no_second_stage_rerank,
            log_enabled=not args.no_log,
            log_file=args.log_file,
            enable_bm25_rescore=not args.no_bm25_rescore,
            enable_auto_merge=not args.no_auto_merge,
            enable_grade_rewrite=not args.no_grade_rewrite,
            auto_merge_config=args.auto_merge_config,
            auto_merge_threshold=args.auto_merge_threshold,
            auto_merge_max_steps=args.auto_merge_max_steps,
            grade_max_retry=args.grade_max_retry,
            grade_use_structured_output=not args.grade_no_structured,
            bm25_state_path=args.bm25_state_path,
        )

        result = pipeline.run(
            question=question,
            document_names=args.document_name,
            chunk_types=args.chunk_type,
        )

        if args.output:
            Path(args.output).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            print_answer(result, show_trace=args.show_trace)

        if args.trace_dir:
            trace_path = save_trace(result, args.trace_dir)
            print(f"\nTrace 已保存: {trace_path}")
