"""
EnhancedRAGPipeline — 增强型 RAG 管线。

在 RAGPipeline 基础上集成三项新增能力（不修改 RAGPipeline.py）：

1. BM25State  — BM25 词表持久化，用于检索后 BM25 重打分（跨进程 IDF 一致）
2. AutoMerger — 父子层级自动合并（多子块命中同一父块时，替换为父块扩大上下文）
3. GradeAndRewrite — 文档相关性评估 + 不通过时查询改写回流重检

新增 Stage 顺序：
  stage_route → stage_rewrite → stage_pseudo_answer → stage_prepare_retrieval
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rags.RAGPipeline import (  # noqa: E402
    DEFAULT_MAX_BLOCKS,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_PER_QUERY_K,
    DEFAULT_TOP_K,
    RAGPipeline,
    PipelineContext,
    build_direct_answer,
    normalize_list,
    unique_keep_order,
)

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
# 默认配置
# ---------------------------------------------------------------------------

DEFAULT_AUTO_MERGE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "auto_merge_config.json"


# ---------------------------------------------------------------------------
# EnhancedRAGPipeline
# ---------------------------------------------------------------------------

class EnhancedRAGPipeline(RAGPipeline):
    """增强型 RAG Pipeline：BM25 持久化 + AutoMerge + Grade&Rewrite 回流。"""

    def __init__(
        self,
        # ---- RAGPipeline 原有参数 ----
        top_k: int = DEFAULT_TOP_K,
        per_query_k: int = DEFAULT_PER_QUERY_K,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_blocks: int = DEFAULT_MAX_BLOCKS,
        temperature: float = 0.2,
        skip_preprocess: bool = False,
        pipeline_mode: str = "standard",
        permission_level: str = "L1",
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
        # 初始化父类（RAGPipeline）
        super().__init__(
            top_k=top_k,
            per_query_k=per_query_k,
            max_context_chars=max_context_chars,
            max_blocks=max_blocks,
            temperature=temperature,
            skip_preprocess=skip_preprocess,
            pipeline_mode=pipeline_mode,
            permission_level=permission_level,
            permission_config=permission_config or "",
            rerank_config=rerank_config or "",
            rerank_top_n=rerank_top_n,
            disable_second_stage_rerank=disable_second_stage_rerank,
            log_enabled=log_enabled,
            log_file=log_file or "",
        )

        # ---- 功能开关 ----
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
            self.bm25_manager = get_bm25_state_manager() if bm25_state_path is None else BM25StateManager(state_path=bm25_state_path)

    # ------------------------------------------------------------------
    # Stage 列表（覆盖父类，插入 3 个新 stage）
    # ------------------------------------------------------------------

    def build_stages(self) -> list:
        """返回增强后的 stage 列表，在检索与重排之间插入 BM25Rescore → AutoMerge → Grade。"""
        return [
            self.stage_route,
            self.stage_rewrite,
            self.stage_pseudo_answer,
            self.stage_prepare_retrieval,
            self.stage_apply_permission,
            self.stage_retrieve,
            self.stage_filter_retrieval_permission,
            self.stage_bm25_rescore,       # ← 新增 1
            self.stage_auto_merge,         # ← 新增 2
            self.stage_grade,              # ← 新增 3
            self.stage_rerank,
            self.stage_compress,
            self.stage_prompt,
            self.stage_answer,
        ]

    # ------------------------------------------------------------------
    # 新增 Stage 1：BM25 持久化重打分
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
            # 词表为空，跳过（未做过 BM25 持久化 ingestion）
            retrieval_result["bm25_rescore"] = {
                "enabled": True,
                "skipped": True,
                "reason": "bm25_vocab_empty",
            }
            context["retrieval_result"] = retrieval_result
            return context

        # 使用原始问题 + 所有检索 query 进行 BM25 重打分
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
    # 新增 Stage 2：父子层级自动合并
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

        # 调用 AutoMerger.integrate_auto_merge_to_retrieval
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
    # 新增 Stage 3：文档相关性评估
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

        # 评估相关性
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

        # 记录改写历史
        rewrite_history: list = context.setdefault("grade_rewrite_history", [])
        rewrite_history.append(rewrite_result)

        # 提取扩展查询并注入到 retrieval_inputs
        expanded_queries: list = rewrite_result.get("expanded_queries") or [question]
        retrieval_inputs = context.get("retrieval_inputs") or {}

        # 用扩展查询替换/扩充 search_queries（去重保持顺序）
        existing = list(retrieval_inputs.get("search_queries") or [])
        new_queries = unique_keep_order(existing + expanded_queries)

        retrieval_inputs["search_queries"] = new_queries
        retrieval_inputs["grade_rewrite_strategy"] = rewrite_result.get("strategy", "simple")
        retrieval_inputs["grade_rewrite_step_back_question"] = rewrite_result.get("step_back_question", "")
        retrieval_inputs["grade_rewrite_step_back_answer"] = rewrite_result.get("step_back_answer", "")
        retrieval_inputs["grade_rewrite_hypothetical_doc"] = rewrite_result.get("hypothetical_document", "")

        context["retrieval_inputs"] = retrieval_inputs
        context["retrieval_result"] = None  # 清空旧检索结果

        self.logger.log(
            event="grade_rewrite_retry",
            trace_id=context["trace_id"],
            strategy=rewrite_result.get("strategy"),
            expanded_query_count=len(expanded_queries),
            attempt=len(rewrite_history),
        )

        return context

    # ------------------------------------------------------------------
    # 增强版 collect_stage_metrics（追加新字段）
    # ------------------------------------------------------------------

    def collect_stage_metrics(self, context: PipelineContext) -> Dict[str, Any]:
        metrics = super().collect_stage_metrics(context)
        retrieval_result = context.get("retrieval_result") or {}

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

    # ------------------------------------------------------------------
    # 增强版 build_result（追加新字段）
    # ------------------------------------------------------------------

    def build_result(self, context: PipelineContext) -> Dict[str, Any]:
        result = super().build_result(context)

        if context.get("stop_pipeline") or context.get("needs_retrieval") is False:
            result["enhanced"] = {
                "bm25_rescore_enabled": self.enable_bm25_rescore,
                "auto_merge_enabled": self.enable_auto_merge,
                "grade_rewrite_enabled": self.enable_grade_rewrite,
            }
            return result

        retrieval_result = context.get("retrieval_result") or {}

        # AutoMerge 元信息
        result["auto_merge"] = retrieval_result.get("auto_merge") or {}

        # Grade 元信息
        result["grade"] = {
            "result": context.get("grade_result"),
            "rewrite_history": context.get("grade_rewrite_history") or [],
        }

        # BM25 重打分元信息
        result["bm25_rescore"] = retrieval_result.get("bm25_rescore") or {}

        # 增强功能总览
        result["enhanced"] = {
            "bm25_rescore_enabled": self.enable_bm25_rescore,
            "auto_merge_enabled": self.enable_auto_merge,
            "grade_rewrite_enabled": self.enable_grade_rewrite,
            "grade_max_retry": self.grade_max_retry,
        }

        return result

    # ------------------------------------------------------------------
    # 核心：覆盖 run() 实现 Grade → Rewrite 回流循环
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        document_names: Optional[Sequence[str]] = None,
        chunk_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """执行增强版 RAG Pipeline。

        在标准流程基础上，检索→评估 阶段支持回流重试：
        如果 stage_grade 判定不通过，调用 rewrite_question 改写查询，
        然后回到 stage_retrieve 重新检索，最多重试 grade_max_retry 次。
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
            # ---- Phase 1: 预处理阶段（路由 → 改写 → 伪答案 → 准备检索 → 权限） ----
            pre_retrieval_stages = [
                self.stage_route,
                self.stage_rewrite,
                self.stage_pseudo_answer,
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
                # 执行检索→评估链
                for stage in retrieval_stages:
                    context = self.run_stage(stage, context)
                    if context.get("stop_pipeline"):
                        return self._finish(context)

                # 检查评估结果
                if context.get("grade_passed", True):
                    self.logger.log(
                        event="grade_passed",
                        trace_id=context["trace_id"],
                        attempt=attempt + 1,
                    )
                    break

                # 不通过 & 还有重试次数
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

            # ---- Phase 3: 后处理阶段（重排 → 压缩 → Prompt → 回答） ----
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
# 向后兼容的快捷入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    from rags.RAGPipeline import (
        DEFAULT_MAX_BLOCKS,
        DEFAULT_MAX_CONTEXT_CHARS,
        DEFAULT_PER_QUERY_K,
        DEFAULT_TOP_K,
        EXIT_COMMANDS,
        PipelineContext,
        build_direct_answer,
        normalize_list,
        print_answer,
        resolve_permission_level,
        save_trace,
    )
    from nodes.auth.PermissionGuard import DEFAULT_CONFIG_PATH as DEFAULT_PERMISSION_CONFIG_PATH, DEFAULT_PERMISSION_LEVEL
    from nodes.rerank.Reranker import DEFAULT_CONFIG_PATH as DEFAULT_RERANK_CONFIG_PATH

    parser = argparse.ArgumentParser(
        description="增强型 RAG Pipeline：BM25 持久化 + AutoMerge + Grade&Rewrite 回流"
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
    parser.add_argument("--log-file", default="")
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
        # 交互模式
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
        print(f"增强功能: BM25重打分={pipeline.enable_bm25_rescore}, "
              f"AutoMerge={pipeline.enable_auto_merge}, "
              f"Grade&Rewrite={pipeline.enable_grade_rewrite}")
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
        # 单次模式
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
