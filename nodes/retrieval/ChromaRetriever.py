from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.rerank.Reranker import mmr_select, rerank_hits  # noqa: E402
from nodes.retrieval.bm25_recall import bm25_recall  # noqa: E402
from nodes.retrieval.chroma_store import get_collection, vector_recall  # noqa: E402
from nodes.retrieval.context_expander import expand_hit_context  # noqa: E402
from nodes.retrieval.filters import build_where_filter  # noqa: E402
from nodes.retrieval.retrieval_config import (  # noqa: E402
    DEFAULT_BM25_K,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_PER_QUERY_K,
    DEFAULT_RERANK_POOL_SIZE,
    DEFAULT_TOP_K,
)
from nodes.retrieval.retrieval_utils import merge_hits, unique_keep_order  # noqa: E402


def retrieve(
    question: str,
    queries: Optional[Sequence[str]] = None,
    keyword_queries: Optional[Sequence[str]] = None,
    pseudo_answer: Optional[str] = None,
    document_ids: Optional[Sequence[str]] = None,
    document_names: Optional[Sequence[str]] = None,
    chunk_types: Optional[Sequence[str]] = None,
    per_query_k: int = DEFAULT_PER_QUERY_K,
    bm25_k: int = DEFAULT_BM25_K,
    rerank_pool_size: int = DEFAULT_RERANK_POOL_SIZE,
    top_k: int = DEFAULT_TOP_K,
    use_bm25: bool = True,
    use_mmr: bool = True,
    table_aware: bool = True,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    expand_context: bool = True,
) -> Dict[str, Any]:
    query_texts = unique_keep_order(
        [question]
        + list(queries or [])
        + ([pseudo_answer] if pseudo_answer else [])
    )
    bm25_query_texts = unique_keep_order(
        [question]
        + list(queries or [])
        + list(keyword_queries or [])
    )
    where_filter = build_where_filter(
        document_ids=document_ids or [],
        document_names=document_names or [],
        chunk_types=chunk_types or [],
    )

    collection = get_collection()
    raw_hits = vector_recall(
        collection=collection,
        query_texts=query_texts,
        where_filter=where_filter,
        per_query_k=per_query_k,
    )
    bm25_hits = bm25_recall(
        collection=collection,
        query_texts=bm25_query_texts,
        where_filter=where_filter,
        bm25_k=bm25_k,
    ) if use_bm25 else []

    candidates = merge_hits(raw_hits + bm25_hits, top_k=None)
    ranked_hits = rerank_hits(
        candidates,
        query_texts=unique_keep_order(query_texts + bm25_query_texts),
        table_aware=table_aware,
        stage_name="retrieval_initial_rerank",
    )
    rerank_pool = ranked_hits[:max(rerank_pool_size, top_k)]
    hits = mmr_select(rerank_pool, top_k=top_k, lambda_mult=mmr_lambda) if use_mmr else rerank_pool[:top_k]

    if expand_context:
        for hit in hits:
            hit["expanded_context"] = expand_hit_context(hit)

    return {
        "question": question,
        "queries": query_texts,
        "keyword_queries": bm25_query_texts,
        "where_filter": where_filter,
        "per_query_k": per_query_k,
        "bm25_k": bm25_k,
        "rerank_pool_size": rerank_pool_size,
        "top_k": top_k,
        "use_bm25": use_bm25,
        "use_mmr": use_mmr,
        "table_aware": table_aware,
        "mmr_lambda": mmr_lambda,
        "raw_vector_hit_count": len(raw_hits),
        "raw_bm25_hit_count": len(bm25_hits),
        "candidate_count": len(candidates),
        "hit_count": len(hits),
        "hits": hits,
        "retrieval_pipeline": {
            "vector_recall": "nodes/retrieval/chroma_store.py",
            "bm25_recall": "nodes/retrieval/bm25_recall.py",
            "initial_rerank": "nodes/rerank/Reranker.py",
            "context_expand": "nodes/retrieval/context_expander.py",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chroma RAG 检索编排：向量召回 + BM25 + 初排 + MMR + small-to-big 扩展。")
    parser.add_argument("question_arg", nargs="*", help="用户问题，未传 --question 时使用。")
    parser.add_argument("--question", "-q", help="用户问题。")
    parser.add_argument("--query", action="append", default=[], help="额外检索 query，可重复传入。")
    parser.add_argument("--keyword-query", action="append", default=[], help="BM25 关键词 query，可重复传入。")
    parser.add_argument("--pseudo-answer", help="HyDE 伪答案文本。")
    parser.add_argument("--document-id", action="append", default=[], help="按 document_id 过滤，可重复传入。")
    parser.add_argument("--document-name", action="append", default=[], help="按 document_name 过滤，可重复传入。")
    parser.add_argument("--chunk-type", action="append", default=[], help="按 chunk_type 过滤，可重复传入。")
    parser.add_argument("--per-query-k", type=int, default=DEFAULT_PER_QUERY_K, help="每条 query 召回数量。")
    parser.add_argument("--bm25-k", type=int, default=DEFAULT_BM25_K, help="BM25 召回数量。")
    parser.add_argument("--rerank-pool-size", type=int, default=DEFAULT_RERANK_POOL_SIZE, help="进入 MMR 前的候选池大小。")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="合并去重后的返回数量。")
    parser.add_argument("--mmr-lambda", type=float, default=DEFAULT_MMR_LAMBDA, help="MMR 相关性权重，越高越偏相关性。")
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25 关键词召回。")
    parser.add_argument("--no-mmr", action="store_true", help="关闭 MMR 去冗余。")
    parser.add_argument("--no-table-aware", action="store_true", help="关闭表格问题 table_chunk 加权。")
    parser.add_argument("--no-expand", action="store_true", help="关闭 small-to-big 上下文扩展。")
    parser.add_argument("--output", "-o", help="输出 JSON 文件；不传则打印到 stdout。")
    return parser.parse_args()


def get_question(args: argparse.Namespace) -> str:
    question = args.question or " ".join(args.question_arg)
    question = question.strip()
    if not question:
        raise ValueError("请通过 --question 或位置参数传入用户问题")
    return question


def main() -> Dict[str, Any]:
    args = parse_args()
    result = retrieve(
        question=get_question(args),
        queries=args.query,
        keyword_queries=args.keyword_query,
        pseudo_answer=args.pseudo_answer,
        document_ids=args.document_id,
        document_names=args.document_name,
        chunk_types=args.chunk_type,
        per_query_k=args.per_query_k,
        bm25_k=args.bm25_k,
        rerank_pool_size=args.rerank_pool_size,
        top_k=args.top_k,
        use_bm25=not args.no_bm25,
        use_mmr=not args.no_mmr,
        table_aware=not args.no_table_aware,
        mmr_lambda=args.mmr_lambda,
        expand_context=not args.no_expand,
    )
    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)
    return result


if __name__ == "__main__":
    main()
