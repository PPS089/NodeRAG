from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.embeddings.ChromaBailianEmbedding import (  # noqa: E402
    BailianEmbeddingClient,
    DEFAULT_COLLECTION_NAME,
    load_json_file,
    resolve_project_path,
)
from utils.FindProjectRoot import find_project_root as fr  # noqa: E402


DEFAULT_PER_QUERY_K = 12
DEFAULT_TOP_K = 8
DEFAULT_CONTEXT_LIMIT = 30
DEFAULT_BM25_K = 20
DEFAULT_RERANK_POOL_SIZE = 40
DEFAULT_MMR_LAMBDA = 0.72
BM25_K1 = 1.5
BM25_B = 0.75
TABLE_INTENT_TERMS = {
    "表",
    "表格",
    "价格",
    "金额",
    "预算",
    "薪酬",
    "等级",
    "比例",
    "字段",
    "清单",
    "模板",
}


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


def parse_json_metadata(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default

    if isinstance(value, (list, dict)):
        return value

    if not isinstance(value, str):
        return default

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def get_collection():
    import chromadb

    project_root = fr()
    load_dotenv(project_root / ".env")

    persist_dir = resolve_project_path(os.getenv("CHROMA_PERSIST_DIR") or "ChromaDB", project_root)
    collection_name = os.getenv("CHROMA_COLLECTION_NAME") or DEFAULT_COLLECTION_NAME

    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def build_in_filter(field: str, values: Sequence[str]) -> Optional[Dict[str, Any]]:
    clean_values = unique_keep_order(values)
    if not clean_values:
        return None
    if len(clean_values) == 1:
        return {field: clean_values[0]}
    return {field: {"$in": clean_values}}


def build_where_filter(
    document_ids: Sequence[str],
    document_names: Sequence[str],
    chunk_types: Sequence[str],
) -> Optional[Dict[str, Any]]:
    conditions = []
    for condition in (
        build_in_filter("document_id", document_ids),
        build_in_filter("document_name", document_names),
        build_in_filter("chunk_type", chunk_types),
    ):
        if condition:
            conditions.append(condition)

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def metadata_matches_filter(metadata: Dict[str, Any], where_filter: Optional[Dict[str, Any]]) -> bool:
    """
    对 Chroma metadata 做本地过滤，供 BM25 召回复用。
    """

    if not where_filter:
        return True

    if "$and" in where_filter:
        return all(metadata_matches_filter(metadata, condition) for condition in where_filter["$and"])

    for field, expected in where_filter.items():
        actual = metadata.get(field)
        if isinstance(expected, dict) and "$in" in expected:
            if actual not in expected["$in"]:
                return False
        elif actual != expected:
            return False

    return True


def chroma_distance_to_score(distance: Optional[float]) -> Optional[float]:
    if distance is None:
        return None
    return 1.0 - float(distance)


def normalize_hit(
    chroma_id: str,
    document: str,
    metadata: Dict[str, Any],
    distance: Optional[float],
    query_text: str,
) -> Dict[str, Any]:
    return {
        "chroma_id": chroma_id,
        "chunk_id": metadata.get("chunk_id", ""),
        "document": document,
        "metadata": metadata,
        "distance": distance,
        "score": chroma_distance_to_score(distance),
        "vector_score": chroma_distance_to_score(distance),
        "bm25_score": None,
        "rerank_score": chroma_distance_to_score(distance) or 0.0,
        "retrieval_sources": ["vector"],
        "matched_queries": [query_text],
    }


def tokenize_for_search(text: str) -> List[str]:
    """
    轻量 tokenizer：英文/数字按词，中文补充单字和 bigram。
    """

    text = str(text).lower()
    ascii_terms = re.findall(r"[a-z0-9_]+", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    chinese_bigrams = [
        "".join(chinese_chars[index:index + 2])
        for index in range(len(chinese_chars) - 1)
    ]
    return ascii_terms + chinese_chars + chinese_bigrams


def token_set(text: str) -> Set[str]:
    return set(tokenize_for_search(text))


def min_max_normalize_scores(items: Sequence[Dict[str, Any]], field: str, output_field: str) -> None:
    values = [
        float(item[field])
        for item in items
        if item.get(field) is not None
    ]
    if not values:
        for item in items:
            item[output_field] = 0.0
        return

    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        for item in items:
            item[output_field] = 1.0 if item.get(field) is not None else 0.0
        return

    for item in items:
        value = item.get(field)
        item[output_field] = 0.0 if value is None else (float(value) - min_value) / (max_value - min_value)


def looks_like_table_query(query_texts: Sequence[str]) -> bool:
    joined = " ".join(query_texts)
    return any(term in joined for term in TABLE_INTENT_TERMS)


def get_hit_text(hit: Dict[str, Any]) -> str:
    metadata = hit.get("metadata", {})
    title_path = parse_json_metadata(metadata.get("title_path"), default=[])
    return "\n".join(
        [
            str(metadata.get("document_name", "")),
            " ".join(title_path) if isinstance(title_path, list) else str(title_path),
            str(hit.get("document", "")),
        ]
    )


def merge_hits(hits: Sequence[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for hit in hits:
        chroma_id = hit["chroma_id"]
        existing = merged.get(chroma_id)
        if existing is None:
            merged[chroma_id] = hit
            continue

        existing_sources = existing.setdefault("retrieval_sources", [])
        for source in hit.get("retrieval_sources", []):
            if source not in existing_sources:
                existing_sources.append(source)

        existing_queries = existing.setdefault("matched_queries", [])
        for query_text in hit.get("matched_queries", []):
            if query_text not in existing_queries:
                existing_queries.append(query_text)

        if hit.get("bm25_score") is not None:
            existing["bm25_score"] = max(float(existing.get("bm25_score") or 0.0), float(hit["bm25_score"]))

        if hit.get("vector_score") is not None:
            existing["vector_score"] = max(float(existing.get("vector_score") or 0.0), float(hit["vector_score"]))

        if hit.get("distance") is not None and (
            existing.get("distance") is None or hit["distance"] < existing["distance"]
        ):
            existing["distance"] = hit["distance"]
            existing["score"] = hit["score"]
            existing["document"] = hit["document"]
            existing["metadata"] = hit["metadata"]
            existing["chunk_id"] = hit["chunk_id"]

    results = sorted(
        merged.values(),
        key=lambda item: item["distance"] if item.get("distance") is not None else float("inf"),
    )
    return results if top_k is None else results[:top_k]


def bm25_recall(
    collection,
    query_texts: Sequence[str],
    where_filter: Optional[Dict[str, Any]],
    bm25_k: int = DEFAULT_BM25_K,
) -> List[Dict[str, Any]]:
    """
    从 Chroma 中取当前 collection 文档，做轻量 BM25 关键词召回。
    """

    query_tokens = tokenize_for_search(" ".join(query_texts))
    if not query_tokens:
        return []

    result = collection.get(include=["documents", "metadatas"])
    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    corpus = []
    doc_freq: Counter[str] = Counter()
    for chroma_id, document, metadata in zip(ids, documents, metadatas):
        metadata = metadata or {}
        if not metadata_matches_filter(metadata, where_filter):
            continue

        title_path = parse_json_metadata(metadata.get("title_path"), default=[])
        searchable_text = "\n".join(
            [
                str(metadata.get("document_name", "")),
                " ".join(title_path) if isinstance(title_path, list) else str(title_path),
                str(document or ""),
            ]
        )
        tokens = tokenize_for_search(searchable_text)
        if not tokens:
            continue

        token_counts = Counter(tokens)
        corpus.append(
            {
                "chroma_id": chroma_id,
                "document": document or "",
                "metadata": metadata,
                "token_counts": token_counts,
                "length": len(tokens),
            }
        )
        doc_freq.update(set(tokens))

    if not corpus:
        return []

    avg_doc_length = sum(item["length"] for item in corpus) / len(corpus)
    query_counts = Counter(query_tokens)
    scored_hits = []

    for item in corpus:
        score = 0.0
        for token, query_weight in query_counts.items():
            freq = item["token_counts"].get(token, 0)
            if freq == 0:
                continue

            idf = math.log(1 + (len(corpus) - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            denominator = freq + BM25_K1 * (1 - BM25_B + BM25_B * item["length"] / avg_doc_length)
            score += query_weight * idf * (freq * (BM25_K1 + 1) / denominator)

        if score <= 0:
            continue

        scored_hits.append(
            {
                "chroma_id": item["chroma_id"],
                "chunk_id": item["metadata"].get("chunk_id", ""),
                "document": item["document"],
                "metadata": item["metadata"],
                "distance": None,
                "score": None,
                "vector_score": None,
                "bm25_score": score,
                "rerank_score": score,
                "retrieval_sources": ["bm25"],
                "matched_queries": list(query_texts),
            }
        )

    return sorted(scored_hits, key=lambda item: item["bm25_score"], reverse=True)[:bm25_k]


def lexical_overlap_score(query_tokens: Set[str], hit: Dict[str, Any]) -> float:
    if not query_tokens:
        return 0.0

    hit_tokens = token_set(get_hit_text(hit))
    if not hit_tokens:
        return 0.0

    return len(query_tokens & hit_tokens) / len(query_tokens)


def rerank_hits(
    hits: Sequence[Dict[str, Any]],
    query_texts: Sequence[str],
    table_aware: bool = True,
) -> List[Dict[str, Any]]:
    """
    轻量融合排序：向量分 + BM25 分 + 关键词覆盖 + 表格意图加权。
    """

    ranked_hits = [dict(hit) for hit in hits]
    min_max_normalize_scores(ranked_hits, "bm25_score", "bm25_score_norm")
    min_max_normalize_scores(ranked_hits, "vector_score", "vector_score_norm")

    query_tokens = token_set(" ".join(query_texts))
    table_query = table_aware and looks_like_table_query(query_texts)

    for hit in ranked_hits:
        metadata = hit.get("metadata", {})
        lexical_score = lexical_overlap_score(query_tokens, hit)
        source_bonus = 0.08 if len(hit.get("retrieval_sources", [])) > 1 else 0.0
        table_bonus = 0.12 if table_query and metadata.get("chunk_type") == "table_chunk" else 0.0

        rerank_score = (
            0.52 * float(hit.get("vector_score_norm", 0.0))
            + 0.30 * float(hit.get("bm25_score_norm", 0.0))
            + 0.18 * lexical_score
            + source_bonus
            + table_bonus
        )
        hit["lexical_overlap_score"] = lexical_score
        hit["table_bonus"] = table_bonus
        hit["source_bonus"] = source_bonus
        hit["rerank_score"] = rerank_score

    return sorted(ranked_hits, key=lambda item: item["rerank_score"], reverse=True)


def jaccard_similarity(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def mmr_select(
    hits: Sequence[Dict[str, Any]],
    top_k: int,
    lambda_mult: float = DEFAULT_MMR_LAMBDA,
) -> List[Dict[str, Any]]:
    """
    MMR 去冗余，避免结果集中全是同一段相似内容。
    """

    if top_k <= 0:
        return []

    remaining = [dict(hit) for hit in hits]
    selected: List[Dict[str, Any]] = []
    token_cache = {
        hit["chroma_id"]: token_set(get_hit_text(hit))
        for hit in remaining
    }

    while remaining and len(selected) < top_k:
        best_hit = None
        best_score = -float("inf")

        for hit in remaining:
            relevance = float(hit.get("rerank_score", 0.0))
            diversity_penalty = 0.0
            hit_tokens = token_cache.get(hit["chroma_id"], set())

            if selected:
                diversity_penalty = max(
                    jaccard_similarity(hit_tokens, token_cache.get(selected_hit["chroma_id"], set()))
                    for selected_hit in selected
                )

            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_hit = hit

        if best_hit is None:
            break

        best_hit["mmr_score"] = best_score
        selected.append(best_hit)
        remaining = [hit for hit in remaining if hit["chroma_id"] != best_hit["chroma_id"]]

    return selected


def load_chunk_file_index(chunk_file: str | Path) -> Dict[str, Dict[str, Any]]:
    path = Path(chunk_file)
    chunks = load_json_file(path)
    if not isinstance(chunks, list):
        return {}
    return {
        str(chunk.get("id")): chunk
        for chunk in chunks
        if chunk.get("id")
    }


def compact_chunk(chunk: Dict[str, Any], role: str) -> Dict[str, Any]:
    return {
        "role": role,
        "chunk_id": chunk.get("id", ""),
        "type": chunk.get("type", ""),
        "document_id": chunk.get("document_id", ""),
        "document_name": chunk.get("document_name", ""),
        "title_path": chunk.get("title_path", []),
        "line_range": chunk.get("line_range", []),
        "content": chunk.get("content", ""),
        "embedding_text": chunk.get("embedding_text", ""),
        "parent_context": chunk.get("parent_context", {}),
    }


def expand_hit_context(hit: Dict[str, Any], context_limit: int = DEFAULT_CONTEXT_LIMIT) -> List[Dict[str, Any]]:
    metadata = hit.get("metadata", {})
    chunk_file = metadata.get("chunk_file")
    if not chunk_file:
        return []

    chunk_index = load_chunk_file_index(chunk_file)
    if not chunk_index:
        return []

    root_chunk_id = metadata.get("chunk_id")
    context_ids = parse_json_metadata(metadata.get("small_to_big_context_ids"), default=[])
    if not isinstance(context_ids, list):
        context_ids = []

    ordered_ids = unique_keep_order([root_chunk_id] + [str(item) for item in context_ids])
    chunks = []
    for index, chunk_id in enumerate(ordered_ids):
        chunk = chunk_index.get(chunk_id)
        if not chunk:
            continue
        chunks.append(compact_chunk(chunk, role="hit" if index == 0 else "expanded"))

    chunks.sort(key=lambda item: item.get("line_range", [10**9])[0] if item.get("line_range") else 10**9)
    return chunks[:context_limit]


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
    embeddings = BailianEmbeddingClient().embed_texts(query_texts)
    raw_hits = []

    for query_text, embedding in zip(query_texts, embeddings):
        query_result = collection.query(
            query_embeddings=[embedding],
            n_results=per_query_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        ids = query_result.get("ids", [[]])[0]
        documents = query_result.get("documents", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]

        for chroma_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            raw_hits.append(
                normalize_hit(
                    chroma_id=chroma_id,
                    document=document,
                    metadata=metadata or {},
                    distance=distance,
                    query_text=query_text,
                )
            )

    bm25_hits = bm25_recall(
        collection=collection,
        query_texts=bm25_query_texts,
        where_filter=where_filter,
        bm25_k=bm25_k,
    ) if use_bm25 else []

    candidates = merge_hits(raw_hits + bm25_hits, top_k=None)
    ranked_hits = rerank_hits(candidates, query_texts=unique_keep_order(query_texts + bm25_query_texts), table_aware=table_aware)
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第二版 Chroma RAG 检索：向量召回 + BM25 + rerank + MMR + small-to-big 扩展。")
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
