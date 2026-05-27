from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
        "matched_queries": [query_text],
    }


def merge_hits(hits: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for hit in hits:
        chroma_id = hit["chroma_id"]
        existing = merged.get(chroma_id)
        if existing is None:
            merged[chroma_id] = hit
            continue

        existing_queries = existing.setdefault("matched_queries", [])
        for query_text in hit.get("matched_queries", []):
            if query_text not in existing_queries:
                existing_queries.append(query_text)

        if hit.get("distance") is not None and (
            existing.get("distance") is None or hit["distance"] < existing["distance"]
        ):
            existing["distance"] = hit["distance"]
            existing["score"] = hit["score"]
            existing["document"] = hit["document"]
            existing["metadata"] = hit["metadata"]
            existing["chunk_id"] = hit["chunk_id"]

    return sorted(
        merged.values(),
        key=lambda item: item["distance"] if item.get("distance") is not None else float("inf"),
    )[:top_k]


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
    pseudo_answer: Optional[str] = None,
    document_ids: Optional[Sequence[str]] = None,
    document_names: Optional[Sequence[str]] = None,
    chunk_types: Optional[Sequence[str]] = None,
    per_query_k: int = DEFAULT_PER_QUERY_K,
    top_k: int = DEFAULT_TOP_K,
    expand_context: bool = True,
) -> Dict[str, Any]:
    query_texts = unique_keep_order(
        [question]
        + list(queries or [])
        + ([pseudo_answer] if pseudo_answer else [])
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

    hits = merge_hits(raw_hits, top_k=top_k)

    if expand_context:
        for hit in hits:
            hit["expanded_context"] = expand_hit_context(hit)

    return {
        "question": question,
        "queries": query_texts,
        "where_filter": where_filter,
        "per_query_k": per_query_k,
        "top_k": top_k,
        "hit_count": len(hits),
        "hits": hits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第一版 Chroma RAG 检索：向量召回 + 多 query 合并 + small-to-big 扩展。")
    parser.add_argument("question_arg", nargs="*", help="用户问题，未传 --question 时使用。")
    parser.add_argument("--question", "-q", help="用户问题。")
    parser.add_argument("--query", action="append", default=[], help="额外检索 query，可重复传入。")
    parser.add_argument("--pseudo-answer", help="HyDE 伪答案文本。")
    parser.add_argument("--document-id", action="append", default=[], help="按 document_id 过滤，可重复传入。")
    parser.add_argument("--document-name", action="append", default=[], help="按 document_name 过滤，可重复传入。")
    parser.add_argument("--chunk-type", action="append", default=[], help="按 chunk_type 过滤，可重复传入。")
    parser.add_argument("--per-query-k", type=int, default=DEFAULT_PER_QUERY_K, help="每条 query 召回数量。")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="合并去重后的返回数量。")
    parser.add_argument("--no-expand", action="store_true", help="关闭 small-to-big 上下文扩展。")
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
        pseudo_answer=args.pseudo_answer,
        document_ids=args.document_id,
        document_names=args.document_name,
        chunk_types=args.chunk_type,
        per_query_k=args.per_query_k,
        top_k=args.top_k,
        expand_context=not args.no_expand,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
