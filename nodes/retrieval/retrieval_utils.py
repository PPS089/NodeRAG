from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from nodes.retrieval.retrieval_config import TABLE_INTENT_TERMS


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


def jaccard_similarity(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
