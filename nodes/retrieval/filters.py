from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from nodes.retrieval.retrieval_utils import unique_keep_order


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
