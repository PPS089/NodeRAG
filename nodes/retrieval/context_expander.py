from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from nodes.embeddings.ChromaBailianEmbedding import load_json_file
from nodes.retrieval.retrieval_config import DEFAULT_CONTEXT_LIMIT
from nodes.retrieval.retrieval_utils import parse_json_metadata, unique_keep_order


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
