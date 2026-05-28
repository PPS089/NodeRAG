from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from nodes.retrieval.filters import metadata_matches_filter
from nodes.retrieval.retrieval_config import BM25_B, BM25_K1, DEFAULT_BM25_K
from nodes.retrieval.retrieval_utils import parse_json_metadata, tokenize_for_search


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
