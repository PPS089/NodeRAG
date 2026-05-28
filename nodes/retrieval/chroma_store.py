from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

from nodes.embeddings.ChromaBailianEmbedding import (
    DEFAULT_COLLECTION_NAME,
    BailianEmbeddingClient,
    resolve_project_path,
)
from nodes.retrieval.retrieval_utils import normalize_hit
from utils.FindProjectRoot import find_project_root as fr


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


def vector_recall(
    collection,
    query_texts: Sequence[str],
    where_filter: Optional[Dict[str, Any]],
    per_query_k: int,
) -> List[Dict[str, Any]]:
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

    return raw_hits
