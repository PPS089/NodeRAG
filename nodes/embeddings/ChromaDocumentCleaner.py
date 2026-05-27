from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.embeddings.ChromaBailianEmbedding import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CHUNK_FILE_NAME,
    find_hybrid_chunk_files,
    load_json_file,
    resolve_project_path,
)
from utils.FindProjectRoot import find_project_root as fr  # noqa: E402


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


def get_local_document_ids(
    result_dir: str | Path | None = None,
    chunk_file_name: str = DEFAULT_CHUNK_FILE_NAME,
) -> set[str]:
    document_ids = set()

    for chunk_file in find_hybrid_chunk_files(result_dir=result_dir, chunk_file_name=chunk_file_name):
        chunks = load_json_file(chunk_file)
        if not isinstance(chunks, list):
            continue

        for chunk in chunks:
            document_id = str(chunk.get("document_id") or "").strip()
            if document_id:
                document_ids.add(document_id)

    return document_ids


def get_chroma_document_ids(collection) -> set[str]:
    result = collection.get(include=["metadatas"])
    document_ids = set()

    for metadata in result.get("metadatas", []) or []:
        document_id = str((metadata or {}).get("document_id") or "").strip()
        if document_id:
            document_ids.add(document_id)

    return document_ids


def find_document_ids_by_name(collection, document_name: str) -> List[str]:
    result = collection.get(where={"document_name": document_name}, include=["metadatas"])
    document_ids = set()

    for metadata in result.get("metadatas", []) or []:
        document_id = str((metadata or {}).get("document_id") or "").strip()
        if document_id:
            document_ids.add(document_id)

    return sorted(document_ids)


def delete_documents(collection, document_ids: Sequence[str]) -> Dict[str, Any]:
    deleted_count = 0
    details = {}

    for document_id in sorted(set(document_ids)):
        existing = collection.get(where={"document_id": document_id})
        existing_ids = existing.get("ids", [])
        if not existing_ids:
            details[document_id] = 0
            continue

        collection.delete(ids=existing_ids)
        deleted_count += len(existing_ids)
        details[document_id] = len(existing_ids)

    return {
        "deleted_document_count": sum(1 for count in details.values() if count > 0),
        "deleted_vector_count": deleted_count,
        "details": details,
    }


def prune_missing_documents(
    result_dir: str | Path | None = None,
    chunk_file_name: str = DEFAULT_CHUNK_FILE_NAME,
) -> Dict[str, Any]:
    collection = get_collection()
    local_document_ids = get_local_document_ids(result_dir=result_dir, chunk_file_name=chunk_file_name)
    chroma_document_ids = get_chroma_document_ids(collection)
    missing_document_ids = sorted(chroma_document_ids - local_document_ids)

    result = delete_documents(collection, missing_document_ids)
    result.update(
        {
            "mode": "prune_missing",
            "local_document_count": len(local_document_ids),
            "chroma_document_count": len(chroma_document_ids),
            "missing_document_ids": missing_document_ids,
        }
    )
    return result


def delete_by_document_ids(document_ids: Sequence[str]) -> Dict[str, Any]:
    collection = get_collection()
    result = delete_documents(collection, document_ids)
    result.update({"mode": "delete_by_document_id"})
    return result


def delete_by_document_names(document_names: Sequence[str]) -> Dict[str, Any]:
    collection = get_collection()
    document_ids = []

    for document_name in document_names:
        document_ids.extend(find_document_ids_by_name(collection, document_name))

    result = delete_documents(collection, document_ids)
    result.update(
        {
            "mode": "delete_by_document_name",
            "document_names": list(document_names),
            "matched_document_ids": sorted(set(document_ids)),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="删除 Chroma 中指定文档或本地已不存在文档的向量数据。"
    )
    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        help="要删除的 document_id，可重复传入。",
    )
    parser.add_argument(
        "--document-name",
        action="append",
        default=[],
        help="要删除的 document_name，可重复传入。",
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="删除 Chroma 中存在但 MinerUResult 本地已不存在的 document_id。",
    )
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()

    if args.document_id:
        result = delete_by_document_ids(args.document_id)
    elif args.document_name:
        result = delete_by_document_names(args.document_name)
    else:
        # 默认用于删除 PDF 后的清理场景。
        result = prune_missing_documents()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
