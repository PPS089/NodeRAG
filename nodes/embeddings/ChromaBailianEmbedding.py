from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.contracts import validate_hybrid_chunk  # noqa: E402
from utils.FindProjectRoot import find_project_root as fr  # noqa: E402


DEFAULT_CHUNK_FILE_NAME = "DataCleaned.hybrid_chunks.json"
DEFAULT_COLLECTION_NAME = "document_chunks"
DEFAULT_BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 60
DEFAULT_SYNC_MODE = "chunk"
CHROMA_METADATA_VALUE_TYPES = (str, int, float, bool)
VALID_SYNC_MODES = {"chunk", "strict", "incremental", "force"}


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def find_hybrid_chunk_files(
    result_dir: str | Path | None = None,
    chunk_file_name: str = DEFAULT_CHUNK_FILE_NAME,
) -> List[Path]:
    """
    查找 MinerUResult 下的 Hybrid chunks 文件。
    """

    project_root = fr()
    mineru_result_dir = Path(result_dir) if result_dir else project_root / "MinerUResult"

    if not mineru_result_dir.exists():
        raise FileNotFoundError(f"MinerUResult 目录不存在: {mineru_result_dir}")

    if not mineru_result_dir.is_dir():
        raise ValueError(f"不是有效目录: {mineru_result_dir}")

    return sorted(
        path
        for path in mineru_result_dir.glob(f"*/{chunk_file_name}")
        if path.is_file()
    )


def keep_chroma_metadata_value(value: Any) -> str | int | float | bool:
    """
    Chroma metadata 保持可过滤字段为标量，复杂结构序列化为 JSON 字符串。
    """

    if value is None:
        return ""

    if isinstance(value, CHROMA_METADATA_VALUE_TYPES):
        return value

    return json.dumps(value, ensure_ascii=False)


def build_metadata(chunk: Dict[str, Any], chunk_file: Path) -> Dict[str, str | int | float | bool]:
    """
    构造 Chroma metadata。
    """

    metadata = {
        "chunk_id": chunk.get("id", ""),
        "chunk_type": chunk.get("type", ""),
        "document_id": chunk.get("document_id", ""),
        "document_name": chunk.get("document_name", ""),
        "source_file": chunk.get("source_file", ""),
        "source_path": chunk.get("source_path", ""),
        "chunk_file": str(chunk_file),
        "retrieval_role": chunk.get("retrieval_role", ""),
        "semantic_unit_type": chunk.get("semantic_unit_type", ""),
        "title_path": chunk.get("title_path", []),
        "line_range": chunk.get("line_range", []),
        "parent_context": chunk.get("parent_context", {}),
        "neighbor_chunk_ids": chunk.get("neighbor_chunk_ids", {}),
        "related_ref_chunk_ids": chunk.get("related_ref_chunk_ids", {}),
        "small_to_big_context_ids": chunk.get("small_to_big_context_ids", []),
        "content_hash": chunk.get("content_hash", ""),
        "chunk_strategy": chunk.get("chunk_strategy", ""),
        "chunk_schema_version": chunk.get("chunk_schema_version", ""),
    }

    return {
        key: keep_chroma_metadata_value(value)
        for key, value in metadata.items()
    }


def build_chroma_id(chunk: Dict[str, Any]) -> str:
    """
    构造稳定 Chroma id，避免重新生成 chunk 随机 id 后重复入库。
    """

    line_range = chunk.get("line_range", [])
    if isinstance(line_range, list) and len(line_range) == 2:
        line_part = f"{line_range[0]}-{line_range[1]}"
    else:
        line_part = "unknown"

    return "::".join(
        [
            str(chunk.get("document_id") or chunk.get("document_name") or "document"),
            str(chunk.get("type") or "chunk"),
            line_part,
            str(chunk.get("content_hash") or chunk.get("id") or "")[:16],
        ]
    )


def get_document_ids(records: Sequence[Tuple[Path, Dict[str, Any]]]) -> List[str]:
    document_ids = []
    for _, chunk in records:
        document_id = str(chunk.get("document_id") or chunk.get("document_name") or "").strip()
        if document_id:
            document_ids.append(document_id)

    return sorted(set(document_ids))


def get_current_ids_by_document(records: Sequence[Tuple[Path, Dict[str, Any]]]) -> Dict[str, set[str]]:
    ids_by_document: Dict[str, set[str]] = {}

    for _, chunk in records:
        document_id = str(chunk.get("document_id") or chunk.get("document_name") or "").strip()
        if not document_id:
            continue

        ids_by_document.setdefault(document_id, set()).add(build_chroma_id(chunk))

    return ids_by_document


def load_embed_chunks(chunk_files: Sequence[Path]) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    读取 should_embed=True 的 chunks。
    """

    records: List[Tuple[Path, Dict[str, Any]]] = []

    for chunk_file in chunk_files:
        chunks = load_json_file(chunk_file)
        if not isinstance(chunks, list):
            raise ValueError(f"chunks 文件格式错误，应为 list: {chunk_file}")

        for chunk in chunks:
            if chunk.get("should_embed") is True:
                validate_hybrid_chunk(chunk, source=str(chunk_file))
                embedding_text = str(chunk.get("embedding_text", "")).strip()
                if embedding_text:
                    records.append((chunk_file, chunk))

    return records


class BailianEmbeddingClient:
    """
    百炼 OpenAI 兼容 Embedding 客户端。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        dimensions: Optional[int] = None,
    ) -> None:
        project_root = fr()
        load_dotenv(project_root / ".env")

        self.api_key = (
            api_key
            or os.getenv("BAILIAN_API_KEY")
            or os.getenv("EMBEDDING_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        self.base_url = (
            base_url
            or os.getenv("BAILIAN_BASE_URL")
            or os.getenv("EMBEDDING_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or DEFAULT_BAILIAN_BASE_URL
        )
        self.model = (
            model
            or os.getenv("BAILIAN_EMBEDDING_MODEL")
            or os.getenv("EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL
        )
        self.timeout = int(timeout or os.getenv("EMBEDDING_TIMEOUT", DEFAULT_TIMEOUT))
        env_dimensions = os.getenv("BAILIAN_EMBEDDING_DIMENSIONS") or os.getenv("EMBEDDING_DIMENSIONS")
        self.dimensions = dimensions if dimensions is not None else int(env_dimensions) if env_dimensions else None

        if not self.api_key:
            raise ValueError("请在 .env 中配置 BAILIAN_API_KEY 或 EMBEDDING_API_KEY")

    @property
    def embeddings_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/embeddings"):
            return base_url
        return f"{base_url}/embeddings"

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        response = requests.post(
            self.embeddings_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()

        data = result.get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"百炼 Embedding 响应格式异常: {result}")

        sorted_data = sorted(data, key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding") for item in sorted_data]

        if len(embeddings) != len(texts) or any(not embedding for embedding in embeddings):
            raise RuntimeError(f"百炼 Embedding 数量异常: 输入 {len(texts)}，返回 {len(embeddings)}")

        return embeddings


class ChromaBailianIndexer:
    """
    使用百炼生成 embedding，并写入本地 Chroma。
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        embedding_client: Optional[BailianEmbeddingClient] = None,
        force_reembed: Optional[bool] = None,
        sync_mode: Optional[str] = None,
    ) -> None:
        project_root = fr()
        load_dotenv(project_root / ".env")

        self.persist_dir = resolve_project_path(
            persist_dir
            or os.getenv("CHROMA_PERSIST_DIR")
            or "ChromaDB",
            project_root,
        )
        self.collection_name = (
            collection_name
            or os.getenv("CHROMA_COLLECTION_NAME")
            or DEFAULT_COLLECTION_NAME
        )
        self.batch_size = int(batch_size or os.getenv("EMBEDDING_BATCH_SIZE", DEFAULT_BATCH_SIZE))
        self.embedding_client = embedding_client or BailianEmbeddingClient()
        self.force_reembed = (
            force_reembed
            if force_reembed is not None
            else env_bool("CHROMA_FORCE_REEMBED", default=False)
        )
        self.sync_mode = self.resolve_sync_mode(sync_mode)

    def resolve_sync_mode(self, sync_mode: Optional[str]) -> str:
        if self.force_reembed:
            return "force"

        mode = (
            sync_mode
            or os.getenv("CHROMA_SYNC_MODE")
            or DEFAULT_SYNC_MODE
        ).strip().lower()

        if mode == "chunk_sync":
            mode = "chunk"

        if mode not in VALID_SYNC_MODES:
            raise ValueError(f"CHROMA_SYNC_MODE 仅支持 {sorted(VALID_SYNC_MODES)}，当前值: {mode}")

        return mode

    def get_collection(self):
        import chromadb

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def get_existing_ids(collection, ids: Sequence[str]) -> set[str]:
        if not ids:
            return set()

        result = collection.get(ids=list(ids))
        return set(result.get("ids", []))

    @staticmethod
    def delete_documents(collection, document_ids: Sequence[str]) -> int:
        deleted_count = 0

        for document_id in document_ids:
            existing = collection.get(where={"document_id": document_id})
            existing_ids = existing.get("ids", [])
            if not existing_ids:
                continue

            collection.delete(ids=existing_ids)
            deleted_count += len(existing_ids)

        return deleted_count

    @staticmethod
    def delete_stale_chunks(collection, current_ids_by_document: Dict[str, set[str]]) -> int:
        deleted_count = 0

        for document_id, current_ids in current_ids_by_document.items():
            existing = collection.get(where={"document_id": document_id})
            existing_ids = set(existing.get("ids", []))
            stale_ids = sorted(existing_ids - current_ids)
            if not stale_ids:
                continue

            collection.delete(ids=stale_ids)
            deleted_count += len(stale_ids)

        return deleted_count

    def upsert_chunks(self, records: Sequence[Tuple[Path, Dict[str, Any]]]) -> Dict[str, Any]:
        collection = self.get_collection()
        upsert_count = 0
        skipped_count = 0
        deleted_count = 0
        document_ids = get_document_ids(records)
        current_ids_by_document = get_current_ids_by_document(records)

        if self.sync_mode in {"strict", "force"}:
            deleted_count = self.delete_documents(collection, document_ids)
        elif self.sync_mode == "chunk":
            deleted_count = self.delete_stale_chunks(collection, current_ids_by_document)

        for batch in batched(list(records), self.batch_size):
            ids = [build_chroma_id(chunk) for _, chunk in batch]
            existing_ids = (
                self.get_existing_ids(collection, ids)
                if self.sync_mode in {"chunk", "incremental"}
                else set()
            )
            pending_items = [
                (chroma_id, chunk_file, chunk)
                for chroma_id, (chunk_file, chunk) in zip(ids, batch)
                if chroma_id not in existing_ids
            ]

            skipped_count += len(batch) - len(pending_items)
            if not pending_items:
                continue

            pending_ids = [chroma_id for chroma_id, _, _ in pending_items]
            texts = [str(chunk["embedding_text"]) for _, _, chunk in pending_items]
            documents = [str(chunk.get("content", "")) for _, _, chunk in pending_items]
            metadatas = [build_metadata(chunk, chunk_file) for _, chunk_file, chunk in pending_items]
            embeddings = self.embedding_client.embed_texts(texts)

            collection.upsert(
                ids=pending_ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            upsert_count += len(pending_items)

        return {
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "upsert_count": upsert_count,
            "skipped_existing_count": skipped_count,
            "deleted_count": deleted_count,
            "document_count": len(document_ids),
            "sync_mode": self.sync_mode,
        }


def index_hybrid_chunks(
    result_dir: str | Path | None = None,
    chunk_file_name: str = DEFAULT_CHUNK_FILE_NAME,
) -> Dict[str, Any]:
    chunk_files = find_hybrid_chunk_files(result_dir=result_dir, chunk_file_name=chunk_file_name)
    if not chunk_files:
        target_dir = Path(result_dir) if result_dir else fr() / "MinerUResult"
        raise FileNotFoundError(f"未找到 Hybrid chunks 文件: {target_dir}/*/{chunk_file_name}")

    records = load_embed_chunks(chunk_files)
    if not records:
        raise ValueError("没有找到 should_embed=True 且 embedding_text 非空的 chunk")

    indexer = ChromaBailianIndexer()
    result = indexer.upsert_chunks(records)
    result.update(
        {
            "chunk_file_count": len(chunk_files),
            "embed_record_count": len(records),
        }
    )
    return result


def main() -> Dict[str, Any]:
    result = index_hybrid_chunks()
    print("Chroma 入库完成:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
