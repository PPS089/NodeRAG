"""
Auto-Merging Retriever: 按父子层级自动合并召回结果。

- 当多个叶子块（L3）命中同一父块（L2/L1）时，将其替换为父块，
  扩大上下文范围，提升回答完整性。
- 支持两级合并：L3→L2，L2→L1（或自定义层级）
- 仅在合并收益大于阈值时触发，避免无意义合并。
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from dotenv import load_dotenv

from nodes.retrieval.retrieval_config import DEFAULT_TOP_K  # noqa: E402
from nodes.retrieval.retrieval_utils import parse_json_metadata  # noqa: E402


# 默认配置路径，可通过环境变量覆盖
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "auto_merge_config.json"

# 默认合并阈值：命中子块数 >= threshold 时触发合并
DEFAULT_MERGE_THRESHOLD = 2

# 默认目标层级（叶子层 = 最高细粒度子块）
DEFAULT_LEAF_LEVEL = 3


def load_auto_merge_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """加载 Auto-Merging 配置，支持 JSON 文件覆盖。"""
    import json

    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        return {
            "enabled": True,
            "merge_threshold": DEFAULT_MERGE_THRESHOLD,
            "max_merge_steps": 2,
            "merge_levels": [3, 2, 1],
        }

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {
            "enabled": raw.get("enabled", True),
            "merge_threshold": int(raw.get("merge_threshold", DEFAULT_MERGE_THRESHOLD)),
            "max_merge_steps": int(raw.get("max_merge_steps", 2)),
            "merge_levels": raw.get("merge_levels", [3, 2, 1]),
        }
    except (json.JSONDecodeError, OSError):
        return {
            "enabled": True,
            "merge_threshold": DEFAULT_MERGE_THRESHOLD,
            "max_merge_steps": 2,
            "merge_levels": [3, 2, 1],
        }


def _get_collection():
    """获取 Chroma collection，与 ChromaRetriever 保持一致。"""
    from utils.FindProjectRoot import find_project_root as fr

    project_root = fr()
    load_dotenv(project_root / ".env")

    persist_dir = project_root / os.getenv("CHROMA_PERSIST_DIR", "ChromaDB")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")

    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def fetch_parent_chunks(
    parent_ids: List[str],
    collection,
) -> Dict[str, Dict[str, Any]]:
    """根据 parent_chunk_id 批量拉取父块信息。"""
    if not parent_ids:
        return {}

    ids_to_fetch = [pid for pid in parent_ids if pid]
    if not ids_to_fetch:
        return {}

    # Chroma 的 where filter 支持 in 操作
    quoted_ids = ", ".join([f'"{pid}"' for pid in ids_to_fetch])
    filter_expr = f'chunk_id in [{quoted_ids}]'

    result = collection.get(
        where=filter_expr,
        include=["documents", "metadatas", "metadatas"],
    )

    chunks: Dict[str, Dict[str, Any]] = {}
    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    for chroma_id, document, metadata in zip(ids, documents, metadatas):
        chunk_id = (metadata or {}).get("chunk_id", chroma_id or "")
        if chunk_id:
            chunks[chunk_id] = {
                "chroma_id": chroma_id,
                "document": document or "",
                "metadata": metadata or {},
            }

    return chunks


def merge_to_parent_level(
    hits: Sequence[Dict[str, Any]],
    parent_level: int,
    threshold: int,
    collection,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    将命中同一父块的多个子块合并为父块。

    :param hits: 原始召回结果列表
    :param parent_level: 目标父块层级（如 2 表示合并到 L2 父块）
    :param threshold: 子块命中数 >= threshold 时触发合并
    :param collection: Chroma collection（用于拉取父块信息）
    :return: (合并后结果, 合并次数)
    """
    # 按 parent_chunk_id 分组
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        metadata = hit.get("metadata") or {}
        parent_id = str(metadata.get("parent_chunk_id", "") or "").strip()
        if not parent_id:
            continue
        # 检查父块层级是否匹配
        chunk_level = int(metadata.get("chunk_level", 0) or 0)
        if chunk_level != parent_level + 1:
            continue
        groups[parent_id].append(hit)

    # 仅对满足阈值的父块触发合并
    merge_parent_ids = [pid for pid, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return list(hits), 0

    # 批量拉取父块信息
    parent_chunks = fetch_parent_chunks(merge_parent_ids, collection)
    parent_map = {pid: parent_chunks.get(pid) for pid in merge_parent_ids if pid in parent_chunks}

    merged: List[Dict[str, Any]] = []
    merged_count = 0

    for hit in hits:
        metadata = hit.get("metadata") or {}
        parent_id = str(metadata.get("parent_chunk_id", "") or "").strip()
        chunk_level = int(metadata.get("chunk_level", 0) or 0)

        # 如果满足合并条件且父块存在
        if (
            parent_id
            and parent_id in parent_map
            and chunk_level == parent_level + 1
            and len(groups[parent_id]) >= threshold
        ):
            parent_info = parent_map[parent_id]
            if not parent_info:
                merged.append(hit)
                continue

            merged_doc = dict(hit)
            # 用父块信息替换子块
            merged_doc["document"] = parent_info.get("document", hit.get("document", ""))
            merged_doc["metadata"] = parent_info.get("metadata", hit.get("metadata", {}))
            merged_doc["chroma_id"] = parent_info.get("chroma_id", hit.get("chroma_id", ""))

            # 标记合并来源
            merged_doc["merged_from_children"] = True
            merged_doc["merged_child_count"] = len(groups[parent_id])
            merged_doc["merged_parent_id"] = parent_id
            merged_doc["merge_level"] = parent_level

            # 保留最高分数
            existing_score = hit.get("bm25_score") or hit.get("rerank_score") or hit.get("score", 0.0)
            parent_score = merged_doc.get("bm25_score") or merged_doc.get("rerank_score") or 0.0
            merged_doc["score"] = max(float(existing_score), float(parent_score))
            if merged_doc.get("bm25_score"):
                merged_doc["bm25_score"] = max(float(merged_doc.get("bm25_score", 0.0)), float(existing_score))
            if merged_doc.get("rerank_score"):
                merged_doc["rerank_score"] = max(float(merged_doc.get("rerank_score", 0.0)), float(existing_score))

            merged.append(merged_doc)
            merged_count += 1
        else:
            merged.append(hit)

    # 去重（同一父块可能多次出现）
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in merged:
        key = item.get("chroma_id") or item.get("chunk_id") or str(item.get("metadata", {}).get("chunk_id", ""))
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped, merged_count


def auto_merge_hits(
    hits: Sequence[Dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    merge_threshold: int = DEFAULT_MERGE_THRESHOLD,
    max_steps: int = 2,
    merge_levels: Optional[List[int]] = None,
    collection=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Auto-Merging 主函数：执行多级父子层级合并。

    :param hits: 原始召回结果
    :param top_k: 返回结果数量
    :param merge_threshold: 子块命中阈值
    :param max_steps: 最大合并步数（默认 2，即 L3→L2→L1）
    :param merge_levels: 合并层级顺序，默认 [3, 2, 1]
    :param collection: Chroma collection（可不传，内部获取）
    :return: (合并后结果, 合并元信息)
    """
    if not hits:
        return [], {
            "enabled": True,
            "merge_threshold": merge_threshold,
            "merge_steps": 0,
            "total_merged_count": 0,
            "per_level_merged": {},
        }

    if collection is None:
        collection = _get_collection()

    if merge_levels is None:
        merge_levels = [3, 2, 1]

    # 执行多级合并：每步将叶子层合并到上一层父块
    merged_docs = list(hits)
    total_merged = 0
    per_level_merged = {}

    # 合并步数 = min(max_steps, len(merge_levels) - 1)
    steps = min(max_steps, len(merge_levels) - 1)
    for step in range(steps):
        from_level = merge_levels[step]  # 当前层
        to_level = merge_levels[step + 1]  # 目标父层级

        merged_docs, merged_count = merge_to_parent_level(
            hits=merged_docs,
            parent_level=to_level,
            threshold=merge_threshold,
            collection=collection,
        )
        if merged_count > 0:
            per_level_merged[f"L{from_level}_to_L{to_level}"] = merged_count
            total_merged += merged_count

    # 按分数排序并截取 top_k
    merged_docs.sort(key=lambda item: item.get("rerank_score") or item.get("score", 0.0), reverse=True)
    final_hits = merged_docs[:top_k]

    return final_hits, {
        "enabled": True,
        "merge_threshold": merge_threshold,
        "merge_steps": steps,
        "total_merged_count": total_merged,
        "per_level_merged": per_level_merged,
        "input_hit_count": len(hits),
        "output_hit_count": len(final_hits),
    }


def integrate_auto_merge_to_retrieval(
    retrieval_result: Dict[str, Any],
    collection=None,
    top_k: int = DEFAULT_TOP_K,
    merge_threshold: int = DEFAULT_MERGE_THRESHOLD,
    max_steps: int = 2,
    config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    将 Auto-Merging 集成到检索结果中。

    :param retrieval_result: ChromaRetriever 返回的检索结果
    :param collection: Chroma collection（可不传）
    :param top_k: 返回结果数量
    :param merge_threshold: 子块命中阈值
    :param max_steps: 最大合并步数
    :param config_path: 配置文件路径（用于加载 merge_levels 等）
    :return: 增加 auto_merge 字段的检索结果
    """
    config = load_auto_merge_config(config_path)
    if not config.get("enabled", True):
        retrieval_result["auto_merge"] = {
            "enabled": False,
            "skipped_reason": "auto_merge_disabled",
        }
        return retrieval_result

    hits = list(retrieval_result.get("hits") or [])
    if not hits:
        retrieval_result["auto_merge"] = {
            "enabled": True,
            "merge_steps": 0,
            "total_merged_count": 0,
        }
        return retrieval_result

    merged_hits, merge_meta = auto_merge_hits(
        hits=hits,
        top_k=top_k,
        merge_threshold=merge_threshold,
        max_steps=max_steps,
        merge_levels=config.get("merge_levels", [3, 2, 1]),
        collection=collection,
    )

    retrieval_result["hits"] = merged_hits
    retrieval_result["hit_count"] = len(merged_hits)
    retrieval_result["auto_merge"] = merge_meta

    return retrieval_result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Auto-Merging 测试：从 JSON 文件读取检索结果并执行合并")
    parser.add_argument("input", nargs="?", help="检索结果 JSON 文件路径")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=int, default=DEFAULT_MERGE_THRESHOLD)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    args = parser.parse_args()

    if args.input:
        retrieval_result = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        import sys

        retrieval_result = json.loads(sys.stdin.read())

    result = integrate_auto_merge_to_retrieval(
        retrieval_result=retrieval_result,
        top_k=args.top_k,
        merge_threshold=args.threshold,
        max_steps=args.max_steps,
    )

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)