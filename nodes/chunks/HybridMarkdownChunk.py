from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.chunks.MarkDownChunk import (  # noqa: E402
    count_chunk_types,
    find_datacleaned_md_files,
    split_markdown,
)
from utils.FindProjectRoot import find_project_root as fr  # noqa: E402


HYBRID_CHUNK_SCHEMA_VERSION = "1.0"
HYBRID_CHUNK_STRATEGY = "markdown_header_parent_child_table_aware_small_to_big"
DEFAULT_INPUT_NAME = "DataCleaned.md"
DEFAULT_OUTPUT_SUFFIX = ".hybrid_chunks.json"
RETRIEVAL_CHILD_TYPES = {"text_chunk", "table_chunk", "image_chunk"}
NEIGHBOR_WINDOW = 1


def infer_document_id(input_md: str | Path) -> str:
    """
    使用 MinerUResult 的文档目录名作为稳定 document_id。
    """

    path = Path(input_md)
    return path.parent.name or path.stem


def find_parent_section_id(
    chunk: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """
    沿 parent_id 向上寻找最近的 section_chunk。
    """

    parent_id = chunk.get("parent_id")
    while parent_id:
        parent = chunks_by_id.get(parent_id)
        if not parent:
            return None
        if parent.get("type") == "section_chunk":
            return parent.get("id")
        parent_id = parent.get("parent_id")

    return None


def collect_ref_target_ids(chunk: Dict[str, Any]) -> List[str]:
    """
    收集 text_chunk 中引用到的表格/图片 chunk。
    """

    ref_ids = []
    for ref in chunk.get("refs", []) or []:
        target_id = ref.get("target_chunk_id")
        if target_id:
            ref_ids.append(target_id)

    for target_id in chunk.get("ref_target_ids", []) or []:
        if target_id:
            ref_ids.append(target_id)

    return sorted(set(ref_ids))


def collect_back_reference_ids(chunk: Dict[str, Any]) -> List[str]:
    """
    收集 table/image chunk 被哪些 text_chunk 引用。
    """

    return sorted(set(chunk.get("referenced_by_text_chunk_ids", []) or []))


def get_retrieval_chunk_ids(chunks: Sequence[Dict[str, Any]]) -> List[str]:
    return [
        chunk["id"]
        for chunk in chunks
        if chunk.get("type") in RETRIEVAL_CHILD_TYPES
    ]


def find_neighbor_chunk_ids(
    chunk_id: str,
    retrieval_chunk_ids: Sequence[str],
    window: int = NEIGHBOR_WINDOW,
) -> Tuple[List[str], List[str]]:
    """
    基于 retrieval child 的顺序，返回前后邻居。
    """

    if chunk_id not in retrieval_chunk_ids:
        return [], []

    index = retrieval_chunk_ids.index(chunk_id)
    previous_ids = list(retrieval_chunk_ids[max(0, index - window):index])
    next_ids = list(retrieval_chunk_ids[index + 1:index + 1 + window])
    return previous_ids, next_ids


def build_parent_context(
    chunk: Dict[str, Any],
    parent_section: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    构造 parent section 上下文，不复制大段正文。
    """

    if not parent_section:
        return {
            "parent_section_id": None,
            "parent_section_title": None,
            "parent_section_title_path": chunk.get("title_path", []),
            "parent_section_line_range": None,
        }

    return {
        "parent_section_id": parent_section.get("id"),
        "parent_section_title": parent_section.get("title"),
        "parent_section_title_path": parent_section.get("title_path", []),
        "parent_section_line_range": parent_section.get("section_line_range"),
    }


def build_search_text(
    chunk: Dict[str, Any],
    parent_context: Dict[str, Any],
) -> str:
    """
    生成用于向量化的检索文本。
    """

    parts = [
        f"文档: {chunk.get('document_name', '')}",
    ]

    title_path = parent_context.get("parent_section_title_path") or chunk.get("title_path", [])
    if title_path:
        parts.append("章节: " + " > ".join(title_path))

    chunk_type = chunk.get("type")
    if chunk_type == "table_chunk":
        if chunk.get("table_description"):
            parts.append(str(chunk["table_description"]))
        if chunk.get("table_text"):
            parts.append(str(chunk["table_text"]))
    elif chunk_type == "image_chunk":
        if chunk.get("alt_text"):
            parts.append(f"图片说明: {chunk['alt_text']}")
        if chunk.get("image_path"):
            parts.append(f"图片路径: {chunk['image_path']}")
    elif chunk_type == "section_chunk":
        parts.append(f"标题: {chunk.get('title', chunk.get('content', ''))}")
    else:
        parts.append(str(chunk.get("content", "")))

    return "\n".join(part for part in parts if part).strip()


def enrich_chunk(
    chunk: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    retrieval_chunk_ids: Sequence[str],
    input_md: str | Path,
) -> Dict[str, Any]:
    """
    在基础 chunk 上增加成熟 RAG 检索需要的元数据。
    """

    chunk_type = chunk.get("type")
    parent_section_id = find_parent_section_id(chunk, chunks_by_id)
    parent_section = chunks_by_id.get(parent_section_id) if parent_section_id else None
    parent_context = build_parent_context(chunk, parent_section)
    previous_ids, next_ids = find_neighbor_chunk_ids(chunk["id"], retrieval_chunk_ids)
    ref_target_ids = collect_ref_target_ids(chunk)
    back_reference_ids = collect_back_reference_ids(chunk)

    small_to_big_context_ids = []
    if parent_context["parent_section_id"]:
        small_to_big_context_ids.append(parent_context["parent_section_id"])
    small_to_big_context_ids.extend(previous_ids)
    small_to_big_context_ids.extend(next_ids)
    small_to_big_context_ids.extend(ref_target_ids)
    small_to_big_context_ids.extend(back_reference_ids)

    enriched = dict(chunk)
    enriched.update(
        {
            "chunk_schema_version": HYBRID_CHUNK_SCHEMA_VERSION,
            "chunk_strategy": HYBRID_CHUNK_STRATEGY,
            "document_id": infer_document_id(input_md),
            "retrieval_role": (
                "parent_section"
                if chunk_type == "section_chunk"
                else "retrieval_child"
                if chunk_type in RETRIEVAL_CHILD_TYPES
                else "supporting_chunk"
            ),
            "should_embed": chunk_type in RETRIEVAL_CHILD_TYPES,
            "semantic_unit_type": chunk_type.replace("_chunk", "") if chunk_type else "unknown",
            "parent_context": parent_context,
            "neighbor_chunk_ids": {
                "previous": previous_ids,
                "next": next_ids,
            },
            "related_ref_chunk_ids": {
                "outgoing": ref_target_ids,
                "incoming": back_reference_ids,
            },
            "small_to_big_context_ids": sorted(set(small_to_big_context_ids)),
            "embedding_text": build_search_text(chunk, parent_context),
        }
    )

    return enriched


def build_hybrid_chunks(input_md: str | Path, skip_toc: bool = True) -> List[Dict[str, Any]]:
    """
    构建 Hybrid 分片结果。

    基础解析仍由 MarkDownChunk.split_markdown 完成；本函数只增强检索策略层。
    """

    base_chunks = split_markdown(input_md, skip_toc=skip_toc)
    chunks_by_id = {chunk["id"]: chunk for chunk in base_chunks}
    retrieval_chunk_ids = get_retrieval_chunk_ids(base_chunks)

    return [
        enrich_chunk(
            chunk=chunk,
            chunks_by_id=chunks_by_id,
            retrieval_chunk_ids=retrieval_chunk_ids,
            input_md=input_md,
        )
        for chunk in base_chunks
    ]


def hybrid_chunk_md_file(
    input_md: str | Path,
    output_path: str | Path | None = None,
    skip_toc: bool = True,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """
    对单个 DataCleaned.md 生成 Hybrid chunks JSON。
    """

    input_file = Path(input_md)
    chunks = build_hybrid_chunks(input_file, skip_toc=skip_toc)

    if output_path is None:
        output_file = input_file.with_suffix(DEFAULT_OUTPUT_SUFFIX)
    else:
        output_file = Path(output_path)

    output_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file, chunks


def hybrid_chunk_mineru_result_dir(
    result_dir: str | Path | None = None,
    input_name: str = DEFAULT_INPUT_NAME,
    skip_toc: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    批量处理 MinerUResult 下所有 DataCleaned.md。
    """

    input_files = find_datacleaned_md_files(result_dir=result_dir, input_name=input_name)
    if not input_files:
        target_dir = Path(result_dir) if result_dir else fr() / "MinerUResult"
        raise FileNotFoundError(f"未找到待分片 Markdown: {target_dir}/*/{input_name}")

    outputs: Dict[str, Dict[str, Any]] = {}
    for input_file in input_files:
        output_file, chunks = hybrid_chunk_md_file(input_file, skip_toc=skip_toc)
        outputs[str(input_file)] = {
            "output_path": str(output_file),
            "chunk_count": len(chunks),
            "chunk_type_counts": count_chunk_types(chunks),
            "embed_chunk_count": sum(1 for chunk in chunks if chunk.get("should_embed")),
        }

    return outputs


def main(input_md: str | Path | None = None) -> List[Dict[str, Any]] | Dict[str, Dict[str, Any]]:
    """
    传入文件路径时处理单个文件；不传时批量处理 MinerUResult/*/DataCleaned.md。
    """

    if input_md is not None:
        output_file, chunks = hybrid_chunk_md_file(input_md)
        print(f"Hybrid 分片完成: {input_md}")
        print(f"输出文件: {output_file}")
        print(f"chunk 数量: {len(chunks)}")
        print(f"类型分布: {count_chunk_types(chunks)}")
        print(f"默认向量化 chunk 数量: {sum(1 for chunk in chunks if chunk.get('should_embed'))}")
        return chunks

    outputs = hybrid_chunk_mineru_result_dir()
    print(f"Hybrid 批量分片完成，共处理 {len(outputs)} 个 Markdown:")
    for input_file, summary in outputs.items():
        print(f"- {input_file}")
        print(f"  输出文件: {summary['output_path']}")
        print(f"  chunk 数量: {summary['chunk_count']}")
        print(f"  类型分布: {summary['chunk_type_counts']}")
        print(f"  默认向量化 chunk 数量: {summary['embed_chunk_count']}")

    return outputs


if __name__ == "__main__":
    cli_input = sys.argv[1] if len(sys.argv) > 1 else None
    main(cli_input)
