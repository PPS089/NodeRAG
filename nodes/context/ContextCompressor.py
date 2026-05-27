from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.contracts import validate_retrieval_result  # noqa: E402


DEFAULT_MAX_CONTEXT_CHARS = 12000
DEFAULT_MAX_BLOCK_CHARS = 1800
DEFAULT_MAX_BLOCKS = 12


def load_json_input(input_path: Optional[str]) -> Dict[str, Any]:
    if input_path:
        path = Path(input_path)
        for encoding in ("utf-8", "utf-8-sig", "utf-16"):
            try:
                return json.loads(path.read_text(encoding=encoding))
            except UnicodeDecodeError:
                continue
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))

    raw = sys.stdin.read().strip()
    if not raw:
        raise ValueError("请通过 --input 传入检索 JSON 文件，或通过 stdin 输入 JSON")

    return json.loads(raw)


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n...[TRUNCATED {len(text) - max_chars} chars]"


def normalize_line_range(value: Any) -> List[int]:
    if isinstance(value, list) and len(value) == 2:
        return value
    return []


def build_citation_id(index: int) -> str:
    return f"C{index}"


def chunk_identity(chunk: Dict[str, Any]) -> str:
    return "::".join(
        [
            str(chunk.get("document_id", "")),
            str(chunk.get("chunk_id", "")),
            str(chunk.get("type", "")),
            str(chunk.get("line_range", "")),
        ]
    )


def get_chunk_text(chunk: Dict[str, Any]) -> str:
    chunk_type = chunk.get("type")

    if chunk_type == "table_chunk":
        return str(chunk.get("embedding_text") or chunk.get("content") or "")

    return str(chunk.get("content") or chunk.get("embedding_text") or "")


def score_context_chunk(hit: Dict[str, Any], chunk: Dict[str, Any], order: int) -> float:
    base_score = float(hit.get("rerank_score") or hit.get("score") or 0.0)
    role_bonus = 0.12 if chunk.get("role") == "hit" else 0.0
    table_bonus = 0.08 if chunk.get("type") == "table_chunk" else 0.0
    order_penalty = order * 0.01
    return base_score + role_bonus + table_bonus - order_penalty


def collect_context_candidates(retrieval_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = []

    for hit_index, hit in enumerate(retrieval_result.get("hits", []), start=1):
        expanded_context = hit.get("expanded_context") or []
        if not expanded_context:
            metadata = hit.get("metadata", {})
            expanded_context = [
                {
                    "role": "hit",
                    "chunk_id": hit.get("chunk_id", ""),
                    "type": metadata.get("chunk_type", ""),
                    "document_id": metadata.get("document_id", ""),
                    "document_name": metadata.get("document_name", ""),
                    "title_path": metadata.get("title_path", []),
                    "line_range": metadata.get("line_range", []),
                    "content": hit.get("document", ""),
                    "embedding_text": "",
                    "parent_context": metadata.get("parent_context", {}),
                }
            ]

        for order, chunk in enumerate(expanded_context):
            text = get_chunk_text(chunk)
            if not text.strip():
                continue

            candidates.append(
                {
                    "hit_rank": hit_index,
                    "score": score_context_chunk(hit, chunk, order),
                    "source_hit": {
                        "chroma_id": hit.get("chroma_id", ""),
                        "rerank_score": hit.get("rerank_score"),
                        "mmr_score": hit.get("mmr_score"),
                        "retrieval_sources": hit.get("retrieval_sources", []),
                    },
                    "chunk": chunk,
                    "text": text,
                    "identity": chunk_identity(chunk),
                }
            )

    return candidates


def compress_context(
    retrieval_result: Dict[str, Any],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_block_chars: int = DEFAULT_MAX_BLOCK_CHARS,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
) -> Dict[str, Any]:
    validate_retrieval_result(retrieval_result)

    candidates = collect_context_candidates(retrieval_result)
    candidates.sort(key=lambda item: item["score"], reverse=True)

    selected = []
    used_identities = set()
    used_chars = 0

    for candidate in candidates:
        if len(selected) >= max_blocks:
            break
        if candidate["identity"] in used_identities:
            continue

        chunk = candidate["chunk"]
        text = truncate_text(candidate["text"], max_block_chars)
        if used_chars + len(text) > max_context_chars:
            remaining = max_context_chars - used_chars
            if remaining <= 200:
                break
            text = truncate_text(text, remaining)

        citation_id = build_citation_id(len(selected) + 1)
        block = {
            "citation_id": citation_id,
            "score": candidate["score"],
            "role": chunk.get("role", ""),
            "chunk_id": chunk.get("chunk_id", ""),
            "chunk_type": chunk.get("type", ""),
            "document_id": chunk.get("document_id", ""),
            "document_name": chunk.get("document_name", ""),
            "title_path": chunk.get("title_path", []),
            "line_range": normalize_line_range(chunk.get("line_range")),
            "content": text,
            "source_hit": candidate["source_hit"],
        }
        selected.append(block)
        used_identities.add(candidate["identity"])
        used_chars += len(text)

    citations = [
        {
            "citation_id": block["citation_id"],
            "document_name": block["document_name"],
            "title_path": block["title_path"],
            "line_range": block["line_range"],
            "chunk_id": block["chunk_id"],
            "chunk_type": block["chunk_type"],
        }
        for block in selected
    ]

    return {
        "question": retrieval_result.get("question", ""),
        "context_blocks": selected,
        "citations": citations,
        "compression_stats": {
            "input_hit_count": len(retrieval_result.get("hits", [])),
            "candidate_count": len(candidates),
            "selected_block_count": len(selected),
            "context_chars": used_chars,
            "max_context_chars": max_context_chars,
            "max_block_chars": max_block_chars,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="压缩 ChromaRetriever 输出的检索上下文。")
    parser.add_argument("--input", "-i", help="检索结果 JSON 文件；不传则从 stdin 读取。")
    parser.add_argument("--output", "-o", help="输出 JSON 文件；不传则打印到 stdout。")
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--max-block-chars", type=int, default=DEFAULT_MAX_BLOCK_CHARS)
    parser.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    retrieval_result = load_json_input(args.input)
    result = compress_context(
        retrieval_result=retrieval_result,
        max_context_chars=args.max_context_chars,
        max_block_chars=args.max_block_chars,
        max_blocks=args.max_blocks,
    )

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)

    return result


if __name__ == "__main__":
    main()
