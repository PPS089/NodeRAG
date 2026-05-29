from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.retrieval.ChromaRetriever import retrieve  # noqa: E402


DEFAULT_EVAL_FILE = PROJECT_ROOT / "eval" / "rag_retrieval_eval.jsonl"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "eval" / "retrieval_eval_result.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"评测集 JSONL 第 {line_no} 行格式错误: {exc}") from exc
            rows.append(row)
    return rows


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict):
        return " ".join(f"{normalize_text(k)} {normalize_text(v)}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(normalize_text(item) for item in value)
    return str(value).lower()


def hit_document_name(hit: Dict[str, Any]) -> str:
    metadata = hit.get("metadata") or {}
    return str(
        metadata.get("document_name")
        or hit.get("document_name")
        or metadata.get("source_document")
        or ""
    )


def hit_chunk_id(hit: Dict[str, Any]) -> str:
    metadata = hit.get("metadata") or {}
    return str(hit.get("chunk_id") or metadata.get("chunk_id") or hit.get("chroma_id") or "")


def hit_text(hit: Dict[str, Any]) -> str:
    metadata = hit.get("metadata") or {}
    parts = [
        hit.get("document"),
        hit.get("content"),
        hit.get("text"),
        hit.get("expanded_context"),
        metadata,
    ]
    return normalize_text(parts)


def parse_k_values(raw: str) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("--k-values 只能包含正整数")
        values.append(value)
    return sorted(set(values))


def first_expected_doc_rank(hits: Sequence[Dict[str, Any]], expected_documents: Sequence[str]) -> int | None:
    expected = set(expected_documents)
    for index, hit in enumerate(hits, start=1):
        if hit_document_name(hit) in expected:
            return index
    return None


def doc_recall_at_k(hits: Sequence[Dict[str, Any]], expected_documents: Sequence[str], k: int) -> float:
    if not expected_documents:
        return 0.0
    found = {hit_document_name(hit) for hit in hits[:k]}
    expected = set(expected_documents)
    return len(found & expected) / len(expected)


def doc_hit_at_k(hits: Sequence[Dict[str, Any]], expected_documents: Sequence[str], k: int) -> bool:
    if not expected_documents:
        return False
    found = {hit_document_name(hit) for hit in hits[:k]}
    return bool(found & set(expected_documents))


def keyword_recall_at_k(hits: Sequence[Dict[str, Any]], expected_keywords: Sequence[str], k: int) -> float:
    if not expected_keywords:
        return 0.0
    joined = normalize_text([hit_text(hit) for hit in hits[:k]])
    matched = [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) in joined
    ]
    return len(matched) / len(expected_keywords)


def expected_chunk_hit_at_k(hits: Sequence[Dict[str, Any]], expected_chunk_ids: Sequence[str], k: int) -> bool | None:
    if not expected_chunk_ids:
        return None
    found = {hit_chunk_id(hit) for hit in hits[:k]}
    return bool(found & set(expected_chunk_ids))


def evaluate_case(
    case: Dict[str, Any],
    k_values: Sequence[int],
    per_query_k: int,
    bm25_k: int,
    rerank_pool_size: int,
    top_k: int,
    use_bm25: bool,
    use_mmr: bool,
    table_aware: bool,
    expand_context: bool,
) -> Dict[str, Any]:
    question = str(case.get("question", "")).strip()
    if not question:
        raise ValueError(f"评测用例缺少 question: {case}")

    result = retrieve(
        question=question,
        per_query_k=per_query_k,
        bm25_k=bm25_k,
        rerank_pool_size=rerank_pool_size,
        top_k=top_k,
        use_bm25=use_bm25,
        use_mmr=use_mmr,
        table_aware=table_aware,
        expand_context=expand_context,
    )
    hits = result.get("hits", [])
    expected_documents = list(case.get("expected_documents") or [])
    expected_keywords = list(case.get("expected_keywords") or [])
    expected_chunk_ids = list(case.get("expected_chunk_ids") or [])

    doc_recalls = {f"doc_recall@{k}": doc_recall_at_k(hits, expected_documents, k) for k in k_values}
    doc_hits = {f"doc_hit@{k}": doc_hit_at_k(hits, expected_documents, k) for k in k_values}
    keyword_recalls = {
        f"keyword_recall@{k}": keyword_recall_at_k(hits, expected_keywords, k)
        for k in k_values
    }
    chunk_hits = {
        f"chunk_hit@{k}": expected_chunk_hit_at_k(hits, expected_chunk_ids, k)
        for k in k_values
    }

    first_rank = first_expected_doc_rank(hits, expected_documents)
    mrr = 0.0 if first_rank is None else 1.0 / first_rank

    return {
        "id": case.get("id"),
        "question": question,
        "category": case.get("category"),
        "required_permission_level": case.get("required_permission_level"),
        "expected_documents": expected_documents,
        "expected_keywords": expected_keywords,
        "first_expected_doc_rank": first_rank,
        "mrr": mrr,
        "top_documents": [hit_document_name(hit) for hit in hits],
        "top_chunk_ids": [hit_chunk_id(hit) for hit in hits],
        "hit_count": len(hits),
        **doc_recalls,
        **doc_hits,
        **keyword_recalls,
        **chunk_hits,
    }


def aggregate(results: Sequence[Dict[str, Any]], k_values: Sequence[int]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "case_count": len(results),
        "mrr": mean([float(item["mrr"]) for item in results]) if results else 0.0,
    }
    for k in k_values:
        summary[f"doc_hit@{k}"] = mean([1.0 if item[f"doc_hit@{k}"] else 0.0 for item in results]) if results else 0.0
        summary[f"doc_recall@{k}"] = mean([float(item[f"doc_recall@{k}"]) for item in results]) if results else 0.0
        summary[f"keyword_recall@{k}"] = mean([float(item[f"keyword_recall@{k}"]) for item in results]) if results else 0.0

        chunk_values = [
            item[f"chunk_hit@{k}"]
            for item in results
            if item.get(f"chunk_hit@{k}") is not None
        ]
        if chunk_values:
            summary[f"chunk_hit@{k}"] = mean([1.0 if value else 0.0 for value in chunk_values])

    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        by_category.setdefault(str(item.get("category") or "unknown"), []).append(item)

    summary["by_category"] = {}
    for category, items in sorted(by_category.items()):
        summary["by_category"][category] = {
            "case_count": len(items),
            "mrr": mean([float(item["mrr"]) for item in items]),
        }
        for k in k_values:
            summary["by_category"][category][f"doc_hit@{k}"] = mean(
                [1.0 if item[f"doc_hit@{k}"] else 0.0 for item in items]
            )
            summary["by_category"][category][f"doc_recall@{k}"] = mean(
                [float(item[f"doc_recall@{k}"]) for item in items]
            )
            summary["by_category"][category][f"keyword_recall@{k}"] = mean(
                [float(item[f"keyword_recall@{k}"]) for item in items]
            )

    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 ChromaRetriever 的文档召回率、关键词召回率和 MRR。")
    parser.add_argument("--eval-file", default=str(DEFAULT_EVAL_FILE), help="评测集 JSONL 文件。")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT_FILE), help="评测结果 JSON 输出路径。")
    parser.add_argument("--k-values", default="1,3,5,8", help="逗号分隔的 K 值，例如 1,3,5,8。")
    parser.add_argument("--limit", type=int, help="只评估前 N 条，用于快速验证。")
    parser.add_argument("--category", action="append", default=[], help="只评估指定 category，可重复传入。")
    parser.add_argument("--per-query-k", type=int, default=12)
    parser.add_argument("--bm25-k", type=int, default=20)
    parser.add_argument("--rerank-pool-size", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25 召回。")
    parser.add_argument("--no-mmr", action="store_true", help="关闭 MMR。")
    parser.add_argument("--no-table-aware", action="store_true", help="关闭表格问题加权。")
    parser.add_argument("--no-expand", action="store_true", help="关闭 small-to-big 上下文扩展。")
    parser.add_argument("--dry-run", action="store_true", help="只校验评测集格式，不执行检索。")
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    eval_file = Path(args.eval_file)
    cases = load_jsonl(eval_file)
    if args.category:
        categories = set(args.category)
        cases = [case for case in cases if case.get("category") in categories]
    if args.limit is not None:
        cases = cases[:args.limit]

    k_values = parse_k_values(args.k_values)
    if args.top_k < max(k_values):
        raise ValueError(f"--top-k 不能小于最大 K 值 {max(k_values)}")

    if args.dry_run:
        result = {
            "eval_file": str(eval_file),
            "case_count": len(cases),
            "k_values": k_values,
            "dry_run": True,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    details = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.get('id')} {case.get('question')}", file=sys.stderr)
        details.append(
            evaluate_case(
                case=case,
                k_values=k_values,
                per_query_k=args.per_query_k,
                bm25_k=args.bm25_k,
                rerank_pool_size=args.rerank_pool_size,
                top_k=args.top_k,
                use_bm25=not args.no_bm25,
                use_mmr=not args.no_mmr,
                table_aware=not args.no_table_aware,
                expand_context=not args.no_expand,
            )
        )

    payload = {
        "eval_file": str(eval_file),
        "k_values": k_values,
        "retrieval_config": {
            "per_query_k": args.per_query_k,
            "bm25_k": args.bm25_k,
            "rerank_pool_size": args.rerank_pool_size,
            "top_k": args.top_k,
            "use_bm25": not args.no_bm25,
            "use_mmr": not args.no_mmr,
            "table_aware": not args.no_table_aware,
            "expand_context": not args.no_expand,
        },
        "summary": aggregate(details, k_values),
        "details": details,
    }
    output_path = Path(args.output)
    write_json(output_path, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"结果已写入: {output_path}", file=sys.stderr)
    return payload


if __name__ == "__main__":
    main()
