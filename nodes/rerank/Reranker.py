from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.retrieval.retrieval_config import DEFAULT_MMR_LAMBDA, DEFAULT_TOP_K  # noqa: E402
from nodes.retrieval.retrieval_utils import (  # noqa: E402
    get_hit_text,
    jaccard_similarity,
    looks_like_table_query,
    min_max_normalize_scores,
    parse_json_metadata,
    token_set,
    unique_keep_order,
)


DEFAULT_RERANK_TOP_N = 24
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "rerank_config.json"
DEFAULT_RERANK_CONFIG = {
    "enabled": True,
    "mode": "rule",
    "rerank_top_n": DEFAULT_RERANK_TOP_N,
    "final_top_k": DEFAULT_TOP_K,
    "use_mmr": True,
    "table_aware": True,
    "mmr_lambda": DEFAULT_MMR_LAMBDA,
    "weights": {
        "vector": 0.44,
        "bm25": 0.26,
        "lexical": 0.16,
        "title_path": 0.14,
    },
    "bonuses": {
        "multi_source": 0.08,
        "table_chunk": 0.12,
    },
}


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_rerank_config(config_path: str | Path | None = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    if not config_path:
        return dict(DEFAULT_RERANK_CONFIG)

    path = Path(config_path)
    if not path.exists():
        return dict(DEFAULT_RERANK_CONFIG)

    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"重排配置必须是 JSON object: {path}")
    return deep_merge_dict(DEFAULT_RERANK_CONFIG, config)


def lexical_overlap_score(query_tokens: Set[str], hit: Dict[str, Any]) -> float:
    if not query_tokens:
        return 0.0

    hit_tokens = token_set(get_hit_text(hit))
    if not hit_tokens:
        return 0.0

    return len(query_tokens & hit_tokens) / len(query_tokens)


def title_path_score(query_tokens: Set[str], hit: Dict[str, Any]) -> float:
    if not query_tokens:
        return 0.0

    metadata = hit.get("metadata", {})
    title_path = parse_json_metadata(metadata.get("title_path"), default=[])
    title_text = " ".join(title_path) if isinstance(title_path, list) else str(title_path)
    title_tokens = token_set(title_text)
    if not title_tokens:
        return 0.0

    return len(query_tokens & title_tokens) / len(query_tokens)


def rerank_hits(
    hits: Sequence[Dict[str, Any]],
    query_texts: Sequence[str],
    table_aware: bool = True,
    stage_name: str = "rule_rerank",
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    规则重排：融合向量分、BM25、词面覆盖、标题匹配、来源和表格意图。
    """

    ranked_hits = [dict(hit) for hit in hits]
    min_max_normalize_scores(ranked_hits, "bm25_score", "bm25_score_norm")
    min_max_normalize_scores(ranked_hits, "vector_score", "vector_score_norm")

    query_tokens = token_set(" ".join(query_texts))
    table_query = table_aware and looks_like_table_query(query_texts)
    rerank_config = config or DEFAULT_RERANK_CONFIG
    weights = rerank_config.get("weights") or {}
    bonuses = rerank_config.get("bonuses") or {}

    for hit in ranked_hits:
        metadata = hit.get("metadata", {})
        lexical_score = lexical_overlap_score(query_tokens, hit)
        title_score = title_path_score(query_tokens, hit)
        source_bonus = float(bonuses.get("multi_source", 0.0)) if len(hit.get("retrieval_sources", [])) > 1 else 0.0
        table_bonus = float(bonuses.get("table_chunk", 0.0)) if table_query and metadata.get("chunk_type") == "table_chunk" else 0.0

        rerank_score = (
            float(weights.get("vector", 0.0)) * float(hit.get("vector_score_norm", 0.0))
            + float(weights.get("bm25", 0.0)) * float(hit.get("bm25_score_norm", 0.0))
            + float(weights.get("lexical", 0.0)) * lexical_score
            + float(weights.get("title_path", 0.0)) * title_score
            + source_bonus
            + table_bonus
        )
        features = {
            "vector_score_norm": float(hit.get("vector_score_norm", 0.0)),
            "bm25_score_norm": float(hit.get("bm25_score_norm", 0.0)),
            "lexical_overlap_score": lexical_score,
            "title_path_score": title_score,
            "source_bonus": source_bonus,
            "table_bonus": table_bonus,
        }
        hit["lexical_overlap_score"] = lexical_score
        hit["title_path_score"] = title_score
        hit["table_bonus"] = table_bonus
        hit["source_bonus"] = source_bonus
        hit["rerank_score"] = rerank_score
        hit["rerank_stage"] = stage_name
        hit["rerank_features"] = features
        hit["rerank_weights"] = {
            "vector": float(weights.get("vector", 0.0)),
            "bm25": float(weights.get("bm25", 0.0)),
            "lexical": float(weights.get("lexical", 0.0)),
            "title_path": float(weights.get("title_path", 0.0)),
        }
        hit["rerank_reason"] = build_rerank_reason(features)

    return sorted(ranked_hits, key=lambda item: item["rerank_score"], reverse=True)


def build_rerank_reason(features: Dict[str, float]) -> str:
    ordered = sorted(features.items(), key=lambda item: item[1], reverse=True)
    top_features = [
        f"{name}={value:.3f}"
        for name, value in ordered[:3]
        if value > 0
    ]
    return "；".join(top_features) if top_features else "无明显重排加分信号"


def mmr_select(
    hits: Sequence[Dict[str, Any]],
    top_k: int,
    lambda_mult: float = DEFAULT_MMR_LAMBDA,
) -> List[Dict[str, Any]]:
    """
    MMR 去冗余，避免结果集中全是同一段相似内容。
    """

    if top_k <= 0:
        return []

    remaining = [dict(hit) for hit in hits]
    selected: List[Dict[str, Any]] = []
    token_cache = {
        hit["chroma_id"]: token_set(get_hit_text(hit))
        for hit in remaining
    }

    while remaining and len(selected) < top_k:
        best_hit = None
        best_score = -float("inf")

        for hit in remaining:
            relevance = float(hit.get("rerank_score", 0.0))
            diversity_penalty = 0.0
            hit_tokens = token_cache.get(hit["chroma_id"], set())

            if selected:
                diversity_penalty = max(
                    jaccard_similarity(hit_tokens, token_cache.get(selected_hit["chroma_id"], set()))
                    for selected_hit in selected
                )

            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_hit = hit

        if best_hit is None:
            break

        best_hit["mmr_score"] = best_score
        selected.append(best_hit)
        remaining = [hit for hit in remaining if hit["chroma_id"] != best_hit["chroma_id"]]

    return selected


def rerank_retrieval_result(
    retrieval_result: Dict[str, Any],
    query_texts: Sequence[str] | None = None,
    rerank_top_n: Optional[int] = None,
    final_top_k: Optional[int] = None,
    use_mmr: Optional[bool] = None,
    table_aware: Optional[bool] = None,
    mmr_lambda: Optional[float] = None,
    config_path: str | Path | None = DEFAULT_CONFIG_PATH,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rerank_config = deep_merge_dict(load_rerank_config(config_path), config or {})
    rerank_top_n = int(rerank_top_n if rerank_top_n is not None else rerank_config.get("rerank_top_n", DEFAULT_RERANK_TOP_N))
    final_top_k = int(final_top_k if final_top_k is not None else rerank_config.get("final_top_k", DEFAULT_TOP_K))
    use_mmr = bool(use_mmr if use_mmr is not None else rerank_config.get("use_mmr", True))
    table_aware = bool(table_aware if table_aware is not None else rerank_config.get("table_aware", True))
    mmr_lambda = float(mmr_lambda if mmr_lambda is not None else rerank_config.get("mmr_lambda", DEFAULT_MMR_LAMBDA))

    result = dict(retrieval_result)
    hits = list(result.get("hits") or [])
    if not rerank_config.get("enabled", True):
        result["hits"] = hits[:final_top_k]
        result["hit_count"] = len(result["hits"])
        result["rerank"] = {
            "enabled": False,
            "mode": rerank_config.get("mode", "rule"),
            "rerank_top_n": rerank_top_n,
            "final_top_k": final_top_k,
            "input_hit_count": len(hits),
            "output_hit_count": len(result["hits"]),
        }
        return result

    if not hits:
        result["hit_count"] = 0
        result["rerank"] = {
            "enabled": True,
            "rerank_top_n": rerank_top_n,
            "final_top_k": final_top_k,
            "input_hit_count": 0,
            "output_hit_count": 0,
        }
        return result

    queries = unique_keep_order(
        list(query_texts or [])
        or [str(result.get("question", ""))]
        + list(result.get("queries") or [])
        + list(result.get("keyword_queries") or [])
    )
    pool = hits[:max(rerank_top_n, final_top_k)]
    ranked_hits = rerank_hits(
        pool,
        query_texts=queries,
        table_aware=table_aware,
        stage_name="second_stage_rule_rerank",
        config=rerank_config,
    )
    selected_hits = mmr_select(ranked_hits, top_k=final_top_k, lambda_mult=mmr_lambda) if use_mmr else ranked_hits[:final_top_k]

    result["hits"] = selected_hits
    result["hit_count"] = len(selected_hits)
    result["rerank"] = {
        "enabled": True,
        "mode": rerank_config.get("mode", "rule"),
        "config_path": str(config_path) if config_path else None,
        "rerank_top_n": rerank_top_n,
        "final_top_k": final_top_k,
        "use_mmr": use_mmr,
        "table_aware": table_aware,
        "mmr_lambda": mmr_lambda,
        "weights": rerank_config.get("weights", {}),
        "bonuses": rerank_config.get("bonuses", {}),
        "input_hit_count": len(hits),
        "output_hit_count": len(selected_hits),
    }
    return result


def load_json_input(input_path: str | None) -> Dict[str, Any]:
    if input_path:
        return json.loads(Path(input_path).read_text(encoding="utf-8"))

    raw = sys.stdin.read().strip()
    if not raw:
        raise ValueError("请通过 --input 传入检索 JSON 文件，或通过 stdin 输入 JSON")
    return json.loads(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立重排节点：对 RetrievalResult 做二阶段规则重排。")
    parser.add_argument("--input", "-i", help="检索结果 JSON 文件；不传则从 stdin 读取。")
    parser.add_argument("--output", "-o", help="输出 JSON 文件；不传则打印到 stdout。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="重排配置 JSON 路径。")
    parser.add_argument("--query", action="append", default=[], help="额外重排 query，可重复传入。")
    parser.add_argument("--rerank-top-n", type=int, help="覆盖配置中的 rerank_top_n。")
    parser.add_argument("--final-top-k", type=int, help="覆盖配置中的 final_top_k。")
    parser.add_argument("--no-mmr", action="store_true", help="关闭二阶段 MMR。")
    parser.add_argument("--no-table-aware", action="store_true", help="关闭表格问题加权。")
    parser.add_argument("--mmr-lambda", type=float, help="覆盖配置中的 mmr_lambda。")
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    retrieval_result = load_json_input(args.input)
    result = rerank_retrieval_result(
        retrieval_result=retrieval_result,
        query_texts=args.query,
        rerank_top_n=args.rerank_top_n,
        final_top_k=args.final_top_k,
        use_mmr=False if args.no_mmr else None,
        table_aware=False if args.no_table_aware else None,
        mmr_lambda=args.mmr_lambda,
        config_path=args.config,
    )
    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)
    return result


if __name__ == "__main__":
    main()
