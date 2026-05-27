from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "knowledge_permissions.json"
DEFAULT_PERMISSION_LEVEL = "L1"


def unique_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def load_permission_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    path = Path(config_path)
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_permission_level(level: str) -> str:
    normalized = str(level or "").strip().upper()
    if normalized and normalized[0].isdigit():
        normalized = f"L{normalized}"
    return normalized or DEFAULT_PERMISSION_LEVEL


def level_rank(level: str, levels: Dict[str, int]) -> int:
    normalized = normalize_permission_level(level)
    if normalized not in levels:
        raise ValueError(f"未知权限等级: {level}，可选值: {sorted(levels)}")
    return int(levels[normalized])


def normalize_document_name(name: str) -> str:
    return str(name or "").strip()


def build_permission_context(
    permission_level: str = DEFAULT_PERMISSION_LEVEL,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> Dict[str, Any]:
    config = load_permission_config(config_path)
    levels = config.get("levels") or {}
    user_level = normalize_permission_level(permission_level)
    user_rank = level_rank(user_level, levels)

    allowed_rules = []
    denied_rules = []
    for rule in config.get("knowledge_bases", []):
        required_level = normalize_permission_level(rule.get("required_level", DEFAULT_PERMISSION_LEVEL))
        required_rank = level_rank(required_level, levels)
        normalized_rule = dict(rule)
        normalized_rule["required_level"] = required_level
        normalized_rule["required_rank"] = required_rank
        if required_rank <= user_rank:
            allowed_rules.append(normalized_rule)
        else:
            denied_rules.append(normalized_rule)

    allowed_document_names = []
    for rule in allowed_rules:
        allowed_document_names.extend(rule.get("document_names") or [])

    return {
        "permission_level": user_level,
        "permission_rank": user_rank,
        "default_access": str(config.get("default_access", "deny")).lower(),
        "levels": levels,
        "allowed_rules": allowed_rules,
        "denied_rules": denied_rules,
        "allowed_knowledge_bases": [rule.get("knowledge_base", "") for rule in allowed_rules],
        "denied_knowledge_bases": [rule.get("knowledge_base", "") for rule in denied_rules],
        "allowed_permission_codes": [rule.get("permission_code", "") for rule in allowed_rules],
        "allowed_document_names": unique_keep_order(allowed_document_names),
    }


def document_matches_rule(document_name: str, rule: Dict[str, Any]) -> bool:
    name = normalize_document_name(document_name)
    if not name:
        return False

    candidates = [
        str(rule.get("knowledge_base", "")),
        *[str(item) for item in rule.get("document_names") or []],
    ]
    if name in {normalize_document_name(item) for item in candidates if normalize_document_name(item)}:
        return True

    return any(
        fnmatch.fnmatchcase(name, str(pattern))
        for pattern in rule.get("document_patterns") or []
        if str(pattern).strip()
    )


def find_document_rule(document_name: str, permission_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for rule in permission_context.get("allowed_rules", []) + permission_context.get("denied_rules", []):
        if document_matches_rule(document_name, rule):
            return rule
    return None


def can_access_document(document_name: str, permission_context: Dict[str, Any]) -> bool:
    rule = find_document_rule(document_name, permission_context)
    if rule:
        return int(rule.get("required_rank", 999)) <= int(permission_context.get("permission_rank", 0))

    return permission_context.get("default_access") == "allow"


def split_document_names_by_permission(
    document_names: Sequence[str],
    permission_context: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    allowed = []
    denied = []
    for document_name in document_names:
        if can_access_document(document_name, permission_context):
            allowed.append(document_name)
        else:
            denied.append(document_name)
    return unique_keep_order(allowed), unique_keep_order(denied)


def apply_permission_to_retrieval_inputs(
    retrieval_inputs: Dict[str, Any],
    permission_context: Dict[str, Any],
) -> Dict[str, Any]:
    result = dict(retrieval_inputs)
    requested_names = result.get("document_names") or []

    if requested_names:
        allowed_names, denied_names = split_document_names_by_permission(requested_names, permission_context)
        result["document_names"] = allowed_names
    else:
        denied_names = []
        result["document_names"] = list(permission_context.get("allowed_document_names") or [])

    result["permission_level"] = permission_context.get("permission_level")
    result["allowed_permission_codes"] = list(permission_context.get("allowed_permission_codes") or [])
    result["denied_document_names"] = denied_names
    result["permission_denied"] = bool(requested_names and not result["document_names"])
    return result


def filter_expanded_context(
    expanded_context: Sequence[Dict[str, Any]],
    permission_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        item
        for item in expanded_context
        if can_access_document(str(item.get("document_name", "")), permission_context)
    ]


def filter_retrieval_result_by_permission(
    retrieval_result: Dict[str, Any],
    permission_context: Dict[str, Any],
) -> Dict[str, Any]:
    result = dict(retrieval_result)
    allowed_hits = []
    denied_hits = []

    for hit in result.get("hits") or []:
        metadata = hit.get("metadata") or {}
        document_name = str(metadata.get("document_name") or "")
        if can_access_document(document_name, permission_context):
            copied_hit = dict(hit)
            if isinstance(copied_hit.get("expanded_context"), list):
                copied_hit["expanded_context"] = filter_expanded_context(
                    copied_hit["expanded_context"],
                    permission_context,
                )
            allowed_hits.append(copied_hit)
        else:
            denied_hits.append(
                {
                    "chroma_id": hit.get("chroma_id"),
                    "chunk_id": hit.get("chunk_id"),
                    "document_name": document_name,
                }
            )

    result["hits"] = allowed_hits
    result["hit_count"] = len(allowed_hits)
    result["permission"] = {
        "permission_level": permission_context.get("permission_level"),
        "allowed_permission_codes": permission_context.get("allowed_permission_codes", []),
        "denied_hit_count": len(denied_hits),
        "denied_hits": denied_hits,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="知识库权限配置检查工具。")
    parser.add_argument("--permission-level", default=DEFAULT_PERMISSION_LEVEL, help="模拟权限等级：L1-L5。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="权限配置 JSON 路径。")
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    permission_context = build_permission_context(args.permission_level, args.config)
    print(json.dumps(permission_context, ensure_ascii=False, indent=2))
    return permission_context


if __name__ == "__main__":
    main()
