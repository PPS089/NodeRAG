from __future__ import annotations

from typing import Any, Mapping, Sequence


class ContractError(ValueError):
    """
    节点输入输出协议错误。
    """


HYBRID_CHUNK_REQUIRED_FIELDS = (
    "id",
    "type",
    "document_id",
    "document_name",
    "should_embed",
    "embedding_text",
    "small_to_big_context_ids",
)

RETRIEVAL_RESULT_REQUIRED_FIELDS = (
    "question",
    "hits",
)

RETRIEVAL_HIT_REQUIRED_FIELDS = (
    "chroma_id",
    "chunk_id",
    "metadata",
)

COMPRESSED_CONTEXT_REQUIRED_FIELDS = (
    "question",
    "context_blocks",
    "citations",
)

PROMPT_PAYLOAD_REQUIRED_FIELDS = (
    "question",
    "messages",
    "citations",
)


def validate_mapping(value: Any, contract_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{contract_name} 必须是 dict，当前类型: {type(value).__name__}")
    return value


def validate_required_fields(
    value: Mapping[str, Any],
    required_fields: Sequence[str],
    contract_name: str,
) -> None:
    missing_fields = [
        field
        for field in required_fields
        if field not in value
    ]
    if missing_fields:
        raise ContractError(f"{contract_name} 缺少必要字段: {missing_fields}")


def validate_list_field(value: Mapping[str, Any], field: str, contract_name: str) -> None:
    if not isinstance(value.get(field), list):
        raise ContractError(f"{contract_name}.{field} 必须是 list")


def validate_hybrid_chunk(chunk: Mapping[str, Any], source: str = "") -> None:
    name = f"HybridChunk({source})" if source else "HybridChunk"
    validate_required_fields(chunk, HYBRID_CHUNK_REQUIRED_FIELDS, name)

    if chunk.get("should_embed") is True and not str(chunk.get("embedding_text", "")).strip():
        raise ContractError(f"{name}.embedding_text 不能为空")


def validate_retrieval_result(result: Mapping[str, Any]) -> None:
    validate_required_fields(result, RETRIEVAL_RESULT_REQUIRED_FIELDS, "RetrievalResult")
    validate_list_field(result, "hits", "RetrievalResult")

    for index, hit in enumerate(result.get("hits", [])):
        hit = validate_mapping(hit, f"RetrievalResult.hits[{index}]")
        validate_required_fields(hit, RETRIEVAL_HIT_REQUIRED_FIELDS, f"RetrievalResult.hits[{index}]")


def validate_compressed_context(result: Mapping[str, Any]) -> None:
    validate_required_fields(result, COMPRESSED_CONTEXT_REQUIRED_FIELDS, "CompressedContext")
    validate_list_field(result, "context_blocks", "CompressedContext")
    validate_list_field(result, "citations", "CompressedContext")


def validate_prompt_payload(result: Mapping[str, Any]) -> None:
    validate_required_fields(result, PROMPT_PAYLOAD_REQUIRED_FIELDS, "PromptPayload")
    validate_list_field(result, "messages", "PromptPayload")
    validate_list_field(result, "citations", "PromptPayload")
