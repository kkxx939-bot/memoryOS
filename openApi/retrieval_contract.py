"""HTTP、MCP 和 SDK 共用的结构化检索参数契约。

本模块负责公开 JSON Schema 与 Python ``RetrievalOptions`` 之间的转换，让所有
外部通道使用同一组字段、枚举和数量上限，不在这里执行实际检索。
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from infrastructure.context.retrieval.query_plan import (
    MAX_CANDIDATE_LIMIT,
    MAX_FILTER_VALUES,
    MAX_FINAL_LIMIT,
    MAX_TARGET_PATHS,
    MAX_TARGET_URIS,
    RetrievalOptions,
    RetrievalQueryIntent,
)
from infrastructure.store.model.context.context_type import ContextType

_STRING_ARRAY = {
    "type": "array",
    "items": {"type": "string", "minLength": 1},
    "maxItems": MAX_FILTER_VALUES,
    "uniqueItems": True,
}

RETRIEVAL_OPTIONS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target_uris": {**_STRING_ARRAY, "maxItems": MAX_TARGET_URIS},
        "target_paths": {**_STRING_ARRAY, "maxItems": MAX_TARGET_PATHS},
        "context_types": {
            **_STRING_ARRAY,
            "items": {"type": "string", "enum": [item.value for item in ContextType]},
        },
        "source_kinds": _STRING_ARRAY,
        "record_kinds": _STRING_ARRAY,
        "document_ids": _STRING_ARRAY,
        "document_kinds": _STRING_ARRAY,
        "owner_user_id": {"type": ["string", "null"]},
        "workspace_ids": _STRING_ARRAY,
        "session_ids": _STRING_ARRAY,
        "adapter_id": {"type": ["string", "null"]},
        "event_time_from": {"type": ["string", "null"]},
        "event_time_to": {"type": ["string", "null"]},
        "transaction_time_from": {"type": ["string", "null"]},
        "transaction_time_to": {"type": ["string", "null"]},
        "updated_at_from": {"type": ["string", "null"]},
        "updated_at_to": {"type": ["string", "null"]},
        "timezone": {"type": "string", "minLength": 1},
        "query_intent": {
            "type": "string",
            "enum": [item.value for item in RetrievalQueryIntent],
        },
        "relation_expansion": {"type": "boolean"},
        "candidate_limit": {"type": "integer", "minimum": 1, "maximum": MAX_CANDIDATE_LIMIT},
        "final_limit": {"type": "integer", "minimum": 1, "maximum": MAX_FINAL_LIMIT},
        "metadata_filters": {"type": "object"},
    },
}


def retrieval_options_json_schema() -> dict[str, Any]:
    """返回供 HTTP、MCP 和工具定义独立修改的 Schema 副本。"""

    return deepcopy(RETRIEVAL_OPTIONS_JSON_SCHEMA)


def parse_retrieval_options(value: Any) -> RetrievalOptions | None:
    """把外部字典规范化为核心可消费的检索选项。"""

    if value is None:
        return None
    if isinstance(value, RetrievalOptions):
        if value.tenant_id not in {None, "default"}:
            raise ValueError("tenant selection is unavailable in local single-user mode")
        return value
    if not isinstance(value, Mapping):
        raise TypeError("options must be a retrieval options object")
    if "tenant_id" in value:
        raise ValueError("tenant_id is not a public retrieval option in local single-user mode")
    return RetrievalOptions.from_dict(value)


def serialize_retrieval_options(value: RetrievalOptions | Mapping[str, Any] | None) -> dict[str, Any] | None:
    """把检索选项转换为可通过 HTTP 或 MCP 传输的字典。"""

    options = parse_retrieval_options(value)
    if options is None:
        return None
    payload = options.to_dict()
    payload.pop("tenant_id", None)
    return payload


__all__ = [
    "RETRIEVAL_OPTIONS_JSON_SCHEMA",
    "parse_retrieval_options",
    "retrieval_options_json_schema",
    "serialize_retrieval_options",
]
