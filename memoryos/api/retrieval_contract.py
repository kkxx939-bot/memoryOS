"""One transport-neutral schema for structured retrieval options."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.query_plan import (
    MAX_CANDIDATE_LIMIT,
    MAX_FILTER_VALUES,
    MAX_FINAL_LIMIT,
    MAX_TARGET_PATHS,
    MAX_TARGET_URIS,
    MAX_TOKEN_BUDGET,
    RetrievalOptions,
    RetrievalQueryIntent,
)

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
        "tenant_id": {"type": ["string", "null"]},
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
        "token_budget": {"type": "integer", "minimum": 1, "maximum": MAX_TOKEN_BUDGET},
        "metadata_filters": {"type": "object"},
    },
}


def retrieval_options_json_schema() -> dict[str, Any]:
    """Return an isolated schema copy for HTTP/MCP/tool definitions."""

    return deepcopy(RETRIEVAL_OPTIONS_JSON_SCHEMA)


def parse_retrieval_options(value: Any) -> RetrievalOptions | None:
    if value is None:
        return None
    if isinstance(value, RetrievalOptions):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("options must be a retrieval options object")
    return RetrievalOptions.from_dict(value)


def serialize_retrieval_options(value: RetrievalOptions | Mapping[str, Any] | None) -> dict[str, Any] | None:
    options = parse_retrieval_options(value)
    return options.to_dict() if options is not None else None


__all__ = [
    "RETRIEVAL_OPTIONS_JSON_SCHEMA",
    "parse_retrieval_options",
    "retrieval_options_json_schema",
    "serialize_retrieval_options",
]
