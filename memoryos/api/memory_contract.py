"""Transport-neutral Markdown memory command contracts.

HTTP, the Python SDK and MCP all use these operation names, payload fields and
result shapes.  Authentication identity is deliberately not part of a memory
command payload: transports bind tenant and owner from ``TrustedRequestContext``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

_DIGEST = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_OPTIONAL_DIGEST = {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"}
_RESTORE_DIGEST = {"type": "string", "pattern": "^(?:[0-9a-f]{64})?$", "allowEmpty": True}
_OPTIONAL_STRING = {"type": ["string", "null"]}

MEMORY_COMMAND_REQUEST_SCHEMAS: dict[str, dict[str, Any]] = {
    "adopt": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relative_path": {"type": "string", "minLength": 1},
            "expected_raw_sha256": _DIGEST,
        },
        "required": ["relative_path", "expected_raw_sha256"],
    },
    "remember": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "content": {"type": "string", "minLength": 1},
            "occurred_at": _OPTIONAL_STRING,
            "target_hint": _OPTIONAL_STRING,
            "expected_document_digest": _OPTIONAL_DIGEST,
        },
        "required": ["content"],
    },
    "edit": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_uri": {"type": "string", "minLength": 1},
            "edit": {"type": "string", "minLength": 1},
            "expected_digest": _DIGEST,
        },
        "required": ["document_uri", "edit", "expected_digest"],
    },
    "rename": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_uri": {"type": "string", "minLength": 1},
            "new_relative_path": {"type": "string", "minLength": 1},
            "expected_digest": _DIGEST,
            "edit": {"type": ["string", "null"], "minLength": 1},
        },
        "required": ["document_uri", "new_relative_path", "expected_digest"],
    },
    "merge": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_document_uri": {"type": "string", "minLength": 1},
            "merged_edit": {"type": "string", "minLength": 1},
            "expected_target_digest": _DIGEST,
            "source_documents": {
                "type": "array",
                "minItems": 1,
                "maxItems": 100,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "document_uri": {"type": "string", "minLength": 1},
                        "expected_digest": _DIGEST,
                    },
                    "required": ["document_uri", "expected_digest"],
                },
            },
        },
        "required": [
            "target_document_uri",
            "merged_edit",
            "expected_target_digest",
            "source_documents",
        ],
    },
    "merge_resume": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"saga_id": {"type": "string", "minLength": 1}},
        "required": ["saga_id"],
    },
    "forget": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_uri": {"type": "string", "minLength": 1},
            "section_anchor": _OPTIONAL_STRING,
            "mode": {"type": "string", "enum": ["SOFT_FORGET", "HARD_ERASE"]},
            "expected_digest": _OPTIONAL_DIGEST,
        },
        "required": ["document_uri"],
    },
    "history": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"document_uri": {"type": "string", "minLength": 1}},
        "required": ["document_uri"],
    },
    "restore": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_uri": {"type": "string", "minLength": 1},
            "revision": {"type": "integer", "minimum": 1},
            "expected_digest": _RESTORE_DIGEST,
        },
        "required": ["document_uri", "revision", "expected_digest"],
    },
    "review": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "proposal_id": {"type": "string", "minLength": 1},
            "decision": {"type": "string", "enum": ["APPROVE", "REJECT", "CORRECT"]},
            "corrected_edit": _OPTIONAL_STRING,
        },
        "required": ["proposal_id", "decision"],
    },
    "review_preview": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"proposal_id": {"type": "string", "minLength": 1}},
        "required": ["proposal_id"],
    },
}

# A Dreams/summary consolidation proposal accepts the same exact target/source
# bindings as direct merge, but seals a copy-on-write review instead of writing.
MEMORY_COMMAND_REQUEST_SCHEMAS["merge_propose"] = deepcopy(
    MEMORY_COMMAND_REQUEST_SCHEMAS["merge"]
)

_CONSOLIDATION_SOURCE_RESULT: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "document_uri": {"type": "string", "minLength": 1},
        "document_id": {"type": "string", "minLength": 1},
        "relative_path": {"type": "string", "minLength": 1},
        "source_digest": _DIGEST,
        "size": {"type": "integer", "minimum": 0},
    },
    "required": [
        "document_uri",
        "document_id",
        "relative_path",
        "source_digest",
        "size",
    ],
}

_DOCUMENT_RESULT_PROPERTIES: dict[str, Any] = {
    "document_uri": {"type": "string"},
    "document_id": {"type": "string"},
    "document_kind": {"type": "string"},
    "relative_path": {"type": "string"},
    "document_revision": {"type": "integer", "minimum": 0},
    "source_digest": {"type": "string"},
    "changed": {"type": "boolean"},
    "edit_summary": {"type": "string"},
    "projection_status": {"type": "string"},
}
_DOCUMENT_RESULT_REQUIRED = list(_DOCUMENT_RESULT_PROPERTIES)

MEMORY_COMMAND_RESPONSE_SCHEMAS: dict[str, dict[str, Any]] = {
    "adopt": {
        "type": "object",
        "additionalProperties": False,
        "properties": deepcopy(_DOCUMENT_RESULT_PROPERTIES),
        "required": list(_DOCUMENT_RESULT_REQUIRED),
    },
    "remember": {
        "type": "object",
        "additionalProperties": False,
        "properties": deepcopy(_DOCUMENT_RESULT_PROPERTIES),
        "required": list(_DOCUMENT_RESULT_REQUIRED),
    },
    "edit": {
        "type": "object",
        "additionalProperties": False,
        "properties": deepcopy(_DOCUMENT_RESULT_PROPERTIES),
        "required": list(_DOCUMENT_RESULT_REQUIRED),
    },
    "rename": {
        "type": "object",
        "additionalProperties": False,
        "properties": deepcopy(_DOCUMENT_RESULT_PROPERTIES),
        "required": list(_DOCUMENT_RESULT_REQUIRED),
    },
    "merge": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "saga_id": {"type": "string", "minLength": 1},
            "status": {"type": "string"},
            "target_document_id": {"type": "string", "minLength": 1},
            "target_projection_generation": {"type": "integer", "minimum": 0},
            "target_projection_confirmed": {"type": "boolean"},
            "soft_forgotten_document_ids": {"type": "array", "items": {"type": "string"}},
            "pending_document_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "saga_id",
            "status",
            "target_document_id",
            "target_projection_generation",
            "target_projection_confirmed",
            "soft_forgotten_document_ids",
            "pending_document_ids",
        ],
    },
    "merge_resume": {},
    "forget": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            **deepcopy(_DOCUMENT_RESULT_PROPERTIES),
            "mode": {"type": "string", "enum": ["SOFT_FORGET", "HARD_ERASE"]},
            "recoverable": {"type": "boolean"},
            "erasure_status": {"type": "string"},
            "erasure_epoch": {"type": "string"},
            "pending_backends": {"type": "array", "items": {"type": "string"}},
            "independent_evidence_retained": {"type": "array", "items": {"type": "string"}},
            "media_disclaimer": {"type": "string"},
        },
        "required": [*_DOCUMENT_RESULT_REQUIRED, "mode", "recoverable"],
    },
    "history": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_uri": {"type": "string"},
            "document_id": {"type": "string"},
            "document_kind": {"type": "string"},
            "relative_path": {"type": "string"},
            "revisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "document_revision": {"type": "integer", "minimum": 1},
                        "projection_generation": {"type": "integer", "minimum": 1},
                        "edit_kind": {"type": "string"},
                        "relative_path": {"type": "string", "minLength": 1},
                        "source_digest": _DIGEST,
                        "state": {"type": "string"},
                        "created_at": {"type": "string", "minLength": 1},
                        "restorable": {"type": "boolean"},
                    },
                    "required": [
                        "document_revision",
                        "projection_generation",
                        "edit_kind",
                        "relative_path",
                        "source_digest",
                        "state",
                        "created_at",
                        "restorable",
                    ],
                },
            },
        },
        "required": ["document_uri", "document_id", "document_kind", "relative_path", "revisions"],
    },
    "restore": {
        "type": "object",
        "additionalProperties": False,
        "properties": deepcopy(_DOCUMENT_RESULT_PROPERTIES),
        "required": list(_DOCUMENT_RESULT_REQUIRED),
    },
    "review": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "proposal_id": {"type": "string"},
            "status": {"type": "string"},
            **deepcopy(_DOCUMENT_RESULT_PROPERTIES),
            "proposed_source_digest": {"type": "string"},
            "proposed_diff_digest": {"type": "string"},
            "replacement_proposal_id": {"type": "string"},
            "workflow_kind": {
                "type": "string",
                "enum": ["DOCUMENT_EDIT", "CONSOLIDATION"],
            },
            "consolidation_sources": {
                "type": "array",
                "items": deepcopy(_CONSOLIDATION_SOURCE_RESULT),
            },
            "consolidation_saga_id": {"type": "string"},
            "consolidation_status": {"type": "string"},
            "target_projection_generation": {"type": "integer", "minimum": 0},
            "target_projection_confirmed": {"type": "boolean"},
            "soft_forgotten_document_ids": {"type": "array", "items": {"type": "string"}},
            "pending_document_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "proposal_id",
            "status",
            *_DOCUMENT_RESULT_REQUIRED,
            "proposed_source_digest",
            "proposed_diff_digest",
        ],
    },
    "review_preview": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "proposal_id": {"type": "string"},
            "status": {"type": "string"},
            "document_uri": {"type": "string"},
            "document_id": {"type": "string"},
            "document_kind": {"type": "string"},
            "relative_path": {"type": "string"},
            "source_digest": {"type": "string"},
            "proposed_source_digest": {"type": "string"},
            "proposed_diff_digest": {"type": "string"},
            "proposed_diff": {"type": "string"},
            "edit_summary": {"type": "string"},
            "workflow_kind": {
                "type": "string",
                "enum": ["DOCUMENT_EDIT", "CONSOLIDATION"],
            },
            "consolidation_sources": {
                "type": "array",
                "items": deepcopy(_CONSOLIDATION_SOURCE_RESULT),
            },
        },
        "required": [
            "proposal_id",
            "status",
            "document_uri",
            "document_id",
            "document_kind",
            "relative_path",
            "source_digest",
            "proposed_source_digest",
            "proposed_diff_digest",
            "proposed_diff",
            "edit_summary",
        ],
    },
}

# Resume returns the exact same content-free saga shape as the initial merge.
MEMORY_COMMAND_RESPONSE_SCHEMAS["merge_resume"] = deepcopy(
    MEMORY_COMMAND_RESPONSE_SCHEMAS["merge"]
)
MEMORY_COMMAND_RESPONSE_SCHEMAS["merge_propose"] = deepcopy(
    MEMORY_COMMAND_RESPONSE_SCHEMAS["review_preview"]
)


def memory_request_schema(operation: str) -> dict[str, Any]:
    """Return an isolated JSON-schema copy for one memory operation."""

    try:
        return deepcopy(MEMORY_COMMAND_REQUEST_SCHEMAS[operation])
    except KeyError as exc:
        raise ValueError(f"unknown memory operation: {operation}") from exc


def memory_response_schema(operation: str) -> dict[str, Any]:
    """Return an isolated result schema for documentation and contract tests."""

    try:
        return deepcopy(MEMORY_COMMAND_RESPONSE_SCHEMAS[operation])
    except KeyError as exc:
        raise ValueError(f"unknown memory operation: {operation}") from exc


def validate_memory_request(operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the small command schema without a runtime JSON-schema dependency."""

    schema = MEMORY_COMMAND_REQUEST_SCHEMAS.get(operation)
    if schema is None:
        raise ValueError(f"unknown memory operation: {operation}")
    data = dict(payload)
    allowed = set(schema["properties"])
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{operation} contains unsupported fields: {', '.join(unknown)}")
    missing = [key for key in schema.get("required", ()) if key not in data]
    if missing:
        raise ValueError(f"{operation} requires fields: {', '.join(missing)}")
    for key, value in data.items():
        _validate_value(operation, key, value, schema["properties"][key])
    if operation == "review":
        decision = str(data["decision"]).upper()
        data["decision"] = decision
        if decision == "CORRECT" and not str(data.get("corrected_edit") or "").strip():
            raise ValueError("review CORRECT requires corrected_edit")
        if decision != "CORRECT" and data.get("corrected_edit") is not None:
            raise ValueError("only review CORRECT accepts corrected_edit")
    if operation == "forget":
        data["mode"] = str(data.get("mode") or "SOFT_FORGET").upper()
        if data["mode"] == "HARD_ERASE" and data.get("section_anchor") is not None:
            raise ValueError("HARD_ERASE only accepts a whole-document target")
    return data


def memory_result_payload(value: Any) -> dict[str, Any]:
    """Serialize one application result as an exact JSON-compatible object."""

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
    elif is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError("memory command result must be a mapping or dataclass")
    if not isinstance(payload, dict):
        raise TypeError("memory command result serializer must return an object")
    normalized = _normalize_json(payload, path="memory response")
    if not isinstance(normalized, dict):  # pragma: no cover - guarded above.
        raise TypeError("memory command result serializer must return an object")
    return normalized


def validate_memory_response(operation: str, value: Any) -> dict[str, Any]:
    """Normalize and enforce the declared response schema at every transport exit."""

    schema = MEMORY_COMMAND_RESPONSE_SCHEMAS.get(operation)
    if schema is None:
        raise ValueError(f"unknown memory operation: {operation}")
    payload = memory_result_payload(value)
    _validate_schema_value(payload, schema, path=operation)
    return payload


def _validate_value(operation: str, key: str, value: Any, spec: Mapping[str, Any]) -> None:
    expected = spec.get("type")
    types = set(expected if isinstance(expected, list) else [expected])
    if value is None:
        if "null" in types:
            return
        raise ValueError(f"{operation}.{key} cannot be null")
    if "string" in types:
        if not isinstance(value, str):
            raise ValueError(f"{operation}.{key} must be a string")
        if spec.get("minLength") and not value.strip():
            raise ValueError(f"{operation}.{key} must be non-empty")
        if "enum" in spec and value.upper() not in spec["enum"]:
            raise ValueError(f"{operation}.{key} has an unsupported value")
        if spec.get("allowEmpty") and value == "":
            return
        if spec.get("pattern") and (len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)):
            raise ValueError(f"{operation}.{key} must be a lowercase SHA-256 digest")
        return
    if "integer" in types:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{operation}.{key} must be an integer")
        if value < int(spec.get("minimum", 0)):
            raise ValueError(f"{operation}.{key} is below the minimum")
        return
    if "array" in types or "object" in types:
        _validate_schema_value(value, spec, path=f"{operation}.{key}")
        return
    raise ValueError(f"unsupported contract type for {operation}.{key}")


def _normalize_json(value: Any, *, path: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return _normalize_json(value.value, path=path)
    if isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} must not contain a non-finite number")
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _normalize_json(to_dict(), path=path)
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_json(asdict(value), path=path)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} object keys must be strings")
            normalized[key] = _normalize_json(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, list | tuple):
        return [
            _normalize_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


def _validate_schema_value(value: Any, spec: Mapping[str, Any], *, path: str) -> None:
    expected = spec.get("type")
    types = set(expected if isinstance(expected, list) else [expected])
    if value is None:
        if "null" in types:
            return
        raise ValueError(f"{path} cannot be null")
    if "object" in types:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        properties = spec.get("properties")
        if properties is None:
            return
        if not isinstance(properties, Mapping):
            raise TypeError(f"invalid contract schema at {path}")
        unknown = sorted(set(value) - set(properties))
        if spec.get("additionalProperties") is False and unknown:
            raise ValueError(f"{path} contains unsupported fields: {', '.join(unknown)}")
        missing = [key for key in spec.get("required", ()) if key not in value]
        if missing:
            raise ValueError(f"{path} requires fields: {', '.join(missing)}")
        for key, item in value.items():
            child = properties.get(key)
            if child is not None:
                _validate_schema_value(item, child, path=f"{path}.{key}")
        return
    if "array" in types:
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        if len(value) < int(spec.get("minItems", 0)):
            raise ValueError(f"{path} has too few items")
        if "maxItems" in spec and len(value) > int(spec["maxItems"]):
            raise ValueError(f"{path} has too many items")
        item_spec = spec.get("items")
        if item_spec is not None:
            for index, item in enumerate(value):
                _validate_schema_value(item, item_spec, path=f"{path}[{index}]")
        return
    if "boolean" in types:
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be a boolean")
        return
    if "integer" in types:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} must be an integer")
        if "minimum" in spec and value < int(spec["minimum"]):
            raise ValueError(f"{path} is below the minimum")
        return
    if "number" in types:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path} must be a number")
        if "minimum" in spec and value < float(spec["minimum"]):
            raise ValueError(f"{path} is below the minimum")
        return
    if "string" in types:
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        if len(value) < int(spec.get("minLength", 0)):
            raise ValueError(f"{path} is shorter than the minimum length")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{path} has an unsupported value")
        pattern = spec.get("pattern")
        if pattern and re.fullmatch(str(pattern), value) is None:
            raise ValueError(f"{path} does not match its required pattern")
        return
    raise TypeError(f"unsupported contract schema type at {path}")


__all__ = [
    "MEMORY_COMMAND_REQUEST_SCHEMAS",
    "MEMORY_COMMAND_RESPONSE_SCHEMAS",
    "memory_request_schema",
    "memory_response_schema",
    "memory_result_payload",
    "validate_memory_request",
    "validate_memory_response",
]
