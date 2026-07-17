"""Canonical JSON serialization shared by all durable subsystems."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import date, datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any


class CanonicalSerializationError(ValueError):
    """A value cannot be represented with MemoryOS canonical JSON."""


def canonicalize(value: Any) -> Any:
    """Return the existing JSON-safe deterministic snapshot semantics."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalSerializationError("non-finite floats are not valid evidence values")
        return value
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, datetime):
        resolved = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return resolved.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if key in result:
                raise CanonicalSerializationError(f"mapping keys collide after string normalization: {key!r}")
            result[key] = canonicalize(raw_value)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, list | tuple):
        return [canonicalize(item) for item in value]
    if isinstance(value, set | frozenset):
        items = [canonicalize(item) for item in value]
        return sorted(items, key=canonical_json)
    raise CanonicalSerializationError(f"unsupported evidence value type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def immutable_snapshot(value: Any) -> Any:
    """Deeply snapshot mutable caller data without retaining references."""

    return _freeze_normalized(canonicalize(value))


def _freeze_normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_normalized(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_normalized(item) for item in value)
    return value


__all__ = ["CanonicalSerializationError", "canonical_json", "canonicalize", "immutable_snapshot"]
