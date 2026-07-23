"""统一上下文召回使用的可序列化查询契约。"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo
from datetime import timezone as dt_timezone
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from foundation.identity.workspace import normalize_workspace_id
from infrastructure.store.model.catalog import normalize_timestamp as normalize_catalog_timestamp
from infrastructure.store.model.catalog import normalize_tree_path
from infrastructure.store.model.context.context_type import ContextType

DEFAULT_CANDIDATE_LIMIT = 100
DEFAULT_FINAL_LIMIT = 20
MAX_CANDIDATE_LIMIT = 1_000
MAX_FINAL_LIMIT = 200
MAX_TARGET_URIS = 256
MAX_TARGET_PATHS = 16
MAX_FILTER_VALUES = 256
MAX_SEMANTIC_QUERY_CHARS = 16_384

_MAX_URI_CHARS = 2_048
_MAX_PATH_CHARS = 512
_MAX_IDENTIFIER_CHARS = 512
_MAX_METADATA_DEPTH = 8
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_OFFSET_TIMEZONE = re.compile(r"^(?P<sign>[+-])(?P<hours>\d{2}):(?P<minutes>\d{2})$")
_SOURCE_KIND = re.compile(r"^[a-z][a-z0-9_.-]*$")


class RetrievalQueryIntent(str, Enum):
    """对调用者可见的召回语义。"""

    CURRENT = "CURRENT"
    HISTORY = "HISTORY"
    OPEN_RECALL = "OPEN_RECALL"
    EXACT = "EXACT"


@dataclass(frozen=True)
class RetrievalOptions:
    """所有传输入口共享的结构化召回参数。

    时间范围采用 ``from <= value < to`` 的左闭右开语义。只有日期的下界表示
    ``timezone`` 中当天开始，只有日期的上界表示下一本地日期开始；所有索引时间
    最终序列化为 UTC ISO-8601。
    """

    target_uris: tuple[str, ...] = ()
    target_paths: tuple[str, ...] = ()
    context_types: tuple[ContextType, ...] = ()
    source_kinds: tuple[str, ...] = ()
    record_kinds: tuple[str, ...] = ()

    tenant_id: str | None = None
    owner_user_id: str | None = None
    workspace_ids: tuple[str, ...] = ()
    session_ids: tuple[str, ...] = ()
    adapter_id: str | None = None

    event_time_from: str | date | datetime | None = None
    event_time_to: str | date | datetime | None = None
    transaction_time_from: str | date | datetime | None = None
    transaction_time_to: str | date | datetime | None = None
    updated_at_from: str | date | datetime | None = None
    updated_at_to: str | date | datetime | None = None
    timezone: str = "UTC"

    query_intent: RetrievalQueryIntent = RetrievalQueryIntent.CURRENT
    relation_expansion: bool = False

    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    final_limit: int = DEFAULT_FINAL_LIMIT
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    legacy_search_scope: str | None = None
    legacy_retrieval_views: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        timezone_name, resolved_timezone = _resolve_timezone(self.timezone)
        object.__setattr__(self, "timezone", timezone_name)

        object.__setattr__(
            self,
            "target_uris",
            _normalize_strings(
                self.target_uris,
                label="target_uris",
                maximum=MAX_TARGET_URIS,
                max_chars=_MAX_URI_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "target_paths",
            _normalize_paths(self.target_paths),
        )
        object.__setattr__(
            self,
            "context_types",
            _normalize_context_types(self.context_types),
        )
        object.__setattr__(
            self,
            "source_kinds",
            _normalize_source_kinds(self.source_kinds),
        )
        for field_name in ("record_kinds",):
            object.__setattr__(
                self,
                field_name,
                _normalize_strings(
                    getattr(self, field_name),
                    label=field_name,
                    maximum=MAX_FILTER_VALUES,
                    max_chars=_MAX_IDENTIFIER_CHARS,
                ),
            )

        for field_name in ("tenant_id", "owner_user_id", "adapter_id"):
            object.__setattr__(self, field_name, _normalize_optional_identifier(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "workspace_ids",
            tuple(
                normalize_workspace_id(value)
                for value in _normalize_strings(
                    self.workspace_ids,
                    label="workspace_ids",
                    maximum=MAX_FILTER_VALUES,
                    max_chars=_MAX_IDENTIFIER_CHARS,
                )
            ),
        )
        object.__setattr__(
            self,
            "session_ids",
            _normalize_strings(
                self.session_ids,
                label="session_ids",
                maximum=MAX_FILTER_VALUES,
                max_chars=_MAX_IDENTIFIER_CHARS,
            ),
        )

        event_from = _normalize_timestamp(
            self.event_time_from,
            label="event_time_from",
            caller_timezone=resolved_timezone,
            date_is_upper_bound=False,
        )
        event_to = _normalize_timestamp(
            self.event_time_to,
            label="event_time_to",
            caller_timezone=resolved_timezone,
            date_is_upper_bound=True,
        )
        transaction_from = _normalize_timestamp(
            self.transaction_time_from,
            label="transaction_time_from",
            caller_timezone=resolved_timezone,
            date_is_upper_bound=False,
        )
        transaction_to = _normalize_timestamp(
            self.transaction_time_to,
            label="transaction_time_to",
            caller_timezone=resolved_timezone,
            date_is_upper_bound=True,
        )
        updated_from = _normalize_timestamp(
            self.updated_at_from,
            label="updated_at_from",
            caller_timezone=resolved_timezone,
            date_is_upper_bound=False,
        )
        updated_to = _normalize_timestamp(
            self.updated_at_to,
            label="updated_at_to",
            caller_timezone=resolved_timezone,
            date_is_upper_bound=True,
        )
        _validate_time_range(event_from, event_to, "event_time")
        _validate_time_range(transaction_from, transaction_to, "transaction_time")
        _validate_time_range(updated_from, updated_to, "updated_at")
        object.__setattr__(self, "event_time_from", event_from)
        object.__setattr__(self, "event_time_to", event_to)
        object.__setattr__(self, "transaction_time_from", transaction_from)
        object.__setattr__(self, "transaction_time_to", transaction_to)
        object.__setattr__(self, "updated_at_from", updated_from)
        object.__setattr__(self, "updated_at_to", updated_to)

        object.__setattr__(self, "query_intent", _normalize_query_intent(self.query_intent))
        if not isinstance(self.relation_expansion, bool):
            raise TypeError("relation_expansion must be a bool")

        candidate_limit = _bounded_integer(
            self.candidate_limit,
            label="candidate_limit",
            maximum=MAX_CANDIDATE_LIMIT,
        )
        final_limit = _bounded_integer(self.final_limit, label="final_limit", maximum=MAX_FINAL_LIMIT)
        if final_limit > candidate_limit:
            raise ValueError("final_limit must not exceed candidate_limit")
        object.__setattr__(self, "candidate_limit", candidate_limit)
        object.__setattr__(self, "final_limit", final_limit)

        object.__setattr__(self, "metadata_filters", _normalize_json_mapping(self.metadata_filters))
        object.__setattr__(
            self,
            "legacy_search_scope",
            _normalize_optional_identifier(self.legacy_search_scope, "legacy_search_scope"),
        )
        object.__setattr__(
            self,
            "legacy_retrieval_views",
            _normalize_strings(
                self.legacy_retrieval_views,
                label="legacy_retrieval_views",
                maximum=MAX_FILTER_VALUES,
                max_chars=_MAX_PATH_CHARS,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_uris": list(self.target_uris),
            "target_paths": list(self.target_paths),
            "context_types": [item.value for item in self.context_types],
            "source_kinds": list(self.source_kinds),
            "record_kinds": list(self.record_kinds),
            "tenant_id": self.tenant_id,
            "owner_user_id": self.owner_user_id,
            "workspace_ids": list(self.workspace_ids),
            "session_ids": list(self.session_ids),
            "adapter_id": self.adapter_id,
            "event_time_from": self.event_time_from,
            "event_time_to": self.event_time_to,
            "transaction_time_from": self.transaction_time_from,
            "transaction_time_to": self.transaction_time_to,
            "updated_at_from": self.updated_at_from,
            "updated_at_to": self.updated_at_to,
            "timezone": self.timezone,
            "query_intent": self.query_intent.value,
            "relation_expansion": self.relation_expansion,
            "candidate_limit": self.candidate_limit,
            "final_limit": self.final_limit,
            "metadata_filters": _copy_json(self.metadata_filters),
            "legacy_search_scope": self.legacy_search_scope,
            "legacy_retrieval_views": list(self.legacy_retrieval_views),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RetrievalOptions:
        if not isinstance(payload, Mapping):
            raise TypeError("retrieval options payload must be a mapping")
        return cls(**dict(payload))


@dataclass(frozen=True)
class RetrievalQueryPlan(RetrievalOptions):
    """已经规范化、可以直接执行的在线检索计划。"""

    semantic_query: str = ""
    service_id: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.semantic_query, str):
            raise TypeError("semantic_query must be a string")
        semantic_query = self.semantic_query.strip()
        if "\x00" in semantic_query:
            raise ValueError("semantic_query must not contain NUL")
        if len(semantic_query) > MAX_SEMANTIC_QUERY_CHARS:
            raise ValueError(f"semantic_query exceeds the maximum of {MAX_SEMANTIC_QUERY_CHARS} characters")
        object.__setattr__(self, "semantic_query", semantic_query)
        object.__setattr__(
            self,
            "service_id",
            _normalize_optional_identifier(self.service_id, "service_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"semantic_query": self.semantic_query, "service_id": self.service_id, **super().to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RetrievalQueryPlan:
        if not isinstance(payload, Mapping):
            raise TypeError("retrieval query plan payload must be a mapping")
        return cls(**dict(payload))


def _normalize_query_intent(value: RetrievalQueryIntent | str) -> RetrievalQueryIntent:
    if isinstance(value, RetrievalQueryIntent):
        return value
    try:
        return RetrievalQueryIntent(str(value).strip().upper())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in RetrievalQueryIntent)
        raise ValueError(f"query_intent must be one of: {allowed}") from exc


def _normalize_context_types(values: Sequence[ContextType | str] | ContextType | str | None) -> tuple[ContextType, ...]:
    items = _as_sequence(values, "context_types")
    normalized: list[ContextType] = []
    for item in items:
        try:
            value = item if isinstance(item, ContextType) else ContextType(str(item).strip().lower())
        except ValueError as exc:
            raise ValueError(f"unknown context_type: {item!r}") from exc
        if value not in normalized:
            normalized.append(value)
    if len(normalized) > len(ContextType):
        raise ValueError("context_types contains too many values")
    return tuple(normalized)


def _normalize_source_kinds(values: Sequence[str] | str | None) -> tuple[str, ...]:
    items = _normalize_strings(
        values,
        label="source_kinds",
        maximum=MAX_FILTER_VALUES,
        max_chars=_MAX_IDENTIFIER_CHARS,
    )
    normalized: list[str] = []
    for item in items:
        value = item.lower()
        if not _SOURCE_KIND.fullmatch(value):
            raise ValueError(f"invalid source_kind: {item!r}")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _normalize_strings(
    values: Sequence[str] | str | None,
    *,
    label: str,
    maximum: int,
    max_chars: int,
) -> tuple[str, ...]:
    items = _as_sequence(values, label)
    normalized: list[str] = []
    for raw in items:
        if not isinstance(raw, str):
            raise TypeError(f"{label} values must be strings")
        value = raw.strip()
        if not value:
            raise ValueError(f"{label} values must not be empty")
        if "\x00" in value:
            raise ValueError(f"{label} values must not contain NUL")
        if len(value) > max_chars:
            raise ValueError(f"{label} value exceeds the maximum of {max_chars} characters")
        if value not in normalized:
            normalized.append(value)
    if len(normalized) > maximum:
        raise ValueError(f"{label} exceeds the maximum of {maximum} values")
    return tuple(normalized)


def _normalize_paths(values: Sequence[str] | str | None) -> tuple[str, ...]:
    paths = _normalize_strings(
        values,
        label="target_paths",
        maximum=MAX_TARGET_PATHS,
        max_chars=_MAX_PATH_CHARS,
    )
    normalized: list[str] = []
    for raw in paths:
        if "\\" in raw:
            raise ValueError("target_paths must use forward slashes")
        value = "/".join(part for part in raw.strip("/").split("/") if part)
        parts = value.split("/") if value else []
        if not parts or any(part in {".", ".."} for part in parts):
            raise ValueError(f"invalid target_path: {raw!r}")
        controlled = normalize_tree_path(value)
        if controlled not in normalized:
            normalized.append(controlled)
    return tuple(normalized)


def _as_sequence(value: Any, label: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str | Enum):
        return (value,)
    if not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


def _normalize_optional_identifier(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    normalized = value.strip()
    if not normalized:
        return None
    if "\x00" in normalized:
        raise ValueError(f"{label} must not contain NUL")
    if len(normalized) > _MAX_IDENTIFIER_CHARS:
        raise ValueError(f"{label} exceeds the maximum of {_MAX_IDENTIFIER_CHARS} characters")
    return normalized


def _resolve_timezone(value: str) -> tuple[str, tzinfo]:
    if not isinstance(value, str):
        raise TypeError("timezone must be a string")
    normalized = value.strip()
    if normalized.upper() in {"UTC", "Z"}:
        return "UTC", dt_timezone.utc
    offset_match = _OFFSET_TIMEZONE.fullmatch(normalized)
    if offset_match:
        hours = int(offset_match.group("hours"))
        minutes = int(offset_match.group("minutes"))
        if hours > 14 or minutes > 59 or (hours == 14 and minutes != 0):
            raise ValueError(f"invalid timezone offset: {value!r}")
        delta = timedelta(hours=hours, minutes=minutes)
        if offset_match.group("sign") == "-":
            delta = -delta
        return normalized, dt_timezone(delta)
    try:
        return normalized, ZoneInfo(normalized)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown timezone: {value!r}") from exc


def _normalize_timestamp(
    value: str | date | datetime | None,
    *,
    label: str,
    caller_timezone: tzinfo,
    date_is_upper_bound: bool,
) -> str | None:
    if value is None:
        return None
    parsed: datetime
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        local_date = value + timedelta(days=1) if date_is_upper_bound else value
        parsed = datetime.combine(local_date, time.min, tzinfo=caller_timezone)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError(f"{label} must not be empty")
        if _DATE_ONLY.fullmatch(raw):
            try:
                local_date = date.fromisoformat(raw)
            except ValueError as exc:
                raise ValueError(f"{label} must be an ISO-8601 date or datetime") from exc
            if date_is_upper_bound:
                local_date += timedelta(days=1)
            parsed = datetime.combine(local_date, time.min, tzinfo=caller_timezone)
        else:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{label} must be an ISO-8601 date or datetime") from exc
    else:
        raise TypeError(f"{label} must be a date, datetime, ISO-8601 string, or None")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=caller_timezone)
    # SQLite 按字典序比较 Catalog 中的 TEXT 时间。这里复用写入端的规范 UTC 表示，
    # 确保精确边界与存储值逐字节相等；若混用 ``Z`` 和 ``+00:00``，会导致包含型
    # 下界跳过精确时刻，或排除型上界错误包含该时刻。
    return normalize_catalog_timestamp(parsed.isoformat(), label)


def _validate_time_range(lower: str | None, upper: str | None, label: str) -> None:
    if lower is None or upper is None:
        return
    lower_value = datetime.fromisoformat(lower.replace("Z", "+00:00"))
    upper_value = datetime.fromisoformat(upper.replace("Z", "+00:00"))
    if lower_value >= upper_value:
        raise ValueError(f"{label}_from must be earlier than {label}_to")


def _bounded_integer(value: Any, *, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 1 or value > maximum:
        raise ValueError(f"{label} must be between 1 and {maximum}")
    return value


def _normalize_json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("metadata_filters must be a mapping")
    if any(not isinstance(raw_key, str) for raw_key in value):
        raise TypeError("metadata_filters keys must be strings")
    normalized: dict[str, Any] = {}
    for raw_key in sorted(value):
        key = raw_key.strip()
        if not key or "\x00" in key:
            raise ValueError("metadata_filters keys must be non-empty and must not contain NUL")
        if len(key) > _MAX_IDENTIFIER_CHARS:
            raise ValueError("metadata_filters key is too long")
        normalized[key] = _normalize_json_value(value[raw_key], depth=0)
    if len(normalized) > MAX_FILTER_VALUES:
        raise ValueError(f"metadata_filters exceeds the maximum of {MAX_FILTER_VALUES} keys")
    return normalized


def _normalize_json_value(value: Any, *, depth: int) -> Any:
    if depth > _MAX_METADATA_DEPTH:
        raise ValueError(f"metadata_filters exceeds the maximum depth of {_MAX_METADATA_DEPTH}")
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata_filters numbers must be finite")
        return value
    if isinstance(value, Mapping):
        return _normalize_json_mapping_at_depth(value, depth=depth + 1)
    if isinstance(value, list | tuple):
        if len(value) > MAX_FILTER_VALUES:
            raise ValueError(f"metadata_filters list exceeds the maximum of {MAX_FILTER_VALUES} values")
        return [_normalize_json_value(item, depth=depth + 1) for item in value]
    raise TypeError("metadata_filters values must be JSON serializable")


def _normalize_json_mapping_at_depth(value: Mapping[Any, Any], *, depth: int) -> dict[str, Any]:
    if any(not isinstance(key, str) for key in value):
        raise TypeError("metadata_filters nested keys must be strings")
    normalized: dict[str, Any] = {}
    for key in sorted(value):
        normalized[key] = _normalize_json_value(value[key], depth=depth)
    if len(normalized) > MAX_FILTER_VALUES:
        raise ValueError(f"metadata_filters object exceeds the maximum of {MAX_FILTER_VALUES} keys")
    return normalized


def _copy_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_json(item) for item in value]
    return value
