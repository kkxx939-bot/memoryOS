"""Deterministic planning and legacy adaptation for unified retrieval."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone, tzinfo
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.query_plan import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_FINAL_LIMIT,
    RetrievalOptions,
    RetrievalQueryIntent,
    RetrievalQueryPlan,
)
from memoryos.security.workspace_identity import normalize_workspace_id, normalize_workspace_scope_key

_KNOWN_LEGACY_KEYS = frozenset(
    {
        "adapter_id",
        "applicability_scope_keys",
        "applicability_scopes",
        "candidate_limit",
        "connect_metadata",
        "context_type",
        "context_types",
        "document_ids",
        "document_kinds",
        "event_time_from",
        "event_time_to",
        "expand_relations",
        "final_limit",
        "limit",
        "metadata",
        "metadata_filters",
        "owner_user_id",
        "project_id",
        "query_intent",
        "record_kinds",
        "relation_expansion",
        "retrieval_views",
        "search_scope",
        "session_id",
        "session_ids",
        "source_kind",
        "source_kinds",
        "target_paths",
        "target_uris",
        "tenant_id",
        "timezone",
        "token_budget",
        "transaction_time_from",
        "transaction_time_to",
        "updated_at_from",
        "updated_at_to",
        "user_id",
        "workspace_id",
        "workspace_ids",
    }
)

_PRINCIPAL_ONLY_WORKSPACE = "__memoryos_principal_only__"
_CHINESE_CALENDAR_DATE = re.compile(
    r"(?<!\d)(?:(?P<year>\d{4})\s*年\s*)?(?:(?P<month>\d{1,2})\s*月\s*)?"
    r"(?P<day>\d{1,2})\s*(?:号|日)(?!\d)"
)
_TRANSACTION_TIME_CUES = (
    "系统新增",
    "新增了哪些记忆",
    "写入了哪些记忆",
    "录入了哪些记忆",
    "存入了哪些记忆",
    "创建了哪些记忆",
    "系统写入",
    "系统记录",
    "transaction time",
    "transaction_time",
)
_PAST_CHAT_CUES = (
    "之前讨论",
    "之前聊",
    "讨论过",
    "聊过",
    "回顾",
    "还记得吗",
    "过去对话",
    "past chat",
    "discussed",
)


class RetrievalScopeViolation(ValueError):
    """Raised when an explicit request conflicts with trusted caller scope."""


@dataclass(frozen=True)
class TrustedRetrievalScope:
    """Non-user-controlled scope constraints applied before query planning.

    ``None`` on a collection means that the trusted caller supplied no
    constraint for that dimension. An empty collection is a deny-all scope.
    """

    tenant_id: str | None = None
    owner_user_id: str | None = None
    workspace_ids: tuple[str, ...] | None = None
    session_ids: tuple[str, ...] | None = None
    adapter_id: str | None = None
    service_id: str | None = None
    authorized_scope_keys: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for field_name in ("tenant_id", "owner_user_id", "adapter_id", "service_id"):
            object.__setattr__(self, field_name, _optional_string(getattr(self, field_name), field_name))
        if self.workspace_ids is not None:
            object.__setattr__(
                self,
                "workspace_ids",
                tuple(normalize_workspace_id(item) for item in _string_tuple(self.workspace_ids, "workspace_ids")),
            )
        if self.session_ids is not None:
            object.__setattr__(self, "session_ids", _string_tuple(self.session_ids, "session_ids"))
        if self.authorized_scope_keys is not None:
            object.__setattr__(
                self,
                "authorized_scope_keys",
                tuple(
                    normalize_workspace_scope_key(item)
                    for item in _string_tuple(self.authorized_scope_keys, "authorized_scope_keys")
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "owner_user_id": self.owner_user_id,
            "workspace_ids": None if self.workspace_ids is None else list(self.workspace_ids),
            "session_ids": None if self.session_ids is None else list(self.session_ids),
            "adapter_id": self.adapter_id,
            "service_id": self.service_id,
            "authorized_scope_keys": (None if self.authorized_scope_keys is None else list(self.authorized_scope_keys)),
        }


class QueryPlanner:
    """Build normalized plans without delegating safety decisions to an LLM."""

    def __init__(self, *, now_provider: Callable[[], datetime] | None = None) -> None:
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))

    def build(
        self,
        semantic_query: str,
        *,
        options: RetrievalOptions | None = None,
        trusted_scope: TrustedRetrievalScope | None = None,
    ) -> RetrievalQueryPlan:
        if not isinstance(semantic_query, str):
            raise TypeError("semantic_query must be a string")
        bound = bind_trusted_scope(options or RetrievalOptions(), trusted_scope)
        inferred = _apply_deterministic_query_filters(
            semantic_query,
            bound,
            now=self._now_provider(),
        )
        return RetrievalQueryPlan(
            semantic_query=semantic_query,
            service_id=(trusted_scope.service_id if trusted_scope is not None else None),
            **inferred.to_dict(),
        )

    def plan(
        self,
        semantic_query: str,
        *,
        options: RetrievalOptions | None = None,
        trusted_scope: TrustedRetrievalScope | None = None,
    ) -> RetrievalQueryPlan:
        """Alias retained for callers that name the planning operation directly."""

        return self.build(semantic_query, options=options, trusted_scope=trusted_scope)

    def build_from_legacy(
        self,
        semantic_query: str,
        flat_kwargs: Mapping[str, Any],
        *,
        trusted_scope: TrustedRetrievalScope | None = None,
    ) -> RetrievalQueryPlan:
        return self.build(
            semantic_query,
            options=retrieval_options_from_legacy(flat_kwargs),
            trusted_scope=trusted_scope,
        )


def _apply_deterministic_query_filters(
    semantic_query: str,
    options: RetrievalOptions,
    *,
    now: datetime,
) -> RetrievalOptions:
    """Infer one explicit local calendar day without widening caller filters.

    The parser intentionally recognizes only a small, deterministic grammar.
    Multiple or invalid dates are left untouched. Trusted scope is already
    bound before this function runs, and every inferred field is filled only
    when the caller did not provide that field. In particular an explicit
    Tree path is never combined with an inferred path because path filters are
    unioned by the Catalog and such a merge would widen the caller's request.
    """

    matches = tuple(_CHINESE_CALENDAR_DATE.finditer(semantic_query))
    if len(matches) != 1:
        return options
    match = matches[0]
    try:
        local_today = _today_in_timezone(now, options.timezone)
        raw_month = match.group("month")
        raw_year = match.group("year")
        day = int(match.group("day"))
        if raw_month is None:
            normalized_query = semantic_query.casefold()
            if not any(cue in normalized_query for cue in _PAST_CHAT_CUES):
                return options
            try:
                candidate = date(local_today.year, local_today.month, day)
            except ValueError:
                candidate = None
            if candidate is None or candidate > local_today:
                previous_month_end = local_today.replace(day=1) - timedelta(days=1)
                candidate = date(previous_month_end.year, previous_month_end.month, day)
            inferred_date = candidate
        else:
            inferred_date = date(
                int(raw_year or local_today.year),
                int(raw_month),
                day,
            )
    except (TypeError, ValueError):
        return options

    local_day = inferred_date.isoformat()
    normalized_query = semantic_query.casefold()
    if any(cue in normalized_query for cue in _TRANSACTION_TIME_CUES):
        # Transaction-time questions ask what the system wrote on that day,
        # including immutable historical rows that may no longer be the
        # latest source-time value. Preserve any non-default caller intent; otherwise
        # convert the public default to the unified historical view.
        inferred_intent = (
            RetrievalQueryIntent.HISTORY
            if options.query_intent == RetrievalQueryIntent.CURRENT
            else options.query_intent
        )
        if options.transaction_time_from or options.transaction_time_to:
            return replace(options, query_intent=inferred_intent)
        return replace(
            options,
            transaction_time_from=local_day,
            transaction_time_to=local_day,
            query_intent=inferred_intent,
        )

    if options.event_time_from or options.event_time_to:
        return options
    inferred_intent = (
        RetrievalQueryIntent.OPEN_RECALL
        if options.query_intent == RetrievalQueryIntent.CURRENT
        else options.query_intent
    )
    return replace(
        options,
        event_time_from=local_day,
        event_time_to=local_day,
        query_intent=inferred_intent,
    )


def _today_in_timezone(now: datetime, timezone_name: str) -> date:
    if not isinstance(now, datetime):
        raise TypeError("QueryPlanner now_provider must return a datetime")
    if now.tzinfo is None or now.utcoffset() is None:
        now = now.replace(tzinfo=timezone.utc)
    resolved: tzinfo
    if timezone_name == "UTC":
        resolved = timezone.utc
    elif re.fullmatch(r"[+-]\d{2}:\d{2}", timezone_name):
        sign = 1 if timezone_name[0] == "+" else -1
        hours, minutes = (int(value) for value in timezone_name[1:].split(":"))
        resolved = timezone(sign * timedelta(hours=hours, minutes=minutes))
    else:
        # RetrievalOptions has already validated this IANA timezone.
        resolved = ZoneInfo(timezone_name)
    return now.astimezone(resolved).date()


def bind_trusted_scope(
    options: RetrievalOptions,
    trusted_scope: TrustedRetrievalScope | None,
) -> RetrievalOptions:
    """Apply trusted constraints without permitting an explicit scope expansion.

    Scalar conflicts and collection values outside the trusted allow-list fail
    closed. Missing explicit filters inherit the trusted constraint. This keeps
    an empty result from being accidentally represented as an unbounded query.
    """

    if not isinstance(options, RetrievalOptions):
        raise TypeError("options must be RetrievalOptions")
    if trusted_scope is None:
        return options
    if not isinstance(trusted_scope, TrustedRetrievalScope):
        raise TypeError("trusted_scope must be TrustedRetrievalScope or None")

    tenant_id = _bind_scalar(options.tenant_id, trusted_scope.tenant_id, "tenant_id")
    owner_user_id = _bind_scalar(options.owner_user_id, trusted_scope.owner_user_id, "owner_user_id")
    adapter_id = _bind_scalar(options.adapter_id, trusted_scope.adapter_id, "adapter_id")
    workspace_ids = _bind_collection(options.workspace_ids, trusted_scope.workspace_ids, "workspace_ids")
    session_ids = _bind_collection(options.session_ids, trusted_scope.session_ids, "session_ids")
    target_paths = _bind_agent_paths(options.target_paths, adapter_id=adapter_id)
    metadata_filters = _with_bound_scope_keys(
        options.metadata_filters,
        owner_user_id=owner_user_id,
        workspace_ids=workspace_ids,
        authorized_scope_keys=trusted_scope.authorized_scope_keys,
    )
    return replace(
        options,
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
        workspace_ids=workspace_ids,
        session_ids=session_ids,
        adapter_id=adapter_id,
        target_paths=target_paths,
        metadata_filters=metadata_filters,
    )


def _bind_agent_paths(paths: tuple[str, ...], *, adapter_id: str | None) -> tuple[str, ...]:
    """Prevent a trusted Agent caller from enumerating another Agent tree."""

    if not any(path == "agents" or path.startswith("agents/") for path in paths):
        return paths
    if not adapter_id:
        raise RetrievalScopeViolation("agents target_paths require a trusted adapter_id")
    allowed_root = f"agents/{adapter_id}"
    bound: list[str] = []
    for path in paths:
        if path == "agents":
            path = allowed_root
        elif path.startswith("agents/"):
            if path != allowed_root and not path.startswith(f"{allowed_root}/"):
                raise RetrievalScopeViolation("target_paths cannot expand the trusted adapter scope")
        if path not in bound:
            bound.append(path)
    return tuple(bound)


def merge_retrieval_options(primary: RetrievalOptions, fallback: RetrievalOptions) -> RetrievalOptions:
    """Merge a structured request with non-default legacy constraints.

    Structured ranking/budget choices win. Scope, target, type, and time
    constraints are inherited only when absent and conflicting explicit values
    fail closed instead of silently widening the query.
    """

    if not isinstance(primary, RetrievalOptions) or not isinstance(fallback, RetrievalOptions):
        raise TypeError("primary and fallback must be RetrievalOptions")
    return replace(
        primary,
        target_uris=_merge_compatible_collection(primary.target_uris, fallback.target_uris, "target_uris"),
        target_paths=_merge_compatible_collection(primary.target_paths, fallback.target_paths, "target_paths"),
        context_types=_merge_compatible_collection(primary.context_types, fallback.context_types, "context_types"),
        source_kinds=_merge_compatible_collection(primary.source_kinds, fallback.source_kinds, "source_kinds"),
        record_kinds=_merge_compatible_collection(primary.record_kinds, fallback.record_kinds, "record_kinds"),
        document_ids=_merge_compatible_collection(primary.document_ids, fallback.document_ids, "document_ids"),
        document_kinds=_merge_compatible_collection(
            primary.document_kinds,
            fallback.document_kinds,
            "document_kinds",
        ),
        tenant_id=_merge_explicit_scalar(primary.tenant_id, fallback.tenant_id, "tenant_id"),
        owner_user_id=_merge_explicit_scalar(
            primary.owner_user_id,
            fallback.owner_user_id,
            "owner_user_id",
        ),
        workspace_ids=_merge_compatible_collection(
            primary.workspace_ids,
            fallback.workspace_ids,
            "workspace_ids",
        ),
        session_ids=_merge_compatible_collection(primary.session_ids, fallback.session_ids, "session_ids"),
        adapter_id=_merge_explicit_scalar(primary.adapter_id, fallback.adapter_id, "adapter_id"),
        event_time_from=_merge_explicit_scalar(
            primary.event_time_from,
            fallback.event_time_from,
            "event_time_from",
        ),
        event_time_to=_merge_explicit_scalar(primary.event_time_to, fallback.event_time_to, "event_time_to"),
        transaction_time_from=_merge_explicit_scalar(
            primary.transaction_time_from,
            fallback.transaction_time_from,
            "transaction_time_from",
        ),
        transaction_time_to=_merge_explicit_scalar(
            primary.transaction_time_to,
            fallback.transaction_time_to,
            "transaction_time_to",
        ),
        updated_at_from=_merge_explicit_scalar(
            primary.updated_at_from,
            fallback.updated_at_from,
            "updated_at_from",
        ),
        updated_at_to=_merge_explicit_scalar(
            primary.updated_at_to,
            fallback.updated_at_to,
            "updated_at_to",
        ),
        metadata_filters=_merge_metadata(fallback.metadata_filters, primary.metadata_filters),
        legacy_search_scope=_merge_explicit_scalar(
            primary.legacy_search_scope,
            fallback.legacy_search_scope,
            "legacy_search_scope",
        ),
        legacy_retrieval_views=_merge_compatible_collection(
            primary.legacy_retrieval_views,
            fallback.legacy_retrieval_views,
            "legacy_retrieval_views",
        ),
    )


def retrieval_options_from_legacy(flat_kwargs: Mapping[str, Any]) -> RetrievalOptions:
    """Convert the former flat retrieval kwargs to one structured object.

    Unknown keys and conflicting aliases are rejected so transports cannot
    silently diverge. ``None`` values are treated as omitted legacy defaults.
    """

    if not isinstance(flat_kwargs, Mapping):
        raise TypeError("legacy retrieval kwargs must be a mapping")
    unknown = sorted(set(flat_kwargs) - _KNOWN_LEGACY_KEYS)
    if unknown:
        raise ValueError(f"unknown legacy retrieval options: {', '.join(unknown)}")
    raw = {key: value for key, value in flat_kwargs.items() if value is not None}
    owner_user_id = _coalesce_scalar_alias(raw, "owner_user_id", "user_id")
    workspace_ids = _coalesce_collection_aliases(raw, "workspace_ids", ("workspace_id", "project_id"))
    session_ids = _coalesce_collection_aliases(raw, "session_ids", ("session_id",))
    context_types = _coalesce_type_filters(raw)
    source_kinds = _coalesce_sequence_alias(raw, "source_kinds", "source_kind")
    record_kinds = _pop_sequence(raw, "record_kinds")
    document_ids = _pop_sequence(raw, "document_ids")
    document_kinds = _pop_sequence(raw, "document_kinds")

    target_uris = _pop_sequence(raw, "target_uris")
    target_paths = _pop_sequence(raw, "target_paths")

    final_limit = _coalesce_limit(raw)
    candidate_default = (
        max(DEFAULT_CANDIDATE_LIMIT, final_limit) if isinstance(final_limit, int) else DEFAULT_CANDIDATE_LIMIT
    )
    candidate_limit = raw.pop("candidate_limit", candidate_default)
    relation_expansion = _coalesce_bool_alias(raw, "relation_expansion", "expand_relations", default=False)

    metadata_filters = _merged_metadata_filters(raw)
    search_scope_value = raw.pop("search_scope", None)
    search_scope = _optional_string(search_scope_value, "search_scope")
    explicit_views = _pop_sequence(raw, "retrieval_views")
    retrieval_views = explicit_views
    adapter_id = _optional_string(raw.get("adapter_id"), "adapter_id")
    query_intent_raw = raw.pop("query_intent", RetrievalQueryIntent.CURRENT)
    try:
        query_intent = (
            query_intent_raw
            if isinstance(query_intent_raw, RetrievalQueryIntent)
            else RetrievalQueryIntent(str(query_intent_raw).strip().upper())
        )
    except ValueError as exc:
        raise ValueError(f"unknown legacy query_intent: {query_intent_raw!r}") from exc
    options = RetrievalOptions(
        target_uris=target_uris,
        target_paths=target_paths,
        context_types=context_types,
        source_kinds=source_kinds,
        record_kinds=record_kinds,
        document_ids=document_ids,
        document_kinds=document_kinds,
        tenant_id=raw.pop("tenant_id", None),
        owner_user_id=owner_user_id,
        workspace_ids=workspace_ids,
        session_ids=session_ids,
        adapter_id=raw.pop("adapter_id", adapter_id),
        event_time_from=raw.pop("event_time_from", None),
        event_time_to=raw.pop("event_time_to", None),
        transaction_time_from=raw.pop("transaction_time_from", None),
        transaction_time_to=raw.pop("transaction_time_to", None),
        updated_at_from=raw.pop("updated_at_from", None),
        updated_at_to=raw.pop("updated_at_to", None),
        timezone=raw.pop("timezone", "UTC"),
        query_intent=query_intent,
        relation_expansion=relation_expansion,
        candidate_limit=candidate_limit,
        final_limit=final_limit,
        token_budget=raw.pop("token_budget", 4_096),
        metadata_filters=metadata_filters,
        legacy_search_scope=search_scope,
        legacy_retrieval_views=retrieval_views,
    )
    if raw:
        # Every accepted key must have a deterministic destination above.
        raise AssertionError(f"unconsumed legacy retrieval options: {', '.join(sorted(raw))}")
    return options


def _coalesce_scalar_alias(raw: dict[str, Any], primary: str, alias: str) -> str | None:
    primary_value = _optional_string(raw.pop(primary, None), primary)
    alias_value = _optional_string(raw.pop(alias, None), alias)
    return _merge_scalar_constraint(primary_value, alias_value, primary)


def _coalesce_collection_aliases(
    raw: dict[str, Any],
    primary: str,
    aliases: tuple[str, ...],
) -> tuple[str, ...]:
    values = _pop_sequence(raw, primary)
    for alias in aliases:
        alias_value = _optional_string(raw.pop(alias, None), alias)
        if alias_value is None:
            continue
        if values and alias_value not in values:
            raise ValueError(f"conflicting legacy {primary} and {alias}")
        values = (alias_value,)
    return values


def _coalesce_type_filters(raw: dict[str, Any]) -> tuple[ContextType, ...]:
    plural = _pop_sequence(raw, "context_types")
    singular = raw.pop("context_type", None)
    values = plural
    if singular is not None:
        if plural and _context_type_value(singular) not in tuple(_context_type_value(item) for item in plural):
            raise ValueError("conflicting legacy context_type and context_types")
        values = plural or (singular,)
    try:
        return tuple(ContextType(_context_type_value(item)) for item in values)
    except ValueError as exc:
        raise ValueError("legacy context_types contains an unknown value") from exc


def _coalesce_sequence_alias(raw: dict[str, Any], primary: str, alias: str) -> tuple[Any, ...]:
    plural = _pop_sequence(raw, primary)
    singular = raw.pop(alias, None)
    if singular is None:
        return plural
    if plural and singular not in plural:
        raise ValueError(f"conflicting legacy {primary} and {alias}")
    return plural or (singular,)


def _coalesce_limit(raw: dict[str, Any]) -> int:
    final_limit = raw.pop("final_limit", None)
    legacy_limit = raw.pop("limit", None)
    if final_limit is not None and legacy_limit is not None and final_limit != legacy_limit:
        raise ValueError("conflicting legacy final_limit and limit")
    value = final_limit if final_limit is not None else legacy_limit
    return DEFAULT_FINAL_LIMIT if value is None else value


def _coalesce_bool_alias(raw: dict[str, Any], primary: str, alias: str, *, default: bool) -> bool:
    primary_value = raw.pop(primary, None)
    alias_value = raw.pop(alias, None)
    if primary_value is not None and alias_value is not None and primary_value != alias_value:
        raise ValueError(f"conflicting legacy {primary} and {alias}")
    value = primary_value if primary_value is not None else alias_value
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"{primary} must be a bool")
    return value


def _merged_metadata_filters(raw: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("metadata_filters", "metadata", "connect_metadata"):
        value = raw.pop(key, None)
        if value is not None:
            if not isinstance(value, Mapping):
                raise TypeError(f"{key} must be a mapping")
            merged = _merge_metadata(merged, value)
    for legacy_key in ("applicability_scope_keys", "applicability_scopes"):
        value = raw.pop(legacy_key, None)
        if value is not None:
            merged = _merge_metadata(merged, {legacy_key: value})
    return merged


def _merge_metadata(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if key in merged and merged[key] != value:
            raise ValueError(f"conflicting metadata filter for {key!r}")
        merged[key] = value
    return merged


def _with_bound_scope_keys(
    metadata_filters: Mapping[str, Any],
    *,
    owner_user_id: str | None,
    workspace_ids: tuple[str, ...],
    authorized_scope_keys: tuple[str, ...] | None,
) -> dict[str, Any]:
    merged = dict(metadata_filters)
    explicit_declared = "applicability_scope_keys" in merged and merged["applicability_scope_keys"] is not None
    raw_keys = merged.get("applicability_scope_keys", ())
    if raw_keys is None:
        raw_keys = ()
    if isinstance(raw_keys, str):
        explicit = (raw_keys,)
    elif isinstance(raw_keys, Sequence):
        explicit = tuple(raw_keys)
    else:
        raise TypeError("applicability_scope_keys must be a sequence")
    keys: list[str] = []
    for item in explicit:
        if not isinstance(item, str) or not item.strip() or "\x00" in item:
            raise TypeError("applicability_scope_keys must contain non-empty strings")
        keys.append(item.strip())
    required: list[str] = []
    if owner_user_id:
        required.append(f"memoryos:principal:{owner_user_id}")
    required.extend(
        f"memoryos:workspace:{workspace_id}"
        for workspace_id in workspace_ids
        if workspace_id and workspace_id != _PRINCIPAL_ONLY_WORKSPACE
    )
    if authorized_scope_keys is None:
        # Embedded/local callers predate a trusted authorization envelope.
        # Preserve their established principal/workspace + explicit scope
        # behavior instead of treating the absence of grants as deny-all.
        keys.extend(required)
    else:
        allowed = set(authorized_scope_keys)
        missing_required = set(required) - allowed
        if missing_required:
            raise RetrievalScopeViolation("trusted authorized scope keys omit the bound principal or workspace")
        if explicit_declared:
            unauthorized = set(keys) - allowed
            if unauthorized:
                raise RetrievalScopeViolation("applicability_scope_keys exceed trusted caller scope")
        else:
            # No caller filter means all authenticated grants are available.
            # An explicit filter is retained exactly and therefore can only
            # narrow this set.
            keys = list(authorized_scope_keys)
    if keys:
        merged["applicability_scope_keys"] = list(dict.fromkeys(keys))
    elif explicit_declared or authorized_scope_keys is not None:
        merged["applicability_scope_keys"] = []
        merged["require_unscoped"] = True
    else:
        merged.pop("applicability_scope_keys", None)
    return merged


def _merge_explicit_scalar(primary: Any, fallback: Any, label: str) -> Any:
    if primary not in (None, "") and fallback not in (None, "") and primary != fallback:
        raise ValueError(f"structured options conflict with legacy {label}")
    return primary if primary not in (None, "") else fallback


def _merge_compatible_collection(primary: tuple[Any, ...], fallback: tuple[Any, ...], label: str) -> tuple[Any, ...]:
    if primary and fallback and primary != fallback:
        raise ValueError(f"structured options conflict with legacy {label}")
    return primary or fallback


def _bind_scalar(requested: str | None, trusted: str | None, label: str) -> str | None:
    if trusted is None:
        return requested
    if requested is not None and requested != trusted:
        raise RetrievalScopeViolation(f"requested {label} is outside trusted caller scope")
    return trusted


def _bind_collection(
    requested: tuple[str, ...],
    trusted: tuple[str, ...] | None,
    label: str,
) -> tuple[str, ...]:
    if trusted is None:
        return requested
    if not trusted:
        raise RetrievalScopeViolation(f"trusted caller has no authorized {label}")
    if not requested:
        return trusted
    unauthorized = [value for value in requested if value not in trusted]
    if unauthorized:
        raise RetrievalScopeViolation(f"requested {label} is outside trusted caller scope")
    return requested


def _intersect_or_inherit(
    explicit: tuple[str, ...],
    derived: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    if not derived:
        return explicit
    if not explicit:
        return derived
    overlap = tuple(value for value in explicit if value in derived)
    if not overlap:
        raise ValueError(f"legacy {label} conflicts with retrieval_views")
    return overlap


def _merge_scalar_constraint(left: str | None, right: str | None, label: str) -> str | None:
    if left is not None and right is not None and left != right:
        raise ValueError(f"conflicting legacy {label} constraints")
    return left if left is not None else right


def _one_or_none(values: list[str], label: str) -> str | None:
    unique = _dedupe(values)
    if len(unique) > 1:
        raise ValueError(f"legacy retrieval_views contain multiple {label} values")
    return unique[0] if unique else None


def _pop_sequence(raw: dict[str, Any], key: str) -> tuple[Any, ...]:
    value = raw.pop(key, None)
    return _sequence_value(value, key)


def _sequence_value(value: Any, key: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Enum)):
        return (value,)
    if not isinstance(value, Sequence):
        raise TypeError(f"{key} must be a sequence")
    return _dedupe(list(value))


def _merge_sequences(*values: tuple[Any, ...]) -> tuple[Any, ...]:
    merged: list[Any] = []
    for items in values:
        for item in items:
            if item not in merged:
                merged.append(item)
    return tuple(merged)


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    normalized = value.strip()
    return normalized or None


def _string_tuple(values: Sequence[str] | str, label: str) -> tuple[str, ...]:
    raw_values: Sequence[str] = (values,) if isinstance(values, str) else values
    normalized: list[str] = []
    for value in raw_values:
        item = _optional_string(value, label)
        if item is None:
            raise ValueError(f"{label} values must not be empty")
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _dedupe(values: Sequence[Any]) -> tuple[Any, ...]:
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(unique)


def _context_type_value(value: Any) -> str:
    return value.value if isinstance(value, ContextType) else str(value).strip().lower()
