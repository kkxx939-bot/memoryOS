"""统一检索的确定性查询规划与参数归一化。"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone, tzinfo
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from infrastructure.context.retrieval.query_plan import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_FINAL_LIMIT,
    RetrievalOptions,
    RetrievalQueryIntent,
    RetrievalQueryPlan,
)
from infrastructure.store.model.context.context_type import ContextType

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
        "transaction_time_from",
        "transaction_time_to",
        "updated_at_from",
        "updated_at_to",
        "user_id",
        "workspace_id",
        "workspace_ids",
    }
)

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


class QueryPlanner:
    """只根据查询文本和已确认的检索条件生成确定性计划。

    固定存储命名空间、本地用户和工作区由上层写入 ``RetrievalOptions``；Planner
    只生成查询计划，不承担身份绑定。
    """

    def __init__(self, *, now_provider: Callable[[], datetime] | None = None) -> None:
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))

    def build(
        self,
        semantic_query: str,
        *,
        options: RetrievalOptions | None = None,
    ) -> RetrievalQueryPlan:
        if not isinstance(semantic_query, str):
            raise TypeError("semantic_query must be a string")
        inferred = _apply_deterministic_query_filters(
            semantic_query,
            options or RetrievalOptions(),
            now=self._now_provider(),
        )
        return RetrievalQueryPlan(
            semantic_query=semantic_query,
            **inferred.to_dict(),
        )

    def plan(
        self,
        semantic_query: str,
        *,
        options: RetrievalOptions | None = None,
    ) -> RetrievalQueryPlan:
        """供显式使用 ``plan`` 命名的调用者复用同一规划入口。"""

        return self.build(semantic_query, options=options)

    def build_from_legacy(
        self,
        semantic_query: str,
        flat_kwargs: Mapping[str, Any],
    ) -> RetrievalQueryPlan:
        return self.build(
            semantic_query,
            options=retrieval_options_from_legacy(flat_kwargs),
        )


def _apply_deterministic_query_filters(
    semantic_query: str,
    options: RetrievalOptions,
    *,
    now: datetime,
) -> RetrievalOptions:
    """识别一个明确的本地日历日期，并且不扩大上层传入的过滤条件。

    这里只支持一小组确定性语法；日期不合法或同时出现多个日期时不做推断。
    所有推断字段只填充调用者没有提供的值，尤其不会把显式 Tree 路径和推断
    路径合并，因为 Catalog 对路径取并集会扩大查询范围。
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
        # 事务时间问题询问系统当天写入了什么，包括已经不是最新源时间值的不可变
        # 历史记录。保留调用者显式的非默认意图，否则把公开默认值转换为统一历史视图。
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
        # RetrievalOptions 已经校验过该 IANA 时区。
        resolved = ZoneInfo(timezone_name)
    return now.astimezone(resolved).date()


def merge_retrieval_options(primary: RetrievalOptions, fallback: RetrievalOptions) -> RetrievalOptions:
    """合并结构化请求与旧式平铺参数，并对冲突条件执行失败关闭。

    排序和数量选择以结构化请求为准；范围、目标、类型和时间条件只在缺失时
    继承，两个入口同时提供不同值时直接报错，不能静默扩大查询。
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
    """把旧的平铺召回参数转换为一个结构化对象。

    未知字段和冲突别名会被拒绝，避免不同传输入口静默产生偏差；``None`` 按省略
    的旧默认值处理。
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
        metadata_filters=metadata_filters,
        legacy_search_scope=search_scope,
        legacy_retrieval_views=retrieval_views,
    )
    if raw:
        # 每个允许的字段都必须在上面有确定的归宿。
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


def _merge_explicit_scalar(primary: Any, fallback: Any, label: str) -> Any:
    if primary not in (None, "") and fallback not in (None, "") and primary != fallback:
        raise ValueError(f"structured options conflict with legacy {label}")
    return primary if primary not in (None, "") else fallback


def _merge_compatible_collection(primary: tuple[Any, ...], fallback: tuple[Any, ...], label: str) -> tuple[Any, ...]:
    if primary and fallback and primary != fallback:
        raise ValueError(f"structured options conflict with legacy {label}")
    return primary or fallback


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
    if isinstance(value, str | Enum):
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
