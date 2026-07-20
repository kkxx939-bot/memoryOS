"""Catalog 读取器共享的小型查询过滤组件。"""

from __future__ import annotations

from infrastructure.store.sqlite._common import Any, Mapping


class QueryFilterBuilder:
    """编译精确元数据和关系条件，禁止缺少租户约束的关联查询。"""

    def __init__(self, store: Any) -> None:
        self._store = store

    def _connect_filter_sql(
        self,
        alias: str,
        filters: Mapping[str, Any],
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for filter_name, column in (
            ("adapter_id", "adapter_id"),
            ("source_kind", "source_kind"),
            ("session_id", "session_id"),
        ):
            if filter_name not in filters:
                continue
            values = self._store._filter_values(filters[filter_name])
            placeholders = ", ".join("?" for _ in values)
            clauses.append(f"{alias}.{column} IN ({placeholders})")
            params.extend(values)
        return "".join(f" AND {clause}" for clause in clauses), params

    def _target_identity_sql(
        self,
        filters: Mapping[str, Any],
        target_identity_uris: Any,
    ) -> tuple[str, list[Any]]:
        identities = self._store._filter_values(target_identity_uris)
        normalized = dict(filters)
        normalized["target_uris"] = identities
        predicate, params = self._store._base_filter_sql(normalized)
        placeholders = ", ".join("?" for _ in identities)
        return f"{predicate} AND (c.uri IN ({placeholders}) OR c.source_uri IN ({placeholders}))", [
            *params,
            *identities,
            *identities,
        ]


__all__ = ["QueryFilterBuilder"]
