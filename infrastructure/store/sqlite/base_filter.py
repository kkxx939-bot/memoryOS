"""结构化且具备租户隔离的 SQLite Catalog 查询条件。"""

from __future__ import annotations

from infrastructure.store.sqlite._common import (
    _PLURAL_FILTER_ALIASES,
    _SIMPLE_FILTER_FIELDS,
    Any,
    Mapping,
    normalize_tree_path,
)


class BaseFilterBuilder:
    """针对 ``contexts`` 的别名 ``c`` 编译可信结构化过滤条件。"""

    def __init__(self, store: Any) -> None:
        self._store = store

    def _base_filter_sql(
        self,
        filters: Mapping[str, Any],
        *,
        path_candidate_limit: int = 100,
        path_fts_query: str = "",
        path_exact_value: str = "",
    ) -> tuple[str, list[Any]]:
        del path_candidate_limit, path_fts_query, path_exact_value
        normalized = dict(filters)
        tenant_values = self._store._filter_values(normalized.get("tenant_id"), allow_empty=True)
        if len(tenant_values) != 1 or not tenant_values[0]:
            raise ValueError("structured Catalog queries require exactly one tenant_id")

        clauses: list[str] = ["c.tenant_id = ?"]
        params: list[Any] = [tenant_values[0]]

        def add_values(column: str, value: Any) -> None:
            values = self._store._filter_values(value)
            placeholders = ", ".join("?" for _ in values)
            clauses.append(f"c.{column} IN ({placeholders})")
            params.extend(values)

        for filter_name, column in _SIMPLE_FILTER_FIELDS.items():
            if filter_name == "tenant_id" or filter_name not in normalized:
                continue
            add_values(column, normalized[filter_name])
        for filter_name, column in _PLURAL_FILTER_ALIASES.items():
            if filter_name in normalized:
                if filter_name == "document_kinds" and normalized.get("document_kinds_apply_to_documents_only"):
                    continue
                add_values(column, normalized[filter_name])
        if normalized.get("document_kinds_apply_to_documents_only"):
            document_kinds = self._store._filter_values(normalized.get("document_kinds"))
            placeholders = ", ".join("?" for _ in document_kinds)
            clauses.append(
                f"(c.record_kind NOT IN ('memory_document', 'memory_block') OR c.document_kind IN ({placeholders}))"
            )
            params.extend(document_kinds)
        for filter_name, column in (
            ("uri", "uri"),
            ("source_uri", "source_uri"),
            ("project_id", "project_id"),
            ("projection_effect_hash", "projection_effect_hash"),
        ):
            if filter_name in normalized:
                add_values(column, normalized[filter_name])

        if not normalized.get("include_inactive") and "lifecycle_state" not in normalized:
            clauses.append("c.lifecycle_state = 'active'")
        if "serving_tier" not in normalized and not normalized.get("include_inactive"):
            clauses.append("c.serving_tier IN ('HOT', 'WARM')")
        if "projection_status" not in normalized:
            clauses.append("c.projection_status IN ('PROJECTED', 'DEGRADED')")
        for filter_name, column, operator in (
            ("event_time_from", "event_time", ">="),
            ("event_time_to", "event_time", "<"),
            ("transaction_time_from", "transaction_time", ">="),
            ("transaction_time_to", "transaction_time", "<"),
            ("updated_at_from", "updated_at", ">="),
            ("updated_at_to", "updated_at", "<"),
        ):
            value = normalized.get(filter_name)
            if value is None:
                continue
            timestamp = self._store._coerce_timestamp(str(value))
            if not timestamp:
                raise ValueError(f"{filter_name} must be an ISO-8601 timestamp")
            clauses.append(f"c.{column} {operator} ?")
            params.append(timestamp)
        if normalized.get("valid_at") is not None:
            raise ValueError("valid_at is not supported by the greenfield Catalog")

        raw_paths = normalized.get("target_paths", normalized.get("path_prefixes"))
        if raw_paths is not None:
            paths = tuple(normalize_tree_path(value) for value in self._store._filter_values(raw_paths))
            placeholders = ", ".join("?" for _ in paths)
            clauses.append(
                "EXISTS (SELECT 1 FROM context_path_closure AS pc "
                "WHERE pc.tenant_id = c.tenant_id AND pc.record_key = c.record_key "
                f"AND pc.ancestor_path IN ({placeholders}))"
            )
            params.extend(paths)

        raw_scope_keys = normalized.get("applicability_scope_keys")
        if raw_scope_keys:
            signatures = self._store._scope_signature_options(self._store._filter_values(raw_scope_keys))
            placeholders = ", ".join("?" for _ in signatures)
            clauses.append(f"c.scope_signature IN ({placeholders})")
            params.extend(signatures)
        elif normalized.get("require_unscoped"):
            clauses.append("c.scope_signature = ?")
            params.append(self._store._scope_signature(()))

        principal = normalized.get("principal_owner_id")
        if principal is not None:
            grants = [
                "(ag.grant_kind = 'principal' AND ag.grant_id = ?)",
                "ag.grant_kind = 'public'",
            ]
            grant_params: list[Any] = [str(principal)]
            workspace_values = normalized.get("workspace_access_ids")
            if workspace_values is not None:
                workspaces = self._store._filter_values(workspace_values, allow_empty=True)
                if workspaces:
                    placeholders = ", ".join("?" for _ in workspaces)
                    grants.append(f"(ag.grant_kind = 'workspace' AND ag.grant_id IN ({placeholders}))")
                    grant_params.extend(workspaces)
            clauses.append(
                "EXISTS (SELECT 1 FROM context_acl_grants AS ag "
                "WHERE ag.tenant_id = c.tenant_id AND ag.record_key = c.record_key AND (" + " OR ".join(grants) + "))"
            )
            params.extend(grant_params)
        elif normalized.get("require_public"):
            clauses.append(
                "EXISTS (SELECT 1 FROM context_acl_grants AS ag "
                "WHERE ag.tenant_id = c.tenant_id AND ag.record_key = c.record_key "
                "AND ag.grant_kind = 'public')"
            )

        return "".join(f" AND {clause}" for clause in clauses), params


__all__ = ["BaseFilterBuilder"]
