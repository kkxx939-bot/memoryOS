"""SQLite catalog QueryFilterBuilder responsibility component."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _INVALID_SCOPE_KEY,
    _MAX_QUERY_LIMIT,
    _MAX_TARGET_PATHS,
    _PLURAL_FILTER_ALIASES,
    _SIMPLE_FILTER_FIELDS,
    Any,
    Mapping,
    ServingTier,
    normalize_tree_path,
)


class QueryFilterBuilder:
    """Own one stable subset of SQLite catalog behavior."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def _legacy_filter_sql(self, filters: Mapping[str, Any]) -> tuple[str, list[Any]]:
        """Build the conservative flat-index predicate used only by rollback reads."""

        sql = ""
        params: list[Any] = []
        for filter_name, column in _SIMPLE_FILTER_FIELDS.items():
            if filters.get(filter_name) is None:
                continue
            values = self._store._filter_values(filters[filter_name])
            sql += f" AND c.{column} IN ({','.join('?' for _ in values)})"
            params.extend(values)
        for filter_name, column in _PLURAL_FILTER_ALIASES.items():
            if filters.get(filter_name) is None:
                continue
            values = self._store._filter_values(filters[filter_name], allow_empty=True)
            if not values:
                return " AND 1 = 0", []
            sql += f" AND c.{column} IN ({','.join('?' for _ in values)})"
            params.extend(values)
        target_identity_uris = filters.get("target_identity_uris")
        if target_identity_uris is not None:
            identity_sql, identity_params = self._store._target_identity_sql(
                filters,
                target_identity_uris,
                legacy=True,
            )
            sql += identity_sql
            params.extend(identity_params)
        connect_sql, connect_params = self._store._connect_filter_sql("c", filters)
        sql += connect_sql
        params.extend(connect_params)
        principal = filters.get("principal_owner_id")
        if principal is not None:
            # Fail closed for cross-owner canonical sharing on rollback.  The
            # Unified route can evaluate Visibility grants; the legacy route
            # intentionally exposes only the owner's rows plus unowned public
            # resources/skills.
            sql += " AND (c.owner_user_id = ? OR (c.owner_user_id = '' AND c.context_type IN ('resource', 'skill')))"
            params.append(str(principal))
        elif filters.get("owner_user_id") == "" and filters.get("require_unscoped"):
            sql += " AND c.owner_user_id = '' AND c.context_type IN ('resource', 'skill')"
        workspace_access = filters.get("workspace_access_ids")
        if workspace_access is not None:
            values = self._store._filter_values(workspace_access, allow_empty=True)
            if not values:
                return " AND 1 = 0", []
            sql += f" AND c.workspace_id IN ({','.join('?' for _ in values)})"
            params.extend(values)
        adapter_access = filters.get("adapter_access_id")
        if adapter_access is not None:
            sql += (
                " AND (c.adapter_id IN ('', ?) OR c.context_type IN ('session', 'resource', 'skill') "
                "OR c.record_kind = 'current_slot')"
            )
            params.append(str(adapter_access))
        project_id = filters.get("project_id")
        if project_id is not None:
            sql += " AND (c.project_id = ? OR c.project_id = '')"
            params.append(str(project_id))
        if not bool(filters.get("include_inactive", False)):
            sql += " AND c.lifecycle_state NOT IN ('deleted', 'archived', 'obsolete')"
            sql += " AND c.serving_tier != ?"
            params.append(ServingTier.ARCHIVED.value)
            if filters.get("include_candidates"):
                sql += " AND c.admission_status NOT IN ('restricted', 'archive_only', 'reject')"
            else:
                sql += " AND c.admission_status NOT IN ('pending', 'restricted', 'archive_only', 'reject')"
        for filter_name, column, operator in (
            ("event_time_from", "event_time", ">="),
            ("event_time_to", "event_time", "<"),
            ("transaction_time_from", "transaction_time", ">="),
            ("transaction_time_to", "transaction_time", "<"),
            ("updated_at_from", "updated_at", ">="),
            ("updated_at_to", "updated_at", "<"),
        ):
            if filters.get(filter_name) is not None:
                sql += f" AND c.{column} {operator} ?"
                params.append(str(filters[filter_name]))
        if filters.get("valid_at") is not None:
            valid_at = str(filters["valid_at"])
            sql += " AND c.valid_from <= ? AND (c.valid_to = '' OR c.valid_to > ?)"
            params.extend((valid_at, valid_at))
        available_scopes = filters.get("applicability_scope_keys")
        if available_scopes:
            scopes = self._store._filter_values(available_scopes)
            sql += (
                " AND NOT EXISTS (SELECT 1 FROM json_each("
                "CASE WHEN json_valid(c.scope_keys) AND json_type(c.scope_keys) = 'array' "
                "THEN c.scope_keys ELSE '[\"__invalid__\"]' END) "
                f"WHERE value NOT IN ({','.join('?' for _ in scopes)}))"
            )
            params.extend(scopes)
        if filters.get("require_unscoped"):
            sql += (
                " AND json_array_length(CASE WHEN json_valid(c.scope_keys) "
                "AND json_type(c.scope_keys) = 'array' THEN c.scope_keys ELSE '[\"__invalid__\"]' END) = 0"
            )
        raw_paths = filters.get("target_paths", filters.get("path_prefixes"))
        if raw_paths is not None:
            paths = self._store._filter_values(raw_paths, allow_empty=True)
            if not paths:
                return " AND 1 = 0", []
            if len(paths) > _MAX_TARGET_PATHS:
                raise ValueError(f"path filters cannot exceed {_MAX_TARGET_PATHS} values")
            branches: list[str] = []
            for raw_path in paths:
                path = normalize_tree_path(raw_path)
                branches.append("(lp.path = ? OR (lp.path >= ? AND lp.path < ?))")
                params.extend((path, f"{path}/", f"{path}/\uffff"))
            sql += (
                " AND EXISTS (SELECT 1 FROM context_paths AS lp "
                "WHERE lp.tenant_id = c.tenant_id AND lp.record_key = c.record_key AND (" + " OR ".join(branches) + "))"
            )
        return sql, params

    def _connect_filter_sql(
        self,
        alias: str,
        filters: Mapping[str, Any],
    ) -> tuple[str, list[Any]]:
        """Compile trusted connect metadata into a pre-LIMIT predicate."""

        raw_filters = filters.get("connect_filters")
        if raw_filters is None:
            return "", []
        if not isinstance(raw_filters, Mapping):
            raise ValueError("connect_filters must be a mapping")
        expressions = {
            "adapter_id": f"{alias}.adapter_id",
            "source_kind": f"{alias}.source_kind",
            "connect_type": f"json_extract({alias}.metadata_json, '$.connect.connect_type')",
            "run_mode": f"json_extract({alias}.metadata_json, '$.connect.run_mode')",
            "world_domain": f"json_extract({alias}.metadata_json, '$.connect.world_domain')",
        }
        sql = ""
        params: list[Any] = []
        for raw_key, value in raw_filters.items():
            key = str(raw_key)
            if key not in expressions:
                raise ValueError(f"unsupported connect filter: {key}")
            if value in (None, ""):
                continue
            # CurrentSlot is the authoritative state overlay shared by all
            # permitted adapters; connect metadata describes evidence
            # provenance and must not hide that state after ACL/scope checks.
            sql += f" AND ({alias}.record_kind = 'current_slot' OR {expressions[key]} = ?)"
            params.append(value)
        return sql, params

    def _target_identity_sql(
        self,
        filters: Mapping[str, Any],
        target_identity_uris: Any,
        *,
        legacy: bool = False,
    ) -> tuple[str, list[Any]]:
        """Build a bounded equality union for serving, Slot, and Claim URI.

        Each branch applies the complete trusted eligibility predicate --
        access, lifecycle, scope, path, time, validity, and caller type --
        *before* its LIMIT.  This prevents a stable Slot URI with many
        immutable Claim revisions from being materialized in full, and keeps
        newer unauthorized/archived/out-of-range rows from crowding an older
        eligible identity out of the bounded candidate set.  The outer query
        repeats every predicate as a defense-in-depth validation before its
        final Top-K.
        """

        values = self._store._filter_values(target_identity_uris, allow_empty=True)
        if not values:
            return " AND 1 = 0", []
        tenant_values = self._store._filter_values(filters["tenant_id"]) if filters.get("tenant_id") is not None else []
        if not tenant_values:
            raise ValueError("exact Catalog identity lookup requires tenant_id")
        raw_candidate_limit = filters.get("_identity_candidate_limit", _MAX_QUERY_LIMIT)
        if isinstance(raw_candidate_limit, bool):
            raise ValueError("exact Catalog identity candidate limit must be an integer")
        try:
            candidate_limit = int(raw_candidate_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("exact Catalog identity candidate limit must be an integer") from exc
        if not 1 <= candidate_limit <= _MAX_QUERY_LIMIT:
            raise ValueError(f"exact Catalog identity candidate limit must be between 1 and {_MAX_QUERY_LIMIT}")
        tenant_placeholders = ",".join("?" for _ in tenant_values)
        value_placeholders = ",".join("?" for _ in values)
        branches: list[str] = []
        params: list[Any] = []

        def branch_filters(alias: str) -> tuple[str, list[Any]]:
            branch_sql = ""
            branch_params: list[Any] = []
            if not legacy:
                branch_sql += (
                    " AND NOT EXISTS (SELECT 1 FROM json_each("
                    f"CASE WHEN json_valid({alias}.scope_keys) THEN "
                    f"CASE WHEN json_type({alias}.scope_keys) = 'array' "
                    f"THEN {alias}.scope_keys ELSE '[\"{_INVALID_SCOPE_KEY}\"]' END "
                    f"ELSE '[\"{_INVALID_SCOPE_KEY}\"]' END) WHERE value = ?)"
                )
                branch_params.append(_INVALID_SCOPE_KEY)
            # These are the direct, trusted Catalog constraints that can be
            # evaluated without inference.  In particular record_kind keeps a
            # CURRENT lookup away from Claim history and a HISTORY lookup away
            # from the mutable CurrentSlot overlay.
            for filter_name, column in _SIMPLE_FILTER_FIELDS.items():
                if filter_name == "tenant_id" or filters.get(filter_name) is None:
                    continue
                selected = self._store._filter_values(filters[filter_name])
                branch_sql += f" AND {alias}.{column} IN ({','.join('?' for _ in selected)})"
                branch_params.extend(selected)
            for filter_name, column in _PLURAL_FILTER_ALIASES.items():
                if filters.get(filter_name) is None:
                    continue
                selected = self._store._filter_values(filters[filter_name], allow_empty=True)
                if not selected:
                    return " AND 1 = 0", []
                branch_sql += f" AND {alias}.{column} IN ({','.join('?' for _ in selected)})"
                branch_params.extend(selected)

            if filters.get("uri") is not None:
                selected = self._store._filter_values(filters["uri"])
                branch_sql += f" AND {alias}.uri IN ({','.join('?' for _ in selected)})"
                branch_params.extend(selected)
            project_filter = filters.get("project_id")
            if project_filter is not None:
                if legacy:
                    branch_sql += f" AND ({alias}.project_id = ? OR {alias}.project_id = '')"
                    branch_params.append(str(project_filter))
                else:
                    branch_sql += (
                        f" AND ({alias}.project_id = ? OR ({alias}.project_id = '' "
                        f"AND {alias}.memory_type NOT IN (?, ?, ?)))"
                    )
                    branch_params.extend(
                        (
                            str(project_filter),
                            "project_rule",
                            "project_decision",
                            "agent_experience",
                        )
                    )
            allowed_uris = filters.get("allowed_uris")
            if allowed_uris is not None:
                selected = self._store._filter_values(allowed_uris, allow_empty=True)
                if not selected:
                    return " AND 1 = 0", []
                branch_sql += f" AND {alias}.uri IN ({','.join('?' for _ in selected)})"
                branch_params.extend(selected)

            connect_sql, connect_params = self._store._connect_filter_sql(alias, filters)
            branch_sql += connect_sql
            branch_params.extend(connect_params)
            adapter_access = filters.get("adapter_access_id")
            if adapter_access is not None:
                branch_sql += (
                    f" AND ({alias}.adapter_id IN ('', ?) "
                    f"OR {alias}.context_type IN ('session', 'resource', 'skill') "
                    f"OR {alias}.record_kind = 'current_slot')"
                )
                branch_params.append(str(adapter_access))

            include_inactive = bool(filters.get("include_inactive", False))
            if filters.get("admission_status") is None and not include_inactive:
                excluded_admission = (
                    ("restricted", "archive_only", "reject")
                    if filters.get("include_candidates")
                    else ("pending", "restricted", "archive_only", "reject")
                )
                branch_sql += f" AND {alias}.admission_status NOT IN ({','.join('?' for _ in excluded_admission)})"
                branch_params.extend(excluded_admission)
            if filters.get("lifecycle_state") is None and not include_inactive:
                branch_sql += f" AND {alias}.lifecycle_state NOT IN (?, ?, ?)"
                branch_params.extend(("deleted", "archived", "obsolete"))
            if filters.get("serving_tier") is None and not include_inactive:
                branch_sql += f" AND {alias}.serving_tier != ?"
                branch_params.append(ServingTier.ARCHIVED.value)

            available_scopes = filters.get("applicability_scope_keys")
            if available_scopes:
                scopes = self._store._filter_values(available_scopes)
                if not legacy:
                    signatures = self._store._scope_signature_options(scopes)
                    branch_sql += f" AND {alias}.scope_signature IN ({','.join('?' for _ in signatures)})"
                    branch_params.extend(signatures)
                invalid_fallback = "'[\"__invalid__\"]'" if legacy else "'[]'"
                branch_sql += (
                    " AND NOT EXISTS (SELECT 1 FROM json_each("
                    f"CASE WHEN json_valid({alias}.scope_keys) "
                    + (f"AND json_type({alias}.scope_keys) = 'array' " if legacy else "")
                    + f"THEN {alias}.scope_keys ELSE {invalid_fallback} END) "
                    f"WHERE value NOT IN ({','.join('?' for _ in scopes)}))"
                )
                branch_params.extend(scopes)
            if filters.get("require_unscoped"):
                if not legacy:
                    branch_sql += f" AND {alias}.scope_signature = ?"
                    branch_params.append(self._store._scope_signature(()))
                invalid_fallback = "'[\"__invalid__\"]'" if legacy else "'[]'"
                branch_sql += (
                    " AND json_array_length(CASE WHEN "
                    f"json_valid({alias}.scope_keys) "
                    f"AND json_type({alias}.scope_keys) = 'array' "
                    f"THEN {alias}.scope_keys ELSE {invalid_fallback} END) = 0"
                )

            for filter_name, column, operator in (
                ("event_time_from", "event_time", ">="),
                ("event_time_to", "event_time", "<"),
                ("transaction_time_from", "transaction_time", ">="),
                ("transaction_time_to", "transaction_time", "<"),
                ("updated_at_from", "updated_at", ">="),
                ("updated_at_to", "updated_at", "<"),
            ):
                if filters.get(filter_name) is not None:
                    branch_sql += f" AND {alias}.{column} {operator} ?"
                    branch_params.append(str(filters[filter_name]))
            if filters.get("valid_at") is not None:
                valid_at = str(filters["valid_at"])
                branch_sql += f" AND {alias}.valid_from <= ? AND ({alias}.valid_to = '' OR {alias}.valid_to > ?)"
                branch_params.extend((valid_at, valid_at))

            raw_paths = filters.get("target_paths", filters.get("path_prefixes"))
            if raw_paths is not None:
                paths = self._store._filter_values(raw_paths, allow_empty=True)
                if not paths:
                    return " AND 1 = 0", []
                if len(paths) > _MAX_TARGET_PATHS:
                    raise ValueError(f"path filters cannot exceed {_MAX_TARGET_PATHS} values")
                if legacy:
                    path_predicates: list[str] = []
                    for raw_path in paths:
                        path = normalize_tree_path(raw_path)
                        path_predicates.append(
                            "(identity_path.path = ? OR (identity_path.path >= ? AND identity_path.path < ?))"
                        )
                        branch_params.extend((path, f"{path}/", f"{path}/\uffff"))
                    branch_sql += (
                        " AND EXISTS (SELECT 1 FROM context_paths AS identity_path "
                        f"WHERE identity_path.tenant_id = {alias}.tenant_id "
                        f"AND identity_path.record_key = {alias}.record_key AND (" + " OR ".join(path_predicates) + "))"
                    )
                else:
                    normalized_paths = tuple(normalize_tree_path(path) for path in paths)
                    branch_sql += (
                        " AND EXISTS (SELECT 1 FROM context_path_closure AS identity_path "
                        f"WHERE identity_path.tenant_id = {alias}.tenant_id "
                        f"AND identity_path.record_key = {alias}.record_key "
                        f"AND identity_path.ancestor_path IN "
                        f"({','.join('?' for _ in normalized_paths)}))"
                    )
                    branch_params.extend(normalized_paths)

            if legacy:
                principal = filters.get("principal_owner_id")
                if principal is not None:
                    branch_sql += (
                        f" AND ({alias}.owner_user_id = ? OR "
                        f"({alias}.owner_user_id = '' AND "
                        f"{alias}.context_type IN ('resource', 'skill')))"
                    )
                    branch_params.append(str(principal))
                elif filters.get("owner_user_id") == "" and filters.get("require_unscoped"):
                    branch_sql += f" AND {alias}.owner_user_id = '' AND {alias}.context_type IN ('resource', 'skill')"
                return branch_sql, branch_params

            principal = filters.get("principal_owner_id")
            public_only = bool(
                principal is None and filters.get("owner_user_id") == "" and filters.get("require_unscoped")
            )
            workspace_access = tuple(
                self._store._filter_values(filters.get("workspace_access_ids", ()), allow_empty=True)
            )
            has_workspace_constraint = filters.get("workspace_access_ids") is not None
            shared_workspaces = tuple(
                value for value in workspace_access if value not in {"", "__memoryos_principal_only__"}
            )
            if principal is not None:
                access_predicates = [
                    "(identity_acl.grant_kind = 'principal' AND identity_acl.grant_id = ?)",
                    "identity_acl.grant_kind = 'public'",
                ]
                access_params: list[Any] = [str(principal)]
                service_access_id = filters.get("service_access_id")
                if service_access_id:
                    access_predicates.append("(identity_acl.grant_kind = 'service' AND identity_acl.grant_id = ?)")
                    access_params.append(str(service_access_id))
                access_predicates.append("identity_acl.grant_kind = 'tenant'")
                if shared_workspaces:
                    access_predicates.append(
                        "(identity_acl.grant_kind = 'workspace' AND identity_acl.grant_id IN ("
                        + ",".join("?" for _ in shared_workspaces)
                        + "))"
                    )
                    access_params.extend(shared_workspaces)
                workspace_sql = ""
                if has_workspace_constraint:
                    workspace_sql = (
                        " AND identity_acl.workspace_id IN (" + ",".join("?" for _ in workspace_access) + ")"
                    )
                    access_params.extend(workspace_access)
                branch_sql += (
                    " AND EXISTS (SELECT 1 FROM context_acl_grants AS identity_acl "
                    "INDEXED BY idx_context_acl_grants_record "
                    f"WHERE identity_acl.tenant_id = {alias}.tenant_id "
                    f"AND identity_acl.record_key = {alias}.record_key AND ("
                    + " OR ".join(access_predicates)
                    + ")"
                    + workspace_sql
                    + ")"
                )
                branch_params.extend(access_params)
            elif public_only:
                workspace_sql = ""
                if has_workspace_constraint:
                    workspace_sql = (
                        " AND identity_acl.workspace_id IN (" + ",".join("?" for _ in workspace_access) + ")"
                    )
                branch_sql += (
                    " AND EXISTS (SELECT 1 FROM context_acl_grants AS identity_acl "
                    "INDEXED BY idx_context_acl_grants_record "
                    f"WHERE identity_acl.tenant_id = {alias}.tenant_id "
                    f"AND identity_acl.record_key = {alias}.record_key "
                    "AND identity_acl.grant_kind = 'public'" + workspace_sql + ")"
                )
                if has_workspace_constraint:
                    branch_params.extend(workspace_access)
            return branch_sql, branch_params

        for ordinal, (column, index_name) in enumerate(
            (
                ("uri", "idx_contexts_tenant_uri_kind_updated"),
                ("canonical_slot_uri", "idx_contexts_tenant_canonical_slot_uri"),
                ("canonical_claim_uri", "idx_contexts_tenant_canonical_claim_uri"),
            )
        ):
            alias = f"identity_{ordinal}"
            inner_filter_sql, inner_filter_params = branch_filters(alias)
            branches.append(
                f"SELECT bounded_identity_{ordinal}.record_key FROM ("
                f"SELECT {alias}.record_key FROM contexts AS {alias} INDEXED BY {index_name} "
                f"WHERE {alias}.tenant_id IN ({tenant_placeholders}) "
                f"AND {alias}.{column} IN ({value_placeholders})"
                + inner_filter_sql
                + f" ORDER BY {alias}.updated_at DESC, {alias}.record_key LIMIT ?"
                f") AS bounded_identity_{ordinal}"
            )
            params.extend((*tenant_values, *values, *inner_filter_params, candidate_limit))
        return (
            " AND c.record_key IN (SELECT exact_identity.record_key FROM ("
            + " UNION ALL ".join(branches)
            + ") AS exact_identity)",
            params,
        )


__all__ = ["QueryFilterBuilder"]
