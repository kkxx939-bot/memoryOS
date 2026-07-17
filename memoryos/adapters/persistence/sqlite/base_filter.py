"""SQLite catalog BaseFilterBuilder responsibility component."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _FTS_BM25,
    _INVALID_SCOPE_KEY,
    _MAX_QUERY_LIMIT,
    _MAX_TARGET_PATHS,
    _PLURAL_FILTER_ALIASES,
    _SIMPLE_FILTER_FIELDS,
    Any,
    ServingTier,
    normalize_tree_path,
)


class BaseFilterBuilder:
    """Own one stable subset of SQLite catalog behavior."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def _base_filter_sql(
        self,
        filters: dict[str, Any],
        *,
        path_candidate_limit: int = 100,
        path_fts_query: str = "",
        path_exact_value: str = "",
    ) -> tuple[str, list[Any]]:
        sql = (
            " AND NOT EXISTS (SELECT 1 FROM json_each("
            "CASE WHEN json_valid(c.scope_keys) THEN "
            "CASE WHEN json_type(c.scope_keys) = 'array' THEN c.scope_keys "
            f"ELSE '[\"{_INVALID_SCOPE_KEY}\"]' END "
            f"ELSE '[\"{_INVALID_SCOPE_KEY}\"]' END"
            ") WHERE value = ?)"
        )
        params: list[Any] = [_INVALID_SCOPE_KEY]
        principal_owner = filters.get("principal_owner_id")
        has_path_filter = filters.get("target_paths", filters.get("path_prefixes")) is not None
        public_only = bool(
            principal_owner is None and filters.get("owner_user_id") == "" and filters.get("require_unscoped")
        )
        shared_workspaces = tuple(
            value
            for value in self._store._filter_values(filters.get("workspace_access_ids", ()), allow_empty=True)
            if value not in {"", "__memoryos_principal_only__"}
        )
        has_workspace_constraint = filters.get("workspace_access_ids") is not None
        workspace_access = tuple(self._store._filter_values(filters.get("workspace_access_ids", ()), allow_empty=True))
        raw_available_scopes = filters.get("applicability_scope_keys")
        has_scope_filter = raw_available_scopes is not None or bool(filters.get("require_unscoped"))
        if raw_available_scopes:
            scope_signature_options = self._store._scope_signature_options(
                self._store._filter_values(raw_available_scopes)
            )
        elif has_scope_filter:
            scope_signature_options = (self._store._scope_signature(()),)
        else:
            scope_signature_options = ()
        if principal_owner is not None:
            grant_predicates = ["(cg.grant_kind = 'principal' AND cg.grant_id = ?)", "cg.grant_kind = 'public'"]
            catalog_grant_params: list[Any] = [str(principal_owner)]
            service_access_id = filters.get("service_access_id")
            if service_access_id:
                grant_predicates.append("(cg.grant_kind = 'service' AND cg.grant_id = ?)")
                catalog_grant_params.append(str(service_access_id))
            grant_predicates.append("cg.grant_kind = 'tenant'")
            if shared_workspaces:
                grant_predicates.append(
                    "(cg.grant_kind = 'workspace' AND cg.grant_id IN ("
                    + ",".join("?" for _ in shared_workspaces)
                    + "))"
                )
                catalog_grant_params.extend(shared_workspaces)
            grant_workspace_sql = ""
            if has_workspace_constraint:
                grant_workspace_sql = " AND cg.workspace_id IN (" + ",".join("?" for _ in workspace_access) + ")"
                catalog_grant_params.extend(workspace_access)
            sql += (
                " AND EXISTS (SELECT 1 FROM context_acl_grants AS cg "
                "WHERE cg.tenant_id = c.tenant_id AND cg.record_key = c.record_key AND ("
                + " OR ".join(grant_predicates)
                + ")"
                + grant_workspace_sql
                + ")"
            )
            params.extend(catalog_grant_params)
        elif public_only:
            public_workspace_sql = ""
            if has_workspace_constraint:
                public_workspace_sql = " AND cg.workspace_id IN (" + ",".join("?" for _ in workspace_access) + ")"
            sql += (
                " AND EXISTS (SELECT 1 FROM context_acl_grants AS cg "
                "WHERE cg.tenant_id = c.tenant_id AND cg.record_key = c.record_key "
                "AND cg.grant_kind = 'public'" + public_workspace_sql + ")"
            )
            if has_workspace_constraint:
                params.extend(workspace_access)
        adapter_access = filters.get("adapter_access_id")
        if adapter_access is not None:
            sql += (
                " AND (c.adapter_id IN ('', ?) OR c.context_type IN ('session', 'resource', 'skill') "
                "OR c.record_kind = 'current_slot')"
            )
            params.append(str(adapter_access))
        for filter_name, column in _SIMPLE_FILTER_FIELDS.items():
            if filters.get(filter_name) is None:
                continue
            values = self._store._filter_values(filters[filter_name])
            sql += f" AND c.{column} IN ({','.join('?' for _ in values)})"
            params.extend(values)
        for filter_name, column in _PLURAL_FILTER_ALIASES.items():
            if filters.get(filter_name) is None:
                continue
            values = self._store._filter_values(filters[filter_name])
            sql += f" AND c.{column} IN ({','.join('?' for _ in values)})"
            params.extend(values)
        target_identity_uris = filters.get("target_identity_uris")
        if target_identity_uris is not None:
            identity_sql, identity_params = self._store._target_identity_sql(
                filters,
                target_identity_uris,
            )
            sql += identity_sql
            params.extend(identity_params)
        connect_sql, connect_params = self._store._connect_filter_sql("c", filters)
        sql += connect_sql
        params.extend(connect_params)
        if filters.get("uri") is not None:
            values = self._store._filter_values(filters["uri"])
            sql += f" AND c.uri IN ({','.join('?' for _ in values)})"
            params.extend(values)
        project_filter = filters.get("project_id")
        if project_filter is not None:
            sql += " AND (c.project_id = ? OR (c.project_id = '' AND c.memory_type NOT IN (?, ?, ?)))"
            params.append(str(project_filter))
            params.extend(("project_rule", "project_decision", "agent_experience"))
        allowed_uris = filters.get("allowed_uris")
        if allowed_uris is not None:
            values = self._store._filter_values(allowed_uris, allow_empty=True)
            if values:
                sql += f" AND c.uri IN ({','.join('?' for _ in values)})"
                params.extend(values)
            else:
                sql += " AND 1 = 0"
        include_inactive = bool(filters.get("include_inactive", False))
        if filters.get("admission_status") is None and not include_inactive:
            excluded_admission = (
                ("restricted", "archive_only", "reject")
                if filters.get("include_candidates")
                else ("pending", "restricted", "archive_only", "reject")
            )
            sql += f" AND c.admission_status NOT IN ({','.join('?' for _ in excluded_admission)})"
            params.extend(excluded_admission)
        if filters.get("lifecycle_state") is None and not include_inactive:
            sql += " AND c.lifecycle_state NOT IN (?, ?, ?)"
            params.extend(("deleted", "archived", "obsolete"))
        if filters.get("serving_tier") is None and not include_inactive:
            sql += " AND c.serving_tier != ?"
            params.append(ServingTier.ARCHIVED.value)
        available_scopes = filters.get("applicability_scope_keys")
        if available_scopes:
            scopes = self._store._filter_values(available_scopes)
            sql += f" AND c.scope_signature IN ({','.join('?' for _ in scope_signature_options)})"
            params.extend(scope_signature_options)
            sql += (
                " AND NOT EXISTS (SELECT 1 FROM json_each("
                "CASE WHEN json_valid(c.scope_keys) THEN c.scope_keys ELSE '[]' END) "
                f"WHERE value NOT IN ({','.join('?' for _ in scopes)}))"
            )
            params.extend(scopes)
        if filters.get("require_unscoped"):
            sql += " AND c.scope_signature = ?"
            params.append(self._store._scope_signature(()))
            sql += (
                " AND json_array_length(CASE WHEN json_valid(c.scope_keys) "
                "AND json_type(c.scope_keys) = 'array' THEN c.scope_keys ELSE '[]' END) = 0"
            )
        time_filters = (
            ("event_time_from", "event_time", ">="),
            ("event_time_to", "event_time", "<"),
            ("transaction_time_from", "transaction_time", ">="),
            ("transaction_time_to", "transaction_time", "<"),
            ("updated_at_from", "updated_at", ">="),
            ("updated_at_to", "updated_at", "<"),
        )
        for filter_name, column, operator in time_filters:
            if filters.get(filter_name) is not None:
                sql += f" AND c.{column} {operator} ?"
                params.append(str(filters[filter_name]))
        if filters.get("valid_at") is not None:
            valid_at = str(filters["valid_at"])
            valid_point = self._store._timestamp_number(valid_at, lower=True)
            validity_tenants = (
                self._store._filter_values(filters["tenant_id"]) if filters.get("tenant_id") is not None else []
            )
            correlated_tenant_sql = ""
            correlated_tenant_params: list[Any] = []
            validity_map_index = "sqlite_autoindex_context_validity_map_1"
            if validity_tenants:
                validity_map_index = "idx_context_validity_map_tenant_record"
                correlated_tenant_sql = f" AND vm.tenant_id IN ({','.join('?' for _ in validity_tenants)})"
                correlated_tenant_params.extend(validity_tenants)
            # Drive the correlated lookup from the unique record identity and
            # only then probe its one RTree row.  Reversing this CROSS JOIN
            # scans every interval that contains ``valid_at`` for every ACL
            # candidate and becomes quadratic when most rows are open-ended.
            sql += (
                " AND EXISTS (SELECT 1 FROM context_validity_map AS vm INDEXED BY "
                + validity_map_index
                + " CROSS JOIN context_validity_rtree AS vr "
                "WHERE vm.record_key = c.record_key" + correlated_tenant_sql + " AND vr.validity_id = vm.validity_id "
                "AND vr.valid_from_min <= ? AND vr.valid_to_max > ? LIMIT 1)"
            )
            params.extend((*correlated_tenant_params, valid_point, valid_point))
            # RTree coordinates are float32 and intentionally rounded outward.
            # Recheck the exact half-open ISO interval before LIMIT so boundary
            # rounding can never admit a false-positive serving record.
            sql += " AND c.valid_from <= ? AND (c.valid_to = '' OR c.valid_to > ?)"
            params.extend((valid_at, valid_at))
        # Preserve the complete deterministic filter for the path branches
        # before adding the general ACL candidate driver below.  Principal
        # path branches already use owner/public/shared partial indexes and
        # must not recursively embed the ACL driver inside every path branch.
        path_inner_filter_sql = sql
        path_inner_filter_params = list(params)
        if (
            (principal_owner is not None or public_only)
            and not has_path_filter
            and filters.get("record_keys") is None
            and filters.get("target_identity_uris") is None
            and not filters.get("_fts_bound_candidates")
            and not filters.get("_exact_bound_candidates")
        ):
            tenant_values = (
                self._store._filter_values(filters["tenant_id"]) if filters.get("tenant_id") is not None else []
            )
            if not tenant_values:
                raise ValueError("principal-scoped Catalog queries require tenant_id")
            bounded_acl_limit = min(_MAX_QUERY_LIMIT, max(1, int(path_candidate_limit)))
            acl_filter_sql = sql.replace("c.", "ac.")
            acl_filter_params = tuple(params)
            acl_join = ""
            acl_match_sql = ""
            acl_match_params: tuple[Any, ...] = ()
            acl_order = "ag.updated_at DESC, ag.record_key DESC"
            if path_fts_query:
                acl_join = " CROSS JOIN contexts_fts ON contexts_fts.record_key = ac.record_key"
                acl_match_sql = " AND contexts_fts MATCH ?"
                acl_match_params = (path_fts_query,)
                acl_order = f"{_FTS_BM25}, ag.updated_at DESC, ag.record_key DESC"
            elif path_exact_value:
                acl_match_sql = " AND (ac.scene_key = ? OR ac.action = ? OR ac.memory_anchor_uri = ?)"
                acl_match_params = (path_exact_value, path_exact_value, path_exact_value)
            if any(filters.get(name) is not None for name in ("event_time_from", "event_time_to")):
                acl_time_dimension = "event"
            elif any(filters.get(name) is not None for name in ("transaction_time_from", "transaction_time_to")):
                acl_time_dimension = "transaction"
            elif any(filters.get(name) is not None for name in ("updated_at_from", "updated_at_to")):
                acl_time_dimension = "updated"
            else:
                acl_time_dimension = "updated"
            if not path_fts_query:
                acl_order_column = {
                    "event": "event_time",
                    "transaction": "transaction_time",
                    "updated": "updated_at",
                }[acl_time_dimension]
                acl_order = f"ag.{acl_order_column} DESC, ag.record_key DESC"
            raw_acl_types = filters.get("context_types", filters.get("context_type"))
            acl_type_values = self._store._filter_values(raw_acl_types) if raw_acl_types is not None else []
            acl_source_values = (
                self._store._filter_values(filters["source_kinds"]) if filters.get("source_kinds") is not None else []
            )
            acl_session_values = (
                self._store._filter_values(filters["session_ids"]) if filters.get("session_ids") is not None else []
            )
            acl_adapter_values = (
                self._store._filter_values(filters["adapter_id"]) if filters.get("adapter_id") is not None else []
            )
            acl_adapter_access_values = (
                self._store._filter_values(filters["adapter_access_id"])
                if filters.get("adapter_access_id") is not None
                else []
            )
            acl_target_values = (
                self._store._filter_values(filters["target_uris"]) if filters.get("target_uris") is not None else []
            )
            raw_record_kinds = filters.get("record_kinds", filters.get("record_kind"))
            acl_record_kinds = self._store._filter_values(raw_record_kinds) if raw_record_kinds is not None else []
            if has_workspace_constraint and acl_target_values:
                grant_index = "idx_context_acl_grants_workspace_uri"
            elif not has_workspace_constraint:
                if acl_target_values:
                    grant_index = "idx_context_acl_grants_access_uri"
                elif has_scope_filter:
                    grant_index = f"idx_context_acl_grants_access_scope_{acl_time_dimension}"
                elif acl_adapter_access_values:
                    grant_index = f"idx_context_acl_grants_access_adapter_access_{acl_time_dimension}"
                elif acl_adapter_values:
                    grant_index = f"idx_context_acl_grants_access_adapter_{acl_time_dimension}"
                elif acl_session_values:
                    grant_index = f"idx_context_acl_grants_access_session_{acl_time_dimension}"
                elif acl_source_values:
                    grant_index = f"idx_context_acl_grants_access_source_{acl_time_dimension}"
                elif acl_type_values:
                    grant_index = f"idx_context_acl_grants_access_type_{acl_time_dimension}"
                else:
                    grant_index = f"idx_context_acl_grants_access_{acl_time_dimension}"
            elif not any(
                (
                    has_scope_filter,
                    acl_adapter_access_values,
                    acl_adapter_values,
                    acl_session_values,
                    acl_source_values,
                )
            ):
                grant_index = f"idx_context_acl_grants_workspace_access_{acl_time_dimension}"
            else:
                grant_index = "idx_context_acl_grants_"
                grant_index += "workspace_" if has_workspace_constraint else "kind_"
                if has_workspace_constraint and has_scope_filter:
                    grant_index += "scope_"
                elif has_workspace_constraint and acl_adapter_access_values:
                    grant_index += "adapter_access_"
                elif has_workspace_constraint and acl_adapter_values:
                    grant_index += "adapter_"
                elif has_workspace_constraint and acl_session_values and acl_time_dimension == "updated":
                    grant_index += "session_"
                    if acl_type_values:
                        grant_index += "type_"
                elif has_workspace_constraint and acl_source_values and acl_time_dimension == "updated":
                    grant_index += "source_"
                    if acl_type_values:
                        grant_index += "type_"
                elif has_workspace_constraint and acl_type_values:
                    grant_index += "type_"
                grant_index += acl_time_dimension
            grant_predicates = [
                f"ag.tenant_id IN ({','.join('?' for _ in tenant_values)})",
            ]
            grant_filter_params: list[Any] = list(tenant_values)
            if acl_record_kinds:
                grant_predicates.append(f"ag.record_kind IN ({','.join('?' for _ in acl_record_kinds)})")
                grant_filter_params.extend(acl_record_kinds)
            if has_scope_filter:
                grant_predicates.append(f"ag.scope_signature IN ({','.join('?' for _ in scope_signature_options)})")
                grant_filter_params.extend(scope_signature_options)
            if acl_type_values:
                grant_predicates.append(f"ag.context_type IN ({','.join('?' for _ in acl_type_values)})")
                grant_filter_params.extend(acl_type_values)
            if acl_source_values:
                grant_predicates.append(f"ag.source_kind IN ({','.join('?' for _ in acl_source_values)})")
                grant_filter_params.extend(acl_source_values)
            if acl_session_values:
                grant_predicates.append(f"ag.session_id IN ({','.join('?' for _ in acl_session_values)})")
                grant_filter_params.extend(acl_session_values)
            if acl_adapter_values:
                grant_predicates.append(f"ag.adapter_id IN ({','.join('?' for _ in acl_adapter_values)})")
                grant_filter_params.extend(acl_adapter_values)
            if acl_adapter_access_values:
                grant_predicates.append(
                    f"ag.adapter_access_id IN ({','.join('?' for _ in range(len(acl_adapter_access_values) + 1))})"
                )
                grant_filter_params.extend(("*", *acl_adapter_access_values))
            if acl_target_values:
                grant_predicates.append(f"ag.uri IN ({','.join('?' for _ in acl_target_values)})")
                grant_filter_params.extend(acl_target_values)
            for filter_name, column, operator in (
                ("event_time_from", "event_time", ">="),
                ("event_time_to", "event_time", "<"),
                ("transaction_time_from", "transaction_time", ">="),
                ("transaction_time_to", "transaction_time", "<"),
                ("updated_at_from", "updated_at", ">="),
                ("updated_at_to", "updated_at", "<"),
            ):
                if filters.get(filter_name) is not None:
                    grant_predicates.append(f"ag.{column} {operator} ?")
                    grant_filter_params.append(str(filters[filter_name]))
            grant_access: list[tuple[str, tuple[Any, ...]]] = []
            if not public_only:
                grant_access.append(("ag.grant_kind = 'principal' AND ag.grant_id = ?", (str(principal_owner),)))
                if filters.get("service_access_id"):
                    grant_access.append(
                        (
                            "ag.grant_kind = 'service' AND ag.grant_id = ?",
                            (str(filters["service_access_id"]),),
                        )
                    )
                grant_access.append(("ag.grant_kind = 'tenant' AND ag.grant_id = ''", ()))
            grant_access.append(("ag.grant_kind = 'public' AND ag.grant_id = ''", ()))
            if not public_only and shared_workspaces:
                grant_access.append(
                    (
                        f"ag.grant_kind = 'workspace' AND ag.grant_id IN ({','.join('?' for _ in shared_workspaces)})",
                        tuple(shared_workspaces),
                    )
                )
            acl_selects: list[str] = []
            acl_params: list[Any] = []
            grant_filter_sql = " AND ".join(grant_predicates)
            grant_access_variants: list[tuple[str, tuple[Any, ...]]] = []
            for access_predicate, grant_access_params in grant_access:
                if has_workspace_constraint:
                    grant_access_variants.extend(
                        (
                            access_predicate + " AND ag.workspace_id = ?",
                            (*grant_access_params, workspace_id),
                        )
                        for workspace_id in workspace_access
                    )
                else:
                    grant_access_variants.append((access_predicate, grant_access_params))
            for access_predicate, grant_access_params in grant_access_variants:
                acl_selects.append(
                    "SELECT one_acl.record_key FROM ("
                    "SELECT ac.record_key FROM context_acl_grants AS ag INDEXED BY "
                    + grant_index
                    + " CROSS JOIN contexts AS ac INDEXED BY idx_contexts_record_key "
                    "ON ac.record_key = ag.record_key"
                    + acl_join
                    + " WHERE 1=1 "
                    + " AND "
                    + access_predicate
                    + " AND "
                    + grant_filter_sql
                    + acl_filter_sql
                    + acl_match_sql
                    + " ORDER BY "
                    + acl_order
                    + " LIMIT ?) AS one_acl"
                )
                acl_params.extend(
                    (
                        *grant_access_params,
                        *grant_filter_params,
                        *acl_filter_params,
                        *acl_match_params,
                        bounded_acl_limit,
                    )
                )
            sql += (
                " AND c.record_key IN (SELECT bounded_acl.record_key FROM ("
                + " UNION ALL ".join(acl_selects)
                + ") AS bounded_acl)"
            )
            params.extend(acl_params)
        raw_paths = filters.get("target_paths", filters.get("path_prefixes"))
        if filters.get("_fts_bound_candidates") or filters.get("_exact_bound_candidates"):
            # FTS and exact-identity branches already have a selective outer
            # candidate driver.  Probe the materialized closure by the
            # candidate record key instead of independently scanning and
            # sorting every matching path branch.  All trusted path
            # predicates still execute before the branch LIMIT.
            paths = self._store._filter_values(raw_paths, allow_empty=True) if raw_paths is not None else []
            if raw_paths is not None and not paths:
                sql += " AND 1 = 0"
            elif paths:
                normalized_paths = tuple(normalize_tree_path(path) for path in paths)
                sql += (
                    " AND EXISTS (SELECT 1 FROM context_path_closure AS fp "
                    "WHERE fp.tenant_id = c.tenant_id AND fp.record_key = c.record_key "
                    f"AND fp.ancestor_path IN ({','.join('?' for _ in normalized_paths)}))"
                )
                params.extend(normalized_paths)
            raw_paths = None
        if raw_paths is not None:
            paths = self._store._filter_values(raw_paths, allow_empty=True)
            if not paths:
                sql += " AND 1 = 0"
            else:
                if len(paths) > _MAX_TARGET_PATHS:
                    raise ValueError(f"path filters cannot exceed {_MAX_TARGET_PATHS} values")
                inner_filter_sql = path_inner_filter_sql.replace("c.", "pc.")
                inner_filter_params = list(path_inner_filter_params)
                tenant_values = (
                    self._store._filter_values(filters["tenant_id"]) if filters.get("tenant_id") is not None else []
                )
                owner_values = (
                    self._store._filter_values(filters["owner_user_id"])
                    if filters.get("owner_user_id") is not None
                    else []
                )
                raw_types = filters.get("context_types", filters.get("context_type"))
                type_values = self._store._filter_values(raw_types) if raw_types is not None else []
                raw_path_kinds = filters.get("record_kinds", filters.get("record_kind"))
                path_kind_values = self._store._filter_values(raw_path_kinds) if raw_path_kinds is not None else []
                path_source_values = (
                    self._store._filter_values(filters["source_kinds"])
                    if filters.get("source_kinds") is not None
                    else []
                )
                path_session_values = (
                    self._store._filter_values(filters["session_ids"]) if filters.get("session_ids") is not None else []
                )
                path_adapter_access_values = (
                    self._store._filter_values(filters["adapter_access_id"])
                    if filters.get("adapter_access_id") is not None
                    else []
                )
                path_target_values = (
                    self._store._filter_values(filters["target_uris"]) if filters.get("target_uris") is not None else []
                )
                has_event_time = any(filters.get(name) is not None for name in ("event_time_from", "event_time_to"))
                has_transaction_time = any(
                    filters.get(name) is not None for name in ("transaction_time_from", "transaction_time_to")
                )
                has_updated_time = any(filters.get(name) is not None for name in ("updated_at_from", "updated_at_to"))
                if has_event_time:
                    path_time_dimension = "event"
                elif has_transaction_time:
                    path_time_dimension = "transaction"
                elif has_updated_time:
                    path_time_dimension = "updated"
                else:
                    path_time_dimension = ""
                path_predicates: list[str] = []
                path_params: list[Any] = []
                if tenant_values:
                    path_predicates.append(f"p.tenant_id IN ({','.join('?' for _ in tenant_values)})")
                    path_params.extend(tenant_values)
                if type_values:
                    path_predicates.append(f"p.context_type IN ({','.join('?' for _ in type_values)})")
                    path_params.extend(type_values)
                if path_kind_values:
                    path_predicates.append(f"p.record_kind IN ({','.join('?' for _ in path_kind_values)})")
                    path_params.extend(path_kind_values)
                if path_source_values:
                    path_predicates.append(f"p.source_kind IN ({','.join('?' for _ in path_source_values)})")
                    path_params.extend(path_source_values)
                if path_session_values:
                    path_predicates.append(f"p.session_id IN ({','.join('?' for _ in path_session_values)})")
                    path_params.extend(path_session_values)
                if path_adapter_access_values:
                    path_predicates.append(
                        f"p.adapter_access_id IN ({','.join('?' for _ in range(len(path_adapter_access_values) + 1))})"
                    )
                    path_params.extend(("*", *path_adapter_access_values))
                if path_target_values:
                    path_predicates.append(f"p.uri IN ({','.join('?' for _ in path_target_values)})")
                    path_params.extend(path_target_values)
                if owner_values:
                    path_predicates.append(f"p.owner_user_id IN ({','.join('?' for _ in owner_values)})")
                    path_params.extend(owner_values)
                if has_workspace_constraint:
                    path_predicates.append(f"p.workspace_id IN ({','.join('?' for _ in workspace_access)})")
                    path_params.extend(workspace_access)
                if has_scope_filter:
                    path_predicates.append(f"p.scope_signature IN ({','.join('?' for _ in scope_signature_options)})")
                    path_params.extend(scope_signature_options)
                for filter_name, operator in (("event_time_from", ">="), ("event_time_to", "<")):
                    if filters.get(filter_name) is not None:
                        path_predicates.append(f"p.event_time {operator} ?")
                        path_params.append(str(filters[filter_name]))
                for filter_name, column, operator in (
                    ("transaction_time_from", "transaction_time", ">="),
                    ("transaction_time_to", "transaction_time", "<"),
                    ("updated_at_from", "updated_at", ">="),
                    ("updated_at_to", "updated_at", "<"),
                ):
                    if filters.get(filter_name) is not None:
                        path_predicates.append(f"p.{column} {operator} ?")
                        path_params.append(str(filters[filter_name]))
                path_selects: list[str] = []
                path_select_params: list[Any] = []
                base_path_sql = " AND ".join(path_predicates)
                base_path_prefix = f"{base_path_sql} AND " if base_path_sql else ""
                bounded_path_limit = min(_MAX_QUERY_LIMIT, max(1, int(path_candidate_limit)))
                principal_path_branches: list[tuple[str, str, tuple[Any, ...]]] = []
                if principal_owner is not None:
                    owner_prefix = (
                        "idx_context_path_closure_owner_workspace"
                        if has_workspace_constraint
                        else "idx_context_path_closure_owner"
                    )
                    if type_values:
                        owner_prefix += "_type"
                    owner_index = owner_prefix + "_ancestor"
                    if path_time_dimension:
                        owner_index += f"_{path_time_dimension}"
                    owner_predicate = "p.owner_user_id = ?"
                    owner_params: tuple[Any, ...] = (str(principal_owner),)
                    if has_workspace_constraint:
                        owner_predicate += f" AND p.workspace_id IN ({','.join('?' for _ in workspace_access)})"
                        owner_params = (*owner_params, *workspace_access)
                    public_prefix = (
                        "idx_context_path_closure_public_workspace_ancestor"
                        if has_workspace_constraint
                        else "idx_context_path_closure_public_ancestor"
                    )
                    public_index = public_prefix + (f"_{path_time_dimension}" if path_time_dimension else "")
                    public_predicate = "p.owner_user_id = '' AND p.context_type IN ('resource', 'skill')"
                    owner_public_params: tuple[Any, ...] = ()
                    if has_workspace_constraint:
                        public_predicate += f" AND p.workspace_id IN ({','.join('?' for _ in workspace_access)})"
                        owner_public_params = workspace_access
                    principal_path_branches.extend(
                        (
                            (owner_index, owner_predicate, owner_params),
                            (public_index, public_predicate, owner_public_params),
                        )
                    )
                    grant_index = (
                        f"idx_context_path_closure_ancestor_{path_time_dimension}"
                        if path_time_dimension
                        else "idx_context_path_closure_tenant_ancestor"
                    )
                    grant_predicate = (
                        "EXISTS (SELECT 1 FROM context_acl_grants AS pg "
                        "WHERE pg.tenant_id = p.tenant_id AND pg.record_key = p.record_key "
                        "AND pg.grant_kind = 'principal' AND pg.grant_id = ?"
                    )
                    path_grant_params: tuple[Any, ...] = (str(principal_owner),)
                    if has_workspace_constraint:
                        grant_predicate += f" AND pg.workspace_id IN ({','.join('?' for _ in workspace_access)})"
                        path_grant_params = (*path_grant_params, *workspace_access)
                    grant_predicate += ")"
                    principal_path_branches.append((grant_index, grant_predicate, path_grant_params))
                    tenant_predicate = (
                        "EXISTS (SELECT 1 FROM context_acl_grants AS tg "
                        "WHERE tg.tenant_id = p.tenant_id AND tg.record_key = p.record_key "
                        "AND tg.grant_kind = 'tenant'"
                    )
                    tenant_params: tuple[Any, ...] = ()
                    if has_workspace_constraint:
                        tenant_predicate += f" AND tg.workspace_id IN ({','.join('?' for _ in workspace_access)})"
                        tenant_params = workspace_access
                    tenant_predicate += ")"
                    principal_path_branches.append((grant_index, tenant_predicate, tenant_params))
                    if filters.get("service_access_id"):
                        service_predicate = (
                            "EXISTS (SELECT 1 FROM context_acl_grants AS sg "
                            "WHERE sg.tenant_id = p.tenant_id AND sg.record_key = p.record_key "
                            "AND sg.grant_kind = 'service' AND sg.grant_id = ?"
                        )
                        service_params: tuple[Any, ...] = (str(filters["service_access_id"]),)
                        if has_workspace_constraint:
                            service_predicate += f" AND sg.workspace_id IN ({','.join('?' for _ in workspace_access)})"
                            service_params = (*service_params, *workspace_access)
                        service_predicate += ")"
                        principal_path_branches.append((grant_index, service_predicate, service_params))
                    if shared_workspaces:
                        principal_path_branches.append(
                            (
                                (
                                    f"idx_context_path_closure_shared_workspace_ancestor_{path_time_dimension}"
                                    if path_time_dimension
                                    else "idx_context_path_closure_shared_workspace_ancestor"
                                ),
                                "p.context_type = 'memory' "
                                "AND p.record_kind IN ('current_slot', 'claim_revision') "
                                "AND p.canonical_slot_id != '' AND p.canonical_claim_id != '' "
                                "AND p.workspace_shared = 1 "
                                f"AND p.workspace_id IN ({','.join('?' for _ in shared_workspaces)})",
                                tuple(shared_workspaces),
                            )
                        )
                elif public_only:
                    public_prefix = (
                        "idx_context_path_closure_public_workspace_ancestor"
                        if has_workspace_constraint
                        else "idx_context_path_closure_public_ancestor"
                    )
                    public_index = public_prefix + (f"_{path_time_dimension}" if path_time_dimension else "")
                    public_predicate = "p.owner_user_id = '' AND p.context_type IN ('resource', 'skill')"
                    unscoped_public_params: tuple[Any, ...] = ()
                    if has_workspace_constraint:
                        public_predicate += f" AND p.workspace_id IN ({','.join('?' for _ in workspace_access)})"
                        unscoped_public_params = workspace_access
                    principal_path_branches.append((public_index, public_predicate, unscoped_public_params))
                else:
                    if owner_values:
                        path_index = (
                            "idx_context_path_closure_owner_workspace"
                            if has_workspace_constraint
                            else "idx_context_path_closure_owner"
                        )
                        if type_values:
                            path_index += "_type"
                        path_index += "_ancestor"
                        if path_time_dimension:
                            path_index += f"_{path_time_dimension}"
                    elif path_time_dimension:
                        path_index = f"idx_context_path_closure_ancestor_{path_time_dimension}"
                    elif type_values:
                        path_index = "idx_context_path_closure_type_ancestor"
                    else:
                        path_index = "idx_context_path_closure_tenant_ancestor"
                    principal_path_branches.append((path_index, "", ()))
                if has_scope_filter:
                    scope_index = "idx_context_path_closure_scope_kind_ancestor"
                    if path_time_dimension:
                        scope_index += f"_{path_time_dimension}"
                    access_parts: list[str] = []
                    scope_access_params: list[Any] = []
                    if principal_owner is not None:
                        access_parts.append("(sg.grant_kind = 'principal' AND sg.grant_id = ?)")
                        scope_access_params.append(str(principal_owner))
                        if filters.get("service_access_id"):
                            access_parts.append("(sg.grant_kind = 'service' AND sg.grant_id = ?)")
                            scope_access_params.append(str(filters["service_access_id"]))
                        if shared_workspaces:
                            access_parts.append(
                                "(sg.grant_kind = 'workspace' AND sg.grant_id IN ("
                                + ",".join("?" for _ in shared_workspaces)
                                + "))"
                            )
                            scope_access_params.extend(shared_workspaces)
                        access_parts.append("sg.grant_kind = 'tenant'")
                    access_parts.append("sg.grant_kind = 'public'")
                    scope_access_predicate = (
                        "EXISTS (SELECT 1 FROM context_acl_grants AS sg "
                        "WHERE sg.tenant_id = p.tenant_id AND sg.record_key = p.record_key AND ("
                        + " OR ".join(access_parts)
                        + ")"
                    )
                    if has_workspace_constraint:
                        scope_access_predicate += f" AND sg.workspace_id IN ({','.join('?' for _ in workspace_access)})"
                        scope_access_params.extend(workspace_access)
                    scope_access_predicate += ")"
                    principal_path_branches = [(scope_index, scope_access_predicate, tuple(scope_access_params))]
                path_table = "context_path_closure"
                if principal_owner is not None or public_only:
                    path_table = "context_path_acl"
                    effective_path_time = path_time_dimension or "updated"
                    if path_target_values:
                        path_acl_index = "idx_context_path_acl_workspace_uri"
                    elif has_scope_filter:
                        path_acl_index = f"idx_context_path_acl_workspace_scope_{effective_path_time}"
                    elif path_adapter_access_values and effective_path_time == "updated":
                        path_acl_index = "idx_context_path_acl_workspace_adapter_access_updated"
                    elif path_session_values and effective_path_time == "updated":
                        path_acl_index = "idx_context_path_acl_workspace_session_updated"
                    elif path_source_values and effective_path_time == "updated":
                        path_acl_index = "idx_context_path_acl_workspace_source_updated"
                    elif type_values and effective_path_time == "updated":
                        path_acl_index = "idx_context_path_acl_workspace_type_updated"
                    else:
                        path_acl_index = f"idx_context_path_acl_workspace_{effective_path_time}"
                    path_acl_branches: list[tuple[str, str, tuple[Any, ...]]] = []
                    if principal_owner is not None:
                        path_acl_branches.append(
                            (
                                path_acl_index,
                                "p.grant_kind = 'principal' AND p.grant_id = ?",
                                (str(principal_owner),),
                            )
                        )
                        path_acl_branches.append((path_acl_index, "p.grant_kind = 'tenant' AND p.grant_id = ''", ()))
                        if filters.get("service_access_id"):
                            path_acl_branches.append(
                                (
                                    path_acl_index,
                                    "p.grant_kind = 'service' AND p.grant_id = ?",
                                    (str(filters["service_access_id"]),),
                                )
                            )
                        if shared_workspaces:
                            path_acl_branches.append(
                                (
                                    path_acl_index,
                                    f"p.grant_kind = 'workspace' AND p.grant_id IN ({','.join('?' for _ in shared_workspaces)})",
                                    tuple(shared_workspaces),
                                )
                            )
                    path_acl_branches.append((path_acl_index, "p.grant_kind = 'public' AND p.grant_id = ''", ()))
                    principal_path_branches = path_acl_branches
                for raw_path in paths:
                    path = normalize_tree_path(raw_path)
                    path_join = ""
                    path_match_sql = ""
                    path_match_params: tuple[Any, ...] = ()
                    path_rank_sql = "0.0"
                    if path_fts_query:
                        path_join = " JOIN contexts_fts ON contexts_fts.record_key = pc.record_key"
                        path_match_sql = " AND contexts_fts MATCH ?"
                        path_match_params = (path_fts_query,)
                        path_rank_sql = _FTS_BM25
                    elif path_exact_value:
                        path_match_sql = " AND (pc.scene_key = ? OR pc.action = ? OR pc.memory_anchor_uri = ?)"
                        path_match_params = (path_exact_value, path_exact_value, path_exact_value)
                    # Every physical path materializes its bounded ancestor
                    # closure at write time. Prefix matching is therefore an
                    # indexed equality, never an online path-range scan.
                    ordered_suffix = " ORDER BY path_rank, path_updated_at DESC, p.record_key DESC LIMIT ?"
                    for path_index, acl_predicate, path_acl_params in principal_path_branches:
                        predicates = base_path_prefix
                        if acl_predicate:
                            predicates += acl_predicate + " AND "
                        select_prefix = (
                            "SELECT DISTINCT p.record_key AS record_key, "
                            + path_rank_sql
                            + " AS path_rank, p.updated_at AS path_updated_at FROM "
                            + path_table
                            + " AS p INDEXED BY "
                            + path_index
                            + " CROSS JOIN contexts AS pc ON pc.tenant_id = p.tenant_id "
                            "AND pc.record_key = p.record_key"
                            + path_join
                            + " WHERE 1=1 "
                            + inner_filter_sql
                            + path_match_sql
                            + " AND "
                            + predicates
                        )
                        path_selects.append(
                            "SELECT one_path.record_key FROM ("
                            + select_prefix
                            + "p.ancestor_path = ?"
                            + ordered_suffix
                            + ") AS one_path"
                        )
                        path_select_params.extend(
                            (
                                *inner_filter_params,
                                *path_match_params,
                                *path_params,
                                *path_acl_params,
                                path,
                                bounded_path_limit,
                            )
                        )
                sql += (
                    " AND c.record_key IN (SELECT bounded_path.record_key FROM ("
                    + " UNION ALL ".join(path_selects)
                    + ") AS bounded_path)"
                )
                params.extend(path_select_params)
        return sql, params


__all__ = ["BaseFilterBuilder"]
