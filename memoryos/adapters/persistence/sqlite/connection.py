"""SQLite catalog SQLiteConnectionManager responsibility component."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _BOUNDED_FTS_OVERFETCH,
    _MAX_QUERY_LIMIT,
    _ONLINE_PROGRESS_GRANULARITY,
    Any,
    CatalogCandidateBoundExceeded,
    Mapping,
    Sequence,
    sqlite3,
)


class SQLiteConnectionManager:
    """Own one stable subset of SQLite catalog behavior."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._store.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _online_fetchall(
        self,
        conn: sqlite3.Connection,
        sql: str,
        params: Sequence[Any],
    ) -> list[sqlite3.Row]:
        """Execute one serving query under a fixed SQLite VM-step ceiling.

        Indexes make normal Top-K queries stop early.  This final guard covers
        adversarial combinations (for example, a long run of newer expired
        records before an older valid record) without returning a truncated
        result as an empty success.  Migration, repair, audit, and keyset GC
        deliberately use their separate unguarded administrative methods.
        """

        limit = max(_ONLINE_PROGRESS_GRANULARITY, int(self._store.online_vm_step_limit))
        ticks = 0
        interrupted = False

        def progress() -> int:
            nonlocal ticks, interrupted
            ticks += _ONLINE_PROGRESS_GRANULARITY
            if ticks >= limit:
                interrupted = True
                return 1
            return 0

        conn.set_progress_handler(progress, _ONLINE_PROGRESS_GRANULARITY)
        try:
            return conn.execute(sql, list(params)).fetchall()
        except sqlite3.OperationalError as exc:
            if interrupted and "interrupted" in str(exc).casefold():
                raise CatalogCandidateBoundExceeded(f"online Catalog query exceeded {limit} SQLite VM steps") from exc
            raise
        finally:
            conn.set_progress_handler(None, 0)

    def _narrow_online_validity_filters(
        self,
        filters: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Use the RTree as a sparse validity-first candidate driver.

        Up to a fixed threshold, the returned record identities are the
        complete tenant/ACL-valid set and can safely constrain the normal SQL
        before Top-K.  Dense valid sets fall back to the access/time index,
        which stops early.  Thus both the common dense case and the adversarial
        "many newer expired, one older valid" case remain bounded.
        """

        narrowed = dict(filters)
        if filters.get("valid_at") is None:
            return narrowed
        valid_point = self._store._timestamp_number(str(filters["valid_at"]), lower=True)
        tenants = self._store._filter_values(filters["tenant_id"]) if filters.get("tenant_id") is not None else []
        sql = (
            "SELECT vm.record_key FROM context_validity_rtree AS vr "
            "CROSS JOIN context_validity_map AS vm "
            "CROSS JOIN contexts AS vc INDEXED BY idx_contexts_record_key "
            "WHERE vm.validity_id = vr.validity_id "
            "AND vc.record_key = vm.record_key "
            "AND vr.valid_from_min <= ? AND vr.valid_to_max > ?"
        )
        params: list[Any] = [valid_point, valid_point]
        if tenants:
            ranges = " OR ".join(
                "(vr.tenant_min <= (SELECT tenant_key FROM context_tenants WHERE tenant_id = ?) "
                "AND vr.tenant_max >= (SELECT tenant_key FROM context_tenants WHERE tenant_id = ?))"
                for _ in tenants
            )
            sql += f" AND ({ranges})"
            for tenant in tenants:
                params.extend((tenant, tenant))
            sql += f" AND vm.tenant_id IN ({','.join('?' for _ in tenants)})"
            params.extend(tenants)
        valid_at = str(filters["valid_at"])
        sql += " AND vc.valid_from <= ? AND (vc.valid_to = '' OR vc.valid_to > ?)"
        params.extend((valid_at, valid_at))

        principal_owner = filters.get("principal_owner_id")
        public_only = bool(
            principal_owner is None and filters.get("owner_user_id") == "" and filters.get("require_unscoped")
        )
        workspace_access = tuple(self._store._filter_values(filters.get("workspace_access_ids", ()), allow_empty=True))
        shared_workspaces = tuple(
            value for value in workspace_access if value not in {"", "__memoryos_principal_only__"}
        )
        access_predicates: list[str] = []
        access_params: list[Any] = []
        if principal_owner is not None:
            access_predicates.extend(("(vg.grant_kind = 'principal' AND vg.grant_id = ?)", "vg.grant_kind = 'tenant'"))
            access_params.append(str(principal_owner))
            if filters.get("service_access_id"):
                access_predicates.append("(vg.grant_kind = 'service' AND vg.grant_id = ?)")
                access_params.append(str(filters["service_access_id"]))
            if shared_workspaces:
                access_predicates.append(
                    "(vg.grant_kind = 'workspace' AND vg.grant_id IN ("
                    + ",".join("?" for _ in shared_workspaces)
                    + "))"
                )
                access_params.extend(shared_workspaces)
            access_predicates.append("vg.grant_kind = 'public'")
        elif public_only:
            access_predicates.append("vg.grant_kind = 'public'")
        if access_predicates:
            workspace_sql = ""
            if filters.get("workspace_access_ids") is not None:
                workspace_sql = " AND vg.workspace_id IN (" + ",".join("?" for _ in workspace_access) + ")"
                access_params.extend(workspace_access)
            sql += (
                " AND EXISTS (SELECT 1 FROM context_acl_grants AS vg "
                "INDEXED BY idx_context_acl_grants_record "
                "WHERE vg.tenant_id = vc.tenant_id AND vg.record_key = vc.record_key AND ("
                + " OR ".join(access_predicates)
                + ")"
                + workspace_sql
                + ")"
            )
            params.extend(access_params)
        threshold = min(_BOUNDED_FTS_OVERFETCH, _MAX_QUERY_LIMIT)
        sql += " LIMIT ?"
        params.append(threshold + 1)
        with self._store._connect() as conn:
            rows = self._store._online_fetchall(conn, sql, params)
        record_keys = tuple(dict.fromkeys(str(row["record_key"]) for row in rows))
        if not record_keys:
            return None
        if len(record_keys) > threshold:
            return narrowed
        existing = filters.get("record_keys")
        if existing is not None:
            allowed = set(self._store._filter_values(existing, allow_empty=True))
            record_keys = tuple(key for key in record_keys if key in allowed)
            if not record_keys:
                return None
        narrowed["record_keys"] = record_keys
        return narrowed


__all__ = ["SQLiteConnectionManager"]
