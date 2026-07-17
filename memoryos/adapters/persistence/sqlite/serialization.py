"""SQLite catalog CatalogSerializer responsibility component."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _CONTEXT_COLUMNS,
    _INVALID_SCOPE_KEY,
    _MAX_FILTER_VALUES,
    _MAX_FTS_METADATA_TEXT,
    _MAX_QUERY_LIMIT,
    _MAX_SCOPE_KEYS_PER_RECORD,
    _MAX_SCOPE_SIGNATURE_OPTIONS,
    _SAFE_FTS_METADATA_KEYS,
    Any,
    CatalogCandidateBoundExceeded,
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    Mapping,
    Path,
    Sequence,
    ServingTier,
    _path_ancestors,
    combinations,
    datetime,
    hashlib,
    json,
    lexical_match_count,
    lexical_relevance,
    lexical_terms,
    math,
    normalize_tree_path,
    sqlite3,
    timezone,
)
from memoryos.core.types import ContextScope, scope_key_from_payload


class CatalogSerializer:
    """Own one stable subset of SQLite catalog behavior."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def _catalog_record_from_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> CatalogRecord:
        paths = tuple(
            str(item["path"])
            for item in conn.execute(
                "SELECT path FROM context_paths WHERE tenant_id = ? AND record_key = ? ORDER BY is_primary DESC, path",
                (str(row["tenant_id"]), str(row["record_key"])),
            ).fetchall()
        )
        record_kind = str(row["record_kind"])
        if record_kind not in {kind.value for kind in CatalogRecordKind}:
            record_kind = CatalogRecordKind.CONTEXT.value
        serving_tier = str(row["serving_tier"]).upper()
        if serving_tier not in {tier.value for tier in ServingTier}:
            serving_tier = ServingTier.HOT.value
        projection_status = str(row["projection_status"]).upper()
        if projection_status not in {status.value for status in CatalogProjectionStatus}:
            projection_status = CatalogProjectionStatus.FAILED.value
        return CatalogRecord(
            record_key=str(row["record_key"]),
            uri=str(row["uri"]),
            tenant_id=str(row["tenant_id"]),
            owner_user_id=str(row["owner_user_id"]),
            workspace_id=str(row["workspace_id"]),
            session_id=str(row["session_id"]),
            adapter_id=str(row["adapter_id"]),
            context_type=str(row["context_type"]),
            source_kind=str(row["source_kind"]),
            record_kind=record_kind,
            lifecycle_state=str(row["lifecycle_state"]),
            parent_uri=str(row["parent_uri"]),
            primary_tree_path=str(row["primary_tree_path"]),
            tree_paths=paths,
            created_at=self._store._coerce_timestamp(str(row["created_at"])),
            updated_at=self._store._coerce_timestamp(str(row["updated_at"])),
            event_time=self._store._coerce_timestamp(str(row["event_time"])),
            ingested_at=self._store._coerce_timestamp(str(row["ingested_at"])),
            transaction_time=self._store._coerce_timestamp(str(row["transaction_time"])),
            valid_from=self._store._coerce_timestamp(str(row["valid_from"])),
            valid_to=self._store._coerce_timestamp(str(row["valid_to"])),
            title=str(row["title"]),
            l0_text=str(row["l0_text"]),
            l1_text=str(row["l1_text"]),
            l2_uri=str(row["l2_uri"]),
            source_uri=str(row["source_uri"]),
            source_digest=str(row["source_digest"]),
            source_revision=int(row["source_revision"]),
            canonical_slot_id=str(row["canonical_slot_id"]),
            canonical_slot_uri=str(row["canonical_slot_uri"]),
            canonical_claim_id=str(row["canonical_claim_id"]),
            canonical_claim_uri=str(row["canonical_claim_uri"]),
            canonical_revision=int(row["canonical_revision"]),
            canonical_state=str(row["canonical_state"]),
            canonical_head_digest=str(row["canonical_head_digest"]),
            receipt_digest=str(row["receipt_digest"]),
            projection_effect_hash=str(row["projection_effect_hash"]),
            hotness=float(row["hotness"]),
            semantic_hotness=float(row["semantic_hotness"]),
            behavior_support_hotness=float(row["behavior_support_hotness"]),
            serving_tier=serving_tier,
            projection_status=projection_status,
            metadata=self._store._json_mapping(row["metadata_json"]),
        )

    def _catalog_records_from_rows(
        self,
        conn: sqlite3.Connection,
        rows: Sequence[sqlite3.Row],
    ) -> list[CatalogRecord]:
        return [self._store._catalog_record_from_row(conn, row) for row in rows]

    def _scope_keys_from_metadata(self, metadata: Mapping[str, Any]) -> list[str]:
        if metadata.get("canonical_kind") == "current_slot_projection":
            explicit = metadata.get("scope_keys")
            if not isinstance(explicit, list | tuple) or any(
                not isinstance(item, str) or not item or len(item) > 1_000 for item in explicit
            ):
                raise ValueError("Current Slot projection scope keys are invalid")
            return list(dict.fromkeys(explicit))
        if metadata.get("canonical_kind") in {"claim", "slot", "pending_proposal"}:
            raw_scope = metadata.get("scope")
            if not isinstance(raw_scope, Mapping):
                raise ValueError("canonical scope must be an object")
            canonical_scope = ContextScope.from_dict(raw_scope)
            if canonical_scope.canonical_subject is None:
                raise ValueError("canonical scope requires a subject")
            return [scope.key for scope in canonical_scope.applicability.all_of]
        raw_scope = metadata.get("scope")
        if raw_scope is None:
            return []
        if not isinstance(raw_scope, Mapping):
            raise ValueError("scope must be an object")
        raw_applicability = raw_scope.get("applicability")
        if raw_applicability is None:
            return []
        if not isinstance(raw_applicability, Mapping):
            raise ValueError("scope applicability must be an object")
        items = raw_applicability.get("all_of", [])
        if not isinstance(items, list | tuple) or any(not isinstance(item, Mapping) for item in items):
            raise ValueError("scope applicability must contain scope objects")
        return list(dict.fromkeys(scope_key_from_payload(item) for item in items))

    def _safe_metadata_text(self, metadata: Mapping[str, Any]) -> str:
        values: list[str] = []

        def collect(value: Any) -> None:
            if value is None or isinstance(value, bool):
                return
            if isinstance(value, str | int | float):
                values.append(str(value))
                return
            if isinstance(value, Mapping):
                for nested in value.values():
                    collect(nested)
                return
            if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
                for nested in value:
                    collect(nested)

        for key in _SAFE_FTS_METADATA_KEYS:
            if key in metadata:
                collect(metadata[key])
        text = " ".join(values)[:_MAX_FTS_METADATA_TEXT]
        self._store.sanitizer.assert_safe(text)
        return text

    def _safe_reference_uri(self, value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        if raw.startswith("file://") or Path(raw).is_absolute():
            name, location = self._store.sanitizer.sanitize_path(raw)
            return f"resource://{location or 'external'}/{name}" if name else "resource://redacted"
        return str(self._store.sanitizer.sanitize_trace(raw))

    def _restore_internal_projection_path(self, metadata: dict[str, Any]) -> None:
        """Recreate a canonical control path for the legacy verifier, never for FTS."""

        claimed_path = metadata.get("projection_record_path")
        if not isinstance(claimed_path, Mapping) and Path(str(claimed_path or "")).is_absolute():
            return
        claim_uri = str(metadata.get("claim_uri") or "")
        attempt_id = str(metadata.get("projection_attempt_id") or "")
        revision = metadata.get("projection_source_revision")
        if (
            not claim_uri
            or not attempt_id
            or revision is None
            or any(character not in "0123456789abcdef" for character in attempt_id.casefold())
        ):
            return
        try:
            source_revision = int(revision)
        except (TypeError, ValueError):
            return
        claim_digest = hashlib.sha256(claim_uri.encode("utf-8")).hexdigest()
        artifact_root = self._store.path.parent.parent
        metadata["projection_record_path"] = str(
            artifact_root
            / "system"
            / "projection-state"
            / claim_digest[:2]
            / claim_digest
            / "revisions"
            / f"rev-{source_revision}"
            / f"attempt-{attempt_id}.json"
        )

    def _match_query(self, query: str) -> str:
        escaped = [f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in lexical_terms(query)]
        return " OR ".join(escaped)

    def _acl_bound_fts_query(self, match_query: str, filters: Mapping[str, Any]) -> str:
        content_query = f"{{title content_text metadata_text search_terms}} : ({match_query})"
        principal = filters.get("principal_owner_id")
        tenants = self._store._filter_values(filters.get("tenant_id"), allow_empty=True)
        if not tenants:
            raise ValueError("ACL-bound FTS queries require tenant_id")
        tenant_id = str(tenants[0])
        workspace_values = tuple(
            str(item) for item in self._store._filter_values(filters.get("workspace_access_ids", ()), allow_empty=True)
        )
        workspace_constrained = filters.get("workspace_access_ids") is not None
        acl_tokens: list[str] = []
        if principal is None:
            if filters.get("owner_user_id") == "" and filters.get("require_unscoped"):
                for workspace_id in workspace_values if workspace_constrained else ("*",):
                    acl_tokens.append(self._store._grant_acl_token(tenant_id, "public", "", workspace_id))
            else:
                raw_record_keys = filters.get("record_keys")
                if raw_record_keys is None:
                    return content_query
                record_tokens = [
                    self._store._acl_token(tenant_id, "record_key", value)
                    for value in self._store._filter_values(raw_record_keys)
                ]
                return "acl_tokens : (" + " OR ".join(record_tokens) + f") AND ({content_query})"
        else:
            visible_workspaces = workspace_values if workspace_constrained else ("*",)
            for workspace_id in visible_workspaces:
                acl_tokens.append(self._store._grant_acl_token(tenant_id, "principal", str(principal), workspace_id))
                acl_tokens.append(self._store._grant_acl_token(tenant_id, "public", "", workspace_id))
                acl_tokens.append(self._store._grant_acl_token(tenant_id, "tenant", "", workspace_id))
                if filters.get("service_access_id"):
                    acl_tokens.append(
                        self._store._grant_acl_token(
                            tenant_id,
                            "service",
                            str(filters["service_access_id"]),
                            workspace_id,
                        )
                    )
            for workspace_id in workspace_values:
                if workspace_id not in {"", "__memoryos_principal_only__"}:
                    acl_tokens.append(self._store._grant_acl_token(tenant_id, "workspace", workspace_id, workspace_id))
        clauses = ["acl_tokens : (" + " OR ".join(dict.fromkeys(acl_tokens)) + ")"]
        raw_record_keys = filters.get("record_keys")
        if raw_record_keys is not None:
            record_tokens = [
                self._store._acl_token(tenant_id, "record_key", value)
                for value in self._store._filter_values(raw_record_keys)
            ]
            clauses.append("acl_tokens : (" + " OR ".join(record_tokens) + ")")
        for filter_name, token_kind in (
            ("context_types", "context_type"),
            ("source_kinds", "source_kind"),
            ("session_ids", "session_id"),
            ("target_uris", "uri"),
        ):
            raw_values = filters.get(filter_name)
            if raw_values is None:
                continue
            field_tokens = [
                self._store._acl_token(tenant_id, token_kind, value) for value in self._store._filter_values(raw_values)
            ]
            clauses.append("acl_tokens : (" + " OR ".join(field_tokens) + ")")
        raw_paths = filters.get("target_paths", filters.get("path_prefixes"))
        if raw_paths is not None:
            path_tokens = [
                self._store._acl_token(tenant_id, "tree_path", normalize_tree_path(value))
                for value in self._store._filter_values(raw_paths)
            ]
            clauses.append("acl_tokens : (" + " OR ".join(path_tokens) + ")")
        raw_kinds = filters.get("record_kinds", filters.get("record_kind"))
        if raw_kinds is not None:
            kind_tokens = [
                self._store._acl_token(tenant_id, "record_kind", value)
                for value in self._store._filter_values(raw_kinds)
            ]
            clauses.append("acl_tokens : (" + " OR ".join(kind_tokens) + ")")
        if filters.get("adapter_id") is not None:
            adapter_tokens = [
                self._store._acl_token(tenant_id, "adapter", value)
                for value in self._store._filter_values(filters["adapter_id"])
            ]
            clauses.append("acl_tokens : (" + " OR ".join(adapter_tokens) + ")")
        if filters.get("adapter_access_id") is not None:
            adapter_access_tokens = [self._store._acl_token(tenant_id, "adapter_access", "*")]
            adapter_access_tokens.extend(
                self._store._acl_token(tenant_id, "adapter_access", value)
                for value in self._store._filter_values(filters["adapter_access_id"])
            )
            clauses.append("acl_tokens : (" + " OR ".join(adapter_access_tokens) + ")")
        raw_scopes = filters.get("applicability_scope_keys")
        if raw_scopes:
            scope_tokens = [
                self._store._acl_token(tenant_id, "scope_signature", signature)
                for signature in self._store._scope_signature_options(self._store._filter_values(raw_scopes))
            ]
            clauses.append("acl_tokens : (" + " OR ".join(scope_tokens) + ")")
        elif filters.get("require_unscoped"):
            clauses.append(
                "acl_tokens : ("
                + self._store._acl_token(tenant_id, "scope_signature", self._store._scope_signature(()))
                + ")"
            )
        clauses.append(f"({content_query})")
        return " AND ".join(clauses)

    def _fts_acl_tokens(self, record: CatalogRecord, *, scope_signature: str) -> str:
        tokens: list[str] = []
        for grant_kind, grant_id, workspace_id in self._store._acl_grants_for_record(record):
            tokens.append(self._store._grant_acl_token(record.tenant_id, grant_kind, grant_id, workspace_id))
            tokens.append(self._store._grant_acl_token(record.tenant_id, grant_kind, grant_id, "*"))
        tokens.append(self._store._acl_token(record.tenant_id, "record_kind", record.record_kind))
        tokens.append(self._store._acl_token(record.tenant_id, "record_key", record.record_key))
        tokens.append(self._store._acl_token(record.tenant_id, "context_type", record.context_type))
        tokens.append(self._store._acl_token(record.tenant_id, "source_kind", record.source_kind))
        tokens.append(self._store._acl_token(record.tenant_id, "session_id", record.session_id))
        tokens.append(self._store._acl_token(record.tenant_id, "uri", record.uri))
        for path in record.tree_paths:
            tokens.extend(
                self._store._acl_token(record.tenant_id, "tree_path", ancestor) for ancestor in _path_ancestors(path)
            )
        tokens.append(self._store._acl_token(record.tenant_id, "adapter", record.adapter_id))
        tokens.append(
            self._store._acl_token(record.tenant_id, "adapter_access", self._store._adapter_access_value(record))
        )
        tokens.append(self._store._acl_token(record.tenant_id, "scope_signature", scope_signature))
        return " ".join(dict.fromkeys(tokens))

    def _grant_acl_token(self, tenant_id: str, grant_kind: str, grant_id: str, workspace_id: str) -> str:
        return self._store._acl_token(tenant_id, f"grant:{grant_kind}:{workspace_id}", grant_id)

    @staticmethod
    def _acl_token(tenant_id: str, scope_kind: str, scope_id: str) -> str:
        payload = "\x00".join((str(tenant_id), str(scope_kind), str(scope_id)))
        return "acl" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _filter_values(self, value: Any, *, allow_empty: bool = False) -> list[str]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence | set | frozenset):
            values = [value]
        else:
            values = list(value)
        if not values and not allow_empty:
            raise ValueError("structured filter values cannot be empty")
        if len(values) > _MAX_FILTER_VALUES:
            raise ValueError("structured filter exceeds the bounded value limit")
        return [str(item) for item in values]

    @staticmethod
    def _scope_signature(scope_keys: Sequence[str]) -> str:
        normalized = tuple(sorted(dict.fromkeys(str(item) for item in scope_keys)))
        payload = "\x00".join(normalized)
        return "scope" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _scope_signature_options(self, available_scope_keys: Sequence[str]) -> tuple[str, ...]:
        available = tuple(sorted(dict.fromkeys(str(item) for item in available_scope_keys)))
        signatures: list[str] = []
        maximum_size = min(len(available), _MAX_SCOPE_KEYS_PER_RECORD)
        for size in range(maximum_size + 1):
            for subset in combinations(available, size):
                signatures.append(self._store._scope_signature(subset))
                if len(signatures) > _MAX_SCOPE_SIGNATURE_OPTIONS:
                    raise CatalogCandidateBoundExceeded(
                        "authorized scope combinations exceed the bounded online query plan"
                    )
        return tuple(signatures)

    def _bounded_limit(self, limit: int) -> int:
        return min(max(1, int(limit)), _MAX_QUERY_LIMIT)

    def _lexical_relevance(self, query: str, haystack: str) -> float:
        return lexical_relevance(query, haystack)

    def _lexical_match_count(self, query: str, haystack: str) -> float:
        return float(lexical_match_count(query, haystack))

    def _bounded(self, value: Any) -> float:
        if isinstance(value, bool):
            return 0.0
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(number):
            return 0.0
        return max(0.0, min(1.0, number))

    def _finite_rank(self, value: Any) -> float:
        if isinstance(value, bool):
            return 0.0
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(number) or number < 0:
            return 0.0
        return number

    def _coerce_record(self, value: CatalogRecord | Mapping[str, Any]) -> CatalogRecord:
        if isinstance(value, CatalogRecord):
            return value
        if isinstance(value, Mapping):
            return CatalogRecord(**dict(value))
        raise TypeError("catalog records must be CatalogRecord or mapping")

    def _insert_context_row(
        self,
        conn: sqlite3.Connection,
        values: Mapping[str, Any],
        *,
        table_name: str,
    ) -> None:
        columns = ", ".join(_CONTEXT_COLUMNS)
        placeholders = ", ".join("?" for _ in _CONTEXT_COLUMNS)
        conn.execute(
            f"INSERT INTO {table_name}({columns}) VALUES ({placeholders})",
            tuple(values[column] for column in _CONTEXT_COLUMNS),
        )

    def _delete_fts_record(self, conn: sqlite3.Connection, record_key: str) -> None:
        mapping = conn.execute(
            "SELECT fts_rowid FROM context_fts_map WHERE record_key = ?",
            (str(record_key),),
        ).fetchone()
        if mapping is None:
            return
        fts_rowid = int(mapping["fts_rowid"])
        current = conn.execute(
            "SELECT record_key FROM contexts_fts WHERE rowid = ?",
            (fts_rowid,),
        ).fetchone()
        if current is None or str(current["record_key"]) != str(record_key):
            raise RuntimeError("FTS rowid map failed integrity validation")
        conn.execute("DELETE FROM contexts_fts WHERE rowid = ?", (fts_rowid,))
        conn.execute(
            "DELETE FROM context_fts_map WHERE record_key = ? AND fts_rowid = ?",
            (str(record_key), fts_rowid),
        )

    @staticmethod
    def _content_digest(content: str) -> str:
        encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _json_dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    def _json_mapping(self, value: Any) -> dict[str, Any]:
        try:
            decoded = json.loads(str(value or "{}"))
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        return self._store._mapping(decoded)

    @staticmethod
    def _json_list(value: Any) -> list[str]:
        try:
            decoded = json.loads(str(value or "[]"))
        except (json.JSONDecodeError, TypeError, ValueError):
            return [_INVALID_SCOPE_KEY]
        if not isinstance(decoded, list):
            return [_INVALID_SCOPE_KEY]
        return [str(item) for item in decoded]

    @staticmethod
    def _safe_exact_value(value: Any) -> str:
        return str(value) if isinstance(value, str | int | float) and not isinstance(value, bool) else ""

    @staticmethod
    def _legacy_value(row: sqlite3.Row, columns: set[str], name: str) -> str:
        return str(row[name] if name in columns and row[name] is not None else "")

    @staticmethod
    def _coerce_timestamp(value: str) -> str:
        raw = str(value or "")
        if not raw:
            return ""
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return ""
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _timestamp_number(value: str, *, lower: bool) -> float:
        """Map a normalized timestamp to an RTree-safe interval endpoint."""

        raw = str(value or "")
        if not raw:
            return -1.0e12 if lower else 1.0e12
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("validity timestamps must be ISO-8601") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).timestamp()

    @staticmethod
    def _tenant_rtree_key(conn: sqlite3.Connection, tenant_id: str) -> int:
        conn.execute(
            "INSERT INTO context_tenants(tenant_id) VALUES (?) ON CONFLICT(tenant_id) DO NOTHING",
            (str(tenant_id),),
        )
        row = conn.execute("SELECT tenant_key FROM context_tenants WHERE tenant_id = ?", (str(tenant_id),)).fetchone()
        if row is None:
            raise RuntimeError("failed to allocate a collision-free tenant validity key")
        return int(row["tenant_key"])

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _row_dict(row: sqlite3.Row, *, json_fields: Sequence[str] = ()) -> dict[str, Any]:
        result = {str(key): row[key] for key in row.keys()}
        for field in json_fields:
            try:
                result[field] = json.loads(str(result.get(field) or "{}"))
            except (json.JSONDecodeError, TypeError, ValueError):
                result[field] = {}
        return result


__all__ = ["CatalogSerializer"]
