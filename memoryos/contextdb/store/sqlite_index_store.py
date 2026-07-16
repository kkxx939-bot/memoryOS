"""SQLite-backed, rebuildable Unified Context Catalog serving index."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from memoryos.contextdb.catalog import (
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    ServingTier,
    normalize_tree_path,
)
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.retrieval.errors import CatalogCandidateBoundExceeded
from memoryos.contextdb.store.source_store import IndexHit
from memoryos.security.context_projection import ContextProjectionSanitizer
from memoryos.security.workspace_identity import normalize_workspace_id

_SCOPE_KEY_SCHEMA_VERSION = 2
_CATALOG_SCHEMA_VERSION = 10
_INVALID_SCOPE_KEY = "__memoryos_invalid_scope__"
_MAX_FILTER_VALUES = 900
_MAX_QUERY_LIMIT = 1_000
_MAX_TARGET_PATHS = 16
_BOUNDED_FTS_OVERFETCH = 256
_MIGRATION_BATCH_SIZE = 256
_UNIFIED_CATALOG_MIGRATION_NAME = "unified-context-catalog-v1"
_SCHEMA_UPGRADE_BOOTSTRAP_TENANT = "__memoryos_schema_upgrade__"
_GREENFIELD_CATALOG_ORIGIN_NAME = "__memoryos_greenfield_catalog_v10__"
_MAX_FTS_METADATA_TEXT = 4_000
_MAX_SCOPE_KEYS_PER_RECORD = 8
_MAX_SCOPE_SIGNATURE_OPTIONS = 256
_FTS_BM25 = "bm25(contexts_fts, 0.0, 0.0, 5.0, 4.0, 2.0, 1.0, 0.0)"
_FTS_RANK_CONFIG = "bm25(0.0, 0.0, 5.0, 4.0, 2.0, 1.0, 0.0)"
_ONLINE_VM_STEP_LIMIT = 1_000_000
_ONLINE_PROGRESS_GRANULARITY = 1_000
_MIGRATION_STATES = frozenset(
    {
        "NOT_STARTED",
        "SCHEMA_READY",
        "BACKFILLING",
        "DUAL_WRITE",
        "SHADOW_VALIDATING",
        "READY_TO_CUTOVER",
        "CUTOVER",
        "ROLLBACK",
        "COMPLETED",
        "FAILED",
    }
)
_SAFE_FTS_METADATA_KEYS = frozenset(
    {
        "action",
        "dimension",
        "file_name",
        "filename",
        "keywords",
        "memory_type",
        "resource_location",
        "resource_name",
        "scene_key",
        "subject",
        "summary",
        "tags",
        "topic",
    }
)

_CONTEXT_COLUMNS = (
    "record_key",
    "uri",
    "tenant_id",
    "owner_user_id",
    "project_id",
    "workspace_id",
    "workspace_shared",
    "session_id",
    "adapter_id",
    "context_type",
    "source_kind",
    "record_kind",
    "lifecycle_state",
    "admission_status",
    "claim_state",
    "slot_id",
    "memory_type",
    "scope_keys",
    "scope_signature",
    "parent_uri",
    "primary_tree_path",
    "path_depth",
    "created_at",
    "updated_at",
    "event_time",
    "ingested_at",
    "transaction_time",
    "valid_from",
    "valid_to",
    "title",
    "l0_text",
    "l1_text",
    "l2_uri",
    "source_uri",
    "source_digest",
    "source_revision",
    "canonical_slot_id",
    "canonical_slot_uri",
    "canonical_claim_id",
    "canonical_claim_uri",
    "canonical_revision",
    "canonical_state",
    "canonical_head_digest",
    "receipt_digest",
    "projection_effect_hash",
    "hotness",
    "semantic_hotness",
    "behavior_support_hotness",
    "serving_tier",
    "projection_status",
    "metadata_json",
    "content_digest",
    "stored_content_digest",
    "content_text",
    "scene_key",
    "action",
    "memory_anchor_uri",
)

_ALTER_COLUMN_DEFINITIONS = {
    "project_id": "TEXT NOT NULL DEFAULT ''",
    "workspace_id": "TEXT NOT NULL DEFAULT ''",
    "workspace_shared": "INTEGER NOT NULL DEFAULT 0",
    "session_id": "TEXT NOT NULL DEFAULT ''",
    "adapter_id": "TEXT NOT NULL DEFAULT ''",
    "source_kind": "TEXT NOT NULL DEFAULT ''",
    "record_kind": "TEXT NOT NULL DEFAULT 'context'",
    "admission_status": "TEXT NOT NULL DEFAULT ''",
    "claim_state": "TEXT NOT NULL DEFAULT ''",
    "slot_id": "TEXT NOT NULL DEFAULT ''",
    "memory_type": "TEXT NOT NULL DEFAULT ''",
    "scope_keys": "TEXT NOT NULL DEFAULT '[]'",
    "scope_signature": "TEXT NOT NULL DEFAULT ''",
    "parent_uri": "TEXT NOT NULL DEFAULT ''",
    "primary_tree_path": "TEXT NOT NULL DEFAULT ''",
    "path_depth": "INTEGER NOT NULL DEFAULT 0",
    "created_at": "TEXT NOT NULL DEFAULT ''",
    "event_time": "TEXT NOT NULL DEFAULT ''",
    "ingested_at": "TEXT NOT NULL DEFAULT ''",
    "transaction_time": "TEXT NOT NULL DEFAULT ''",
    "valid_from": "TEXT NOT NULL DEFAULT ''",
    "valid_to": "TEXT NOT NULL DEFAULT ''",
    "l0_text": "TEXT NOT NULL DEFAULT ''",
    "l1_text": "TEXT NOT NULL DEFAULT ''",
    "l2_uri": "TEXT NOT NULL DEFAULT ''",
    "source_uri": "TEXT NOT NULL DEFAULT ''",
    "source_digest": "TEXT NOT NULL DEFAULT ''",
    "source_revision": "INTEGER NOT NULL DEFAULT 0",
    "canonical_slot_id": "TEXT NOT NULL DEFAULT ''",
    "canonical_slot_uri": "TEXT NOT NULL DEFAULT ''",
    "canonical_claim_id": "TEXT NOT NULL DEFAULT ''",
    "canonical_claim_uri": "TEXT NOT NULL DEFAULT ''",
    "canonical_revision": "INTEGER NOT NULL DEFAULT 0",
    "canonical_state": "TEXT NOT NULL DEFAULT ''",
    "canonical_head_digest": "TEXT NOT NULL DEFAULT ''",
    "receipt_digest": "TEXT NOT NULL DEFAULT ''",
    "projection_effect_hash": "TEXT NOT NULL DEFAULT ''",
    "serving_tier": "TEXT NOT NULL DEFAULT 'HOT'",
    "projection_status": "TEXT NOT NULL DEFAULT 'PROJECTED'",
    "content_digest": "TEXT NOT NULL DEFAULT ''",
    "stored_content_digest": "TEXT NOT NULL DEFAULT ''",
    "scene_key": "TEXT NOT NULL DEFAULT ''",
    "action": "TEXT NOT NULL DEFAULT ''",
    "memory_anchor_uri": "TEXT NOT NULL DEFAULT ''",
}

_SIMPLE_FILTER_FIELDS = {
    "record_key": "record_key",
    "tenant_id": "tenant_id",
    "owner_user_id": "owner_user_id",
    "workspace_id": "workspace_id",
    "session_id": "session_id",
    "adapter_id": "adapter_id",
    "context_type": "context_type",
    "source_kind": "source_kind",
    "record_kind": "record_kind",
    "lifecycle_state": "lifecycle_state",
    "admission_status": "admission_status",
    "claim_state": "claim_state",
    "slot_id": "slot_id",
    "canonical_slot_id": "canonical_slot_id",
    "canonical_claim_id": "canonical_claim_id",
    "canonical_state": "canonical_state",
    "memory_type": "memory_type",
    "serving_tier": "serving_tier",
    "projection_status": "projection_status",
    "scene_key": "scene_key",
    "action": "action",
    "memory_anchor_uri": "memory_anchor_uri",
}

_PLURAL_FILTER_ALIASES = {
    "record_keys": "record_key",
    "target_uris": "uri",
    "workspace_ids": "workspace_id",
    "workspace_access_ids": "workspace_id",
    "session_ids": "session_id",
    "context_types": "context_type",
    "source_kinds": "source_kind",
    "source_uris": "source_uri",
    "record_kinds": "record_kind",
    "canonical_slot_ids": "canonical_slot_id",
    "canonical_claim_ids": "canonical_claim_id",
}


def lexical_terms(text: object) -> tuple[str, ...]:
    """Return deterministic complete Latin tokens and CJK character bigrams."""

    normalized = str(text).casefold()
    terms = re.findall(r"[a-z0-9_]+", normalized)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(sequence) == 1:
            terms.append(sequence)
        else:
            terms.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tuple(dict.fromkeys(term for term in terms if term))


def lexical_match_count(query: object, haystack: object) -> int:
    query_terms = lexical_terms(query)
    if not query_terms:
        return 0
    haystack_terms = set(lexical_terms(haystack))
    return sum(1 for term in query_terms if term in haystack_terms)


def lexical_relevance(query: object, haystack: object) -> float:
    query_terms = lexical_terms(query)
    if not query_terms:
        return 0.0
    return lexical_match_count(query, haystack) / len(query_terms)


def _path_ancestors(path: str) -> tuple[str, ...]:
    """Return the bounded taxonomy closure used by online prefix queries."""

    segments = path.split("/")
    return tuple("/".join(segments[:depth]) for depth in range(1, len(segments) + 1))


@dataclass(frozen=True)
class _PreparedCatalogRecord:
    record: CatalogRecord
    values: dict[str, Any]
    scope_signature: str
    fts_metadata_text: str
    fts_search_terms: str


class SQLiteIndexStore:
    """Serve the catalog from SQLite while keeping SourceStore authoritative."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fts_enabled = True
        self.online_vm_step_limit = _ONLINE_VM_STEP_LIMIT
        self.sanitizer = ContextProjectionSanitizer()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self._init_db()
        os.chmod(self.path, 0o600)

    # ------------------------------------------------------------------
    # Compatibility API

    def upsert_index(self, obj: ContextObject, content: str = "") -> None:
        """Project a legacy ContextObject through the same sanitized catalog writer."""

        record = CatalogRecord.from_context_object(obj, content=content)
        # Legacy callers pass the searchable L2 projection explicitly.  Keep it
        # searchable even when metadata also carries a shorter L1 summary.
        if content:
            record = replace(record, l1_text=content)
        self.upsert_catalog(record)

    def delete_index(self, uri: str) -> None:
        """Delete every serving record for a legacy URI without touching SourceStore."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tenant_id, record_key FROM contexts WHERE uri = ?",
                (str(uri),),
            ).fetchall()
            for row in rows:
                self._delete_catalog_in_transaction(
                    conn,
                    str(row["record_key"]),
                    tenant_id=str(row["tenant_id"]),
                )

    def indexed_uris(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT uri FROM contexts ORDER BY uri").fetchall()
        return [str(row["uri"]) for row in rows]

    def get_index_metadata(self, uri: str) -> dict[str, Any] | None:
        """Return the legacy record first, then a deterministic projection for the URI."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM contexts WHERE uri = ? "
                "ORDER BY CASE WHEN record_key = uri THEN 0 WHEN record_kind = 'current_slot' THEN 1 ELSE 2 END, "
                "updated_at DESC, record_key LIMIT 1",
                (str(uri),),
            ).fetchone()
        if row is None:
            return None
        metadata = self._json_mapping(row["metadata_json"])
        self._restore_internal_projection_path(metadata)
        return {
            **metadata,
            "record_key": str(row["record_key"]),
            "tenant_id": str(row["tenant_id"]),
            "owner_user_id": str(row["owner_user_id"]),
            "context_type": str(row["context_type"]),
            "claim_state": str(row["claim_state"]),
            "slot_id": str(row["slot_id"]),
            "memory_type": str(row["memory_type"]),
            "index_content_digest": (
                str(row["content_digest"])
                if self._content_digest(str(row["content_text"])) == str(row["stored_content_digest"])
                else self._content_digest(str(row["content_text"]))
            ),
        }

    def ordinary_relation_endpoint_state(
        self,
        uri: str,
        *,
        tenant_id: str,
        session_id: str = "",
    ) -> str:
        """Resolve relation endpoint liveness through durable delete barriers."""

        safe_uri = self._safe_reference_uri(str(uri))
        safe_session_id = str(session_id or "")
        with self._connect() as conn:
            blocked = conn.execute(
                "SELECT 1 FROM context_tombstones WHERE tenant_id = ? "
                "AND ((status IN ('PENDING', 'FAILED', 'CLEANING') AND uri = ?) OR ("
                "? <> '' AND status IN ('PENDING', 'FAILED', 'CLEANING', 'APPLIED') "
                "AND json_extract(payload_json, '$.record_kind') = 'session_delete_barrier' "
                "AND json_extract(payload_json, '$.session_id') = ?)) LIMIT 1",
                (str(tenant_id), safe_uri, safe_session_id, safe_session_id),
            ).fetchone()
            if blocked is not None:
                return "retired"
            row = conn.execute(
                "SELECT lifecycle_state FROM contexts WHERE tenant_id = ? "
                "AND (uri = ? OR source_uri = ? OR (? <> '' AND session_id = ?)) "
                "ORDER BY CASE WHEN lifecycle_state = 'active' THEN 0 ELSE 1 END, updated_at DESC LIMIT 1",
                (str(tenant_id), safe_uri, safe_uri, safe_session_id, safe_session_id),
            ).fetchone()
            if row is not None and str(row["lifecycle_state"]) == "active":
                return "active"
            retired = conn.execute(
                "SELECT 1 FROM context_tombstones WHERE tenant_id = ? AND uri = ? "
                "AND status = 'APPLIED' LIMIT 1",
                (str(tenant_id), safe_uri),
            ).fetchone()
            if retired is not None:
                return "retired"
        if row is None:
            return "missing"
        return "inactive"

    def clear(self) -> None:
        """Clear rebuildable serving data while retaining migration and tombstone journals."""

        with self._connect() as conn:
            conn.execute("DELETE FROM context_links")
            conn.execute("DELETE FROM context_acl_grants")
            conn.execute("DELETE FROM context_path_acl")
            conn.execute("DELETE FROM context_path_closure")
            conn.execute("DELETE FROM context_paths")
            conn.execute("DELETE FROM context_validity_rtree")
            conn.execute("DELETE FROM context_validity_map")
            conn.execute("DELETE FROM context_projection_state")
            conn.execute("DELETE FROM contexts")
            conn.execute("DELETE FROM contexts_fts")
            conn.execute("DELETE FROM context_fts_map")

    def begin_tenant_serving_rebuild(
        self,
        migration_name: str,
        *,
        tenant_id: str,
        batch_size: int,
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically gate and clear one tenant's rebuildable serving rows.

        The durable gate and destructive Catalog mutation share one SQLite
        transaction.  A process crash therefore observes either the old
        serving snapshot or an explicit BACKFILLING row that startup can
        resume; it can never observe a cleared Catalog with a COMPLETED gate.
        Tombstones, migration journals, Session frontiers and immutable Source
        evidence are intentionally retained.
        """

        if not migration_name or not tenant_id:
            raise ValueError("tenant serving rebuild requires migration_name and tenant_id")
        bounded_batch_size = int(batch_size)
        if not 1 <= bounded_batch_size <= _MAX_QUERY_LIMIT:
            raise ValueError("tenant serving rebuild batch_size must be between 1 and 1000")
        safe_details = self.sanitizer.sanitize_trace(dict(details))
        if not isinstance(safe_details, Mapping) or not str(safe_details.get("rebuild_epoch") or ""):
            raise ValueError("tenant serving rebuild requires a sanitized rebuild_epoch")
        now = self._now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM migration_state WHERE migration_name = ? AND tenant_id = ?",
                (str(migration_name), str(tenant_id)),
            ).fetchone()
            if existing is not None and str(existing["state"]) in {
                "BACKFILLING",
                "FAILED",
                "ROLLBACK",
            }:
                return self._row_dict(existing, json_fields=("details_json",))

            record_keys = tuple(
                str(row["record_key"])
                for row in conn.execute(
                    "SELECT record_key FROM contexts WHERE tenant_id = ? ORDER BY record_key",
                    (str(tenant_id),),
                ).fetchall()
            )
            prepared_details = {
                **dict(safe_details),
                "catalog_cleared": True,
                "cleared_records": len(record_keys),
                "phase": "VECTOR_CLEANUP",
            }
            conn.execute(
                """
                INSERT INTO migration_state(
                  migration_name, tenant_id, state, checkpoint, batch_size,
                  details_json, last_error, updated_at
                ) VALUES (?, ?, 'BACKFILLING', '', ?, ?, '', ?)
                ON CONFLICT(migration_name, tenant_id) DO UPDATE SET
                  state='BACKFILLING', checkpoint='', batch_size=excluded.batch_size,
                  details_json=excluded.details_json, last_error='', updated_at=excluded.updated_at
                """,
                (
                    str(migration_name),
                    str(tenant_id),
                    bounded_batch_size,
                    self._json_dump(prepared_details),
                    now,
                ),
            )
            for record_key in record_keys:
                self._delete_catalog_in_transaction(
                    conn,
                    record_key,
                    tenant_id=str(tenant_id),
                )
            persisted = conn.execute(
                "SELECT * FROM migration_state WHERE migration_name = ? AND tenant_id = ?",
                (str(migration_name), str(tenant_id)),
            ).fetchone()
        if persisted is None:  # pragma: no cover - the transaction either commits or raises.
            raise RuntimeError("tenant serving rebuild gate did not persist")
        return self._row_dict(persisted, json_fields=("details_json",))

    def rebuildable_catalog_records(
        self,
        records: Sequence[CatalogRecord],
    ) -> tuple[CatalogRecord, ...]:
        """Filter an offline rebuild batch through durable delete ownership.

        APPLIED tombstones suppress the same or an older Source revision.
        CLEANING remains a hard retry boundary because its Vector/Relation
        consumers have not reached a durable terminal state.
        """

        selected: list[CatalogRecord] = []
        with self._connect() as conn:
            for record in records:
                rows = conn.execute(
                    "SELECT source_revision, status FROM context_tombstones "
                    "WHERE tenant_id = ? AND (record_key = ? OR ("
                    "json_extract(payload_json, '$.record_kind') = 'session_delete_barrier' "
                    "AND json_extract(payload_json, '$.session_id') = ?)) "
                    "AND status IN ('CLEANING', 'APPLIED') ORDER BY created_at",
                    (record.tenant_id, record.record_key, record.session_id),
                ).fetchall()
                blocked = False
                for row in rows:
                    status = str(row["status"])
                    if status == "CLEANING":
                        raise RuntimeError("Catalog rebuild is blocked by in-progress tombstone cleanup")
                    tombstone_revision = int(row["source_revision"])
                    if (
                        tombstone_revision == 0
                        or record.source_revision == 0
                        or tombstone_revision >= record.source_revision
                    ):
                        blocked = True
                        break
                if not blocked:
                    selected.append(record)
        return tuple(selected)

    def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
        """Legacy online search; structured filters are applied before every LIMIT."""

        hits = self.search_catalog(query, filters=filters, limit=limit)
        deduplicated: dict[str, IndexHit] = {}
        for hit in hits:
            deduplicated.setdefault(hit.uri, hit)
        return list(deduplicated.values())[: self._bounded_limit(limit)]

    def list_legacy_catalog(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 100,
    ) -> list[CatalogRecord]:
        """Bounded rollback reader over the pre-unification flat Catalog shape.

        This deliberately does not use ACL-grant, closure, relation, vector,
        or validity adjuncts.  It is a conservative owner/public compatibility
        route over the same evolved ``contexts`` table, not a second Catalog.
        """

        normalized = dict(filters or {})
        bounded = self._bounded_limit(limit)
        filter_sql, params = self._legacy_filter_sql(normalized)
        sql = (
            "SELECT c.* FROM contexts AS c WHERE 1=1 "
            + filter_sql
            + " ORDER BY c.updated_at DESC, c.record_key DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = self._online_fetchall(conn, sql, [*params, bounded])
            return self._catalog_records_from_rows(conn, rows)

    def search_legacy_catalog(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[IndexHit]:
        """Run the independently bounded legacy lexical read for shadow/rollback."""

        normalized = dict(filters or {})
        bounded = self._bounded_limit(limit)
        value = str(query).strip()
        filter_sql, params = self._legacy_filter_sql(normalized)
        exact_rows: list[sqlite3.Row] = []
        with self._connect() as conn:
            if value:
                exact_rows = self._online_fetchall(
                    conn,
                    "SELECT c.*, c.title AS fts_title, c.content_text AS fts_content, "
                    "'' AS fts_metadata FROM contexts AS c WHERE "
                    "(c.scene_key = ? OR c.action = ? OR c.memory_anchor_uri = ?) "
                    + filter_sql
                    + " ORDER BY c.updated_at DESC, c.record_key DESC LIMIT ?",
                    [value, value, value, *params, bounded],
                )
        merged: dict[str, IndexHit] = {
            str(row["record_key"]): self._hit_from_row(row, identity=1.0, identity_rank=10.0)
            for row in exact_rows
        }
        if not value or len(merged) >= bounded or not self.fts_enabled:
            return list(merged.values())[:bounded]
        match_query = self._match_query(value)
        if not match_query:
            return list(merged.values())[:bounded]
        overfetch = min(max(bounded * 4, bounded), _BOUNDED_FTS_OVERFETCH)
        sql = f"""
            SELECT c.*, contexts_fts.title AS fts_title,
                   contexts_fts.content_text AS fts_content,
                   contexts_fts.metadata_text AS fts_metadata,
                   contexts_fts.rank AS rank
            FROM contexts_fts
            CROSS JOIN contexts AS c INDEXED BY idx_contexts_record_key
              ON c.record_key = contexts_fts.record_key
            WHERE contexts_fts MATCH ? AND contexts_fts.rank MATCH ? {filter_sql}
            ORDER BY contexts_fts.rank
            LIMIT ?
        """
        with self._connect() as conn:
            rows = self._online_fetchall(
                conn,
                sql,
                [match_query, _FTS_RANK_CONFIG, *params, overfetch],
            )
        rows.sort(key=lambda row: str(row["record_key"]), reverse=True)
        rows.sort(key=lambda row: str(row["updated_at"]), reverse=True)
        rows.sort(key=lambda row: float(row["rank"]))
        for row in rows:
            haystack = " ".join(
                (str(row["fts_title"]), str(row["fts_content"]), str(row["fts_metadata"]))
            )
            lexical = self._lexical_relevance(value, haystack)
            if lexical <= 0:
                continue
            merged.setdefault(
                str(row["record_key"]),
                self._hit_from_row(
                    row,
                    lexical=lexical,
                    lexical_rank=self._lexical_match_count(value, haystack),
                ),
            )
            if len(merged) >= bounded:
                break
        return list(merged.values())[:bounded]

    # ------------------------------------------------------------------
    # Unified Context Catalog API

    def upsert_catalog(self, record: CatalogRecord | Mapping[str, Any]) -> None:
        """Atomically sanitize and upsert one rebuildable catalog record."""

        self.upsert_catalog_batch((record,))

    def upsert_catalog_batch(self, records: Sequence[CatalogRecord | Mapping[str, Any]]) -> int:
        """Atomically project a batch; any validation, sanitization, or write error rolls it back."""

        prepared = tuple(self._prepare_record(self._coerce_record(record)) for record in records)
        if not prepared:
            return 0
        with self._connect() as conn:
            for item in prepared:
                self._upsert_prepared(conn, item)
        return len(prepared)

    def get_catalog(self, record_key: str, *, tenant_id: str | None = None) -> CatalogRecord | None:
        sql = "SELECT * FROM contexts WHERE record_key = ?"
        params: list[Any] = [str(record_key)]
        if tenant_id is not None:
            sql += " AND tenant_id = ?"
            params.append(str(tenant_id))
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
            if row is None:
                return None
            return self._catalog_record_from_row(conn, row)

    def get_catalog_by_uri(
        self,
        uri: str,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[CatalogRecord]:
        filters: dict[str, Any] = {"target_uris": (str(uri),)}
        filters["include_inactive"] = True
        if tenant_id is not None:
            filters["tenant_id"] = str(tenant_id)
        return self.list_catalog(filters=filters, limit=limit)

    def list_catalog(self, *, filters: Mapping[str, Any] | None = None, limit: int = 100) -> list[CatalogRecord]:
        narrowed_filters = self._narrow_online_validity_filters(dict(filters or {}))
        if narrowed_filters is None:
            return []
        normalized_filters = narrowed_filters
        bounded_limit = self._bounded_limit(limit)
        filter_sql, params = self._base_filter_sql(
            normalized_filters,
            path_candidate_limit=min(_MAX_QUERY_LIMIT, max(bounded_limit, bounded_limit * 4)),
        )
        from_sql, index_predicate = self._catalog_from_sql(normalized_filters)
        sql = (
            f"SELECT c.* FROM {from_sql} WHERE 1=1 {filter_sql}{index_predicate} "
            "ORDER BY c.updated_at DESC, c.record_key LIMIT ?"
        )
        with self._connect() as conn:
            rows = self._online_fetchall(conn, sql, [*params, bounded_limit])
            return self._catalog_records_from_rows(conn, rows)

    def list_catalog_projection_records(
        self,
        *,
        tenant_id: str,
        source_uri: str,
        projection_effect_hash: str,
        limit: int = 1_001,
    ) -> list[CatalogRecord]:
        """Read one evidence-bound projection set for offline/shadow proof.

        This exact identity lookup is intentionally separate from online
        search.  The extra row above the 1000-record proof bound lets callers
        fail closed instead of certifying a truncated projection.
        """

        bounded = int(limit)
        if not tenant_id or not source_uri or not projection_effect_hash:
            raise ValueError("projection evidence identity is required")
        if not 1 <= bounded <= 1_001:
            raise ValueError("projection proof lookup limit must be between 1 and 1001")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.* FROM contexts c WHERE c.tenant_id = ? AND c.source_uri = ? "
                "AND c.projection_effect_hash = ? ORDER BY c.record_key LIMIT ?",
                (str(tenant_id), str(source_uri), str(projection_effect_hash), bounded),
            ).fetchall()
            return self._catalog_records_from_rows(conn, rows)

    def scan_catalog_batch(
        self,
        *,
        after_record_key: str = "",
        filters: Mapping[str, Any] | None = None,
        limit: int = 256,
    ) -> list[CatalogRecord]:
        """Return a stable keyset-paginated batch for offline repair and GC.

        Online retrieval uses ``search_catalog``.  This administrative API is
        deliberately keyset-paginated so retention and rebuild jobs never
        materialize the full catalog or become sensitive to rows whose
        ``updated_at`` changes while a batch is processed.
        """

        bounded_limit = self._bounded_limit(limit)
        filter_sql, params = self._base_filter_sql(
            dict(filters or {}),
            path_candidate_limit=bounded_limit,
        )
        sql = f"SELECT c.* FROM contexts c WHERE c.record_key > ? {filter_sql} ORDER BY c.record_key LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(
                sql,
                [str(after_record_key), *params, bounded_limit],
            ).fetchall()
            return self._catalog_records_from_rows(conn, rows)

    def catalog_schema_version(self) -> int:
        """Return the durable SQLite schema version used by migration gates."""

        with self._connect() as conn:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])

    def gc_orphan_paths(self, *, limit: int = 256) -> int:
        """Delete a bounded batch of paths whose rebuildable record is gone."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT p.tenant_id, p.record_key, p.path FROM context_paths p "
                "LEFT JOIN contexts c ON c.record_key = p.record_key "
                "WHERE c.record_key IS NULL ORDER BY p.record_key, p.path LIMIT ?",
                (self._bounded_limit(limit),),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "DELETE FROM context_paths WHERE tenant_id = ? AND record_key = ? AND path = ?",
                    (str(row["tenant_id"]), str(row["record_key"]), str(row["path"])),
                )
                conn.execute(
                    "DELETE FROM context_path_closure WHERE tenant_id = ? AND record_key = ? AND path = ?",
                    (str(row["tenant_id"]), str(row["record_key"]), str(row["path"])),
                )
                conn.execute(
                    "DELETE FROM context_path_acl WHERE tenant_id = ? AND record_key = ? AND path = ?",
                    (str(row["tenant_id"]), str(row["record_key"]), str(row["path"])),
                )
        return len(rows)

    def gc_applied_tombstones(self, *, updated_before: str, limit: int = 256) -> int:
        """Expire only old tombstones proven safe to forget.

        Stale tombstones did not delete the newer projection and are safe once
        aged out.  Applied tombstones remain durable by default; a projection
        owner must explicitly persist ``payload.gc_safe=true`` after proving
        that replay cannot resurrect the deleted source revision.
        """

        cutoff = self._coerce_timestamp(str(updated_before))
        if not cutoff:
            raise ValueError("updated_before must be an ISO-8601 timestamp")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tombstone_id FROM context_tombstones "
                "WHERE updated_at < ? AND (status = 'STALE' OR "
                "(status = 'APPLIED' AND json_extract(payload_json, '$.gc_safe') = 1)) "
                "ORDER BY updated_at, tombstone_id LIMIT ?",
                (cutoff, self._bounded_limit(limit)),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "DELETE FROM context_tombstones WHERE tombstone_id = ?",
                    (str(row["tombstone_id"]),),
                )
        return len(rows)

    def search_catalog(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[IndexHit]:
        """Return record-key-distinct exact/FTS candidates without Python row scans."""

        normalized_filters = dict(filters or {})
        bounded_limit = self._bounded_limit(limit)
        exact_hits = self._search_metadata_exact(query, normalized_filters, bounded_limit)
        if len(exact_hits) >= bounded_limit:
            return exact_hits[:bounded_limit]
        hits = []
        if str(query).strip():
            hits = self._search_fts(query, normalized_filters, bounded_limit) if self.fts_enabled else []
        merged: dict[str, IndexHit] = {}
        for hit in (*exact_hits, *hits):
            key = str(hit.metadata.get("catalog_record_key") or hit.uri)
            merged.setdefault(key, hit)
        return list(merged.values())[:bounded_limit]

    def delete_catalog(self, record_key: str, *, tenant_id: str | None = None) -> bool:
        with self._connect() as conn:
            if tenant_id is not None:
                exists = conn.execute(
                    "SELECT 1 FROM contexts WHERE record_key = ? AND tenant_id = ?",
                    (str(record_key), str(tenant_id)),
                ).fetchone()
                if exists is None:
                    return False
            return self._delete_catalog_in_transaction(
                conn,
                str(record_key),
                tenant_id=tenant_id,
            )

    def apply_tombstone(
        self,
        *,
        tenant_id: str,
        record_key: str,
        reason: str,
        uri: str = "",
        source_revision: int = 0,
        tombstone_id: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compatibility helper: durably enqueue, then apply one tombstone."""

        queued = self.enqueue_tombstone(
            tenant_id=tenant_id,
            record_key=record_key,
            reason=reason,
            uri=uri,
            source_revision=source_revision,
            tombstone_id=tombstone_id,
            payload=payload,
        )
        applied = self.mark_tombstone_applied(str(queued["tombstone_id"]))
        if applied is None:  # pragma: no cover - enqueue either commits or raises.
            raise RuntimeError("tombstone disappeared before application")
        return applied

    def enqueue_tombstone(
        self,
        *,
        tenant_id: str,
        record_key: str,
        reason: str,
        uri: str = "",
        source_revision: int = 0,
        tombstone_id: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a replayable projection deletion before a worker touches serving data."""

        if not tenant_id or not record_key or not reason:
            raise ValueError("tenant_id, record_key, and reason are required")
        safe_payload = self.sanitizer.sanitize_trace(dict(payload or {}))
        safe_reason = str(self.sanitizer.sanitize_trace(str(reason)))
        identity = {
            "tenant_id": str(tenant_id),
            "record_key": str(record_key),
            "reason": safe_reason,
            "source_revision": int(source_revision),
        }
        stable_id = tombstone_id or self.sanitizer.digest(identity)
        now = self._now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT tenant_id, record_key, reason, source_revision, status, uri "
                "FROM context_tombstones WHERE tombstone_id = ?",
                (stable_id,),
            ).fetchone()
            if existing is not None:
                existing_identity = (
                    str(existing["tenant_id"]),
                    str(existing["record_key"]),
                    str(existing["reason"]),
                    int(existing["source_revision"]),
                )
                requested_identity = (str(tenant_id), str(record_key), safe_reason, int(source_revision))
                if existing_identity != requested_identity:
                    raise ValueError("tombstone_id is already bound to a different immutable identity")
            current = conn.execute("SELECT uri FROM contexts WHERE record_key = ?", (str(record_key),)).fetchone()
            effective_uri = self._safe_reference_uri(str(uri or (current["uri"] if current is not None else "")))
            conn.execute(
                """
                INSERT INTO context_tombstones(
                  tombstone_id, tenant_id, record_key, uri, reason, source_revision,
                  status, payload_json, created_at, updated_at, retry_count, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, 0, '')
                ON CONFLICT(tombstone_id) DO UPDATE SET
                  payload_json=excluded.payload_json,
                  updated_at=excluded.updated_at
                """,
                (
                    stable_id,
                    str(tenant_id),
                    str(record_key),
                    effective_uri,
                    safe_reason,
                    int(source_revision),
                    self._json_dump(safe_payload),
                    now,
                    now,
                ),
            )
        return {
            "tombstone_id": stable_id,
            "tenant_id": str(tenant_id),
            "record_key": str(record_key),
            "uri": effective_uri,
            "status": str(existing["status"]) if existing is not None else "PENDING",
            "source_revision": int(source_revision),
        }

    def get_pending_tombstones(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM context_tombstones WHERE status IN ('PENDING', 'FAILED', 'CLEANING') "
                "ORDER BY updated_at, tombstone_id LIMIT ?",
                (self._bounded_limit(limit),),
            ).fetchall()
        return [self._row_dict(row, json_fields=("payload_json",)) for row in rows]

    def get_pending_tombstones_for_uri(
        self,
        uri: str,
        *,
        tenant_id: str,
        after_tombstone_id: str = "",
        limit: int = 1_000,
    ) -> list[dict[str, Any]]:
        """Recover one delete target's exact unfinished journal without queue starvation."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM context_tombstones "
                "WHERE tenant_id = ? AND uri = ? AND status IN ('PENDING', 'FAILED', 'CLEANING') "
                "AND tombstone_id > ? ORDER BY tombstone_id LIMIT ?",
                (
                    str(tenant_id),
                    self._safe_reference_uri(str(uri)),
                    str(after_tombstone_id),
                    self._bounded_limit(limit),
                ),
            ).fetchall()
        return [self._row_dict(row, json_fields=("payload_json",)) for row in rows]

    def get_tombstones(self, tombstone_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Read an explicit bounded set of durable tombstones in caller order.

        Delete callers use this exact-ID path after they have durably enqueued
        every affected projection.  It avoids starvation behind unrelated
        failed journal entries and does not depend on the pending queue's
        1,000-row administrative batch limit.
        """

        ordered_ids = tuple(dict.fromkeys(str(item) for item in tombstone_ids if str(item)))
        if not ordered_ids:
            return []
        by_id: dict[str, dict[str, Any]] = {}
        with self._connect() as conn:
            for offset in range(0, len(ordered_ids), _MAX_FILTER_VALUES):
                chunk = ordered_ids[offset : offset + _MAX_FILTER_VALUES]
                placeholders = ", ".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT * FROM context_tombstones WHERE tombstone_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    payload = self._row_dict(row, json_fields=("payload_json",))
                    by_id[str(payload["tombstone_id"])] = payload
        return [by_id[tombstone_id] for tombstone_id in ordered_ids if tombstone_id in by_id]

    def pending_tombstones(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Backward-compatible alias for get_pending_tombstones()."""

        return self.get_pending_tombstones(limit=limit)

    def mark_tombstone_applied(self, tombstone_id: str) -> dict[str, Any] | None:
        """Idempotently apply a queued tombstone and close its durable journal row."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status not in {"APPLIED", "STALE"}:
                current = conn.execute(
                    "SELECT tenant_id, source_revision FROM contexts WHERE record_key = ?",
                    (str(row["record_key"]),),
                ).fetchone()
                if current is not None and str(current["tenant_id"]) != str(row["tenant_id"]):
                    raise ValueError("tombstone tenant does not own the target catalog record")
                if (
                    current is not None
                    and int(row["source_revision"])
                    and int(current["source_revision"]) > int(row["source_revision"])
                ):
                    status = "STALE"
                else:
                    self._delete_catalog_in_transaction(
                        conn,
                        str(row["record_key"]),
                        tenant_id=str(row["tenant_id"]),
                    )
                    status = "APPLIED"
                conn.execute(
                    "UPDATE context_tombstones SET status = ?, updated_at = ?, last_error = '' WHERE tombstone_id = ?",
                    (status, self._now(), str(tombstone_id)),
                )
            refreshed = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
        return self._row_dict(refreshed, json_fields=("payload_json",)) if refreshed is not None else None

    def begin_tombstone_cleanup(self, tombstone_id: str) -> dict[str, Any] | None:
        """Atomically establish deletion ownership before external cleanup.

        ``CLEANING`` is a durable, replayable intermediate state.  The Catalog
        row is removed in the same SQLite transaction that enters this state,
        so an online read can never observe a row whose external projections
        are already being retired.  A newer Catalog revision makes the
        tombstone ``STALE`` before Vector or Relation state is touched.
        """

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status not in {"APPLIED", "STALE", "CLEANING"}:
                current = conn.execute(
                    "SELECT tenant_id, source_revision, source_digest, projection_effect_hash, updated_at "
                    "FROM contexts WHERE record_key = ?",
                    (str(row["record_key"]),),
                ).fetchone()
                if current is not None and str(current["tenant_id"]) != str(row["tenant_id"]):
                    raise ValueError("tombstone tenant does not own the target catalog record")
                payload = self._json_mapping(row["payload_json"])
                expected_digest = str(payload.get("expected_source_digest") or "")
                expected_effect = str(payload.get("expected_projection_effect_hash") or "")
                expected_updated_at = str(payload.get("expected_updated_at") or "")
                stale = bool(
                    current is not None
                    and (
                        (int(row["source_revision"]) and int(current["source_revision"]) > int(row["source_revision"]))
                        or (expected_digest and str(current["source_digest"]) != expected_digest)
                        or (expected_effect and str(current["projection_effect_hash"]) != expected_effect)
                        or (expected_updated_at and str(current["updated_at"]) != expected_updated_at)
                    )
                )
                if stale:
                    status = "STALE"
                else:
                    self._delete_catalog_in_transaction(
                        conn,
                        str(row["record_key"]),
                        tenant_id=str(row["tenant_id"]),
                    )
                    status = "CLEANING"
                conn.execute(
                    "UPDATE context_tombstones SET status = ?, updated_at = ?, last_error = '' WHERE tombstone_id = ?",
                    (status, self._now(), str(tombstone_id)),
                )
            refreshed = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
        return self._row_dict(refreshed, json_fields=("payload_json",)) if refreshed is not None else None

    def finish_tombstone_cleanup(self, tombstone_id: str) -> dict[str, Any] | None:
        """Mark external cleanup complete without weakening terminal states."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status == "CLEANING":
                conn.execute(
                    "UPDATE context_tombstones SET status = 'APPLIED', updated_at = ?, last_error = '' "
                    "WHERE tombstone_id = ? AND status = 'CLEANING'",
                    (self._now(), str(tombstone_id)),
                )
            elif status not in {"APPLIED", "STALE"}:
                raise RuntimeError(f"tombstone cleanup cannot finish from {status}")
            refreshed = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
        return self._row_dict(refreshed, json_fields=("payload_json",)) if refreshed is not None else None

    def mark_tombstone_cleanup_failed(self, tombstone_id: str, error: str) -> dict[str, Any] | None:
        """Record an external cleanup error while retaining deletion ownership."""

        safe_error = str(self.sanitizer.sanitize_trace(str(error or "")))
        with self._connect() as conn:
            conn.execute(
                "UPDATE context_tombstones SET retry_count = retry_count + 1, last_error = ?, updated_at = ? "
                "WHERE tombstone_id = ? AND status = 'CLEANING'",
                (safe_error, self._now(), str(tombstone_id)),
            )
            row = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
        return self._row_dict(row, json_fields=("payload_json",)) if row is not None else None

    def mark_tombstone_failed(self, tombstone_id: str, error: str) -> dict[str, Any] | None:
        safe_error = str(self.sanitizer.sanitize_trace(str(error or "")))
        with self._connect() as conn:
            conn.execute(
                "UPDATE context_tombstones SET status = 'FAILED', retry_count = retry_count + 1, "
                "last_error = ?, updated_at = ? WHERE tombstone_id = ?",
                (safe_error, self._now(), str(tombstone_id)),
            )
            row = conn.execute(
                "SELECT * FROM context_tombstones WHERE tombstone_id = ?",
                (str(tombstone_id),),
            ).fetchone()
        return self._row_dict(row, json_fields=("payload_json",)) if row is not None else None

    def set_migration_state(
        self,
        migration_name: str,
        state: str,
        checkpoint: str = "",
        details: Mapping[str, Any] | None = None,
        *,
        tenant_id: str = "",
        batch_size: int = 0,
        error: str = "",
    ) -> dict[str, Any]:
        normalized = str(state).upper()
        if normalized not in _MIGRATION_STATES:
            raise ValueError(f"unsupported migration state: {state}")
        if not migration_name:
            raise ValueError("migration_name is required")
        safe_checkpoint = str(self.sanitizer.sanitize_trace(str(checkpoint or "")))
        safe_details = self.sanitizer.sanitize_trace(dict(details or {}))
        safe_error = self.sanitizer.sanitize_trace(str(error or ""))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO migration_state(
                  migration_name, tenant_id, state, checkpoint, batch_size, details_json, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(migration_name, tenant_id) DO UPDATE SET
                  state=excluded.state,
                  checkpoint=excluded.checkpoint,
                  batch_size=excluded.batch_size,
                  details_json=excluded.details_json,
                  last_error=excluded.last_error,
                  updated_at=excluded.updated_at
                """,
                (
                    str(migration_name),
                    str(tenant_id),
                    normalized,
                    safe_checkpoint,
                    max(0, int(batch_size)),
                    self._json_dump(safe_details),
                    str(safe_error),
                    self._now(),
                ),
            )
        result = self.get_migration_state(migration_name, tenant_id=tenant_id)
        if result is None:  # pragma: no cover - the transaction above either commits or raises.
            raise RuntimeError("migration state write did not persist")
        return result

    def get_migration_state(self, migration_name: str, *, tenant_id: str = "") -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM migration_state WHERE migration_name = ? AND tenant_id = ?",
                (str(migration_name), str(tenant_id)),
            ).fetchone()
        if row is None:
            return None
        return self._row_dict(row, json_fields=("details_json",))

    def initialize_migration_state_if_absent(
        self,
        migration_name: str,
        state: str,
        details: Mapping[str, Any] | None = None,
        *,
        tenant_id: str,
        batch_size: int = 0,
    ) -> dict[str, Any]:
        """Create a durable migration gate without overwriting live progress."""

        normalized = str(state).upper()
        if normalized not in _MIGRATION_STATES:
            raise ValueError(f"unsupported migration state: {state}")
        if not migration_name:
            raise ValueError("migration_name is required")
        if not tenant_id:
            raise ValueError("tenant_id is required")
        safe_details = self.sanitizer.sanitize_trace(dict(details or {}))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO migration_state(
                  migration_name, tenant_id, state, checkpoint, batch_size,
                  details_json, last_error, updated_at
                ) VALUES (?, ?, ?, '', ?, ?, '', ?)
                ON CONFLICT(migration_name, tenant_id) DO NOTHING
                """,
                (
                    str(migration_name),
                    str(tenant_id),
                    normalized,
                    max(0, int(batch_size)),
                    self._json_dump(safe_details),
                    self._now(),
                ),
            )
        result = self.get_migration_state(migration_name, tenant_id=tenant_id)
        if result is None:  # pragma: no cover - INSERT or an existing row must win.
            raise RuntimeError("migration state initialization did not persist")
        return result

    def record_greenfield_catalog_origin(self, *, tenant_id: str) -> dict[str, Any]:
        """Durably distinguish a safe empty first start from an unbackfilled restart."""

        return self.initialize_migration_state_if_absent(
            _GREENFIELD_CATALOG_ORIGIN_NAME,
            "COMPLETED",
            {
                "schema_version": _CATALOG_SCHEMA_VERSION,
                "greenfield": True,
            },
            tenant_id=tenant_id,
        )

    def has_greenfield_catalog_origin(self, *, tenant_id: str) -> bool:
        return self.get_migration_state(
            _GREENFIELD_CATALOG_ORIGIN_NAME,
            tenant_id=tenant_id,
        ) is not None

    def bind_migration_tenant_from_schema_upgrade(
        self,
        migration_name: str,
        *,
        tenant_id: str,
        batch_size: int = 0,
    ) -> dict[str, Any] | None:
        """Atomically materialize an upgraded-database bootstrap for one tenant.

        The index database is opened before the runtime tenant migration
        coordinator is constructed.  A reserved durable row therefore records
        that the schema came from a pre-v10 database, while this method copies
        it to the concrete tenant without overwriting a concurrently advanced
        migration state.
        """

        if not migration_name:
            raise ValueError("migration_name is required")
        if not tenant_id:
            raise ValueError("tenant_id is required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO migration_state(
                  migration_name, tenant_id, state, checkpoint, batch_size,
                  details_json, last_error, updated_at
                )
                SELECT migration_name, ?, state, checkpoint,
                       CASE WHEN ? > 0 THEN ? ELSE batch_size END,
                       details_json, last_error, ?
                FROM migration_state
                WHERE migration_name = ? AND tenant_id = ?
                ON CONFLICT(migration_name, tenant_id) DO NOTHING
                """,
                (
                    str(tenant_id),
                    int(batch_size),
                    int(batch_size),
                    self._now(),
                    str(migration_name),
                    _SCHEMA_UPGRADE_BOOTSTRAP_TENANT,
                ),
            )
        return self.get_migration_state(migration_name, tenant_id=tenant_id)

    def set_session_projection_frontier(
        self,
        *,
        tenant_id: str,
        archive_uri: str,
        owner_user_id: str = "",
        workspace_id: str = "",
        session_id: str,
        manifest_digest: str,
        status: str,
        error: str = "",
    ) -> dict[str, Any]:
        """Persist the SessionArchive -> Catalog projection substate."""

        normalized = str(status).upper()
        if normalized not in {"PENDING", "PROJECTED", "FAILED", "ABANDONED"}:
            raise ValueError("unsupported Session projection frontier status")
        if not tenant_id or not archive_uri or not session_id:
            raise ValueError("Session projection frontier identity is incomplete")
        safe_error = str(self.sanitizer.sanitize_trace(str(error or "")))
        uri_owner = self._owner_from_session_archive_uri(archive_uri)
        safe_owner = str(owner_user_id or uri_owner)
        if not safe_owner or (uri_owner and safe_owner != uri_owner):
            raise ValueError("Session projection frontier owner does not match archive URI")
        safe_workspace = normalize_workspace_id(workspace_id) if workspace_id else ""
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_projection_frontier(
                  tenant_id, archive_uri, owner_user_id, workspace_id,
                  session_id, manifest_digest, status, last_error,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, archive_uri) DO UPDATE SET
                  owner_user_id=excluded.owner_user_id,
                  workspace_id=excluded.workspace_id,
                  session_id=excluded.session_id,
                  manifest_digest=excluded.manifest_digest,
                  status=excluded.status,
                  last_error=excluded.last_error,
                  updated_at=excluded.updated_at
                """,
                (
                    str(tenant_id),
                    str(archive_uri),
                    safe_owner,
                    safe_workspace,
                    str(session_id),
                    str(manifest_digest or ""),
                    normalized,
                    safe_error,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM session_projection_frontier WHERE tenant_id = ? AND archive_uri = ?",
                (str(tenant_id), str(archive_uri)),
            ).fetchone()
        if row is None:  # pragma: no cover - write transaction either persists or raises.
            raise RuntimeError("Session projection frontier write did not persist")
        return self._row_dict(row)

    def get_session_projection_frontier_summary(
        self,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
        workspace_ids: Sequence[str] | None = None,
    ) -> dict[str, int]:
        if not tenant_id:
            raise ValueError("tenant_id is required for Session projection frontier")
        where = ["tenant_id = ?"]
        params: list[Any] = [str(tenant_id)]
        if owner_user_id is not None:
            where.append("owner_user_id = ?")
            params.append(str(owner_user_id))
        if workspace_ids is not None:
            normalized = tuple(
                dict.fromkeys(normalize_workspace_id(item) if item else "" for item in workspace_ids)
            )
            if not normalized:
                return {}
            if len(normalized) > _MAX_FILTER_VALUES:
                raise CatalogCandidateBoundExceeded("too many Session frontier workspace filters")
            where.append(f"workspace_id IN ({','.join('?' for _ in normalized)})")
            params.extend(normalized)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM session_projection_frontier "
                f"WHERE {' AND '.join(where)} GROUP BY status",
                params,
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    @staticmethod
    def _owner_from_session_archive_uri(archive_uri: str) -> str:
        prefix = "memoryos://user/"
        text = str(archive_uri)
        if not text.startswith(prefix):
            return ""
        return text[len(prefix) :].split("/", 1)[0]

    def list_session_projection_frontier(
        self,
        *,
        tenant_id: str,
        statuses: Sequence[str] = ("PENDING", "FAILED"),
        after_archive_uri: str = "",
        limit: int = 256,
    ) -> list[dict[str, Any]]:
        """Read a bounded, tenant-keyset page for startup/repair replay."""

        if not tenant_id:
            raise ValueError("tenant_id is required for Session projection frontier")
        normalized = tuple(dict.fromkeys(str(item).upper() for item in statuses))
        if not normalized or any(item not in {"PENDING", "PROJECTED", "FAILED", "ABANDONED"} for item in normalized):
            raise ValueError("unsupported Session projection frontier status filter")
        placeholders = ",".join("?" for _ in normalized)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM session_projection_frontier WHERE tenant_id = ? "
                f"AND status IN ({placeholders}) AND archive_uri > ? "
                "ORDER BY archive_uri LIMIT ?",
                (
                    str(tenant_id),
                    *normalized,
                    str(after_archive_uri),
                    self._bounded_limit(limit),
                ),
            ).fetchall()
        return [self._row_dict(row) for row in rows]

    def record_migration_equivalence_proof(
        self,
        migration_name: str,
        proof: Mapping[str, Any],
        *,
        tenant_id: str = "",
    ) -> dict[str, Any]:
        """Append one idempotent, payload-free source-to-Catalog proof."""

        required_text = (
            "plane",
            "source_identity_digest",
            "evidence_digest",
            "expected_digest",
            "actual_digest",
        )
        safe = self.sanitizer.sanitize_trace(dict(proof))
        if not isinstance(safe, Mapping) or any(not str(safe.get(key) or "") for key in required_text):
            raise ValueError("migration equivalence proof is incomplete")
        expected_count = int(safe.get("expected_count") or 0)
        actual_count = int(safe.get("actual_count") or 0)
        if not 0 <= expected_count <= 1_000 or not 0 <= actual_count <= 1_001:
            raise ValueError("migration equivalence proof count is outside its hard bound")
        matched = bool(safe.get("matched")) and not bool(safe.get("overflow"))
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM migration_state WHERE migration_name = ? AND tenant_id = ?",
                (str(migration_name), str(tenant_id)),
            ).fetchone()
            if row is None:
                raise ValueError("migration equivalence proof has no migration state")
            details = self._json_mapping(row["details_json"])
            state = str(row["state"])
            epoch = str(details.get("shadow_validation_epoch") or "") if state == "SHADOW_VALIDATING" else ""
            proof_identity = {
                "migration_name": migration_name,
                "tenant_id": tenant_id,
                "state": state,
                "validation_epoch": epoch,
                **{key: safe.get(key) for key in required_text},
                "expected_count": expected_count,
                "actual_count": actual_count,
                "matched": matched,
            }
            proof_id = hashlib.sha256(self._json_dump(proof_identity).encode("utf-8")).hexdigest()
            created_at = self._now()
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO migration_equivalence_journal(
                  proof_id, migration_name, tenant_id, validation_epoch, plane,
                  source_identity_digest, evidence_digest, expected_count, actual_count,
                  expected_digest, actual_digest, matched, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proof_id,
                    str(migration_name),
                    str(tenant_id),
                    epoch,
                    str(safe["plane"]),
                    str(safe["source_identity_digest"]),
                    str(safe["evidence_digest"]),
                    expected_count,
                    actual_count,
                    str(safe["expected_digest"]),
                    str(safe["actual_digest"]),
                    1 if matched else 0,
                    created_at,
                ),
            )
            inserted = cursor.rowcount == 1
            if inserted:
                details["equivalence_proof_count"] = int(details.get("equivalence_proof_count") or 0) + 1
                details["equivalence_mismatch_count"] = int(details.get("equivalence_mismatch_count") or 0) + (
                    0 if matched else 1
                )
                details["last_equivalence_proof"] = {
                    "proof_id": proof_id,
                    "plane": str(safe["plane"]),
                    "expected_count": expected_count,
                    "actual_count": actual_count,
                    "matched": matched,
                }
                if epoch:
                    summary = self._migration_equivalence_summary_in_transaction(
                        conn,
                        migration_name=str(migration_name),
                        tenant_id=str(tenant_id),
                        validation_epoch=epoch,
                    )
                    details["shadow_sample_count"] = summary["sample_count"]
                    details["shadow_mismatch_count"] = summary["mismatch_count"]
                conn.execute(
                    "UPDATE migration_state SET details_json = ?, updated_at = ? "
                    "WHERE migration_name = ? AND tenant_id = ?",
                    (
                        self._json_dump(details),
                        created_at,
                        str(migration_name),
                        str(tenant_id),
                    ),
                )
        return {
            "proof_id": proof_id,
            "inserted": inserted,
            "matched": matched,
            "validation_epoch": epoch,
        }

    def get_migration_equivalence_summary(
        self,
        migration_name: str,
        *,
        tenant_id: str = "",
        validation_epoch: str,
    ) -> dict[str, int]:
        if not validation_epoch:
            return {"sample_count": 0, "mismatch_count": 0}
        with self._connect() as conn:
            return self._migration_equivalence_summary_in_transaction(
                conn,
                migration_name=str(migration_name),
                tenant_id=str(tenant_id),
                validation_epoch=str(validation_epoch),
            )

    def list_migration_equivalence_proofs(
        self,
        migration_name: str,
        *,
        tenant_id: str = "",
        validation_epoch: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        bounded = self._bounded_limit(limit)
        sql = "SELECT * FROM migration_equivalence_journal WHERE migration_name = ? AND tenant_id = ?"
        params: list[Any] = [str(migration_name), str(tenant_id)]
        if validation_epoch is not None:
            sql += " AND validation_epoch = ?"
            params.append(str(validation_epoch))
        sql += " ORDER BY created_at, proof_id LIMIT ?"
        params.append(bounded)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_dict(row) for row in rows]

    def record_migration_shadow_read(
        self,
        migration_name: str,
        comparison: Mapping[str, Any],
        *,
        tenant_id: str = "",
    ) -> dict[str, Any]:
        """Durably record one payload-free old/new bounded read comparison."""

        safe = self.sanitizer.sanitize_trace(dict(comparison))
        required = ("plan_digest", "legacy_digest", "unified_digest")
        if not isinstance(safe, Mapping) or any(not str(safe.get(key) or "") for key in required):
            raise ValueError("migration shadow read comparison is incomplete")
        legacy_count = int(safe.get("legacy_count") or 0)
        unified_count = int(safe.get("unified_count") or 0)
        overlap_count = int(safe.get("overlap_count") or 0)
        if not all(0 <= value <= _MAX_QUERY_LIMIT for value in (legacy_count, unified_count, overlap_count)):
            raise ValueError("migration shadow read count is outside its hard bound")
        if overlap_count > min(legacy_count, unified_count):
            raise ValueError("migration shadow read overlap exceeds result counts")
        matched = bool(
            legacy_count == unified_count
            and overlap_count == legacy_count
            and str(safe["legacy_digest"]) == str(safe["unified_digest"])
        )
        with self._connect() as conn:
            state_row = conn.execute(
                "SELECT * FROM migration_state WHERE migration_name = ? AND tenant_id = ?",
                (str(migration_name), str(tenant_id)),
            ).fetchone()
            if state_row is None or str(state_row["state"]) != "SHADOW_VALIDATING":
                raise ValueError("shadow read comparison requires SHADOW_VALIDATING state")
            details = self._json_mapping(state_row["details_json"])
            epoch = str(details.get("shadow_validation_epoch") or "")
            if not epoch:
                raise ValueError("shadow read comparison has no validation epoch")
            identity = {
                "migration_name": str(migration_name),
                "tenant_id": str(tenant_id),
                "validation_epoch": epoch,
                **{key: str(safe[key]) for key in required},
                "legacy_count": legacy_count,
                "unified_count": unified_count,
                "overlap_count": overlap_count,
                "matched": matched,
            }
            comparison_id = hashlib.sha256(
                self._json_dump(identity).encode("utf-8")
            ).hexdigest()
            now = self._now()
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO migration_shadow_read_journal(
                  comparison_id, migration_name, tenant_id, validation_epoch,
                  plan_digest, legacy_count, unified_count, overlap_count,
                  legacy_digest, unified_digest, matched, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comparison_id,
                    str(migration_name),
                    str(tenant_id),
                    epoch,
                    str(safe["plan_digest"]),
                    legacy_count,
                    unified_count,
                    overlap_count,
                    str(safe["legacy_digest"]),
                    str(safe["unified_digest"]),
                    1 if matched else 0,
                    now,
                ),
            )
            inserted = cursor.rowcount == 1
            if inserted:
                details["shadow_read_sample_count"] = int(
                    details.get("shadow_read_sample_count") or 0
                ) + 1
                details["shadow_read_mismatch_count"] = int(
                    details.get("shadow_read_mismatch_count") or 0
                ) + (0 if matched else 1)
                details["last_shadow_read"] = {
                    "comparison_id": comparison_id,
                    "plan_digest": str(safe["plan_digest"]),
                    "legacy_count": legacy_count,
                    "unified_count": unified_count,
                    "overlap_count": overlap_count,
                    "matched": matched,
                }
                conn.execute(
                    "UPDATE migration_state SET details_json = ?, updated_at = ? "
                    "WHERE migration_name = ? AND tenant_id = ?",
                    (
                        self._json_dump(details),
                        now,
                        str(migration_name),
                        str(tenant_id),
                    ),
                )
        return {
            "comparison_id": comparison_id,
            "inserted": inserted,
            "matched": matched,
            "validation_epoch": epoch,
        }

    def get_migration_shadow_read_summary(
        self,
        migration_name: str,
        *,
        tenant_id: str = "",
        validation_epoch: str,
    ) -> dict[str, int]:
        if not validation_epoch:
            return {"sample_count": 0, "mismatch_count": 0}
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS sample_count, "
                "COALESCE(SUM(CASE WHEN matched = 0 THEN 1 ELSE 0 END), 0) AS mismatch_count "
                "FROM migration_shadow_read_journal WHERE migration_name = ? "
                "AND tenant_id = ? AND validation_epoch = ?",
                (str(migration_name), str(tenant_id), str(validation_epoch)),
            ).fetchone()
        return {
            "sample_count": int(row["sample_count"]),
            "mismatch_count": int(row["mismatch_count"]),
        }

    def list_migration_shadow_reads(
        self,
        migration_name: str,
        *,
        tenant_id: str = "",
        validation_epoch: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT * FROM migration_shadow_read_journal "
            "WHERE migration_name = ? AND tenant_id = ?"
        )
        params: list[Any] = [str(migration_name), str(tenant_id)]
        if validation_epoch is not None:
            sql += " AND validation_epoch = ?"
            params.append(str(validation_epoch))
        sql += " ORDER BY created_at, comparison_id LIMIT ?"
        params.append(self._bounded_limit(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_dict(row) for row in rows]

    @staticmethod
    def _migration_equivalence_summary_in_transaction(
        conn: sqlite3.Connection,
        *,
        migration_name: str,
        tenant_id: str,
        validation_epoch: str,
    ) -> dict[str, int]:
        row = conn.execute(
            "SELECT COUNT(*) AS sample_count, "
            "COALESCE(SUM(CASE WHEN matched = 0 THEN 1 ELSE 0 END), 0) AS mismatch_count "
            "FROM migration_equivalence_journal WHERE migration_name = ? AND tenant_id = ? "
            "AND validation_epoch = ?",
            (migration_name, tenant_id, validation_epoch),
        ).fetchone()
        return {
            "sample_count": int(row["sample_count"]),
            "mismatch_count": int(row["mismatch_count"]),
        }

    def upsert_context_link(
        self,
        *,
        tenant_id: str,
        source_record_key: str,
        source_uri: str,
        relation_type: str,
        target_uri: str,
        target_record_key: str = "",
        metadata: Mapping[str, Any] | None = None,
        link_key: str = "",
    ) -> str:
        if not all((tenant_id, source_record_key, source_uri, relation_type, target_uri)):
            raise ValueError("context link identity fields are required")
        safe_metadata = self.sanitizer.sanitize_trace(dict(metadata or {}))
        identity = {
            "tenant_id": tenant_id,
            "source_record_key": source_record_key,
            "relation_type": relation_type,
            "target_record_key": target_record_key,
            "target_uri": target_uri,
        }
        stable_key = link_key or self.sanitizer.digest(identity)
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO context_links(
                  link_key, tenant_id, source_record_key, source_uri, relation_type,
                  target_record_key, target_uri, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(link_key) DO UPDATE SET
                  target_record_key=excluded.target_record_key,
                  target_uri=excluded.target_uri,
                  metadata_json=excluded.metadata_json,
                  updated_at=excluded.updated_at
                """,
                (
                    stable_key,
                    str(tenant_id),
                    str(source_record_key),
                    self._safe_reference_uri(str(source_uri)),
                    str(relation_type),
                    str(target_record_key),
                    self._safe_reference_uri(str(target_uri)),
                    self._json_dump(safe_metadata),
                    now,
                    now,
                ),
            )
        return stable_key

    def explain_structured_query(self, filters: Mapping[str, Any], *, limit: int = 10) -> list[str]:
        """Expose SQLite's query plan for integration/performance acceptance tests."""

        normalized_filters = dict(filters)
        bounded_limit = self._bounded_limit(limit)
        filter_sql, params = self._base_filter_sql(
            normalized_filters,
            path_candidate_limit=bounded_limit,
        )
        from_sql, index_predicate = self._catalog_from_sql(normalized_filters)
        sql = f"EXPLAIN QUERY PLAN SELECT c.record_key FROM {from_sql} WHERE 1=1 {filter_sql}{index_predicate} LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(sql, [*params, bounded_limit]).fetchall()
        return [str(row["detail"]) for row in rows]

    def _catalog_from_sql(self, filters: Mapping[str, Any]) -> tuple[str, str]:
        """Use the Current Slot unique key for an exact serving lookup."""

        if filters.get("target_identity_uris") is not None:
            # The identity UNION below is already bounded to exact indexed
            # keys. Drive the outer ACL/type read from those record keys.
            return "contexts AS c INDEXED BY idx_contexts_record_key", ""
        if (
            filters.get("principal_owner_id") is not None
            or (
                filters.get("owner_user_id") == ""
                and filters.get("require_unscoped")
            )
            or filters.get("target_paths") is not None
            or filters.get("path_prefixes") is not None
        ):
            # Path candidates are already bounded inside the normalized path
            # subquery.  Drive the outer read by their record keys; otherwise
            # SQLite may prefer a tenant/time ORDER BY index and scan every
            # tenant row before testing membership in the bounded set.
            return "contexts AS c INDEXED BY idx_contexts_record_key", ""
        raw_slots = filters.get("canonical_slot_ids", filters.get("canonical_slot_id"))
        raw_kinds = filters.get("record_kinds", filters.get("record_kind"))
        slots = self._filter_values(raw_slots, allow_empty=True) if raw_slots is not None else ()
        kinds = self._filter_values(raw_kinds, allow_empty=True) if raw_kinds is not None else ()
        if len(slots) == 1 and kinds == [CatalogRecordKind.CURRENT_SLOT.value]:
            return (
                "contexts AS c INDEXED BY uq_contexts_current_slot",
                " AND c.record_kind = 'current_slot' AND c.canonical_slot_id != '' "
                "AND c.lifecycle_state NOT IN ('deleted', 'obsolete')",
            )
        return "contexts AS c", ""

    # ------------------------------------------------------------------
    # Search internals

    def _legacy_filter_sql(self, filters: Mapping[str, Any]) -> tuple[str, list[Any]]:
        """Build the conservative flat-index predicate used only by rollback reads."""

        sql = ""
        params: list[Any] = []
        for filter_name, column in _SIMPLE_FILTER_FIELDS.items():
            if filters.get(filter_name) is None:
                continue
            values = self._filter_values(filters[filter_name])
            sql += f" AND c.{column} IN ({','.join('?' for _ in values)})"
            params.extend(values)
        for filter_name, column in _PLURAL_FILTER_ALIASES.items():
            if filters.get(filter_name) is None:
                continue
            values = self._filter_values(filters[filter_name], allow_empty=True)
            if not values:
                return " AND 1 = 0", []
            sql += f" AND c.{column} IN ({','.join('?' for _ in values)})"
            params.extend(values)
        target_identity_uris = filters.get("target_identity_uris")
        if target_identity_uris is not None:
            identity_sql, identity_params = self._target_identity_sql(
                filters,
                target_identity_uris,
                legacy=True,
            )
            sql += identity_sql
            params.extend(identity_params)
        connect_sql, connect_params = self._connect_filter_sql("c", filters)
        sql += connect_sql
        params.extend(connect_params)
        principal = filters.get("principal_owner_id")
        if principal is not None:
            # Fail closed for cross-owner canonical sharing on rollback.  The
            # Unified route can evaluate Visibility grants; the legacy route
            # intentionally exposes only the owner's rows plus unowned public
            # resources/skills.
            sql += (
                " AND (c.owner_user_id = ? OR "
                "(c.owner_user_id = '' AND c.context_type IN ('resource', 'skill')))"
            )
            params.append(str(principal))
        elif filters.get("owner_user_id") == "" and filters.get("require_unscoped"):
            sql += " AND c.owner_user_id = '' AND c.context_type IN ('resource', 'skill')"
        workspace_access = filters.get("workspace_access_ids")
        if workspace_access is not None:
            values = self._filter_values(workspace_access, allow_empty=True)
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
            scopes = self._filter_values(available_scopes)
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
            paths = self._filter_values(raw_paths, allow_empty=True)
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
                "WHERE lp.tenant_id = c.tenant_id AND lp.record_key = c.record_key AND ("
                + " OR ".join(branches)
                + "))"
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

        values = self._filter_values(target_identity_uris, allow_empty=True)
        if not values:
            return " AND 1 = 0", []
        tenant_values = (
            self._filter_values(filters["tenant_id"])
            if filters.get("tenant_id") is not None
            else []
        )
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
            raise ValueError(
                f"exact Catalog identity candidate limit must be between 1 and {_MAX_QUERY_LIMIT}"
            )
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
                selected = self._filter_values(filters[filter_name])
                branch_sql += (
                    f" AND {alias}.{column} IN ({','.join('?' for _ in selected)})"
                )
                branch_params.extend(selected)
            for filter_name, column in _PLURAL_FILTER_ALIASES.items():
                if filters.get(filter_name) is None:
                    continue
                selected = self._filter_values(filters[filter_name], allow_empty=True)
                if not selected:
                    return " AND 1 = 0", []
                branch_sql += (
                    f" AND {alias}.{column} IN ({','.join('?' for _ in selected)})"
                )
                branch_params.extend(selected)

            if filters.get("uri") is not None:
                selected = self._filter_values(filters["uri"])
                branch_sql += (
                    f" AND {alias}.uri IN ({','.join('?' for _ in selected)})"
                )
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
                selected = self._filter_values(allowed_uris, allow_empty=True)
                if not selected:
                    return " AND 1 = 0", []
                branch_sql += (
                    f" AND {alias}.uri IN ({','.join('?' for _ in selected)})"
                )
                branch_params.extend(selected)

            connect_sql, connect_params = self._connect_filter_sql(alias, filters)
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
                branch_sql += (
                    f" AND {alias}.admission_status NOT IN "
                    f"({','.join('?' for _ in excluded_admission)})"
                )
                branch_params.extend(excluded_admission)
            if filters.get("lifecycle_state") is None and not include_inactive:
                branch_sql += f" AND {alias}.lifecycle_state NOT IN (?, ?, ?)"
                branch_params.extend(("deleted", "archived", "obsolete"))
            if filters.get("serving_tier") is None and not include_inactive:
                branch_sql += f" AND {alias}.serving_tier != ?"
                branch_params.append(ServingTier.ARCHIVED.value)

            available_scopes = filters.get("applicability_scope_keys")
            if available_scopes:
                scopes = self._filter_values(available_scopes)
                if not legacy:
                    signatures = self._scope_signature_options(scopes)
                    branch_sql += (
                        f" AND {alias}.scope_signature IN "
                        f"({','.join('?' for _ in signatures)})"
                    )
                    branch_params.extend(signatures)
                invalid_fallback = "'[\"__invalid__\"]'" if legacy else "'[]'"
                branch_sql += (
                    " AND NOT EXISTS (SELECT 1 FROM json_each("
                    f"CASE WHEN json_valid({alias}.scope_keys) "
                    + (
                        f"AND json_type({alias}.scope_keys) = 'array' "
                        if legacy
                        else ""
                    )
                    + f"THEN {alias}.scope_keys ELSE {invalid_fallback} END) "
                    f"WHERE value NOT IN ({','.join('?' for _ in scopes)}))"
                )
                branch_params.extend(scopes)
            if filters.get("require_unscoped"):
                if not legacy:
                    branch_sql += f" AND {alias}.scope_signature = ?"
                    branch_params.append(self._scope_signature(()))
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
                branch_sql += (
                    f" AND {alias}.valid_from <= ? "
                    f"AND ({alias}.valid_to = '' OR {alias}.valid_to > ?)"
                )
                branch_params.extend((valid_at, valid_at))

            raw_paths = filters.get("target_paths", filters.get("path_prefixes"))
            if raw_paths is not None:
                paths = self._filter_values(raw_paths, allow_empty=True)
                if not paths:
                    return " AND 1 = 0", []
                if len(paths) > _MAX_TARGET_PATHS:
                    raise ValueError(f"path filters cannot exceed {_MAX_TARGET_PATHS} values")
                if legacy:
                    path_predicates: list[str] = []
                    for raw_path in paths:
                        path = normalize_tree_path(raw_path)
                        path_predicates.append(
                            "(identity_path.path = ? OR "
                            "(identity_path.path >= ? AND identity_path.path < ?))"
                        )
                        branch_params.extend((path, f"{path}/", f"{path}/\uffff"))
                    branch_sql += (
                        " AND EXISTS (SELECT 1 FROM context_paths AS identity_path "
                        f"WHERE identity_path.tenant_id = {alias}.tenant_id "
                        f"AND identity_path.record_key = {alias}.record_key AND ("
                        + " OR ".join(path_predicates)
                        + "))"
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
                    branch_sql += (
                        f" AND {alias}.owner_user_id = '' "
                        f"AND {alias}.context_type IN ('resource', 'skill')"
                    )
                return branch_sql, branch_params

            principal = filters.get("principal_owner_id")
            public_only = bool(
                principal is None
                and filters.get("owner_user_id") == ""
                and filters.get("require_unscoped")
            )
            workspace_access = tuple(
                self._filter_values(filters.get("workspace_access_ids", ()), allow_empty=True)
            )
            has_workspace_constraint = filters.get("workspace_access_ids") is not None
            shared_workspaces = tuple(
                value
                for value in workspace_access
                if value not in {"", "__memoryos_principal_only__"}
            )
            if principal is not None:
                access_predicates = [
                    "(identity_acl.grant_kind = 'principal' AND identity_acl.grant_id = ?)",
                    "identity_acl.grant_kind = 'public'",
                ]
                access_params: list[Any] = [str(principal)]
                service_access_id = filters.get("service_access_id")
                if service_access_id:
                    access_predicates.append(
                        "(identity_acl.grant_kind = 'service' AND identity_acl.grant_id = ?)"
                    )
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
                        " AND identity_acl.workspace_id IN ("
                        + ",".join("?" for _ in workspace_access)
                        + ")"
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
                        " AND identity_acl.workspace_id IN ("
                        + ",".join("?" for _ in workspace_access)
                        + ")"
                    )
                branch_sql += (
                    " AND EXISTS (SELECT 1 FROM context_acl_grants AS identity_acl "
                    "INDEXED BY idx_context_acl_grants_record "
                    f"WHERE identity_acl.tenant_id = {alias}.tenant_id "
                    f"AND identity_acl.record_key = {alias}.record_key "
                    "AND identity_acl.grant_kind = 'public'"
                    + workspace_sql
                    + ")"
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
            principal_owner is None
            and filters.get("owner_user_id") == ""
            and filters.get("require_unscoped")
        )
        shared_workspaces = tuple(
            value
            for value in self._filter_values(filters.get("workspace_access_ids", ()), allow_empty=True)
            if value not in {"", "__memoryos_principal_only__"}
        )
        has_workspace_constraint = filters.get("workspace_access_ids") is not None
        workspace_access = tuple(
            self._filter_values(filters.get("workspace_access_ids", ()), allow_empty=True)
        )
        raw_available_scopes = filters.get("applicability_scope_keys")
        has_scope_filter = raw_available_scopes is not None or bool(filters.get("require_unscoped"))
        if raw_available_scopes:
            scope_signature_options = self._scope_signature_options(
                self._filter_values(raw_available_scopes)
            )
        elif has_scope_filter:
            scope_signature_options = (self._scope_signature(()),)
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
                grant_workspace_sql = (
                    " AND cg.workspace_id IN ("
                    + ",".join("?" for _ in workspace_access)
                    + ")"
                )
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
                public_workspace_sql = (
                    " AND cg.workspace_id IN ("
                    + ",".join("?" for _ in workspace_access)
                    + ")"
                )
            sql += (
                " AND EXISTS (SELECT 1 FROM context_acl_grants AS cg "
                "WHERE cg.tenant_id = c.tenant_id AND cg.record_key = c.record_key "
                "AND cg.grant_kind = 'public'"
                + public_workspace_sql
                + ")"
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
            values = self._filter_values(filters[filter_name])
            sql += f" AND c.{column} IN ({','.join('?' for _ in values)})"
            params.extend(values)
        for filter_name, column in _PLURAL_FILTER_ALIASES.items():
            if filters.get(filter_name) is None:
                continue
            values = self._filter_values(filters[filter_name])
            sql += f" AND c.{column} IN ({','.join('?' for _ in values)})"
            params.extend(values)
        target_identity_uris = filters.get("target_identity_uris")
        if target_identity_uris is not None:
            identity_sql, identity_params = self._target_identity_sql(
                filters,
                target_identity_uris,
            )
            sql += identity_sql
            params.extend(identity_params)
        connect_sql, connect_params = self._connect_filter_sql("c", filters)
        sql += connect_sql
        params.extend(connect_params)
        if filters.get("uri") is not None:
            values = self._filter_values(filters["uri"])
            sql += f" AND c.uri IN ({','.join('?' for _ in values)})"
            params.extend(values)
        project_filter = filters.get("project_id")
        if project_filter is not None:
            sql += " AND (c.project_id = ? OR (c.project_id = '' AND c.memory_type NOT IN (?, ?, ?)))"
            params.append(str(project_filter))
            params.extend(("project_rule", "project_decision", "agent_experience"))
        allowed_uris = filters.get("allowed_uris")
        if allowed_uris is not None:
            values = self._filter_values(allowed_uris, allow_empty=True)
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
            scopes = self._filter_values(available_scopes)
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
            params.append(self._scope_signature(()))
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
            valid_point = self._timestamp_number(valid_at, lower=True)
            validity_tenants = (
                self._filter_values(filters["tenant_id"])
                if filters.get("tenant_id") is not None
                else []
            )
            correlated_tenant_sql = ""
            correlated_tenant_params: list[Any] = []
            validity_map_index = "sqlite_autoindex_context_validity_map_1"
            if validity_tenants:
                validity_map_index = "idx_context_validity_map_tenant_record"
                correlated_tenant_sql = (
                    f" AND vm.tenant_id IN ({','.join('?' for _ in validity_tenants)})"
                )
                correlated_tenant_params.extend(validity_tenants)
            # Drive the correlated lookup from the unique record identity and
            # only then probe its one RTree row.  Reversing this CROSS JOIN
            # scans every interval that contains ``valid_at`` for every ACL
            # candidate and becomes quadratic when most rows are open-ended.
            sql += (
                " AND EXISTS (SELECT 1 FROM context_validity_map AS vm INDEXED BY "
                + validity_map_index
                + " CROSS JOIN context_validity_rtree AS vr "
                "WHERE vm.record_key = c.record_key"
                + correlated_tenant_sql
                + " AND vr.validity_id = vm.validity_id "
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
                self._filter_values(filters["tenant_id"])
                if filters.get("tenant_id") is not None
                else []
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
                acl_match_sql = (
                    " AND (ac.scene_key = ? OR ac.action = ? OR ac.memory_anchor_uri = ?)"
                )
                acl_match_params = (path_exact_value, path_exact_value, path_exact_value)
            if any(filters.get(name) is not None for name in ("event_time_from", "event_time_to")):
                acl_time_dimension = "event"
            elif any(
                filters.get(name) is not None
                for name in ("transaction_time_from", "transaction_time_to")
            ):
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
            acl_type_values = self._filter_values(raw_acl_types) if raw_acl_types is not None else []
            acl_source_values = (
                self._filter_values(filters["source_kinds"])
                if filters.get("source_kinds") is not None
                else []
            )
            acl_session_values = (
                self._filter_values(filters["session_ids"])
                if filters.get("session_ids") is not None
                else []
            )
            acl_adapter_values = (
                self._filter_values(filters["adapter_id"])
                if filters.get("adapter_id") is not None
                else []
            )
            acl_adapter_access_values = (
                self._filter_values(filters["adapter_access_id"])
                if filters.get("adapter_access_id") is not None
                else []
            )
            acl_target_values = (
                self._filter_values(filters["target_uris"])
                if filters.get("target_uris") is not None
                else []
            )
            raw_record_kinds = filters.get("record_kinds", filters.get("record_kind"))
            acl_record_kinds = (
                self._filter_values(raw_record_kinds) if raw_record_kinds is not None else []
            )
            if has_workspace_constraint and acl_target_values:
                grant_index = "idx_context_acl_grants_workspace_uri"
            elif not has_workspace_constraint:
                if acl_target_values:
                    grant_index = "idx_context_acl_grants_access_uri"
                elif has_scope_filter:
                    grant_index = f"idx_context_acl_grants_access_scope_{acl_time_dimension}"
                elif acl_adapter_access_values:
                    grant_index = (
                        f"idx_context_acl_grants_access_adapter_access_{acl_time_dimension}"
                    )
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
                grant_predicates.append(
                    f"ag.record_kind IN ({','.join('?' for _ in acl_record_kinds)})"
                )
                grant_filter_params.extend(acl_record_kinds)
            if has_scope_filter:
                grant_predicates.append(
                    f"ag.scope_signature IN ({','.join('?' for _ in scope_signature_options)})"
                )
                grant_filter_params.extend(scope_signature_options)
            if acl_type_values:
                grant_predicates.append(
                    f"ag.context_type IN ({','.join('?' for _ in acl_type_values)})"
                )
                grant_filter_params.extend(acl_type_values)
            if acl_source_values:
                grant_predicates.append(
                    f"ag.source_kind IN ({','.join('?' for _ in acl_source_values)})"
                )
                grant_filter_params.extend(acl_source_values)
            if acl_session_values:
                grant_predicates.append(
                    f"ag.session_id IN ({','.join('?' for _ in acl_session_values)})"
                )
                grant_filter_params.extend(acl_session_values)
            if acl_adapter_values:
                grant_predicates.append(
                    f"ag.adapter_id IN ({','.join('?' for _ in acl_adapter_values)})"
                )
                grant_filter_params.extend(acl_adapter_values)
            if acl_adapter_access_values:
                grant_predicates.append(
                    f"ag.adapter_access_id IN ({','.join('?' for _ in range(len(acl_adapter_access_values) + 1))})"
                )
                grant_filter_params.extend(("*", *acl_adapter_access_values))
            if acl_target_values:
                grant_predicates.append(
                    f"ag.uri IN ({','.join('?' for _ in acl_target_values)})"
                )
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
            paths = self._filter_values(raw_paths, allow_empty=True) if raw_paths is not None else []
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
            paths = self._filter_values(raw_paths, allow_empty=True)
            if not paths:
                sql += " AND 1 = 0"
            else:
                if len(paths) > _MAX_TARGET_PATHS:
                    raise ValueError(f"path filters cannot exceed {_MAX_TARGET_PATHS} values")
                inner_filter_sql = path_inner_filter_sql.replace("c.", "pc.")
                inner_filter_params = list(path_inner_filter_params)
                tenant_values = (
                    self._filter_values(filters["tenant_id"])
                    if filters.get("tenant_id") is not None
                    else []
                )
                owner_values = (
                    self._filter_values(filters["owner_user_id"])
                    if filters.get("owner_user_id") is not None
                    else []
                )
                raw_types = filters.get("context_types", filters.get("context_type"))
                type_values = self._filter_values(raw_types) if raw_types is not None else []
                raw_path_kinds = filters.get("record_kinds", filters.get("record_kind"))
                path_kind_values = (
                    self._filter_values(raw_path_kinds) if raw_path_kinds is not None else []
                )
                path_source_values = (
                    self._filter_values(filters["source_kinds"])
                    if filters.get("source_kinds") is not None
                    else []
                )
                path_session_values = (
                    self._filter_values(filters["session_ids"])
                    if filters.get("session_ids") is not None
                    else []
                )
                path_adapter_access_values = (
                    self._filter_values(filters["adapter_access_id"])
                    if filters.get("adapter_access_id") is not None
                    else []
                )
                path_target_values = (
                    self._filter_values(filters["target_uris"])
                    if filters.get("target_uris") is not None
                    else []
                )
                has_event_time = any(
                    filters.get(name) is not None for name in ("event_time_from", "event_time_to")
                )
                has_transaction_time = any(
                    filters.get(name) is not None
                    for name in ("transaction_time_from", "transaction_time_to")
                )
                has_updated_time = any(
                    filters.get(name) is not None for name in ("updated_at_from", "updated_at_to")
                )
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
                    path_predicates.append(
                        f"p.record_kind IN ({','.join('?' for _ in path_kind_values)})"
                    )
                    path_params.extend(path_kind_values)
                if path_source_values:
                    path_predicates.append(
                        f"p.source_kind IN ({','.join('?' for _ in path_source_values)})"
                    )
                    path_params.extend(path_source_values)
                if path_session_values:
                    path_predicates.append(
                        f"p.session_id IN ({','.join('?' for _ in path_session_values)})"
                    )
                    path_params.extend(path_session_values)
                if path_adapter_access_values:
                    path_predicates.append(
                        f"p.adapter_access_id IN ({','.join('?' for _ in range(len(path_adapter_access_values) + 1))})"
                    )
                    path_params.extend(("*", *path_adapter_access_values))
                if path_target_values:
                    path_predicates.append(
                        f"p.uri IN ({','.join('?' for _ in path_target_values)})"
                    )
                    path_params.extend(path_target_values)
                if owner_values:
                    path_predicates.append(f"p.owner_user_id IN ({','.join('?' for _ in owner_values)})")
                    path_params.extend(owner_values)
                if has_workspace_constraint:
                    path_predicates.append(
                        f"p.workspace_id IN ({','.join('?' for _ in workspace_access)})"
                    )
                    path_params.extend(workspace_access)
                if has_scope_filter:
                    path_predicates.append(
                        f"p.scope_signature IN ({','.join('?' for _ in scope_signature_options)})"
                    )
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
                    public_index = public_prefix + (
                        f"_{path_time_dimension}" if path_time_dimension else ""
                    )
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
                        grant_predicate += (
                            f" AND pg.workspace_id IN ({','.join('?' for _ in workspace_access)})"
                        )
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
                        tenant_predicate += (
                            f" AND tg.workspace_id IN ({','.join('?' for _ in workspace_access)})"
                        )
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
                            service_predicate += (
                                f" AND sg.workspace_id IN ({','.join('?' for _ in workspace_access)})"
                            )
                            service_params = (*service_params, *workspace_access)
                        service_predicate += ")"
                        principal_path_branches.append(
                            (grant_index, service_predicate, service_params)
                        )
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
                    public_index = public_prefix + (
                        f"_{path_time_dimension}" if path_time_dimension else ""
                    )
                    public_predicate = "p.owner_user_id = '' AND p.context_type IN ('resource', 'skill')"
                    unscoped_public_params: tuple[Any, ...] = ()
                    if has_workspace_constraint:
                        public_predicate += f" AND p.workspace_id IN ({','.join('?' for _ in workspace_access)})"
                        unscoped_public_params = workspace_access
                    principal_path_branches.append(
                        (public_index, public_predicate, unscoped_public_params)
                    )
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
                        scope_access_predicate += (
                            f" AND sg.workspace_id IN ({','.join('?' for _ in workspace_access)})"
                        )
                        scope_access_params.extend(workspace_access)
                    scope_access_predicate += ")"
                    principal_path_branches = [
                        (scope_index, scope_access_predicate, tuple(scope_access_params))
                    ]
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
                        path_acl_branches.append(
                            (path_acl_index, "p.grant_kind = 'tenant' AND p.grant_id = ''", ())
                        )
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
                    path_acl_branches.append(
                        (path_acl_index, "p.grant_kind = 'public' AND p.grant_id = ''", ())
                    )
                    principal_path_branches = path_acl_branches
                for raw_path in paths:
                    path = normalize_tree_path(raw_path)
                    path_join = ""
                    path_match_sql = ""
                    path_match_params: tuple[Any, ...] = ()
                    path_rank_sql = "0.0"
                    if path_fts_query:
                        path_join = (
                            " JOIN contexts_fts ON contexts_fts.record_key = pc.record_key"
                        )
                        path_match_sql = " AND contexts_fts MATCH ?"
                        path_match_params = (path_fts_query,)
                        path_rank_sql = _FTS_BM25
                    elif path_exact_value:
                        path_match_sql = (
                            " AND (pc.scene_key = ? OR pc.action = ? OR pc.memory_anchor_uri = ?)"
                        )
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

    def _search_fts(self, query: str, filters: dict[str, Any], limit: int) -> list[IndexHit]:
        narrowed_filters = self._narrow_online_validity_filters(filters)
        if narrowed_filters is None:
            return []
        filters = narrowed_filters
        match_query = self._match_query(query)
        if not match_query:
            return []
        match_query = self._acl_bound_fts_query(match_query, filters)
        overfetch = min(max(limit * 4, limit), _BOUNDED_FTS_OVERFETCH)
        fts_filters = {**filters, "_fts_bound_candidates": True}
        filter_sql, params = self._base_filter_sql(
            fts_filters,
            path_candidate_limit=overfetch,
        )
        sql = f"""
            SELECT c.*, contexts_fts.title AS fts_title,
                   contexts_fts.content_text AS fts_content,
                   contexts_fts.metadata_text AS fts_metadata,
                   contexts_fts.rank AS rank
            FROM contexts_fts
            CROSS JOIN contexts c INDEXED BY idx_contexts_record_key
              ON contexts_fts.record_key = c.record_key
            WHERE contexts_fts MATCH ? AND contexts_fts.rank MATCH ? {filter_sql}
            ORDER BY contexts_fts.rank
            LIMIT ?
        """
        with self._connect() as conn:
            # Runtime FTS/schema/storage failures are not equivalent to a
            # successful query with zero matches.  Capability selection is
            # decided when the store is initialized; an operational failure
            # after that boundary must remain observable to the public API.
            rows = self._online_fetchall(
                conn,
                sql,
                [match_query, _FTS_RANK_CONFIG, *params, overfetch],
            )
        rows.sort(key=lambda row: str(row["record_key"]), reverse=True)
        rows.sort(key=lambda row: str(row["updated_at"]), reverse=True)
        rows.sort(key=lambda row: float(row["rank"]))
        hits: list[IndexHit] = []
        for row in rows:
            haystack = " ".join((str(row["fts_title"]), str(row["fts_content"]), str(row["fts_metadata"])))
            lexical = self._lexical_relevance(query, haystack)
            if lexical <= 0:
                continue
            hits.append(
                self._hit_from_row(
                    row,
                    lexical=lexical,
                    lexical_rank=self._lexical_match_count(query, haystack),
                )
            )
        # Preserve SQLite FTS5's deterministic BM25 order.  ``hit.score`` is
        # a normalized term-coverage component used later by Fusion; sorting
        # by it here collapses many lexical matches to the same value and
        # silently replaces BM25 relevance with URI order.
        return hits[:limit]

    def _search_metadata_exact(self, query: str, filters: dict[str, Any], limit: int) -> list[IndexHit]:
        narrowed_filters = self._narrow_online_validity_filters(filters)
        if narrowed_filters is None:
            return []
        filters = narrowed_filters
        value = str(query).strip()
        if not value:
            return []
        tenants = (
            self._filter_values(filters["tenant_id"], allow_empty=True)
            if filters.get("tenant_id") is not None
            else ["default"]
        )
        if not tenants:
            raise ValueError("exact Catalog lookup requires tenant_id")
        filters = {**filters, "tenant_id": tuple(tenants)}
        candidate_updates: dict[str, str] = {}
        branch_limit = self._bounded_limit(limit)
        # Exact identities are expected to be selective. Detect an excessive
        # *eligible* identity set explicitly instead of silently truncating
        # it, but only after every trusted ACL/path/time/type predicate has
        # been applied inside the branch.
        identity_limit = _MAX_QUERY_LIMIT + 1
        with self._connect() as conn:
            for column, index_name in (
                ("scene_key", "idx_contexts_tenant_scene_key"),
                ("action", "idx_contexts_tenant_action"),
                ("memory_anchor_uri", "idx_contexts_tenant_anchor"),
            ):
                branch_filters = {
                    **filters,
                    column: value,
                    # The exact equality index is already the selective
                    # candidate driver. Keep the direct ACL EXISTS predicate,
                    # but do not put a broader ACL Top-K in front of it.
                    "_exact_bound_candidates": True,
                }
                filter_sql, params = self._base_filter_sql(
                    branch_filters,
                    path_candidate_limit=min(
                        _MAX_QUERY_LIMIT,
                        max(branch_limit, identity_limit),
                    ),
                )
                rows = self._online_fetchall(
                    conn,
                    "SELECT c.record_key, c.updated_at FROM contexts AS c INDEXED BY "
                    + index_name
                    + " WHERE 1=1 "
                    + filter_sql
                    + " ORDER BY c.updated_at DESC, c.record_key LIMIT ?",
                    [*params, identity_limit],
                )
                if len(rows) >= identity_limit:
                    raise CatalogCandidateBoundExceeded(
                        "eligible exact identity candidates exceed the bounded online lookup"
                    )
                for row in rows:
                    key = str(row["record_key"])
                    candidate_updates[key] = max(
                        candidate_updates.get(key, ""),
                        str(row["updated_at"] or ""),
                    )
        if len(candidate_updates) > _MAX_FILTER_VALUES:
            raise CatalogCandidateBoundExceeded(
                "eligible exact identity candidates exceed the aggregate filter bound"
            )
        ordered_keys = sorted(candidate_updates)
        ordered_keys.sort(key=lambda key: candidate_updates[key], reverse=True)
        bounded_keys = tuple(ordered_keys[:branch_limit])
        if not bounded_keys:
            return []
        raw_records = self.list_catalog(
            filters={**filters, "record_keys": bounded_keys},
            limit=branch_limit,
        )
        return [self._exact_hit_from_record(record) for record in raw_records]

    @staticmethod
    def _exact_hit_from_record(record: CatalogRecord) -> IndexHit:
        metadata = {
            **dict(record.metadata),
            "catalog_record_key": record.record_key,
            "tenant_id": record.tenant_id,
            "owner_user_id": record.owner_user_id,
            "workspace_id": record.workspace_id,
            "workspace_shared": record.workspace_shared,
            "session_id": record.session_id,
            "adapter_id": record.adapter_id,
            "context_type": record.context_type,
            "source_kind": record.source_kind,
            "record_kind": record.record_kind,
            "lifecycle_state": record.lifecycle_state,
            "canonical_slot_id": record.canonical_slot_id,
            "canonical_slot_uri": record.canonical_slot_uri,
            "canonical_claim_id": record.canonical_claim_id,
            "canonical_claim_uri": record.canonical_claim_uri,
            "canonical_revision": record.canonical_revision,
            "canonical_state": record.canonical_state,
            "source_digest": record.source_digest,
            "event_time": record.event_time,
            "transaction_time": record.transaction_time,
            "serving_tier": record.serving_tier,
            "retrieval_scores": {
                "lexical": 0.0,
                "vector": 0.0,
                "identity": 1.0,
                "base_relevance": 1.0,
                "hotness": max(
                    record.hotness,
                    record.semantic_hotness,
                    record.behavior_support_hotness,
                ),
                "score": 10.0,
            },
        }
        return IndexHit(
            uri=record.uri,
            score=10.0,
            context_type=record.context_type,
            title=record.title,
            metadata=metadata,
        )

    def _hit_from_row(
        self,
        row: sqlite3.Row,
        lexical: float = 0.0,
        lexical_rank: float | None = None,
        vector: float = 0.0,
        identity: float = 0.0,
        identity_rank: float | None = None,
    ) -> IndexHit:
        metadata = self._json_mapping(row["metadata_json"])
        # CandidateGenerator deliberately re-checks transport-independent
        # structured filters for stores that only implement the legacy search
        # protocol.  Carry the already-sanitized indexed columns in every hit
        # so that this defence-in-depth check never has to infer them from
        # optional metadata JSON (and never rejects a valid SQL-filtered row).
        metadata.update(
            {
                "catalog_record_key": str(row["record_key"]),
                "tenant_id": str(row["tenant_id"]),
                "owner_user_id": str(row["owner_user_id"]),
                "workspace_id": str(row["workspace_id"]),
                "workspace_shared": bool(row["workspace_shared"]),
                "session_id": str(row["session_id"]),
                "adapter_id": str(row["adapter_id"]),
                "context_type": str(row["context_type"]),
                "source_kind": str(row["source_kind"]),
                "record_kind": str(row["record_kind"]),
                "lifecycle_state": str(row["lifecycle_state"]),
                "canonical_slot_id": str(row["canonical_slot_id"]),
                "canonical_slot_uri": str(row["canonical_slot_uri"]),
                "canonical_claim_id": str(row["canonical_claim_id"]),
                "canonical_claim_uri": str(row["canonical_claim_uri"]),
                "canonical_revision": int(row["canonical_revision"]),
                "canonical_state": str(row["canonical_state"]),
                "source_digest": str(row["source_digest"]),
                "event_time": str(row["event_time"]),
                "transaction_time": str(row["transaction_time"]),
                "serving_tier": str(row["serving_tier"]),
            }
        )
        metadata["retrieval_scores"] = self._score_components(
            row,
            lexical=lexical,
            lexical_rank=lexical_rank,
            vector=vector,
            identity=identity,
            identity_rank=identity_rank,
        )
        return IndexHit(
            uri=str(row["uri"]),
            score=float(metadata["retrieval_scores"]["score"]),
            context_type=str(row["context_type"]),
            title=str(row["title"]),
            metadata=metadata,
        )

    def _score_components(
        self,
        row: sqlite3.Row,
        *,
        lexical: float = 0.0,
        lexical_rank: float | None = None,
        vector: float = 0.0,
        identity: float = 0.0,
        identity_rank: float | None = None,
    ) -> dict[str, float]:
        lexical = self._bounded(lexical)
        vector = self._bounded(vector)
        identity = self._bounded(identity)
        base_relevance = max(lexical, vector, identity)
        hotness = (
            self._bounded(row["hotness"])
            + self._bounded(row["semantic_hotness"])
            + self._bounded(row["behavior_support_hotness"])
        ) / 3.0
        ranking_relevance = max(
            self._finite_rank(lexical_rank if lexical_rank is not None else lexical),
            vector,
            self._finite_rank(identity_rank if identity_rank is not None else identity),
        )
        score = ranking_relevance + (0.05 * hotness if base_relevance > 0 else 0.0)
        return {
            "lexical": lexical,
            "vector": vector,
            "identity": identity,
            "base_relevance": base_relevance,
            "hotness": hotness,
            "score": score,
        }

    # ------------------------------------------------------------------
    # Catalog write internals

    def _prepare_record(
        self,
        record: CatalogRecord,
        *,
        scope_keys_override: Sequence[str] | None = None,
        legacy_overrides: Mapping[str, Any] | None = None,
    ) -> _PreparedCatalogRecord:
        if record.projection_status == CatalogProjectionStatus.TOMBSTONED.value:
            raise ValueError("use apply_tombstone() for tombstoned catalog records")
        scope_keys = (
            list(scope_keys_override)
            if scope_keys_override is not None
            else self._scope_keys_from_metadata(record.metadata)
        )
        if len(scope_keys) > _MAX_SCOPE_KEYS_PER_RECORD:
            raise ValueError(
                f"Catalog scope requirements cannot exceed {_MAX_SCOPE_KEYS_PER_RECORD} keys"
            )
        scope_signature = self._scope_signature(scope_keys)
        safe = record.with_sanitized_projection(self.sanitizer)
        safe = replace(
            safe,
            l2_uri=self._safe_reference_uri(safe.l2_uri),
            source_uri=self._safe_reference_uri(safe.source_uri),
        )
        metadata = dict(safe.metadata)
        scope = self._mapping(metadata.get("scope"))
        fields = self._mapping(metadata.get("fields"))
        connect = self._mapping(metadata.get("connect"))
        admission = self._mapping(metadata.get("admission"))
        project_id = normalize_workspace_id(
            scope.get("project_id") or fields.get("project_id") or metadata.get("project_id") or safe.workspace_id or ""
        )
        values: dict[str, Any] = {
            "record_key": safe.record_key,
            "uri": safe.uri,
            "tenant_id": safe.tenant_id,
            "owner_user_id": safe.owner_user_id,
            "project_id": project_id,
            "workspace_id": normalize_workspace_id(safe.workspace_id or project_id),
            "workspace_shared": 1 if safe.workspace_shared else 0,
            "session_id": safe.session_id,
            "adapter_id": safe.adapter_id or str(connect.get("adapter_id") or ""),
            "context_type": safe.context_type,
            "source_kind": safe.source_kind,
            "record_kind": safe.record_kind,
            "lifecycle_state": safe.lifecycle_state,
            "admission_status": str(admission.get("decision") or ""),
            "claim_state": safe.canonical_state or str(metadata.get("state") or metadata.get("claim_state") or ""),
            "slot_id": safe.canonical_slot_id or str(metadata.get("slot_id") or ""),
            "memory_type": str(metadata.get("memory_type") or ""),
            "scope_keys": self._json_dump(list(dict.fromkeys(str(key) for key in scope_keys))),
            "scope_signature": scope_signature,
            "parent_uri": safe.parent_uri,
            "primary_tree_path": safe.primary_tree_path,
            "path_depth": safe.path_depth,
            "created_at": safe.created_at,
            "updated_at": safe.updated_at,
            "event_time": safe.event_time,
            "ingested_at": safe.ingested_at,
            "transaction_time": safe.transaction_time,
            "valid_from": safe.valid_from,
            "valid_to": safe.valid_to,
            "title": safe.title,
            "l0_text": safe.l0_text,
            "l1_text": safe.l1_text,
            "l2_uri": safe.l2_uri,
            "source_uri": safe.source_uri,
            "source_digest": safe.source_digest,
            "source_revision": int(safe.source_revision),
            "canonical_slot_id": safe.canonical_slot_id,
            "canonical_slot_uri": safe.canonical_slot_uri,
            "canonical_claim_id": safe.canonical_claim_id,
            "canonical_claim_uri": safe.canonical_claim_uri,
            "canonical_revision": int(safe.canonical_revision),
            "canonical_state": safe.canonical_state,
            "canonical_head_digest": safe.canonical_head_digest,
            "receipt_digest": safe.receipt_digest,
            "projection_effect_hash": safe.projection_effect_hash,
            "hotness": safe.hotness,
            "semantic_hotness": safe.semantic_hotness,
            "behavior_support_hotness": safe.behavior_support_hotness,
            "serving_tier": safe.serving_tier,
            "projection_status": safe.projection_status,
            "metadata_json": self._json_dump(metadata),
            # The first digest proves the complete source projection without
            # retaining its potentially huge or sensitive body.  The second
            # detects tampering of the bounded sanitized serving text.
            "content_digest": self._content_digest(record.l1_text),
            "stored_content_digest": self._content_digest(safe.l1_text),
            "content_text": safe.l1_text,
            "scene_key": self._safe_exact_value(metadata.get("scene_key")),
            "action": self._safe_exact_value(metadata.get("action")),
            "memory_anchor_uri": self._safe_exact_value(metadata.get("memory_anchor_uri")),
        }
        for key, value in dict(legacy_overrides or {}).items():
            if key in values:
                values[key] = value
        metadata_text = self._safe_metadata_text(metadata)
        search_terms = " ".join(lexical_terms(" ".join((safe.title, safe.l0_text, safe.l1_text, metadata_text))))
        return _PreparedCatalogRecord(
            record=safe,
            values=values,
            scope_signature=scope_signature,
            fts_metadata_text=metadata_text,
            fts_search_terms=search_terms,
        )

    def _upsert_prepared(self, conn: sqlite3.Connection, item: _PreparedCatalogRecord) -> None:
        tombstones = conn.execute(
            "SELECT source_revision, status FROM context_tombstones "
            "WHERE tenant_id = ? AND record_key = ? AND status IN ('CLEANING', 'APPLIED')",
            (item.record.tenant_id, item.record.record_key),
        ).fetchall()
        for tombstone in tombstones:
            if str(tombstone["status"]) == "CLEANING":
                raise ValueError("catalog projection is blocked by in-progress tombstone cleanup")
            tombstone_revision = int(tombstone["source_revision"])
            if (
                tombstone_revision == 0
                or item.record.source_revision == 0
                or tombstone_revision >= item.record.source_revision
            ):
                raise ValueError("catalog projection is not newer than its applied tombstone")
        columns = ", ".join(_CONTEXT_COLUMNS)
        placeholders = ", ".join("?" for _ in _CONTEXT_COLUMNS)
        updates = ", ".join(f"{column}=excluded.{column}" for column in _CONTEXT_COLUMNS if column != "record_key")
        conn.execute(
            f"INSERT INTO contexts({columns}) VALUES ({placeholders}) ON CONFLICT(record_key) DO UPDATE SET {updates}",
            tuple(item.values[column] for column in _CONTEXT_COLUMNS),
        )
        self._replace_paths(conn, item.record, scope_signature=item.scope_signature)
        self._replace_acl_grants(conn, item.record, scope_signature=item.scope_signature)
        self._replace_validity(conn, item.record)
        self._replace_fts(conn, item)
        conn.execute(
            """
            INSERT INTO context_projection_state(
              tenant_id, record_key, source_revision, projection_status,
              projection_effect_hash, retry_count, last_error, updated_at
            ) VALUES (?, ?, ?, ?, ?, 0, '', ?)
            ON CONFLICT(tenant_id, record_key) DO UPDATE SET
              source_revision=excluded.source_revision,
              projection_status=excluded.projection_status,
              projection_effect_hash=excluded.projection_effect_hash,
              last_error='',
              updated_at=excluded.updated_at
            """,
            (
                item.record.tenant_id,
                item.record.record_key,
                item.record.source_revision,
                item.record.projection_status,
                item.record.projection_effect_hash,
                item.record.updated_at or self._now(),
            ),
        )

    def _replace_paths(
        self,
        conn: sqlite3.Connection,
        record: CatalogRecord,
        *,
        scope_signature: str,
    ) -> None:
        old_created = {
            str(row["path"]): str(row["created_at"])
            for row in conn.execute(
                "SELECT path, created_at FROM context_paths WHERE tenant_id = ? AND record_key = ?",
                (record.tenant_id, record.record_key),
            ).fetchall()
        }
        conn.execute(
            "DELETE FROM context_paths WHERE tenant_id = ? AND record_key = ?",
            (record.tenant_id, record.record_key),
        )
        conn.execute(
            "DELETE FROM context_path_closure WHERE tenant_id = ? AND record_key = ?",
            (record.tenant_id, record.record_key),
        )
        conn.execute(
            "DELETE FROM context_path_acl WHERE tenant_id = ? AND record_key = ?",
            (record.tenant_id, record.record_key),
        )
        now = record.updated_at or record.transaction_time or record.ingested_at or record.created_at or self._now()
        acl_grants = self._acl_grants_for_record(record)
        for index, path in enumerate(record.tree_paths):
            conn.execute(
                """
                INSERT INTO context_paths(
                  tenant_id, record_key, uri, owner_user_id, workspace_id, workspace_shared,
                  context_type, record_kind, canonical_slot_id, canonical_claim_id, event_time,
                  transaction_time, valid_from, valid_to,
                  path, path_kind, depth, is_primary, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.tenant_id,
                    record.record_key,
                    record.uri,
                    record.owner_user_id,
                    record.workspace_id,
                    1 if record.workspace_shared else 0,
                    record.context_type,
                    record.record_kind,
                    record.canonical_slot_id,
                    record.canonical_claim_id,
                    record.event_time,
                    record.transaction_time,
                    record.valid_from,
                    record.valid_to,
                    path,
                    "primary" if index == 0 else "secondary",
                    len(path.split("/")),
                    1 if index == 0 else 0,
                    old_created.get(path, now),
                    now,
                ),
            )
            for ancestor_path in _path_ancestors(path):
                conn.execute(
                    """
                    INSERT INTO context_path_closure(
                      tenant_id, record_key, path, ancestor_path,
                      owner_user_id, workspace_id, workspace_shared, scope_signature,
                      uri, context_type, source_kind, record_kind,
                      adapter_id, adapter_access_id, session_id,
                      canonical_slot_id, canonical_claim_id,
                      event_time, transaction_time, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.tenant_id,
                        record.record_key,
                        path,
                        ancestor_path,
                        record.owner_user_id,
                        record.workspace_id,
                        1 if record.workspace_shared else 0,
                        scope_signature,
                        record.uri,
                        record.context_type,
                        record.source_kind,
                        record.record_kind,
                        record.adapter_id,
                        self._adapter_access_value(record),
                        record.session_id,
                        record.canonical_slot_id,
                        record.canonical_claim_id,
                        record.event_time,
                        record.transaction_time,
                        now,
                    ),
                )
                for grant_kind, grant_id, grant_workspace_id in sorted(acl_grants):
                    conn.execute(
                        """
                        INSERT INTO context_path_acl(
                          tenant_id, record_key, path, ancestor_path,
                          grant_kind, grant_id, workspace_id, owner_user_id,
                          scope_signature, uri, context_type, source_kind, record_kind,
                          adapter_id, adapter_access_id, session_id,
                          event_time, transaction_time, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.tenant_id,
                            record.record_key,
                            path,
                            ancestor_path,
                            grant_kind,
                            grant_id,
                            grant_workspace_id,
                            record.owner_user_id,
                            scope_signature,
                            record.uri,
                            record.context_type,
                            record.source_kind,
                            record.record_kind,
                            record.adapter_id,
                            self._adapter_access_value(record),
                            record.session_id,
                            record.event_time,
                            record.transaction_time,
                            now,
                        ),
                    )

    def _replace_validity(self, conn: sqlite3.Connection, record: CatalogRecord) -> None:
        row = conn.execute(
            "SELECT validity_id FROM context_validity_map WHERE tenant_id = ? AND record_key = ?",
            (record.tenant_id, record.record_key),
        ).fetchone()
        if row is not None:
            conn.execute("DELETE FROM context_validity_rtree WHERE validity_id = ?", (int(row["validity_id"]),))
            conn.execute("DELETE FROM context_validity_map WHERE validity_id = ?", (int(row["validity_id"]),))
        cursor = conn.execute(
            "INSERT INTO context_validity_map(tenant_id, record_key) VALUES (?, ?)",
            (record.tenant_id, record.record_key),
        )
        raw_validity_id = cursor.lastrowid
        if raw_validity_id is None:
            raise RuntimeError("validity map insert did not return a durable rowid")
        validity_id = int(raw_validity_id)
        tenant_key = self._tenant_rtree_key(conn, record.tenant_id)
        valid_from = self._timestamp_number(record.valid_from, lower=True)
        valid_to = self._timestamp_number(record.valid_to, lower=False)
        conn.execute(
            "INSERT INTO context_validity_rtree("
            "validity_id, tenant_min, tenant_max, valid_from_min, valid_from_max, valid_to_min, valid_to_max"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (validity_id, tenant_key, tenant_key, valid_from, valid_from, valid_to, valid_to),
        )

    def _replace_acl_grants(
        self,
        conn: sqlite3.Connection,
        record: CatalogRecord,
        *,
        scope_signature: str,
    ) -> None:
        conn.execute(
            "DELETE FROM context_acl_grants WHERE tenant_id = ? AND record_key = ?",
            (record.tenant_id, record.record_key),
        )
        grants = self._acl_grants_for_record(record)
        now = record.updated_at or record.transaction_time or record.ingested_at or record.created_at or self._now()
        for grant_kind, grant_id, workspace_id in sorted(grants):
            conn.execute(
                "INSERT INTO context_acl_grants("
                "tenant_id, record_key, grant_kind, grant_id, workspace_id, "
                "scope_signature, uri, context_type, source_kind, record_kind, adapter_id, adapter_access_id, session_id, "
                "event_time, transaction_time, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.tenant_id,
                    record.record_key,
                    grant_kind,
                    grant_id,
                    workspace_id,
                    scope_signature,
                    record.uri,
                    record.context_type,
                    record.source_kind,
                    record.record_kind,
                    record.adapter_id,
                    self._adapter_access_value(record),
                    record.session_id,
                    record.event_time,
                    record.transaction_time,
                    now,
                ),
            )

    @staticmethod
    def _acl_grants_for_record(record: CatalogRecord) -> set[tuple[str, str, str]]:
        grants: set[tuple[str, str, str]] = set()
        scope = record.metadata.get("scope")
        visibility = scope.get("visibility") if isinstance(scope, Mapping) else None
        is_canonical = bool(
            record.context_type == "memory"
            and record.record_kind
            in {CatalogRecordKind.CURRENT_SLOT.value, CatalogRecordKind.CLAIM_REVISION.value}
            and record.canonical_slot_id
            and record.canonical_claim_id
        )
        valid_visibility = bool(
            is_canonical
            and isinstance(visibility, Mapping)
            and str(visibility.get("tenant_id") or "") == record.tenant_id
        )
        if valid_visibility and isinstance(visibility, Mapping):
            if str(visibility.get("tenant_id") or "") == record.tenant_id:
                for principal_id in visibility.get("allowed_principal_ids", ()) or ():
                    if isinstance(principal_id, str) and principal_id:
                        grants.add(("principal", principal_id, record.workspace_id))
                for service_id in visibility.get("allowed_service_ids", ()) or ():
                    if isinstance(service_id, str) and service_id:
                        grants.add(("service", service_id, record.workspace_id))
                tenant_public = bool(
                    visibility.get("private") is False
                    and not (visibility.get("allowed_principal_ids", ()) or ())
                    and not (visibility.get("allowed_service_ids", ()) or ())
                )
                if tenant_public and record.owner_user_id:
                    grants.add(("principal", record.owner_user_id, record.workspace_id))
                if tenant_public:
                    grants.add(("tenant", "", record.workspace_id))
        elif not is_canonical and record.owner_user_id:
            grants.add(("principal", record.owner_user_id, record.workspace_id))
        elif not is_canonical and record.context_type in {"resource", "skill"}:
            grants.add(("public", "", record.workspace_id))
        if is_canonical and record.workspace_shared and record.workspace_id:
            grants.add(("workspace", record.workspace_id, record.workspace_id))
        return grants

    @staticmethod
    def _adapter_access_value(record: CatalogRecord) -> str:
        if (
            not record.adapter_id
            or record.context_type in {"session", "resource", "skill"}
            or record.record_kind == CatalogRecordKind.CURRENT_SLOT.value
        ):
            return "*"
        return record.adapter_id

    def _replace_fts(self, conn: sqlite3.Connection, item: _PreparedCatalogRecord) -> None:
        self._delete_fts_record(conn, item.record.record_key)
        if item.record.serving_tier not in {ServingTier.HOT.value, ServingTier.WARM.value}:
            return
        if item.record.projection_status not in {
            CatalogProjectionStatus.PROJECTED.value,
            CatalogProjectionStatus.DEGRADED.value,
        }:
            return
        if item.record.lifecycle_state in {"deleted", "archived", "obsolete"}:
            return
        cursor = conn.execute(
            "INSERT INTO contexts_fts(record_key, uri, title, content_text, metadata_text, search_terms, acl_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                item.record.record_key,
                item.record.uri,
                item.record.title,
                item.record.l1_text,
                item.fts_metadata_text,
                item.fts_search_terms,
                self._fts_acl_tokens(item.record, scope_signature=item.scope_signature),
            ),
        )
        raw_fts_rowid = cursor.lastrowid
        if raw_fts_rowid is None:
            raise RuntimeError("FTS insert did not return a durable rowid")
        fts_rowid = int(raw_fts_rowid)
        if fts_rowid <= 0:
            raise RuntimeError("FTS insert did not return a durable rowid")
        conn.execute(
            "INSERT INTO context_fts_map(record_key, fts_rowid) VALUES (?, ?)",
            (item.record.record_key, fts_rowid),
        )

    def _delete_catalog_in_transaction(
        self,
        conn: sqlite3.Connection,
        record_key: str,
        *,
        tenant_id: str | None = None,
    ) -> bool:
        exists = conn.execute(
            "SELECT tenant_id FROM contexts WHERE record_key = ?",
            (record_key,),
        ).fetchone()
        if exists is not None and tenant_id is not None and str(exists["tenant_id"]) != str(tenant_id):
            raise ValueError("Catalog delete tenant does not own record_key")
        resolved_tenant = str(exists["tenant_id"]) if exists is not None else str(tenant_id or "")
        self._delete_fts_record(conn, record_key)
        if not resolved_tenant:
            return False
        conn.execute(
            "DELETE FROM context_acl_grants WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        )
        conn.execute(
            "DELETE FROM context_path_acl WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        )
        conn.execute(
            "DELETE FROM context_path_closure WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        )
        conn.execute(
            "DELETE FROM context_paths WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        )
        validity = conn.execute(
            "SELECT validity_id FROM context_validity_map WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        ).fetchone()
        if validity is not None:
            conn.execute("DELETE FROM context_validity_rtree WHERE validity_id = ?", (int(validity["validity_id"]),))
            conn.execute("DELETE FROM context_validity_map WHERE validity_id = ?", (int(validity["validity_id"]),))
        conn.execute(
            "DELETE FROM context_links WHERE tenant_id = ? AND source_record_key = ?",
            (resolved_tenant, record_key),
        )
        conn.execute(
            "DELETE FROM context_links WHERE tenant_id = ? AND target_record_key = ?",
            (resolved_tenant, record_key),
        )
        conn.execute(
            "DELETE FROM context_projection_state WHERE tenant_id = ? AND record_key = ?",
            (resolved_tenant, record_key),
        )
        conn.execute("DELETE FROM contexts WHERE record_key = ?", (record_key,))
        return exists is not None

    # ------------------------------------------------------------------
    # Schema and migration

    def _init_db(self) -> None:
        with self._connect() as conn:
            initial_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            had_existing_schema = (
                conn.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
                ).fetchone()
                is not None
            )
            requires_unified_backfill = initial_version < _CATALOG_SCHEMA_VERSION and (
                initial_version > 0 or had_existing_schema
            )
            conn.execute("PRAGMA journal_mode = WAL")
            rebuilt = self._ensure_catalog_table(conn)
            self._create_auxiliary_tables(conn)
            self._migrate_scope_keys(conn)
            fts_recreated = self._ensure_fts_table(conn)
            fts_map_consistent = bool(
                initial_version >= _CATALOG_SCHEMA_VERSION
                and not fts_recreated
                and self._fts_row_map_is_consistent(conn)
            )
            if initial_version < _CATALOG_SCHEMA_VERSION and not rebuilt:
                self._sanitize_existing_rows(conn)
            self._create_indexes(conn)
            if rebuilt or fts_recreated or not fts_map_consistent:
                self._rebuild_fts(conn)
            conn.execute(f"PRAGMA user_version = {_CATALOG_SCHEMA_VERSION}")
            if requires_unified_backfill:
                self._record_unified_catalog_schema_upgrade(
                    conn,
                    upgraded_from_schema_version=initial_version,
                )

    def _record_unified_catalog_schema_upgrade(
        self,
        conn: sqlite3.Connection,
        *,
        upgraded_from_schema_version: int,
    ) -> None:
        """Persist upgrade provenance before a restart can mistake it for greenfield."""

        details = {
            "schema_version": _CATALOG_SCHEMA_VERSION,
            "upgraded_from_schema_version": int(upgraded_from_schema_version),
            "requires_backfill": True,
            "session_backfill_complete": False,
            "backfill_complete": False,
        }
        conn.execute(
            """
            INSERT INTO migration_state(
              migration_name, tenant_id, state, checkpoint, batch_size,
              details_json, last_error, updated_at
            ) VALUES (?, ?, 'SCHEMA_READY', '', ?, ?, '', ?)
            ON CONFLICT(migration_name, tenant_id) DO NOTHING
            """,
            (
                _UNIFIED_CATALOG_MIGRATION_NAME,
                _SCHEMA_UPGRADE_BOOTSTRAP_TENANT,
                _MIGRATION_BATCH_SIZE,
                self._json_dump(details),
                self._now(),
            ),
        )

    def _ensure_catalog_table(self, conn: sqlite3.Connection) -> bool:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'contexts'").fetchone()
        if exists is None:
            self._create_contexts_table(conn, "contexts")
            return False
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(contexts)").fetchall()}
        if "record_key" not in columns:
            self._rebuild_legacy_contexts(conn, columns)
            return True
        for name, definition in _ALTER_COLUMN_DEFINITIONS.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE contexts ADD COLUMN {name} {definition}")
        return False

    def _create_contexts_table(self, conn: sqlite3.Connection, table_name: str) -> None:
        conn.execute(
            f"""
            CREATE TABLE {table_name} (
              record_key TEXT PRIMARY KEY,
              uri TEXT NOT NULL,
              tenant_id TEXT NOT NULL,
              owner_user_id TEXT NOT NULL DEFAULT '',
              project_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              workspace_shared INTEGER NOT NULL DEFAULT 0,
              session_id TEXT NOT NULL DEFAULT '',
              adapter_id TEXT NOT NULL DEFAULT '',
              context_type TEXT NOT NULL DEFAULT '',
              source_kind TEXT NOT NULL DEFAULT '',
              record_kind TEXT NOT NULL DEFAULT 'context',
              lifecycle_state TEXT NOT NULL DEFAULT 'active',
              admission_status TEXT NOT NULL DEFAULT '',
              claim_state TEXT NOT NULL DEFAULT '',
              slot_id TEXT NOT NULL DEFAULT '',
              memory_type TEXT NOT NULL DEFAULT '',
              scope_keys TEXT NOT NULL DEFAULT '[]',
              scope_signature TEXT NOT NULL DEFAULT '',
              parent_uri TEXT NOT NULL DEFAULT '',
              primary_tree_path TEXT NOT NULL DEFAULT '',
              path_depth INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL DEFAULT '',
              event_time TEXT NOT NULL DEFAULT '',
              ingested_at TEXT NOT NULL DEFAULT '',
              transaction_time TEXT NOT NULL DEFAULT '',
              valid_from TEXT NOT NULL DEFAULT '',
              valid_to TEXT NOT NULL DEFAULT '',
              title TEXT NOT NULL DEFAULT '',
              l0_text TEXT NOT NULL DEFAULT '',
              l1_text TEXT NOT NULL DEFAULT '',
              l2_uri TEXT NOT NULL DEFAULT '',
              source_uri TEXT NOT NULL DEFAULT '',
              source_digest TEXT NOT NULL DEFAULT '',
              source_revision INTEGER NOT NULL DEFAULT 0,
              canonical_slot_id TEXT NOT NULL DEFAULT '',
              canonical_slot_uri TEXT NOT NULL DEFAULT '',
              canonical_claim_id TEXT NOT NULL DEFAULT '',
              canonical_claim_uri TEXT NOT NULL DEFAULT '',
              canonical_revision INTEGER NOT NULL DEFAULT 0,
              canonical_state TEXT NOT NULL DEFAULT '',
              canonical_head_digest TEXT NOT NULL DEFAULT '',
              receipt_digest TEXT NOT NULL DEFAULT '',
              projection_effect_hash TEXT NOT NULL DEFAULT '',
              hotness REAL NOT NULL DEFAULT 0,
              semantic_hotness REAL NOT NULL DEFAULT 0,
              behavior_support_hotness REAL NOT NULL DEFAULT 0,
              serving_tier TEXT NOT NULL DEFAULT 'HOT',
              projection_status TEXT NOT NULL DEFAULT 'PROJECTED',
              metadata_json TEXT NOT NULL DEFAULT '{{}}',
              content_digest TEXT NOT NULL DEFAULT '',
              stored_content_digest TEXT NOT NULL DEFAULT '',
              content_text TEXT NOT NULL DEFAULT '',
              scene_key TEXT NOT NULL DEFAULT '',
              action TEXT NOT NULL DEFAULT '',
              memory_anchor_uri TEXT NOT NULL DEFAULT ''
            )
            """
        )

    def _rebuild_legacy_contexts(self, conn: sqlite3.Connection, columns: set[str]) -> None:
        conn.execute("DROP TABLE IF EXISTS contexts_catalog_new")
        self._create_contexts_table(conn, "contexts_catalog_new")
        cursor = conn.execute("SELECT * FROM contexts ORDER BY uri")
        while batch := cursor.fetchmany(_MIGRATION_BATCH_SIZE):
            for row in batch:
                raw_metadata = self._json_mapping(row["metadata_json"] if "metadata_json" in columns else "{}")
                try:
                    scope_keys = self._scope_keys_from_metadata(raw_metadata)
                except (KeyError, TypeError, ValueError):
                    scope_keys = [_INVALID_SCOPE_KEY]
                record = self._legacy_record(row, columns, raw_metadata)
                prepared = self._prepare_record(
                    record,
                    scope_keys_override=scope_keys,
                    legacy_overrides={
                        "project_id": self._legacy_value(row, columns, "project_id"),
                        "admission_status": self._legacy_value(row, columns, "admission_status"),
                        "claim_state": self._legacy_value(row, columns, "claim_state"),
                        "slot_id": self._legacy_value(row, columns, "slot_id"),
                        "memory_type": self._legacy_value(row, columns, "memory_type"),
                    },
                )
                self._insert_context_row(conn, prepared.values, table_name="contexts_catalog_new")
        conn.execute("DROP TABLE contexts")
        conn.execute("ALTER TABLE contexts_catalog_new RENAME TO contexts")

    def _legacy_record(
        self,
        row: sqlite3.Row,
        columns: set[str],
        metadata: Mapping[str, Any],
    ) -> CatalogRecord:
        updated_at = self._coerce_timestamp(self._legacy_value(row, columns, "updated_at"))
        created_at = self._coerce_timestamp(str(metadata.get("created_at") or updated_at))
        raw_paths = metadata.get("tree_paths")
        tree_paths: tuple[str, ...] = ()
        if isinstance(raw_paths, Sequence) and not isinstance(raw_paths, str | bytes):
            try:
                tree_paths = tuple(normalize_tree_path(item) for item in raw_paths)
            except ValueError:
                tree_paths = ()
        primary = str(metadata.get("primary_tree_path") or (tree_paths[0] if tree_paths else ""))
        if primary:
            try:
                primary = normalize_tree_path(primary)
            except ValueError:
                primary = ""
                tree_paths = ()
        record_kind = str(metadata.get("record_kind") or CatalogRecordKind.CONTEXT.value)
        if record_kind not in {kind.value for kind in CatalogRecordKind}:
            record_kind = CatalogRecordKind.CONTEXT.value
        serving_tier = str(metadata.get("serving_tier") or ServingTier.HOT.value).upper()
        if serving_tier not in {tier.value for tier in ServingTier}:
            serving_tier = ServingTier.HOT.value
        projection_status = str(metadata.get("projection_status") or CatalogProjectionStatus.PROJECTED.value).upper()
        if projection_status not in {status.value for status in CatalogProjectionStatus}:
            projection_status = CatalogProjectionStatus.PROJECTED.value
        uri = self._legacy_value(row, columns, "uri")
        content = self._legacy_value(row, columns, "content_text")
        project_id = self._legacy_value(row, columns, "project_id")
        return CatalogRecord(
            record_key=uri,
            uri=uri,
            tenant_id=self._legacy_value(row, columns, "tenant_id") or "default",
            owner_user_id=self._legacy_value(row, columns, "owner_user_id"),
            workspace_id=str(metadata.get("workspace_id") or project_id),
            session_id=str(metadata.get("session_id") or ""),
            adapter_id=self._legacy_value(row, columns, "adapter_id"),
            context_type=self._legacy_value(row, columns, "context_type"),
            source_kind=str(metadata.get("source_kind") or "context"),
            record_kind=record_kind,
            lifecycle_state=self._legacy_value(row, columns, "lifecycle_state") or "active",
            primary_tree_path=primary,
            tree_paths=tree_paths,
            created_at=created_at,
            updated_at=updated_at,
            event_time=self._coerce_timestamp(str(metadata.get("event_time") or created_at)),
            ingested_at=self._coerce_timestamp(str(metadata.get("ingested_at") or created_at)),
            transaction_time=self._coerce_timestamp(str(metadata.get("transaction_time") or updated_at)),
            valid_from=self._coerce_timestamp(str(metadata.get("valid_from") or "")),
            valid_to=self._coerce_timestamp(str(metadata.get("valid_to") or "")),
            title=self._legacy_value(row, columns, "title"),
            l0_text=str(metadata.get("l0_text") or self._legacy_value(row, columns, "title")),
            l1_text=content,
            l2_uri=str(metadata.get("l2_uri") or uri),
            source_uri=str(metadata.get("source_uri") or uri),
            source_digest=str(metadata.get("source_digest") or self.sanitizer.digest(content)),
            source_revision=int(metadata.get("source_revision") or metadata.get("revision") or 0),
            canonical_slot_id=self._legacy_value(row, columns, "slot_id"),
            canonical_slot_uri=str(metadata.get("slot_uri") or metadata.get("canonical_slot_uri") or ""),
            canonical_claim_id=str(metadata.get("claim_id") or metadata.get("canonical_claim_id") or ""),
            canonical_claim_uri=str(metadata.get("canonical_claim_uri") or ""),
            canonical_revision=int(metadata.get("current_revision") or metadata.get("revision") or 0),
            canonical_state=self._legacy_value(row, columns, "claim_state"),
            canonical_head_digest=str(
                metadata.get("canonical_head_digest") or metadata.get("current_head_digest") or ""
            ),
            receipt_digest=str(metadata.get("receipt_digest") or metadata.get("current_receipt_digest") or ""),
            projection_effect_hash=str(metadata.get("projection_effect_hash") or ""),
            hotness=float(self._legacy_value(row, columns, "hotness") or 0.0),
            semantic_hotness=float(self._legacy_value(row, columns, "semantic_hotness") or 0.0),
            behavior_support_hotness=float(self._legacy_value(row, columns, "behavior_support_hotness") or 0.0),
            serving_tier=serving_tier,
            projection_status=projection_status,
            metadata=metadata,
        )

    def _sanitize_existing_rows(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("SELECT record_key FROM contexts ORDER BY record_key")
        while batch := cursor.fetchmany(_MIGRATION_BATCH_SIZE):
            for key_row in batch:
                row = conn.execute(
                    "SELECT * FROM contexts WHERE record_key = ?",
                    (str(key_row["record_key"]),),
                ).fetchone()
                if row is None:
                    continue
                scope_keys = self._json_list(row["scope_keys"])
                record = self._catalog_record_from_row(conn, row)
                legacy_overrides = {
                    "project_id": str(row["project_id"]),
                    "admission_status": str(row["admission_status"]),
                    "claim_state": str(row["claim_state"]),
                    "slot_id": str(row["slot_id"]),
                    "memory_type": str(row["memory_type"]),
                }
                if str(row["content_digest"]):
                    legacy_overrides["content_digest"] = str(row["content_digest"])
                prepared = self._prepare_record(
                    record,
                    scope_keys_override=scope_keys,
                    legacy_overrides=legacy_overrides,
                )
                self._upsert_prepared(conn, prepared)

    def _create_auxiliary_tables(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_paths (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              uri TEXT NOT NULL,
              owner_user_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              workspace_shared INTEGER NOT NULL DEFAULT 0,
              context_type TEXT NOT NULL DEFAULT '',
              record_kind TEXT NOT NULL DEFAULT 'context',
              canonical_slot_id TEXT NOT NULL DEFAULT '',
              canonical_claim_id TEXT NOT NULL DEFAULT '',
              event_time TEXT NOT NULL DEFAULT '',
              transaction_time TEXT NOT NULL DEFAULT '',
              valid_from TEXT NOT NULL DEFAULT '',
              valid_to TEXT NOT NULL DEFAULT '',
              path TEXT NOT NULL,
              path_kind TEXT NOT NULL,
              depth INTEGER NOT NULL,
              is_primary INTEGER NOT NULL CHECK(is_primary IN (0, 1)),
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(tenant_id, record_key, path)
            )
            """
        )
        path_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(context_paths)").fetchall()}
        for name, definition in {
            "workspace_id": "TEXT NOT NULL DEFAULT ''",
            "workspace_shared": "INTEGER NOT NULL DEFAULT 0",
            "record_kind": "TEXT NOT NULL DEFAULT 'context'",
            "canonical_slot_id": "TEXT NOT NULL DEFAULT ''",
            "canonical_claim_id": "TEXT NOT NULL DEFAULT ''",
            "transaction_time": "TEXT NOT NULL DEFAULT ''",
            "valid_from": "TEXT NOT NULL DEFAULT ''",
            "valid_to": "TEXT NOT NULL DEFAULT ''",
        }.items():
            if name not in path_columns:
                conn.execute(f"ALTER TABLE context_paths ADD COLUMN {name} {definition}")
        conn.execute(
            "UPDATE context_paths SET "
            "workspace_id = COALESCE((SELECT c.workspace_id FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), ''), "
            "workspace_shared = COALESCE((SELECT c.workspace_shared FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), 0), "
            "record_kind = COALESCE((SELECT c.record_kind FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), 'context'), "
            "canonical_slot_id = COALESCE((SELECT c.canonical_slot_id FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), ''), "
            "canonical_claim_id = COALESCE((SELECT c.canonical_claim_id FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), ''), "
            "transaction_time = COALESCE((SELECT c.transaction_time FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), ''), "
            "valid_from = COALESCE((SELECT c.valid_from FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), ''), "
            "valid_to = COALESCE((SELECT c.valid_to FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), '')"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_acl_grants (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              grant_kind TEXT NOT NULL,
              grant_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              scope_signature TEXT NOT NULL DEFAULT '',
              uri TEXT NOT NULL DEFAULT '',
              context_type TEXT NOT NULL DEFAULT '',
              source_kind TEXT NOT NULL DEFAULT '',
              record_kind TEXT NOT NULL DEFAULT 'context',
              adapter_id TEXT NOT NULL DEFAULT '',
              adapter_access_id TEXT NOT NULL DEFAULT '',
              session_id TEXT NOT NULL DEFAULT '',
              event_time TEXT NOT NULL DEFAULT '',
              transaction_time TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL,
              PRIMARY KEY(tenant_id, record_key, grant_kind, grant_id, workspace_id)
            )
            """
        )
        grant_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(context_acl_grants)").fetchall()
        }
        for name, definition in {
            "scope_signature": "TEXT NOT NULL DEFAULT ''",
            "uri": "TEXT NOT NULL DEFAULT ''",
            "context_type": "TEXT NOT NULL DEFAULT ''",
            "source_kind": "TEXT NOT NULL DEFAULT ''",
            "record_kind": "TEXT NOT NULL DEFAULT 'context'",
            "adapter_id": "TEXT NOT NULL DEFAULT ''",
            "adapter_access_id": "TEXT NOT NULL DEFAULT ''",
            "session_id": "TEXT NOT NULL DEFAULT ''",
            "event_time": "TEXT NOT NULL DEFAULT ''",
            "transaction_time": "TEXT NOT NULL DEFAULT ''",
        }.items():
            if name not in grant_columns:
                conn.execute(f"ALTER TABLE context_acl_grants ADD COLUMN {name} {definition}")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_path_closure (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              path TEXT NOT NULL,
              ancestor_path TEXT NOT NULL,
              owner_user_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              workspace_shared INTEGER NOT NULL DEFAULT 0,
              scope_signature TEXT NOT NULL DEFAULT '',
              uri TEXT NOT NULL DEFAULT '',
              context_type TEXT NOT NULL DEFAULT '',
              source_kind TEXT NOT NULL DEFAULT '',
              record_kind TEXT NOT NULL DEFAULT 'context',
              adapter_id TEXT NOT NULL DEFAULT '',
              adapter_access_id TEXT NOT NULL DEFAULT '',
              session_id TEXT NOT NULL DEFAULT '',
              canonical_slot_id TEXT NOT NULL DEFAULT '',
              canonical_claim_id TEXT NOT NULL DEFAULT '',
              event_time TEXT NOT NULL DEFAULT '',
              transaction_time TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL DEFAULT '',
              PRIMARY KEY(tenant_id, record_key, path, ancestor_path)
            )
            """
        )
        closure_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(context_path_closure)").fetchall()
        }
        for name, definition in {
            "scope_signature": "TEXT NOT NULL DEFAULT ''",
            "uri": "TEXT NOT NULL DEFAULT ''",
            "source_kind": "TEXT NOT NULL DEFAULT ''",
            "adapter_id": "TEXT NOT NULL DEFAULT ''",
            "adapter_access_id": "TEXT NOT NULL DEFAULT ''",
            "session_id": "TEXT NOT NULL DEFAULT ''",
        }.items():
            if name not in closure_columns:
                conn.execute(
                    f"ALTER TABLE context_path_closure ADD COLUMN {name} {definition}"
                )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_path_acl (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              path TEXT NOT NULL,
              ancestor_path TEXT NOT NULL,
              grant_kind TEXT NOT NULL,
              grant_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              owner_user_id TEXT NOT NULL DEFAULT '',
              scope_signature TEXT NOT NULL DEFAULT '',
              uri TEXT NOT NULL DEFAULT '',
              context_type TEXT NOT NULL DEFAULT '',
              source_kind TEXT NOT NULL DEFAULT '',
              record_kind TEXT NOT NULL DEFAULT 'context',
              adapter_id TEXT NOT NULL DEFAULT '',
              adapter_access_id TEXT NOT NULL DEFAULT '',
              session_id TEXT NOT NULL DEFAULT '',
              event_time TEXT NOT NULL DEFAULT '',
              transaction_time TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL DEFAULT '',
              PRIMARY KEY(
                tenant_id, record_key, path, ancestor_path,
                grant_kind, grant_id, workspace_id
              )
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_tenants (
              tenant_key INTEGER PRIMARY KEY AUTOINCREMENT,
              tenant_id TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_validity_map (
              validity_id INTEGER PRIMARY KEY AUTOINCREMENT,
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS context_validity_rtree USING rtree(
              validity_id,
              tenant_min, tenant_max,
              valid_from_min, valid_from_max,
              valid_to_min, valid_to_max
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_fts_map (
              record_key TEXT PRIMARY KEY,
              fts_rowid INTEGER NOT NULL UNIQUE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_links (
              link_key TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              source_record_key TEXT NOT NULL,
              source_uri TEXT NOT NULL,
              relation_type TEXT NOT NULL,
              target_record_key TEXT NOT NULL DEFAULT '',
              target_uri TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_projection_state (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              source_revision INTEGER NOT NULL DEFAULT 0,
              projection_status TEXT NOT NULL,
              projection_effect_hash TEXT NOT NULL DEFAULT '',
              retry_count INTEGER NOT NULL DEFAULT 0,
              last_error TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL,
              PRIMARY KEY(tenant_id, record_key)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_tombstones (
              tombstone_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              uri TEXT NOT NULL DEFAULT '',
              reason TEXT NOT NULL,
              source_revision INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL,
              payload_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              retry_count INTEGER NOT NULL DEFAULT 0,
              last_error TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS migration_state (
              migration_name TEXT NOT NULL,
              tenant_id TEXT NOT NULL DEFAULT '',
              state TEXT NOT NULL,
              checkpoint TEXT NOT NULL DEFAULT '',
              batch_size INTEGER NOT NULL DEFAULT 0,
              details_json TEXT NOT NULL DEFAULT '{}',
              last_error TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL,
              PRIMARY KEY(migration_name, tenant_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS migration_equivalence_journal (
              proof_id TEXT PRIMARY KEY,
              migration_name TEXT NOT NULL,
              tenant_id TEXT NOT NULL DEFAULT '',
              validation_epoch TEXT NOT NULL DEFAULT '',
              plane TEXT NOT NULL,
              source_identity_digest TEXT NOT NULL,
              evidence_digest TEXT NOT NULL,
              expected_count INTEGER NOT NULL,
              actual_count INTEGER NOT NULL,
              expected_digest TEXT NOT NULL,
              actual_digest TEXT NOT NULL,
              matched INTEGER NOT NULL CHECK(matched IN (0, 1)),
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS migration_shadow_read_journal (
              comparison_id TEXT PRIMARY KEY,
              migration_name TEXT NOT NULL,
              tenant_id TEXT NOT NULL DEFAULT '',
              validation_epoch TEXT NOT NULL,
              plan_digest TEXT NOT NULL,
              legacy_count INTEGER NOT NULL,
              unified_count INTEGER NOT NULL,
              overlap_count INTEGER NOT NULL,
              legacy_digest TEXT NOT NULL,
              unified_digest TEXT NOT NULL,
              matched INTEGER NOT NULL CHECK(matched IN (0, 1)),
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_projection_frontier (
              tenant_id TEXT NOT NULL,
              archive_uri TEXT NOT NULL,
              owner_user_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              session_id TEXT NOT NULL,
              manifest_digest TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL,
              last_error TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(tenant_id, archive_uri)
            )
            """
        )
        frontier_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(session_projection_frontier)").fetchall()
        }
        for name, definition in {
            "owner_user_id": "TEXT NOT NULL DEFAULT ''",
            "workspace_id": "TEXT NOT NULL DEFAULT ''",
        }.items():
            if name not in frontier_columns:
                conn.execute(
                    f"ALTER TABLE session_projection_frontier ADD COLUMN {name} {definition}"
                )
        conn.execute(
            "UPDATE session_projection_frontier SET owner_user_id = "
            "CASE WHEN owner_user_id = '' AND archive_uri LIKE 'memoryos://user/%' "
            "THEN substr(substr(archive_uri, length('memoryos://user/') + 1), 1, "
            "instr(substr(archive_uri, length('memoryos://user/') + 1), '/') - 1) "
            "ELSE owner_user_id END"
        )

    def _ensure_fts_table(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'contexts_fts' AND type IN ('table', 'view')"
        ).fetchone()
        desired = {
            "record_key",
            "uri",
            "title",
            "content_text",
            "metadata_text",
            "search_terms",
            "acl_tokens",
        }
        columns = (
            {str(item[1]) for item in conn.execute("PRAGMA table_info(contexts_fts)").fetchall()}
            if row is not None
            else set()
        )
        if row is not None and columns != desired:
            conn.execute("DROP TABLE contexts_fts")
            row = None
        if row is not None:
            self.fts_enabled = "VIRTUAL TABLE" in str(row["sql"] or "").upper()
            return False
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE contexts_fts USING fts5(
                  record_key UNINDEXED,
                  uri UNINDEXED,
                  title,
                  content_text,
                  metadata_text,
                  search_terms,
                  acl_tokens
                )
                """
            )
            self.fts_enabled = True
        except sqlite3.OperationalError:
            self.fts_enabled = False
            conn.execute(
                """
                CREATE TABLE contexts_fts (
                  record_key TEXT PRIMARY KEY,
                  uri TEXT NOT NULL,
                  title TEXT NOT NULL,
                  content_text TEXT NOT NULL,
                  metadata_text TEXT NOT NULL,
                  search_terms TEXT NOT NULL,
                  acl_tokens TEXT NOT NULL
                )
                """
            )
        return True

    @staticmethod
    def _fts_row_map_is_consistent(conn: sqlite3.Connection) -> bool:
        """Validate the rebuildable FTS rowid map during startup integrity."""

        unmapped = conn.execute(
            "SELECT 1 FROM contexts_fts AS f "
            "LEFT JOIN context_fts_map AS m "
            "ON m.fts_rowid = f.rowid AND m.record_key = f.record_key "
            "WHERE m.record_key IS NULL LIMIT 1"
        ).fetchone()
        if unmapped is not None:
            return False
        orphaned = conn.execute(
            "SELECT 1 FROM context_fts_map AS m "
            "LEFT JOIN contexts_fts AS f ON f.rowid = m.fts_rowid "
            "WHERE f.rowid IS NULL OR f.record_key != m.record_key LIMIT 1"
        ).fetchone()
        return orphaned is None

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        statements = (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_contexts_record_key ON contexts(record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_updated "
            "ON contexts(tenant_id, owner_user_id, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_event "
            "ON contexts(tenant_id, owner_user_id, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_transaction "
            "ON contexts(tenant_id, owner_user_id, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_valid "
            "ON contexts(tenant_id, owner_user_id, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_uri "
            "ON contexts(tenant_id, owner_user_id, uri, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_type_updated "
            "ON contexts(tenant_id, owner_user_id, context_type, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_type_event "
            "ON contexts(tenant_id, owner_user_id, context_type, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_type_transaction "
            "ON contexts(tenant_id, owner_user_id, context_type, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_type_valid "
            "ON contexts(tenant_id, owner_user_id, context_type, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_source_updated "
            "ON contexts(tenant_id, owner_user_id, source_kind, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_source_event "
            "ON contexts(tenant_id, owner_user_id, source_kind, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_source_transaction "
            "ON contexts(tenant_id, owner_user_id, source_kind, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_source_valid "
            "ON contexts(tenant_id, owner_user_id, source_kind, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_session_updated "
            "ON contexts(tenant_id, owner_user_id, session_id, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_session_event "
            "ON contexts(tenant_id, owner_user_id, session_id, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_session_transaction "
            "ON contexts(tenant_id, owner_user_id, session_id, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_session_valid "
            "ON contexts(tenant_id, owner_user_id, session_id, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_public_updated "
            "ON contexts(tenant_id, updated_at DESC, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_public_event "
            "ON contexts(tenant_id, event_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_public_transaction "
            "ON contexts(tenant_id, transaction_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_public_valid "
            "ON contexts(tenant_id, valid_from, valid_to, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_shared_workspace_updated "
            "ON contexts(tenant_id, workspace_id, updated_at DESC, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_shared_workspace_event "
            "ON contexts(tenant_id, workspace_id, event_time, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_shared_workspace_transaction "
            "ON contexts(tenant_id, workspace_id, transaction_time, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_shared_workspace_valid "
            "ON contexts(tenant_id, workspace_id, valid_from, valid_to, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_uri ON contexts(tenant_id, uri, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_uri_kind_updated "
            "ON contexts(tenant_id, uri, record_kind, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_canonical_slot_uri "
            "ON contexts(tenant_id, canonical_slot_uri, record_kind, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_canonical_claim_uri "
            "ON contexts(tenant_id, canonical_claim_uri, record_kind, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_owner_type ON contexts(tenant_id, owner_user_id, context_type, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_workspace ON contexts(tenant_id, workspace_id, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_session ON contexts(tenant_id, session_id, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_record_kind ON contexts(tenant_id, record_kind, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_created_at ON contexts(tenant_id, created_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_event_time ON contexts(tenant_id, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_ingested_at ON contexts(tenant_id, ingested_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_transaction_time ON contexts(tenant_id, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_updated_at ON contexts(tenant_id, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_valid_interval ON contexts(tenant_id, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_scene_key ON contexts(tenant_id, scene_key, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_action ON contexts(tenant_id, action, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_anchor ON contexts(tenant_id, memory_anchor_uri, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_projection_evidence ON contexts(tenant_id, source_uri, projection_effect_hash, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_claim_revision ON contexts(tenant_id, canonical_claim_id, canonical_revision)",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_contexts_current_slot ON contexts(tenant_id, canonical_slot_id) "
            "WHERE record_kind = 'current_slot' AND canonical_slot_id != '' "
            "AND lifecycle_state NOT IN ('deleted', 'obsolete')",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_context_paths_primary ON context_paths(tenant_id, record_key) WHERE is_primary = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_tenant_path ON context_paths(tenant_id, path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_path ON context_paths(tenant_id, owner_user_id, path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_type_path "
            "ON context_paths(tenant_id, owner_user_id, context_type, path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_type_path_event "
            "ON context_paths(tenant_id, owner_user_id, context_type, path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_type_path_transaction "
            "ON context_paths(tenant_id, owner_user_id, context_type, path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_type_path_updated "
            "ON context_paths(tenant_id, owner_user_id, context_type, path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_type_path_valid "
            "ON context_paths(tenant_id, owner_user_id, context_type, path, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_path_event "
            "ON context_paths(tenant_id, owner_user_id, path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_path_transaction "
            "ON context_paths(tenant_id, owner_user_id, path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_path_updated "
            "ON context_paths(tenant_id, owner_user_id, path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_path_valid "
            "ON context_paths(tenant_id, owner_user_id, path, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_public_path ON context_paths(tenant_id, path, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_public_path_event "
            "ON context_paths(tenant_id, path, event_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_public_path_transaction "
            "ON context_paths(tenant_id, path, transaction_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_public_path_updated "
            "ON context_paths(tenant_id, path, updated_at, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_public_path_valid "
            "ON context_paths(tenant_id, path, valid_from, valid_to, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_shared_workspace_path "
            "ON context_paths(tenant_id, workspace_id, path, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_shared_workspace_path_event "
            "ON context_paths(tenant_id, workspace_id, path, event_time, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_shared_workspace_path_transaction "
            "ON context_paths(tenant_id, workspace_id, path, transaction_time, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_shared_workspace_path_updated "
            "ON context_paths(tenant_id, workspace_id, path, updated_at, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_shared_workspace_path_valid "
            "ON context_paths(tenant_id, workspace_id, path, valid_from, valid_to, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_type_path ON context_paths(tenant_id, context_type, path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_time_path ON context_paths(tenant_id, event_time, path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_path_time ON context_paths(tenant_id, path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_path_event "
            "ON context_paths(tenant_id, path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_path_transaction "
            "ON context_paths(tenant_id, path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_path_updated "
            "ON context_paths(tenant_id, path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_path_valid "
            "ON context_paths(tenant_id, path, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_uri ON context_paths(tenant_id, uri, path)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_tenant_ancestor "
            "ON context_path_closure(tenant_id, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_scope_kind_ancestor "
            "ON context_path_closure(tenant_id, scope_signature, record_kind, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_scope_kind_ancestor_event "
            "ON context_path_closure(tenant_id, scope_signature, record_kind, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_scope_kind_ancestor_transaction "
            "ON context_path_closure(tenant_id, scope_signature, record_kind, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_scope_kind_ancestor_updated "
            "ON context_path_closure(tenant_id, scope_signature, record_kind, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_type_ancestor "
            "ON context_path_closure(tenant_id, context_type, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_ancestor_event "
            "ON context_path_closure(tenant_id, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_ancestor_transaction "
            "ON context_path_closure(tenant_id, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_ancestor_updated "
            "ON context_path_closure(tenant_id, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_ancestor "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_ancestor "
            "ON context_path_closure(tenant_id, owner_user_id, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_type_ancestor "
            "ON context_path_closure(tenant_id, owner_user_id, context_type, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_ancestor_event "
            "ON context_path_closure(tenant_id, owner_user_id, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_ancestor_transaction "
            "ON context_path_closure(tenant_id, owner_user_id, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_ancestor_updated "
            "ON context_path_closure(tenant_id, owner_user_id, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_type_ancestor_event "
            "ON context_path_closure(tenant_id, owner_user_id, context_type, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_type_ancestor_transaction "
            "ON context_path_closure(tenant_id, owner_user_id, context_type, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_type_ancestor_updated "
            "ON context_path_closure(tenant_id, owner_user_id, context_type, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_type_ancestor "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, context_type, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_ancestor_event "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_ancestor_transaction "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_ancestor_updated "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_type_ancestor_event "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, context_type, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_type_ancestor_transaction "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, context_type, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_type_ancestor_updated "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, context_type, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_workspace_ancestor "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_ancestor "
            "ON context_path_closure(tenant_id, ancestor_path, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_ancestor_event "
            "ON context_path_closure(tenant_id, ancestor_path, event_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_ancestor_transaction "
            "ON context_path_closure(tenant_id, ancestor_path, transaction_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_ancestor_updated "
            "ON context_path_closure(tenant_id, ancestor_path, updated_at, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_workspace_ancestor_event "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, event_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_workspace_ancestor_transaction "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, transaction_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_workspace_ancestor_updated "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, updated_at, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_shared_workspace_ancestor "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_shared_workspace_ancestor_event "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, event_time, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_shared_workspace_ancestor_transaction "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, transaction_time, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_shared_workspace_ancestor_updated "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, updated_at, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_updated "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_event "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_transaction "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_scope_updated "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, scope_signature, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_scope_event "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, scope_signature, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_scope_transaction "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, scope_signature, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_type_updated "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, context_type, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_adapter_access_updated "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_access_id, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_source_updated "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, source_kind, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_session_updated "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, session_id, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_uri "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, uri, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_validity_map_tenant_record "
            "ON context_validity_map(tenant_id, record_key, validity_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_lookup "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_access_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, updated_at DESC, record_key DESC, workspace_id, record_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_access_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, event_time DESC, record_key DESC, workspace_id, record_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_access_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, transaction_time DESC, record_key DESC, workspace_id, record_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_access_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, updated_at DESC, record_key DESC, record_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_access_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, event_time DESC, record_key DESC, record_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_access_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, transaction_time DESC, record_key DESC, record_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_kind_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, record_kind, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_kind_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, record_kind, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_kind_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, record_kind, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_scope_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, scope_signature, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_scope_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, scope_signature, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_scope_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, scope_signature, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_type_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, context_type, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_type_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, context_type, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_type_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, context_type, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_adapter_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_id, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_adapter_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_id, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_adapter_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_id, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_adapter_access_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_access_id, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_adapter_access_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_access_id, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_adapter_access_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_access_id, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_source_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, source_kind, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_source_type_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, source_kind, context_type, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_session_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, session_id, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_session_type_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, session_id, context_type, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_uri "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, uri, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_record "
            "ON context_acl_grants(tenant_id, record_key, grant_kind, grant_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_links_source ON context_links(tenant_id, source_record_key, relation_type)",
            "CREATE INDEX IF NOT EXISTS idx_context_links_target ON context_links(tenant_id, target_record_key, relation_type)",
            "CREATE INDEX IF NOT EXISTS idx_context_tombstones_status ON context_tombstones(status, updated_at, tombstone_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_tombstones_tenant_uri_status "
            "ON context_tombstones(tenant_id, uri, status, updated_at, tombstone_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_tombstones_tenant_uri_id "
            "ON context_tombstones(tenant_id, uri, tombstone_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_migration_state_status ON migration_state(state, updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_migration_equivalence_epoch ON migration_equivalence_journal(migration_name, tenant_id, validation_epoch, created_at, proof_id)",
            "CREATE INDEX IF NOT EXISTS idx_migration_shadow_read_epoch "
            "ON migration_shadow_read_journal(migration_name, tenant_id, validation_epoch, created_at, comparison_id)",
            "CREATE INDEX IF NOT EXISTS idx_session_projection_frontier_status "
            "ON session_projection_frontier(tenant_id, status, updated_at, archive_uri)",
            "CREATE INDEX IF NOT EXISTS idx_session_projection_frontier_scope_status "
            "ON session_projection_frontier(tenant_id, owner_user_id, workspace_id, status, updated_at, archive_uri)",
            "CREATE INDEX IF NOT EXISTS idx_session_projection_frontier_replay "
            "ON session_projection_frontier(tenant_id, archive_uri, status)",
        )
        for statement in statements:
            conn.execute(statement)
        for dimension, column in (
            ("scope", "scope_signature"),
            ("type", "context_type"),
            ("source", "source_kind"),
            ("session", "session_id"),
            ("adapter", "adapter_id"),
            ("adapter_access", "adapter_access_id"),
        ):
            for time_name, time_column in (
                ("updated", "updated_at"),
                ("event", "event_time"),
                ("transaction", "transaction_time"),
            ):
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS "
                    f"idx_context_acl_grants_access_{dimension}_{time_name} "
                    "ON context_acl_grants("
                    f"tenant_id, grant_kind, grant_id, {column}, "
                    f"{time_column} DESC, record_key DESC, record_kind)"
                )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_access_uri "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, uri, record_key)"
        )

    def _migrate_scope_keys(self, conn: sqlite3.Connection) -> None:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version >= _SCOPE_KEY_SCHEMA_VERSION:
            conn.execute(
                f"UPDATE contexts SET scope_keys = '[\"{_INVALID_SCOPE_KEY}\"]' "
                "WHERE NOT json_valid(scope_keys) OR json_type(scope_keys) != 'array'"
            )
            return
        cursor = conn.execute("SELECT record_key, metadata_json FROM contexts ORDER BY record_key")
        while batch := cursor.fetchmany(_MIGRATION_BATCH_SIZE):
            for row in batch:
                try:
                    metadata = json.loads(str(row["metadata_json"] or "{}"))
                    if not isinstance(metadata, Mapping):
                        raise ValueError("metadata must be an object")
                    keys = self._scope_keys_from_metadata(metadata)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    keys = [_INVALID_SCOPE_KEY]
                conn.execute(
                    "UPDATE contexts SET scope_keys = ? WHERE record_key = ?",
                    (self._json_dump(keys), str(row["record_key"])),
                )

    def _rebuild_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM contexts_fts")
        conn.execute("DELETE FROM context_fts_map")
        cursor = conn.execute("SELECT * FROM contexts ORDER BY record_key")
        while batch := cursor.fetchmany(_MIGRATION_BATCH_SIZE):
            for row in batch:
                metadata = self._json_mapping(row["metadata_json"])
                metadata_text = self._safe_metadata_text(metadata)
                item = _PreparedCatalogRecord(
                    record=self._catalog_record_from_row(conn, row),
                    values={},
                    scope_signature=self._scope_signature(self._json_list(row["scope_keys"])),
                    fts_metadata_text=metadata_text,
                    fts_search_terms=" ".join(
                        lexical_terms(
                            " ".join(
                                (
                                    str(row["title"]),
                                    str(row["l0_text"]),
                                    str(row["content_text"]),
                                    metadata_text,
                                )
                            )
                        )
                    ),
                )
                self._replace_fts(conn, item)

    # ------------------------------------------------------------------
    # Helpers

    def _catalog_record_from_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> CatalogRecord:
        paths = tuple(
            str(item["path"])
            for item in conn.execute(
                "SELECT path FROM context_paths WHERE tenant_id = ? AND record_key = ? "
                "ORDER BY is_primary DESC, path",
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
            created_at=self._coerce_timestamp(str(row["created_at"])),
            updated_at=self._coerce_timestamp(str(row["updated_at"])),
            event_time=self._coerce_timestamp(str(row["event_time"])),
            ingested_at=self._coerce_timestamp(str(row["ingested_at"])),
            transaction_time=self._coerce_timestamp(str(row["transaction_time"])),
            valid_from=self._coerce_timestamp(str(row["valid_from"])),
            valid_to=self._coerce_timestamp(str(row["valid_to"])),
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
            metadata=self._json_mapping(row["metadata_json"]),
        )

    def _catalog_records_from_rows(
        self,
        conn: sqlite3.Connection,
        rows: Sequence[sqlite3.Row],
    ) -> list[CatalogRecord]:
        return [self._catalog_record_from_row(conn, row) for row in rows]

    def _scope_keys_from_metadata(self, metadata: Mapping[str, Any]) -> list[str]:
        from memoryos.memory.canonical.scope import MemoryScope, scope_key_from_payload

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
            canonical_scope = MemoryScope.from_dict(raw_scope)
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
        self.sanitizer.assert_safe(text)
        return text

    def _safe_reference_uri(self, value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        if raw.startswith("file://") or Path(raw).is_absolute():
            name, location = self.sanitizer.sanitize_path(raw)
            return f"resource://{location or 'external'}/{name}" if name else "resource://redacted"
        return str(self.sanitizer.sanitize_trace(raw))

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
        artifact_root = self.path.parent.parent
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
        tenants = self._filter_values(filters.get("tenant_id"), allow_empty=True)
        if not tenants:
            raise ValueError("ACL-bound FTS queries require tenant_id")
        tenant_id = str(tenants[0])
        workspace_values = tuple(
            str(item)
            for item in self._filter_values(filters.get("workspace_access_ids", ()), allow_empty=True)
        )
        workspace_constrained = filters.get("workspace_access_ids") is not None
        acl_tokens: list[str] = []
        if principal is None:
            if filters.get("owner_user_id") == "" and filters.get("require_unscoped"):
                for workspace_id in workspace_values if workspace_constrained else ("*",):
                    acl_tokens.append(self._grant_acl_token(tenant_id, "public", "", workspace_id))
            else:
                raw_record_keys = filters.get("record_keys")
                if raw_record_keys is None:
                    return content_query
                record_tokens = [
                    self._acl_token(tenant_id, "record_key", value)
                    for value in self._filter_values(raw_record_keys)
                ]
                return (
                    "acl_tokens : ("
                    + " OR ".join(record_tokens)
                    + f") AND ({content_query})"
                )
        else:
            visible_workspaces = workspace_values if workspace_constrained else ("*",)
            for workspace_id in visible_workspaces:
                acl_tokens.append(
                    self._grant_acl_token(tenant_id, "principal", str(principal), workspace_id)
                )
                acl_tokens.append(self._grant_acl_token(tenant_id, "public", "", workspace_id))
                acl_tokens.append(self._grant_acl_token(tenant_id, "tenant", "", workspace_id))
                if filters.get("service_access_id"):
                    acl_tokens.append(
                        self._grant_acl_token(
                            tenant_id,
                            "service",
                            str(filters["service_access_id"]),
                            workspace_id,
                        )
                    )
            for workspace_id in workspace_values:
                if workspace_id not in {"", "__memoryos_principal_only__"}:
                    acl_tokens.append(
                        self._grant_acl_token(tenant_id, "workspace", workspace_id, workspace_id)
                    )
        clauses = ["acl_tokens : (" + " OR ".join(dict.fromkeys(acl_tokens)) + ")"]
        raw_record_keys = filters.get("record_keys")
        if raw_record_keys is not None:
            record_tokens = [
                self._acl_token(tenant_id, "record_key", value)
                for value in self._filter_values(raw_record_keys)
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
                self._acl_token(tenant_id, token_kind, value)
                for value in self._filter_values(raw_values)
            ]
            clauses.append("acl_tokens : (" + " OR ".join(field_tokens) + ")")
        raw_paths = filters.get("target_paths", filters.get("path_prefixes"))
        if raw_paths is not None:
            path_tokens = [
                self._acl_token(tenant_id, "tree_path", normalize_tree_path(value))
                for value in self._filter_values(raw_paths)
            ]
            clauses.append("acl_tokens : (" + " OR ".join(path_tokens) + ")")
        raw_kinds = filters.get("record_kinds", filters.get("record_kind"))
        if raw_kinds is not None:
            kind_tokens = [
                self._acl_token(tenant_id, "record_kind", value)
                for value in self._filter_values(raw_kinds)
            ]
            clauses.append("acl_tokens : (" + " OR ".join(kind_tokens) + ")")
        if filters.get("adapter_id") is not None:
            adapter_tokens = [
                self._acl_token(tenant_id, "adapter", value)
                for value in self._filter_values(filters["adapter_id"])
            ]
            clauses.append("acl_tokens : (" + " OR ".join(adapter_tokens) + ")")
        if filters.get("adapter_access_id") is not None:
            adapter_access_tokens = [self._acl_token(tenant_id, "adapter_access", "*")]
            adapter_access_tokens.extend(
                self._acl_token(tenant_id, "adapter_access", value)
                for value in self._filter_values(filters["adapter_access_id"])
            )
            clauses.append("acl_tokens : (" + " OR ".join(adapter_access_tokens) + ")")
        raw_scopes = filters.get("applicability_scope_keys")
        if raw_scopes:
            scope_tokens = [
                self._acl_token(tenant_id, "scope_signature", signature)
                for signature in self._scope_signature_options(self._filter_values(raw_scopes))
            ]
            clauses.append("acl_tokens : (" + " OR ".join(scope_tokens) + ")")
        elif filters.get("require_unscoped"):
            clauses.append(
                "acl_tokens : ("
                + self._acl_token(tenant_id, "scope_signature", self._scope_signature(()))
                + ")"
            )
        clauses.append(f"({content_query})")
        return " AND ".join(clauses)

    def _fts_acl_tokens(self, record: CatalogRecord, *, scope_signature: str) -> str:
        tokens: list[str] = []
        for grant_kind, grant_id, workspace_id in self._acl_grants_for_record(record):
            tokens.append(self._grant_acl_token(record.tenant_id, grant_kind, grant_id, workspace_id))
            tokens.append(self._grant_acl_token(record.tenant_id, grant_kind, grant_id, "*"))
        tokens.append(self._acl_token(record.tenant_id, "record_kind", record.record_kind))
        tokens.append(self._acl_token(record.tenant_id, "record_key", record.record_key))
        tokens.append(self._acl_token(record.tenant_id, "context_type", record.context_type))
        tokens.append(self._acl_token(record.tenant_id, "source_kind", record.source_kind))
        tokens.append(self._acl_token(record.tenant_id, "session_id", record.session_id))
        tokens.append(self._acl_token(record.tenant_id, "uri", record.uri))
        for path in record.tree_paths:
            tokens.extend(
                self._acl_token(record.tenant_id, "tree_path", ancestor)
                for ancestor in _path_ancestors(path)
            )
        tokens.append(self._acl_token(record.tenant_id, "adapter", record.adapter_id))
        tokens.append(
            self._acl_token(record.tenant_id, "adapter_access", self._adapter_access_value(record))
        )
        tokens.append(self._acl_token(record.tenant_id, "scope_signature", scope_signature))
        return " ".join(dict.fromkeys(tokens))

    def _grant_acl_token(self, tenant_id: str, grant_kind: str, grant_id: str, workspace_id: str) -> str:
        return self._acl_token(tenant_id, f"grant:{grant_kind}:{workspace_id}", grant_id)

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
                signatures.append(self._scope_signature(subset))
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
        return self._mapping(decoded)

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
        row = conn.execute(
            "SELECT tenant_key FROM context_tenants WHERE tenant_id = ?", (str(tenant_id),)
        ).fetchone()
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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
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

        limit = max(_ONLINE_PROGRESS_GRANULARITY, int(self.online_vm_step_limit))
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
                raise CatalogCandidateBoundExceeded(
                    f"online Catalog query exceeded {limit} SQLite VM steps"
                ) from exc
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
        valid_point = self._timestamp_number(str(filters["valid_at"]), lower=True)
        tenants = (
            self._filter_values(filters["tenant_id"])
            if filters.get("tenant_id") is not None
            else []
        )
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
            principal_owner is None
            and filters.get("owner_user_id") == ""
            and filters.get("require_unscoped")
        )
        workspace_access = tuple(
            self._filter_values(filters.get("workspace_access_ids", ()), allow_empty=True)
        )
        shared_workspaces = tuple(
            value
            for value in workspace_access
            if value not in {"", "__memoryos_principal_only__"}
        )
        access_predicates: list[str] = []
        access_params: list[Any] = []
        if principal_owner is not None:
            access_predicates.extend(
                ("(vg.grant_kind = 'principal' AND vg.grant_id = ?)", "vg.grant_kind = 'tenant'")
            )
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
                workspace_sql = (
                    " AND vg.workspace_id IN ("
                    + ",".join("?" for _ in workspace_access)
                    + ")"
                )
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
        with self._connect() as conn:
            rows = self._online_fetchall(conn, sql, params)
        record_keys = tuple(dict.fromkeys(str(row["record_key"]) for row in rows))
        if not record_keys:
            return None
        if len(record_keys) > threshold:
            return narrowed
        existing = filters.get("record_keys")
        if existing is not None:
            allowed = set(self._filter_values(existing, allow_empty=True))
            record_keys = tuple(key for key in record_keys if key in allowed)
            if not record_keys:
                return None
        narrowed["record_keys"] = record_keys
        return narrowed


SqliteIndexStore = SQLiteIndexStore

__all__ = [
    "CatalogCandidateBoundExceeded",
    "SQLiteIndexStore",
    "SqliteIndexStore",
    "lexical_match_count",
    "lexical_relevance",
    "lexical_terms",
]
