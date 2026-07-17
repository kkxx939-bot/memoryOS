"""SQLite-backed, rebuildable Unified Context Catalog serving index."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _ONLINE_VM_STEP_LIMIT,
    Any,
    CatalogRecord,
    ContextObject,
    ContextProjectionSanitizer,
    IndexHit,
    Mapping,
    Path,
    Sequence,
    _PreparedCatalogRecord,
    lexical_match_count,
    lexical_relevance,
    lexical_terms,
    os,
    sqlite3,
)
from memoryos.adapters.persistence.sqlite.base_filter import BaseFilterBuilder
from memoryos.adapters.persistence.sqlite.catalog import CatalogStoreOperations
from memoryos.adapters.persistence.sqlite.connection import SQLiteConnectionManager
from memoryos.adapters.persistence.sqlite.migration_state import MigrationStateOperations
from memoryos.adapters.persistence.sqlite.query_filters import QueryFilterBuilder
from memoryos.adapters.persistence.sqlite.schema import SchemaManager
from memoryos.adapters.persistence.sqlite.schema_migrations import SchemaMigrationManager
from memoryos.adapters.persistence.sqlite.search import CatalogSearchOperations
from memoryos.adapters.persistence.sqlite.serialization import CatalogSerializer
from memoryos.adapters.persistence.sqlite.tombstones import TombstoneOperations


class SQLiteIndexStore:
    """Stable facade for the decomposed SQLite catalog adapter.

    The facade owns lifecycle and transaction-visible configuration. Concrete
    catalog, query, schema, migration and serialization behavior is delegated
    explicitly to responsibility components.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fts_enabled = True
        self.online_vm_step_limit = _ONLINE_VM_STEP_LIMIT
        self.sanitizer = ContextProjectionSanitizer()
        self._catalog = CatalogStoreOperations(self)
        self._search = CatalogSearchOperations(self)
        self._tombstones = TombstoneOperations(self)
        self._migration_state = MigrationStateOperations(self)
        self._query_filters = QueryFilterBuilder(self)
        self._base_filter = BaseFilterBuilder(self)
        self._schema = SchemaManager(self)
        self._schema_migrations = SchemaMigrationManager(self)
        self._serialization = CatalogSerializer(self)
        self._connection = SQLiteConnectionManager(self)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self._init_db()
        os.chmod(self.path, 0o600)

    def upsert_index(self, obj: ContextObject, content: str = "") -> None:
        """Project a legacy ContextObject through the same sanitized catalog writer."""
        return self._catalog.upsert_index(obj, content)

    def delete_index(self, uri: str) -> None:
        """Delete every serving record for a legacy URI without touching SourceStore."""
        return self._catalog.delete_index(uri)

    def indexed_uris(self) -> list[str]:
        return self._catalog.indexed_uris()

    def get_index_metadata(self, uri: str) -> dict[str, Any] | None:
        """Return the legacy record first, then a deterministic projection for the URI."""
        return self._catalog.get_index_metadata(uri)

    def ordinary_relation_endpoint_state(self, uri: str, *, tenant_id: str, session_id: str = "") -> str:
        """Resolve relation endpoint liveness through durable delete barriers."""
        return self._catalog.ordinary_relation_endpoint_state(uri, tenant_id=tenant_id, session_id=session_id)

    def clear(self) -> None:
        """Clear rebuildable serving data while retaining migration and tombstone journals."""
        return self._catalog.clear()

    def begin_tenant_serving_rebuild(
        self, migration_name: str, *, tenant_id: str, batch_size: int, details: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Atomically gate and clear one tenant's rebuildable serving rows.

        The durable gate and destructive Catalog mutation share one SQLite
        transaction.  A process crash therefore observes either the old
        serving snapshot or an explicit BACKFILLING row that startup can
        resume; it can never observe a cleared Catalog with a COMPLETED gate.
        Tombstones, migration journals, Session frontiers and immutable Source
        evidence are intentionally retained.
        """
        return self._catalog.begin_tenant_serving_rebuild(
            migration_name, tenant_id=tenant_id, batch_size=batch_size, details=details
        )

    def rebuildable_catalog_records(self, records: Sequence[CatalogRecord]) -> tuple[CatalogRecord, ...]:
        """Filter an offline rebuild batch through durable delete ownership.

        APPLIED tombstones suppress the same or an older Source revision.
        CLEANING remains a hard retry boundary because its Vector/Relation
        consumers have not reached a durable terminal state.
        """
        return self._catalog.rebuildable_catalog_records(records)

    def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
        """Legacy online search; structured filters are applied before every LIMIT."""
        return self._search.search(query, filters, limit)

    def list_legacy_catalog(self, *, filters: Mapping[str, Any] | None = None, limit: int = 100) -> list[CatalogRecord]:
        """Bounded rollback reader over the pre-unification flat Catalog shape.

        This deliberately does not use ACL-grant, closure, relation, vector,
        or validity adjuncts.  It is a conservative owner/public compatibility
        route over the same evolved ``contexts`` table, not a second Catalog.
        """
        return self._search.list_legacy_catalog(filters=filters, limit=limit)

    def search_legacy_catalog(
        self, query: str, *, filters: Mapping[str, Any] | None = None, limit: int = 10
    ) -> list[IndexHit]:
        """Run the independently bounded legacy lexical read for shadow/rollback."""
        return self._search.search_legacy_catalog(query, filters=filters, limit=limit)

    def upsert_catalog(self, record: CatalogRecord | Mapping[str, Any]) -> None:
        """Atomically sanitize and upsert one rebuildable catalog record."""
        return self._catalog.upsert_catalog(record)

    def upsert_catalog_batch(self, records: Sequence[CatalogRecord | Mapping[str, Any]]) -> int:
        """Atomically project a batch; any validation, sanitization, or write error rolls it back."""
        return self._catalog.upsert_catalog_batch(records)

    def get_catalog(self, record_key: str, *, tenant_id: str | None = None) -> CatalogRecord | None:
        return self._catalog.get_catalog(record_key, tenant_id=tenant_id)

    def get_catalog_by_uri(self, uri: str, *, tenant_id: str | None = None, limit: int = 100) -> list[CatalogRecord]:
        return self._catalog.get_catalog_by_uri(uri, tenant_id=tenant_id, limit=limit)

    def list_catalog(self, *, filters: Mapping[str, Any] | None = None, limit: int = 100) -> list[CatalogRecord]:
        return self._catalog.list_catalog(filters=filters, limit=limit)

    def list_catalog_projection_records(
        self, *, tenant_id: str, source_uri: str, projection_effect_hash: str, limit: int = 1001
    ) -> list[CatalogRecord]:
        """Read one evidence-bound projection set for offline/shadow proof.

        This exact identity lookup is intentionally separate from online
        search.  The extra row above the 1000-record proof bound lets callers
        fail closed instead of certifying a truncated projection.
        """
        return self._catalog.list_catalog_projection_records(
            tenant_id=tenant_id, source_uri=source_uri, projection_effect_hash=projection_effect_hash, limit=limit
        )

    def scan_catalog_batch(
        self, *, after_record_key: str = "", filters: Mapping[str, Any] | None = None, limit: int = 256
    ) -> list[CatalogRecord]:
        """Return a stable keyset-paginated batch for offline repair and GC.

        Online retrieval uses ``search_catalog``.  This administrative API is
        deliberately keyset-paginated so retention and rebuild jobs never
        materialize the full catalog or become sensitive to rows whose
        ``updated_at`` changes while a batch is processed.
        """
        return self._catalog.scan_catalog_batch(after_record_key=after_record_key, filters=filters, limit=limit)

    def catalog_schema_version(self) -> int:
        """Return the durable SQLite schema version used by migration gates."""
        return self._catalog.catalog_schema_version()

    def gc_orphan_paths(self, *, limit: int = 256) -> int:
        """Delete a bounded batch of paths whose rebuildable record is gone."""
        return self._catalog.gc_orphan_paths(limit=limit)

    def gc_applied_tombstones(self, *, updated_before: str, limit: int = 256) -> int:
        """Expire only old tombstones proven safe to forget.

        Stale tombstones did not delete the newer projection and are safe once
        aged out.  Applied tombstones remain durable by default; a projection
        owner must explicitly persist ``payload.gc_safe=true`` after proving
        that replay cannot resurrect the deleted source revision.
        """
        return self._catalog.gc_applied_tombstones(updated_before=updated_before, limit=limit)

    def search_catalog(
        self, query: str, *, filters: Mapping[str, Any] | None = None, limit: int = 10
    ) -> list[IndexHit]:
        """Return record-key-distinct exact/FTS candidates without Python row scans."""
        return self._search.search_catalog(query, filters=filters, limit=limit)

    def delete_catalog(self, record_key: str, *, tenant_id: str | None = None) -> bool:
        return self._catalog.delete_catalog(record_key, tenant_id=tenant_id)

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
        return self._tombstones.apply_tombstone(
            tenant_id=tenant_id,
            record_key=record_key,
            reason=reason,
            uri=uri,
            source_revision=source_revision,
            tombstone_id=tombstone_id,
            payload=payload,
        )

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
        return self._tombstones.enqueue_tombstone(
            tenant_id=tenant_id,
            record_key=record_key,
            reason=reason,
            uri=uri,
            source_revision=source_revision,
            tombstone_id=tombstone_id,
            payload=payload,
        )

    def get_pending_tombstones(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return self._tombstones.get_pending_tombstones(limit=limit)

    def get_pending_tombstones_for_uri(
        self, uri: str, *, tenant_id: str, after_tombstone_id: str = "", limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Recover one delete target's exact unfinished journal without queue starvation."""
        return self._tombstones.get_pending_tombstones_for_uri(
            uri, tenant_id=tenant_id, after_tombstone_id=after_tombstone_id, limit=limit
        )

    def get_tombstones(self, tombstone_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Read an explicit bounded set of durable tombstones in caller order.

        Delete callers use this exact-ID path after they have durably enqueued
        every affected projection.  It avoids starvation behind unrelated
        failed journal entries and does not depend on the pending queue's
        1,000-row administrative batch limit.
        """
        return self._tombstones.get_tombstones(tombstone_ids)

    def pending_tombstones(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Backward-compatible alias for get_pending_tombstones()."""
        return self._tombstones.pending_tombstones(limit=limit)

    def mark_tombstone_applied(self, tombstone_id: str) -> dict[str, Any] | None:
        """Idempotently apply a queued tombstone and close its durable journal row."""
        return self._tombstones.mark_tombstone_applied(tombstone_id)

    def begin_tombstone_cleanup(self, tombstone_id: str) -> dict[str, Any] | None:
        """Atomically establish deletion ownership before external cleanup.

        ``CLEANING`` is a durable, replayable intermediate state.  The Catalog
        row is removed in the same SQLite transaction that enters this state,
        so an online read can never observe a row whose external projections
        are already being retired.  A newer Catalog revision makes the
        tombstone ``STALE`` before Vector or Relation state is touched.
        """
        return self._tombstones.begin_tombstone_cleanup(tombstone_id)

    def finish_tombstone_cleanup(self, tombstone_id: str) -> dict[str, Any] | None:
        """Mark external cleanup complete without weakening terminal states."""
        return self._tombstones.finish_tombstone_cleanup(tombstone_id)

    def mark_tombstone_cleanup_failed(self, tombstone_id: str, error: str) -> dict[str, Any] | None:
        """Record an external cleanup error while retaining deletion ownership."""
        return self._tombstones.mark_tombstone_cleanup_failed(tombstone_id, error)

    def mark_tombstone_failed(self, tombstone_id: str, error: str) -> dict[str, Any] | None:
        return self._tombstones.mark_tombstone_failed(tombstone_id, error)

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
        return self._migration_state.set_migration_state(
            migration_name, state, checkpoint, details, tenant_id=tenant_id, batch_size=batch_size, error=error
        )

    def get_migration_state(self, migration_name: str, *, tenant_id: str = "") -> dict[str, Any] | None:
        return self._migration_state.get_migration_state(migration_name, tenant_id=tenant_id)

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
        return self._migration_state.initialize_migration_state_if_absent(
            migration_name, state, details, tenant_id=tenant_id, batch_size=batch_size
        )

    def record_greenfield_catalog_origin(self, *, tenant_id: str) -> dict[str, Any]:
        """Durably distinguish a safe empty first start from an unbackfilled restart."""
        return self._migration_state.record_greenfield_catalog_origin(tenant_id=tenant_id)

    def has_greenfield_catalog_origin(self, *, tenant_id: str) -> bool:
        return self._migration_state.has_greenfield_catalog_origin(tenant_id=tenant_id)

    def bind_migration_tenant_from_schema_upgrade(
        self, migration_name: str, *, tenant_id: str, batch_size: int = 0
    ) -> dict[str, Any] | None:
        """Atomically materialize an upgraded-database bootstrap for one tenant.

        The index database is opened before the runtime tenant migration
        coordinator is constructed.  A reserved durable row therefore records
        that the schema came from a pre-v10 database, while this method copies
        it to the concrete tenant without overwriting a concurrently advanced
        migration state.
        """
        return self._migration_state.bind_migration_tenant_from_schema_upgrade(
            migration_name, tenant_id=tenant_id, batch_size=batch_size
        )

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
        return self._migration_state.set_session_projection_frontier(
            tenant_id=tenant_id,
            archive_uri=archive_uri,
            owner_user_id=owner_user_id,
            workspace_id=workspace_id,
            session_id=session_id,
            manifest_digest=manifest_digest,
            status=status,
            error=error,
        )

    def get_session_projection_frontier_summary(
        self, *, tenant_id: str, owner_user_id: str | None = None, workspace_ids: Sequence[str] | None = None
    ) -> dict[str, int]:
        return self._migration_state.get_session_projection_frontier_summary(
            tenant_id=tenant_id, owner_user_id=owner_user_id, workspace_ids=workspace_ids
        )

    @staticmethod
    def _owner_from_session_archive_uri(archive_uri: str) -> str:
        return MigrationStateOperations._owner_from_session_archive_uri(archive_uri)

    def list_session_projection_frontier(
        self,
        *,
        tenant_id: str,
        statuses: Sequence[str] = ("PENDING", "FAILED"),
        after_archive_uri: str = "",
        limit: int = 256,
    ) -> list[dict[str, Any]]:
        """Read a bounded, tenant-keyset page for startup/repair replay."""
        return self._migration_state.list_session_projection_frontier(
            tenant_id=tenant_id, statuses=statuses, after_archive_uri=after_archive_uri, limit=limit
        )

    def record_migration_equivalence_proof(
        self, migration_name: str, proof: Mapping[str, Any], *, tenant_id: str = ""
    ) -> dict[str, Any]:
        """Append one idempotent, payload-free source-to-Catalog proof."""
        return self._migration_state.record_migration_equivalence_proof(migration_name, proof, tenant_id=tenant_id)

    def get_migration_equivalence_summary(
        self, migration_name: str, *, tenant_id: str = "", validation_epoch: str
    ) -> dict[str, int]:
        return self._migration_state.get_migration_equivalence_summary(
            migration_name, tenant_id=tenant_id, validation_epoch=validation_epoch
        )

    def list_migration_equivalence_proofs(
        self, migration_name: str, *, tenant_id: str = "", validation_epoch: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return self._migration_state.list_migration_equivalence_proofs(
            migration_name, tenant_id=tenant_id, validation_epoch=validation_epoch, limit=limit
        )

    def record_migration_shadow_read(
        self, migration_name: str, comparison: Mapping[str, Any], *, tenant_id: str = ""
    ) -> dict[str, Any]:
        """Durably record one payload-free old/new bounded read comparison."""
        return self._migration_state.record_migration_shadow_read(migration_name, comparison, tenant_id=tenant_id)

    def get_migration_shadow_read_summary(
        self, migration_name: str, *, tenant_id: str = "", validation_epoch: str
    ) -> dict[str, int]:
        return self._migration_state.get_migration_shadow_read_summary(
            migration_name, tenant_id=tenant_id, validation_epoch=validation_epoch
        )

    def list_migration_shadow_reads(
        self, migration_name: str, *, tenant_id: str = "", validation_epoch: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return self._migration_state.list_migration_shadow_reads(
            migration_name, tenant_id=tenant_id, validation_epoch=validation_epoch, limit=limit
        )

    @staticmethod
    def _migration_equivalence_summary_in_transaction(
        conn: sqlite3.Connection, *, migration_name: str, tenant_id: str, validation_epoch: str
    ) -> dict[str, int]:
        return MigrationStateOperations._migration_equivalence_summary_in_transaction(
            conn, migration_name=migration_name, tenant_id=tenant_id, validation_epoch=validation_epoch
        )

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
        return self._migration_state.upsert_context_link(
            tenant_id=tenant_id,
            source_record_key=source_record_key,
            source_uri=source_uri,
            relation_type=relation_type,
            target_uri=target_uri,
            target_record_key=target_record_key,
            metadata=metadata,
            link_key=link_key,
        )

    def explain_structured_query(self, filters: Mapping[str, Any], *, limit: int = 10) -> list[str]:
        """Expose SQLite's query plan for integration/performance acceptance tests."""
        return self._search.explain_structured_query(filters, limit=limit)

    def _catalog_from_sql(self, filters: Mapping[str, Any]) -> tuple[str, str]:
        """Use the Current Slot unique key for an exact serving lookup."""
        return self._search._catalog_from_sql(filters)

    def _legacy_filter_sql(self, filters: Mapping[str, Any]) -> tuple[str, list[Any]]:
        """Build the conservative flat-index predicate used only by rollback reads."""
        return self._query_filters._legacy_filter_sql(filters)

    def _connect_filter_sql(self, alias: str, filters: Mapping[str, Any]) -> tuple[str, list[Any]]:
        """Compile trusted connect metadata into a pre-LIMIT predicate."""
        return self._query_filters._connect_filter_sql(alias, filters)

    def _target_identity_sql(
        self, filters: Mapping[str, Any], target_identity_uris: Any, *, legacy: bool = False
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
        return self._query_filters._target_identity_sql(filters, target_identity_uris, legacy=legacy)

    def _base_filter_sql(
        self,
        filters: dict[str, Any],
        *,
        path_candidate_limit: int = 100,
        path_fts_query: str = "",
        path_exact_value: str = "",
    ) -> tuple[str, list[Any]]:
        return self._base_filter._base_filter_sql(
            filters,
            path_candidate_limit=path_candidate_limit,
            path_fts_query=path_fts_query,
            path_exact_value=path_exact_value,
        )

    def _search_fts(self, query: str, filters: dict[str, Any], limit: int) -> list[IndexHit]:
        return self._search._search_fts(query, filters, limit)

    def _search_metadata_exact(self, query: str, filters: dict[str, Any], limit: int) -> list[IndexHit]:
        return self._search._search_metadata_exact(query, filters, limit)

    @staticmethod
    def _exact_hit_from_record(record: CatalogRecord) -> IndexHit:
        return CatalogSearchOperations._exact_hit_from_record(record)

    def _hit_from_row(
        self,
        row: sqlite3.Row,
        lexical: float = 0.0,
        lexical_rank: float | None = None,
        vector: float = 0.0,
        identity: float = 0.0,
        identity_rank: float | None = None,
    ) -> IndexHit:
        return self._search._hit_from_row(row, lexical, lexical_rank, vector, identity, identity_rank)

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
        return self._search._score_components(
            row,
            lexical=lexical,
            lexical_rank=lexical_rank,
            vector=vector,
            identity=identity,
            identity_rank=identity_rank,
        )

    def _prepare_record(
        self,
        record: CatalogRecord,
        *,
        scope_keys_override: Sequence[str] | None = None,
        legacy_overrides: Mapping[str, Any] | None = None,
    ) -> _PreparedCatalogRecord:
        return self._catalog._prepare_record(
            record, scope_keys_override=scope_keys_override, legacy_overrides=legacy_overrides
        )

    def _upsert_prepared(self, conn: sqlite3.Connection, item: _PreparedCatalogRecord) -> None:
        return self._catalog._upsert_prepared(conn, item)

    def _replace_paths(self, conn: sqlite3.Connection, record: CatalogRecord, *, scope_signature: str) -> None:
        return self._catalog._replace_paths(conn, record, scope_signature=scope_signature)

    def _replace_validity(self, conn: sqlite3.Connection, record: CatalogRecord) -> None:
        return self._catalog._replace_validity(conn, record)

    def _replace_acl_grants(self, conn: sqlite3.Connection, record: CatalogRecord, *, scope_signature: str) -> None:
        return self._catalog._replace_acl_grants(conn, record, scope_signature=scope_signature)

    @staticmethod
    def _acl_grants_for_record(record: CatalogRecord) -> set[tuple[str, str, str]]:
        return CatalogStoreOperations._acl_grants_for_record(record)

    @staticmethod
    def _adapter_access_value(record: CatalogRecord) -> str:
        return CatalogStoreOperations._adapter_access_value(record)

    def _replace_fts(self, conn: sqlite3.Connection, item: _PreparedCatalogRecord) -> None:
        return self._catalog._replace_fts(conn, item)

    def _delete_catalog_in_transaction(
        self, conn: sqlite3.Connection, record_key: str, *, tenant_id: str | None = None
    ) -> bool:
        return self._catalog._delete_catalog_in_transaction(conn, record_key, tenant_id=tenant_id)

    def _init_db(self) -> None:
        return self._schema._init_db()

    def _record_unified_catalog_schema_upgrade(
        self, conn: sqlite3.Connection, *, upgraded_from_schema_version: int
    ) -> None:
        """Persist upgrade provenance before a restart can mistake it for greenfield."""
        return self._schema_migrations._record_unified_catalog_schema_upgrade(
            conn, upgraded_from_schema_version=upgraded_from_schema_version
        )

    def _ensure_catalog_table(self, conn: sqlite3.Connection) -> bool:
        return self._schema._ensure_catalog_table(conn)

    def _create_contexts_table(self, conn: sqlite3.Connection, table_name: str) -> None:
        return self._schema._create_contexts_table(conn, table_name)

    def _rebuild_legacy_contexts(self, conn: sqlite3.Connection, columns: set[str]) -> None:
        return self._schema_migrations._rebuild_legacy_contexts(conn, columns)

    def _legacy_record(self, row: sqlite3.Row, columns: set[str], metadata: Mapping[str, Any]) -> CatalogRecord:
        return self._schema_migrations._legacy_record(row, columns, metadata)

    def _sanitize_existing_rows(self, conn: sqlite3.Connection) -> None:
        return self._schema_migrations._sanitize_existing_rows(conn)

    def _create_auxiliary_tables(self, conn: sqlite3.Connection) -> None:
        return self._schema._create_auxiliary_tables(conn)

    def _ensure_fts_table(self, conn: sqlite3.Connection) -> bool:
        return self._schema._ensure_fts_table(conn)

    @staticmethod
    def _fts_row_map_is_consistent(conn: sqlite3.Connection) -> bool:
        """Validate the rebuildable FTS rowid map during startup integrity."""
        return SchemaManager._fts_row_map_is_consistent(conn)

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        return self._schema._create_indexes(conn)

    def _migrate_scope_keys(self, conn: sqlite3.Connection) -> None:
        return self._schema_migrations._migrate_scope_keys(conn)

    def _rebuild_fts(self, conn: sqlite3.Connection) -> None:
        return self._schema_migrations._rebuild_fts(conn)

    def _catalog_record_from_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> CatalogRecord:
        return self._serialization._catalog_record_from_row(conn, row)

    def _catalog_records_from_rows(self, conn: sqlite3.Connection, rows: Sequence[sqlite3.Row]) -> list[CatalogRecord]:
        return self._serialization._catalog_records_from_rows(conn, rows)

    def _scope_keys_from_metadata(self, metadata: Mapping[str, Any]) -> list[str]:
        return self._serialization._scope_keys_from_metadata(metadata)

    def _safe_metadata_text(self, metadata: Mapping[str, Any]) -> str:
        return self._serialization._safe_metadata_text(metadata)

    def _safe_reference_uri(self, value: str) -> str:
        return self._serialization._safe_reference_uri(value)

    def _restore_internal_projection_path(self, metadata: dict[str, Any]) -> None:
        """Recreate a canonical control path for the legacy verifier, never for FTS."""
        return self._serialization._restore_internal_projection_path(metadata)

    def _match_query(self, query: str) -> str:
        return self._serialization._match_query(query)

    def _acl_bound_fts_query(self, match_query: str, filters: Mapping[str, Any]) -> str:
        return self._serialization._acl_bound_fts_query(match_query, filters)

    def _fts_acl_tokens(self, record: CatalogRecord, *, scope_signature: str) -> str:
        return self._serialization._fts_acl_tokens(record, scope_signature=scope_signature)

    def _grant_acl_token(self, tenant_id: str, grant_kind: str, grant_id: str, workspace_id: str) -> str:
        return self._serialization._grant_acl_token(tenant_id, grant_kind, grant_id, workspace_id)

    @staticmethod
    def _acl_token(tenant_id: str, scope_kind: str, scope_id: str) -> str:
        return CatalogSerializer._acl_token(tenant_id, scope_kind, scope_id)

    def _filter_values(self, value: Any, *, allow_empty: bool = False) -> list[str]:
        return self._serialization._filter_values(value, allow_empty=allow_empty)

    @staticmethod
    def _scope_signature(scope_keys: Sequence[str]) -> str:
        return CatalogSerializer._scope_signature(scope_keys)

    def _scope_signature_options(self, available_scope_keys: Sequence[str]) -> tuple[str, ...]:
        return self._serialization._scope_signature_options(available_scope_keys)

    def _bounded_limit(self, limit: int) -> int:
        return self._serialization._bounded_limit(limit)

    def _lexical_relevance(self, query: str, haystack: str) -> float:
        return self._serialization._lexical_relevance(query, haystack)

    def _lexical_match_count(self, query: str, haystack: str) -> float:
        return self._serialization._lexical_match_count(query, haystack)

    def _bounded(self, value: Any) -> float:
        return self._serialization._bounded(value)

    def _finite_rank(self, value: Any) -> float:
        return self._serialization._finite_rank(value)

    def _coerce_record(self, value: CatalogRecord | Mapping[str, Any]) -> CatalogRecord:
        return self._serialization._coerce_record(value)

    def _insert_context_row(self, conn: sqlite3.Connection, values: Mapping[str, Any], *, table_name: str) -> None:
        return self._serialization._insert_context_row(conn, values, table_name=table_name)

    def _delete_fts_record(self, conn: sqlite3.Connection, record_key: str) -> None:
        return self._serialization._delete_fts_record(conn, record_key)

    @staticmethod
    def _content_digest(content: str) -> str:
        return CatalogSerializer._content_digest(content)

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return CatalogSerializer._mapping(value)

    @staticmethod
    def _json_dump(value: Any) -> str:
        return CatalogSerializer._json_dump(value)

    def _json_mapping(self, value: Any) -> dict[str, Any]:
        return self._serialization._json_mapping(value)

    @staticmethod
    def _json_list(value: Any) -> list[str]:
        return CatalogSerializer._json_list(value)

    @staticmethod
    def _safe_exact_value(value: Any) -> str:
        return CatalogSerializer._safe_exact_value(value)

    @staticmethod
    def _legacy_value(row: sqlite3.Row, columns: set[str], name: str) -> str:
        return CatalogSerializer._legacy_value(row, columns, name)

    @staticmethod
    def _coerce_timestamp(value: str) -> str:
        return CatalogSerializer._coerce_timestamp(value)

    @staticmethod
    def _timestamp_number(value: str, *, lower: bool) -> float:
        """Map a normalized timestamp to an RTree-safe interval endpoint."""
        return CatalogSerializer._timestamp_number(value, lower=lower)

    @staticmethod
    def _tenant_rtree_key(conn: sqlite3.Connection, tenant_id: str) -> int:
        return CatalogSerializer._tenant_rtree_key(conn, tenant_id)

    @staticmethod
    def _now() -> str:
        return CatalogSerializer._now()

    @staticmethod
    def _row_dict(row: sqlite3.Row, *, json_fields: Sequence[str] = ()) -> dict[str, Any]:
        return CatalogSerializer._row_dict(row, json_fields=json_fields)

    def _connect(self) -> sqlite3.Connection:
        return self._connection._connect()

    def _online_fetchall(self, conn: sqlite3.Connection, sql: str, params: Sequence[Any]) -> list[sqlite3.Row]:
        """Execute one serving query under a fixed SQLite VM-step ceiling.

        Indexes make normal Top-K queries stop early.  This final guard covers
        adversarial combinations (for example, a long run of newer expired
        records before an older valid record) without returning a truncated
        result as an empty success.  Migration, repair, audit, and keyset GC
        deliberately use their separate unguarded administrative methods.
        """
        return self._connection._online_fetchall(conn, sql, params)

    def _narrow_online_validity_filters(self, filters: Mapping[str, Any]) -> dict[str, Any] | None:
        """Use the RTree as a sparse validity-first candidate driver.

        Up to a fixed threshold, the returned record identities are the
        complete tenant/ACL-valid set and can safely constrain the normal SQL
        before Top-K.  Dense valid sets fall back to the access/time index,
        which stops early.  Thus both the common dense case and the adversarial
        "many newer expired, one older valid" case remain bounded.
        """
        return self._connection._narrow_online_validity_filters(filters)


SqliteIndexStore = SQLiteIndexStore

__all__ = [
    "SQLiteIndexStore",
    "SqliteIndexStore",
    "lexical_match_count",
    "lexical_relevance",
    "lexical_terms",
]
