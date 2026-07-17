"""Application orchestration for domain-aware serving rebuilds."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from threading import RLock
from typing import Any, cast

from memoryos.contextdb.extensions import ContextDomainOverlay, ContextIndexPolicy
from memoryos.contextdb.ordinary_relations import RelationDomainPolicy
from memoryos.contextdb.store.index_consistency import IndexConsistencyService
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.queue_store import QueueStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.memory.integration.consistency import (
    validate_canonical_authoritative_state,
)
from memoryos.memory.integration.current_slot_backfill import (
    CurrentSlotMigrationBackfill,
)

_LOGGER = logging.getLogger(__name__)


class DerivedServingMaintenanceService:
    """Own domain-aware preflight, rebuild, recovery and verification flows."""

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore,
        *,
        queue_store: QueueStore | None = None,
        projection_store: Any | None = None,
        canonical_projector: Any | None = None,
        current_slot_projector: Any | None = None,
        retention_manager: Any | None = None,
        migration_gate: Any | None = None,
        unified_context_migration: Any | None = None,
        readiness: Any | None = None,
        domain_overlay: ContextDomainOverlay,
        index_policy: ContextIndexPolicy,
        serving_lock: RLock | None = None,
        relation_domain_policy: RelationDomainPolicy,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.queue_store = queue_store
        self.projection_store = projection_store
        self.canonical_projector = canonical_projector
        self.current_slot_projector = current_slot_projector
        self.retention_manager = retention_manager
        self.migration_gate = migration_gate
        self.unified_context_migration = unified_context_migration
        self.readiness = readiness
        self.domain_overlay = domain_overlay
        self.index_policy = index_policy
        self.serving_lock = serving_lock or RLock()
        self.relation_domain_policy = relation_domain_policy

    def _require_ready(self) -> None:
        if self.readiness is not None:
            self.readiness.require_ready()

    @contextmanager
    def _migration_projection_fence(self) -> Iterator[None]:
        """Serialize direct Source/serving mutations with tenant rebuilds."""

        acquire = getattr(self.migration_gate, "acquire_projection_fence", None)
        release = getattr(self.migration_gate, "release_projection_fence", None)
        fence = acquire() if callable(acquire) else None
        try:
            yield
        finally:
            if callable(release):
                release(fence)

    def _mark_not_ready(self, error: BaseException, *, artifact: str) -> None:
        mark_not_ready = getattr(self.readiness, "mark_not_ready", None)
        if callable(mark_not_ready):
            mark_not_ready(
                f"canonical consistency failure: {type(error).__name__}: {error}",
                details={"artifact": artifact, "error_type": type(error).__name__},
            )

    def _canonical_preflight(
        self,
        *,
        projection_fence_held: bool = False,
    ) -> tuple[dict[str, int], Any | None]:
        """Prove authoritative canonical inputs before any rebuild mutation."""

        try:
            result = validate_canonical_authoritative_state(
                self.source_store,
                self.relation_store,
                self.projection_store,
            )
            worker = None
            if self.canonical_projector is not None and self.queue_store is not None:
                from memoryos.memory.canonical.projection import MemoryProjectionWorker

                worker = MemoryProjectionWorker(
                    self.canonical_projector,
                    self.queue_store,
                    migration_gate=self.migration_gate,
                )
                # Tenant rebuild owns the same durable key through
                # UnifiedContextMigration.  All other callers use the public
                # self-fenced dispatcher; only that already-fenced admin path
                # may invoke the private implementation.
                dispatched = (
                    worker._dispatch_outbox_unfenced()
                    if projection_fence_held
                    else worker.dispatch_outbox()
                )
                # This validates only immutable publication/outbox/receipt/queue
                # bindings.  Index/vector/views are rebuildable and are checked
                # separately after a rebuild.
                worker._validate_authoritative_projection_proofs()
                queue_stats = self.queue_store.stats(queue_name="memory_projection")
                if queue_stats.get("dead_letter", 0) or queue_stats.get("quarantine", 0):
                    raise RuntimeError("canonical rebuild queue contains terminal failed work")
                if queue_stats.get("leased", 0):
                    raise RuntimeError("canonical rebuild queue contains an active lease")
                result = {
                    **result,
                    "projection_outbox_transactions": len(dispatched),
                    "projection_queue_pending": int(queue_stats.get("pending", 0) or 0),
                    "projection_queue_done": int(queue_stats.get("done", 0) or 0),
                }
            elif result["canonical_claims"]:
                raise RuntimeError("canonical projection proof validation is unavailable")
            return result, worker
        except Exception as exc:
            self._mark_not_ready(exc, artifact="canonical_rebuild_preflight")
            raise

    def _verify_canonical_projection(self, worker: Any | None) -> tuple[dict[str, Any], str]:
        if worker is None:
            return {"verified": 0, "publications": 0, "completions": 0}, ""
        try:
            current = worker.verify_current_projections()
            proofs = worker.validate_projection_proofs()
            return {**current, **proofs}, ""
        except Exception as exc:
            # A concurrent authoritative failure marks readiness in the
            # committed-read/proof layer and must propagate.  Missing or stale
            # derived rows remain a reportable, rebuildable inconsistency.
            state = str(getattr(getattr(self.readiness, "state", None), "value", "READY"))
            if state != "READY":
                self._require_ready()
            return {}, f"{type(exc).__name__}: {exc}"

    def rebuild_index(self, *, owner_user_id: str | None = None) -> dict:
        """Rebuild serving projections without exposing a partially cleared Catalog.

        Owner-scoped repair preserves the historical non-destructive behavior.
        A tenant-wide repair is a durable, resumable state machine: the
        BACKFILLING gate and tenant Catalog clear commit atomically, every
        later phase checkpoints, and runtime startup resumes an interrupted
        epoch before reads become READY.
        """

        if owner_user_id is not None:
            with self._migration_projection_fence():
                self._require_ready()
                with self.serving_lock:
                    return self._rebuild_owner_index_locked(owner_user_id=owner_user_id)

        row = self._derived_serving_rebuild_row()
        if row is None or str(row.get("state") or "") == "COMPLETED":
            self._require_ready()
        with self.serving_lock:
            fence = getattr(self.unified_context_migration, "derived_rebuild_fence", None)
            if not callable(fence):
                raise RuntimeError("tenant-wide rebuild requires a durable cross-process projection fence")
            with cast(AbstractContextManager[Any], fence()) as projection_fence:
                return self._rebuild_all_serving_locked(
                    existing=row,
                    projection_fence=projection_fence,
                )

    def resume_derived_serving_rebuild_if_needed(self) -> dict[str, Any]:
        """Resume one crash-interrupted tenant rebuild during startup.

        This entry point intentionally does not require READY: the runtime
        calls it while RECOVERING.  Absence/COMPLETED is a no-op and never
        starts a new destructive epoch.
        """

        row = self._derived_serving_rebuild_row()
        if row is None or str(row.get("state") or "") == "COMPLETED":
            return {
                "resumed": False,
                "state": str(row.get("state") or "NOT_STARTED") if row is not None else "NOT_STARTED",
            }
        with self.serving_lock:
            fence = getattr(self.unified_context_migration, "derived_rebuild_fence", None)
            if not callable(fence):
                raise RuntimeError("derived serving rebuild resume requires a durable projection fence")
            with cast(AbstractContextManager[Any], fence()) as projection_fence:
                result = self._rebuild_all_serving_locked(
                    existing=row,
                    projection_fence=projection_fence,
                )
        return {"resumed": True, **result}

    def rollback_derived_serving_rebuild(self, reason: str) -> dict[str, Any]:
        """Pause an unfinished repair on the legacy/fail-closed read route."""

        with self.serving_lock:
            fence = getattr(self.unified_context_migration, "derived_rebuild_fence", None)
            if not callable(fence):
                raise RuntimeError("derived serving rollback requires a durable projection fence")
            with cast(AbstractContextManager[Any], fence()):
                row = self._derived_serving_rebuild_row()
                if row is None:
                    raise RuntimeError("there is no derived serving rebuild to roll back")
                state = str(row.get("state") or "")
                if state == "COMPLETED":
                    raise RuntimeError("a completed derived serving rebuild cannot be rolled back in place")
                details = self._migration_details(row)
                details.update(
                    {
                        "rollback_from": state,
                        "rollback_reason": str(reason),
                        "phase": str(details.get("phase") or "VECTOR_CLEANUP"),
                    }
                )
                return self._persist_derived_serving_rebuild(
                    state="ROLLBACK",
                    checkpoint=str(row.get("checkpoint") or ""),
                    details=details,
                )

    def _rebuild_owner_index_locked(self, *, owner_user_id: str) -> dict:
        authoritative, projection_worker = self._canonical_preflight()
        try:
            consistency = IndexConsistencyService(
                self.source_store,
                self.index_store,
                self.relation_store,
                domain_overlay=self.domain_overlay,
                index_policy=self.index_policy,
                relation_domain_policy=self.relation_domain_policy,
            )
            rebuilt = 0
            for obj in self.source_store.list_objects():
                if obj.owner_user_id != owner_user_id or self._canonical_object(obj):
                    continue
                try:
                    content = self.source_store.read_content(obj.uri)
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    content = obj.title
                self.index_store.upsert_index(obj, content=content)
                rebuilt += 1
            result = consistency.verify()
            payload = self._consistency_payload(result)
            payload["canonical_authoritative"] = authoritative
            if self.canonical_projector is not None:
                payload["canonical_projection"] = self.canonical_projector.rebuild()
            feature_gate = getattr(self.migration_gate, "feature_gate", None)
            if self.current_slot_projector is not None and (
                feature_gate is None or bool(getattr(feature_gate, "dual_write_enabled", False))
            ):
                backfill = CurrentSlotMigrationBackfill(self.source_store, self.current_slot_projector)
                checkpoint = ""
                processed_slots = 0
                projected_records = 0
                while True:
                    batch = backfill(checkpoint, 256)
                    processed_slots += batch.processed_slots
                    projected_records += batch.projected_records
                    checkpoint = batch.checkpoint
                    if batch.complete:
                        break
                payload["current_slot_projection"] = {
                    "processed_slots": processed_slots,
                    "projected_records": projected_records,
                    "checkpoint": checkpoint,
                    "complete": True,
                }
            projection, projection_error = self._verify_canonical_projection(projection_worker)
            if projection_error:
                raise RuntimeError(projection_error)
            payload["canonical_projection_validation"] = projection
            payload["rebuilt_count"] = rebuilt
            self._require_ready()
            return payload
        except Exception as exc:
            self._mark_not_ready(exc, artifact="canonical_rebuild_publication")
            raise

    def _rebuild_all_serving_locked(
        self,
        *,
        existing: dict[str, Any] | None,
        projection_fence: Any,
    ) -> dict[str, Any]:
        """Run or resume the complete disposable serving-plane rebuild."""

        from memoryos.contextdb.catalog import (
            CatalogRecord,
            CatalogRecordKind,
            ServingTier,
            catalog_vector_metadata,
        )
        from memoryos.contextdb.store.vector import vector_capabilities, vector_row_id
        from memoryos.contextdb.unified_migration import (
            DERIVED_SERVING_REBUILD_NAME,
        )
        from memoryos.memory.canonical.visibility import reconcile_committed_relation_store

        tenant_id = self._tenant_id()
        batch_size = self._derived_rebuild_batch_size(existing)
        phase_order = (
            "VECTOR_CLEANUP",
            "GENERIC_SOURCE",
            "SESSION_CATALOG",
            "ORDINARY_RELATIONS",
            "CANONICAL_RELATIONS",
            "CANONICAL_CLAIMS",
            "CURRENT_SLOTS",
            "GENERIC_VECTORS",
            "RETENTION",
            "VERIFY",
        )
        current_phase = "PREFLIGHT"
        row = existing
        details: dict[str, Any] = self._migration_details(row)
        checkpoint = str((row or {}).get("checkpoint") or "")

        def checkpoint_projection_fence() -> None:
            prove = getattr(projection_fence, "checkpoint", None)
            if not callable(prove):
                raise RuntimeError("derived serving rebuild has no renewable projection fence")
            prove()

        try:
            # Immutable Source/receipt/current-head/projection-proof closure is
            # proven before the first destructive derived mutation.
            authoritative, projection_worker = self._canonical_preflight(
                projection_fence_held=True,
            )
            checkpoint_projection_fence()
            if row is None or str(row.get("state") or "") == "COMPLETED":
                starter = getattr(self.index_store, "begin_tenant_serving_rebuild", None)
                if not callable(starter):
                    raise RuntimeError("Catalog store has no atomic tenant rebuild gate")
                started = starter(
                    DERIVED_SERVING_REBUILD_NAME,
                    tenant_id=tenant_id,
                    batch_size=batch_size,
                    details={
                        "rebuild_epoch": uuid.uuid4().hex,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "phase": "VECTOR_CLEANUP",
                        "session_checkpoint": "",
                        "current_slot_checkpoint": "",
                        "canonical_authoritative": authoritative,
                    },
                )
                if not isinstance(started, dict):
                    raise TypeError("atomic tenant rebuild gate returned invalid state")
                row = dict(started)
            details = self._migration_details(row)
            checkpoint = str(row.get("checkpoint") or "")
            state = str(row.get("state") or "")
            if state not in {"BACKFILLING", "FAILED", "ROLLBACK"}:
                raise RuntimeError(f"derived serving rebuild cannot resume from {state or 'UNKNOWN'}")
            current_phase = str(details.get("phase") or "VECTOR_CLEANUP")
            if current_phase not in phase_order:
                raise RuntimeError(f"derived serving rebuild has an invalid phase: {current_phase}")
            if state != "BACKFILLING":
                checkpoint_projection_fence()
                row = self._persist_derived_serving_rebuild(
                    state="BACKFILLING",
                    checkpoint=checkpoint,
                    details={**details, "resumed_from": state},
                    batch_size=batch_size,
                )
                details = self._migration_details(row)

            # Rebuilds started by a pre-ordinary-relation binary may already
            # be past SESSION_CATALOG.  Rewind only the disposable serving
            # phases; canonical Source/receipts are unchanged and will be
            # reconciled again below.
            if (
                phase_order.index(current_phase) > phase_order.index("ORDINARY_RELATIONS")
                and not bool(details.get("ordinary_relations_complete"))
            ):
                current_phase = "ORDINARY_RELATIONS"
                checkpoint = str(details.get("ordinary_relation_checkpoint") or "")
                details = {
                    **details,
                    "phase": current_phase,
                    "rewound_for_ordinary_relations": True,
                }
                checkpoint_projection_fence()
                row = self._persist_derived_serving_rebuild(
                    state="BACKFILLING",
                    checkpoint=checkpoint,
                    details=details,
                    batch_size=batch_size,
                )

            def advance(
                completed_phase: str,
                next_phase: str,
                updates: dict[str, Any],
                *,
                next_checkpoint: str = "",
            ) -> None:
                nonlocal row, details, checkpoint, current_phase
                checkpoint_projection_fence()
                details = {
                    **details,
                    **updates,
                    "phase": next_phase,
                    "last_completed_phase": completed_phase,
                }
                checkpoint = next_checkpoint
                row = self._persist_derived_serving_rebuild(
                    state="BACKFILLING",
                    checkpoint=checkpoint,
                    details=details,
                    batch_size=batch_size,
                )
                current_phase = next_phase

            vector_store = self._serving_vector_store()
            if current_phase == "VECTOR_CLEANUP":
                checkpoint_projection_fence()
                deleted_vectors = 0
                if vector_store is not None:
                    capabilities = vector_capabilities(vector_store)
                    deleter = getattr(vector_store, "delete_by_filter", None)
                    if not capabilities.supports_delete_by_filter or not callable(deleter):
                        raise RuntimeError("tenant vector rebuild requires exact delete-by-filter capability")
                    deleted_vectors = deleter({"tenant_id": tenant_id})
                    if (
                        not isinstance(deleted_vectors, int)
                        or isinstance(deleted_vectors, bool)
                        or deleted_vectors < 0
                    ):
                        raise TypeError("tenant vector delete returned an invalid count")
                advance(
                    "VECTOR_CLEANUP",
                    "GENERIC_SOURCE",
                    {"vectors_deleted": deleted_vectors},
                )

            if current_phase == "GENERIC_SOURCE":
                checkpoint_projection_fence()
                generic_result = IndexConsistencyService(
                    self.source_store,
                    self.index_store,
                    self.relation_store,
                    domain_overlay=self.domain_overlay,
                    index_policy=self.index_policy,
                    relation_domain_policy=self.relation_domain_policy,
                ).rebuild_for_canonical_reprojection(projection_fence_held=True)
                advance(
                    "GENERIC_SOURCE",
                    "SESSION_CATALOG",
                    {"generic_source": self._consistency_payload(generic_result)},
                )

            if current_phase == "SESSION_CATALOG":
                if self.unified_context_migration is None:
                    raise RuntimeError("SessionArchive Catalog rebuild is unavailable")
                session_checkpoint = str(details.get("session_checkpoint") or checkpoint or "")
                session_counts = {
                    "processed_archives": int(details.get("session_archives") or 0),
                    "projected_records": int(details.get("session_records") or 0),
                    "vectors_projected": int(details.get("session_vectors") or 0),
                    "tombstoned_records": int(details.get("session_tombstoned_records") or 0),
                }
                while True:
                    checkpoint_projection_fence()
                    batch = self.unified_context_migration.rebuild_session_catalog_next_batch(
                        session_checkpoint,
                        batch_size=batch_size,
                    )
                    session_counts["processed_archives"] += batch.processed_archives
                    session_counts["projected_records"] += batch.projected_records
                    session_counts["vectors_projected"] += batch.vectors_projected
                    session_counts["tombstoned_records"] += batch.tombstoned_records
                    session_checkpoint = batch.checkpoint
                    checkpoint = session_checkpoint
                    details = {
                        **details,
                        "phase": "SESSION_CATALOG",
                        "session_checkpoint": session_checkpoint,
                        "session_archives": session_counts["processed_archives"],
                        "session_records": session_counts["projected_records"],
                        "session_vectors": session_counts["vectors_projected"],
                        "session_tombstoned_records": session_counts["tombstoned_records"],
                    }
                    checkpoint_projection_fence()
                    row = self._persist_derived_serving_rebuild(
                        state="BACKFILLING",
                        checkpoint=session_checkpoint,
                        details=details,
                        batch_size=batch_size,
                    )
                    if batch.complete:
                        break
                    if batch.processed_archives < 1:
                        raise RuntimeError("Session Catalog rebuild checkpoint made no progress")
                advance(
                    "SESSION_CATALOG",
                    "ORDINARY_RELATIONS",
                    {"session_catalog": session_counts},
                )

            if current_phase == "ORDINARY_RELATIONS":
                clearer = getattr(self.relation_store, "clear_ordinary_relations", None)
                if not callable(clearer):
                    raise RuntimeError("RelationStore has no tenant ordinary clear capability")
                ordinary_deleted = int(details.get("ordinary_relations_deleted") or 0)
                if not bool(details.get("ordinary_relations_cleared")):
                    while True:
                        checkpoint_projection_fence()
                        deleted = clearer(tenant_id=tenant_id, limit=batch_size)
                        if not isinstance(deleted, int) or isinstance(deleted, bool) or deleted < 0:
                            raise TypeError("ordinary RelationStore clear returned an invalid count")
                        ordinary_deleted += deleted
                        if deleted == 0:
                            break
                    details = {
                        **details,
                        "phase": "ORDINARY_RELATIONS",
                        "ordinary_relations_cleared": True,
                        "ordinary_relations_deleted": ordinary_deleted,
                    }
                    checkpoint = ""
                    checkpoint_projection_fence()
                    row = self._persist_derived_serving_rebuild(
                        state="BACKFILLING",
                        checkpoint=checkpoint,
                        details=details,
                        batch_size=batch_size,
                    )

                ordinary_checkpoint = str(details.get("ordinary_relation_checkpoint") or checkpoint or "")
                ordinary_objects = int(details.get("ordinary_relation_objects") or 0)
                ordinary_projected = int(details.get("ordinary_relations_projected") or 0)
                ordinary_written = int(details.get("ordinary_relations_written") or 0)
                ordinary_skipped = int(details.get("ordinary_relations_skipped") or 0)
                relation_rebuilder = IndexConsistencyService(
                    self.source_store,
                    self.index_store,
                    self.relation_store,
                    domain_overlay=self.domain_overlay,
                    index_policy=self.index_policy,
                    relation_domain_policy=self.relation_domain_policy,
                )
                while True:
                    checkpoint_projection_fence()
                    batch = relation_rebuilder.rebuild_ordinary_relations_next_batch(
                        tenant_id=tenant_id,
                        after_uri=ordinary_checkpoint,
                        batch_size=batch_size,
                        projection_fence_held=True,
                    )
                    ordinary_objects += batch.processed_objects
                    ordinary_projected += batch.projected_relations
                    ordinary_written += batch.written_relations
                    ordinary_skipped += batch.skipped_relations
                    ordinary_checkpoint = batch.checkpoint
                    checkpoint = ordinary_checkpoint
                    details = {
                        **details,
                        "phase": "ORDINARY_RELATIONS",
                        "ordinary_relation_checkpoint": ordinary_checkpoint,
                        "ordinary_relation_objects": ordinary_objects,
                        "ordinary_relations_projected": ordinary_projected,
                        "ordinary_relations_written": ordinary_written,
                        "ordinary_relations_skipped": ordinary_skipped,
                    }
                    checkpoint_projection_fence()
                    row = self._persist_derived_serving_rebuild(
                        state="BACKFILLING",
                        checkpoint=checkpoint,
                        details=details,
                        batch_size=batch_size,
                    )
                    if batch.complete:
                        break
                    if batch.processed_objects < 1:
                        raise RuntimeError("ordinary relation rebuild checkpoint made no progress")
                advance(
                    "ORDINARY_RELATIONS",
                    "CANONICAL_RELATIONS",
                    {
                        "ordinary_relations_complete": True,
                        "ordinary_relations": {
                            "objects": ordinary_objects,
                            "projected": ordinary_projected,
                            "written": ordinary_written,
                            "skipped": ordinary_skipped,
                            "deleted": ordinary_deleted,
                        },
                    },
                )

            if current_phase == "CANONICAL_RELATIONS":
                checkpoint_projection_fence()
                relation_result = reconcile_committed_relation_store(
                    self.source_store,
                    self.relation_store,
                )
                advance(
                    "CANONICAL_RELATIONS",
                    "CANONICAL_CLAIMS",
                    {"canonical_relations": relation_result},
                )

            if current_phase == "CANONICAL_CLAIMS":
                checkpoint_projection_fence()
                if self.canonical_projector is None:
                    if int(authoritative.get("canonical_claims", 0) or 0):
                        raise RuntimeError("canonical Claim rebuild is unavailable")
                    claim_result: dict[str, Any] = {
                        "projected": 0,
                        "skipped": 0,
                        "retired": 0,
                        "historical_restored": 0,
                    }
                else:
                    # CanonicalMemoryProjector owns both current Claim serving
                    # publication and receipt-proved Claim revision history.
                    claim_result = dict(self.canonical_projector.rebuild())
                advance(
                    "CANONICAL_CLAIMS",
                    "CURRENT_SLOTS",
                    {"canonical_projection": claim_result},
                )

            if current_phase == "CURRENT_SLOTS":
                slot_checkpoint = str(details.get("current_slot_checkpoint") or checkpoint or "")
                slot_count = int(details.get("current_slots") or 0)
                slot_records = int(details.get("current_slot_records") or 0)
                slot_proofs = int(details.get("current_slot_proofs") or 0)
                if self.current_slot_projector is not None:
                    backfill = CurrentSlotMigrationBackfill(
                        self.source_store,
                        self.current_slot_projector,
                    )
                    while True:
                        checkpoint_projection_fence()
                        batch = backfill(slot_checkpoint, batch_size)
                        for proof in batch.equivalence_proofs:
                            if proof.overflow or not proof.matched:
                                raise RuntimeError("CurrentSlot rebuild equivalence proof failed")
                        slot_count += batch.processed_slots
                        slot_records += batch.projected_records
                        slot_proofs += len(batch.equivalence_proofs)
                        slot_checkpoint = batch.checkpoint
                        checkpoint = slot_checkpoint
                        details = {
                            **details,
                            "phase": "CURRENT_SLOTS",
                            "current_slot_checkpoint": slot_checkpoint,
                            "current_slots": slot_count,
                            "current_slot_records": slot_records,
                            "current_slot_proofs": slot_proofs,
                        }
                        checkpoint_projection_fence()
                        row = self._persist_derived_serving_rebuild(
                            state="BACKFILLING",
                            checkpoint=slot_checkpoint,
                            details=details,
                            batch_size=batch_size,
                        )
                        if batch.complete:
                            break
                        if batch.processed_slots < 1:
                            raise RuntimeError("CurrentSlot rebuild checkpoint made no progress")
                advance(
                    "CURRENT_SLOTS",
                    "GENERIC_VECTORS",
                    {
                        "current_slot_projection": {
                            "processed_slots": slot_count,
                            "projected_records": slot_records,
                            "proofs": slot_proofs,
                            "checkpoint": slot_checkpoint,
                            "complete": True,
                        }
                    },
                )

            if current_phase == "GENERIC_VECTORS":
                generic_vectors = 0
                if vector_store is not None:
                    scanner = getattr(self.index_store, "scan_catalog_batch", None)
                    embedding_provider = self._serving_embedding_provider()
                    if not callable(scanner) or embedding_provider is None:
                        raise RuntimeError("configured VectorStore has no deterministic Catalog rebuild path")
                    cursor = ""
                    while True:
                        checkpoint_projection_fence()
                        raw_records = scanner(
                            after_record_key=cursor,
                            filters={
                                "tenant_id": tenant_id,
                                "record_kind": CatalogRecordKind.CONTEXT.value,
                                "include_inactive": True,
                            },
                            limit=batch_size,
                        )
                        if not isinstance(raw_records, list) or any(
                            not isinstance(record, CatalogRecord) for record in raw_records
                        ):
                            raise TypeError("generic vector rebuild Catalog scan returned invalid records")
                        if not raw_records:
                            break
                        for record in raw_records:
                            cursor = record.record_key
                            if not bool(record.metadata.get("vector_eligible")):
                                continue
                            if record.serving_tier not in {ServingTier.HOT.value, ServingTier.WARM.value}:
                                continue
                            text = "\n".join(
                                part for part in (record.title, record.l0_text, record.l1_text) if part
                            )
                            if not text:
                                raise RuntimeError("vector-eligible Catalog record has no sanitized serving text")
                            checkpoint_projection_fence()
                            vector_store.upsert_vector(
                                vector_row_id(record.tenant_id, record.record_key),
                                embedding_provider.embed(text),
                                metadata={
                                    **catalog_vector_metadata(record),
                                    "public_uri": record.uri,
                                    "embedding_model": embedding_provider.model_name,
                                    "schema_version": "generic_context_vector_rebuild_v1",
                                },
                            )
                            generic_vectors += 1
                advance(
                    "GENERIC_VECTORS",
                    "RETENTION",
                    {"generic_vectors_projected": generic_vectors},
                )

            if current_phase == "RETENTION":
                checkpoint_projection_fence()
                if self.retention_manager is None:
                    retention_result: dict[str, Any] = {"configured": False}
                else:
                    tiers = self.retention_manager.apply_serving_tiers(tenant_id=tenant_id)
                    vectors = self.retention_manager.gc_vectors(tenant_id=tenant_id)
                    stale = self.retention_manager.gc_stale_projections(tenant_id=tenant_id)
                    if vectors.tombstones_failed or stale.tombstones_failed:
                        raise RuntimeError("retention projection cleanup remained incomplete")
                    retention_result = {
                        "configured": True,
                        "tiers": asdict(tiers),
                        "vectors": asdict(vectors),
                        "stale": asdict(stale),
                    }
                advance(
                    "RETENTION",
                    "VERIFY",
                    {"retention": retention_result},
                )

            if current_phase != "VERIFY":
                raise RuntimeError(f"derived serving rebuild stopped at unexpected phase {current_phase}")
            checkpoint_projection_fence()
            projection, projection_error = self._verify_canonical_projection(projection_worker)
            if projection_error:
                raise RuntimeError(projection_error)
            current_slot_verified = 0
            if self.current_slot_projector is not None:
                verifier = CurrentSlotMigrationBackfill(
                    self.source_store,
                    self.current_slot_projector,
                )
                verify_checkpoint = ""
                while True:
                    checkpoint_projection_fence()
                    batch = verifier.prove(verify_checkpoint, batch_size)
                    for proof in batch.equivalence_proofs:
                        if proof.overflow or not proof.matched:
                            raise RuntimeError("CurrentSlot final equivalence proof failed")
                    current_slot_verified += len(batch.equivalence_proofs)
                    verify_checkpoint = batch.checkpoint
                    if batch.complete:
                        break
                    if batch.processed_slots < 1:
                        raise RuntimeError("CurrentSlot verification checkpoint made no progress")
            checkpoint_projection_fence()
            consistency_result = IndexConsistencyService(
                self.source_store,
                self.index_store,
                self.relation_store,
                domain_overlay=self.domain_overlay,
                index_policy=self.index_policy,
                relation_domain_policy=self.relation_domain_policy,
            ).verify()
            consistency_payload = self._consistency_payload(consistency_result)
            if not consistency_payload["consistent"]:
                raise RuntimeError("derived serving rebuild consistency verification failed")
            details = {
                **details,
                "phase": "VERIFY",
                "last_completed_phase": "VERIFY",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "canonical_projection_validation": projection,
                "current_slot_verified": current_slot_verified,
                "consistency": consistency_payload,
            }
            checkpoint_projection_fence()
            completed = self._persist_derived_serving_rebuild(
                state="COMPLETED",
                checkpoint="",
                details=details,
                batch_size=batch_size,
            )
            self._restore_ready_after_rebuild()
            return {
                "state": "COMPLETED",
                "tenant_id": tenant_id,
                "rebuild_epoch": str(details.get("rebuild_epoch") or ""),
                "canonical_authoritative": authoritative,
                "canonical_projection": details.get("canonical_projection", {}),
                "canonical_projection_validation": projection,
                "current_slot_projection": details.get("current_slot_projection", {}),
                "session_catalog": details.get("session_catalog", {}),
                "ordinary_relations": details.get("ordinary_relations", {}),
                "canonical_relations": details.get("canonical_relations", {}),
                "retention": details.get("retention", {}),
                "vectors_deleted": int(details.get("vectors_deleted") or 0),
                "generic_vectors_projected": int(details.get("generic_vectors_projected") or 0),
                **consistency_payload,
                "migration": completed,
            }
        except Exception as exc:
            if row is not None:
                failed_details = {
                    **details,
                    "phase": current_phase,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    checkpoint_projection_fence()
                    self._persist_derived_serving_rebuild(
                        state="FAILED",
                        checkpoint=checkpoint,
                        details=failed_details,
                        batch_size=batch_size,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                except Exception as persistence_error:
                    # Preserve the original integrity failure.  The atomic
                    # initial gate still prevents reads if the failure-state
                    # update itself cannot be persisted.
                    _LOGGER.exception(
                        "failed to persist derived serving rebuild failure state",
                        exc_info=persistence_error,
                    )
            self._mark_not_ready(exc, artifact="derived_serving_rebuild")
            raise

    def _derived_serving_rebuild_row(self) -> dict[str, Any] | None:
        from memoryos.contextdb.unified_migration import DERIVED_SERVING_REBUILD_NAME

        getter = getattr(self.index_store, "get_migration_state", None)
        if not callable(getter):
            return None
        raw = getter(DERIVED_SERVING_REBUILD_NAME, tenant_id=self._tenant_id())
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise TypeError("derived serving rebuild state is invalid")
        return dict(raw)

    def _persist_derived_serving_rebuild(
        self,
        *,
        state: str,
        checkpoint: str,
        details: dict[str, Any],
        batch_size: int | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        from memoryos.contextdb.unified_migration import DERIVED_SERVING_REBUILD_NAME

        setter = getattr(self.index_store, "set_migration_state", None)
        if not callable(setter):
            raise RuntimeError("Catalog store has no durable derived rebuild journal")
        raw = setter(
            DERIVED_SERVING_REBUILD_NAME,
            state,
            checkpoint,
            details,
            tenant_id=self._tenant_id(),
            batch_size=batch_size or self._derived_rebuild_batch_size(None),
            error=error,
        )
        if not isinstance(raw, dict):
            raise TypeError("derived serving rebuild journal returned invalid state")
        return dict(raw)

    @staticmethod
    def _migration_details(row: dict[str, Any] | None) -> dict[str, Any]:
        if row is None:
            return {}
        details = row.get("details_json")
        return dict(details) if isinstance(details, dict) else {}

    def _derived_rebuild_batch_size(self, row: dict[str, Any] | None) -> int:
        raw = int((row or {}).get("batch_size") or 0)
        if raw == 0:
            raw = int(getattr(self.unified_context_migration, "batch_size", 256) or 256)
        if not 1 <= raw <= 1_000:
            raise ValueError("derived serving rebuild batch size must be between 1 and 1000")
        return raw

    def _tenant_id(self) -> str:
        tenant_id = str(getattr(self.source_store, "tenant_id", "default") or "default")
        if not tenant_id or "\x00" in tenant_id:
            raise ValueError("derived serving rebuild requires a valid tenant")
        return tenant_id

    def _serving_vector_store(self) -> Any | None:
        candidates = tuple(
            store
            for store in (
                getattr(self.retention_manager, "vector_store", None),
                getattr(self.canonical_projector, "vector_store", None),
                getattr(self.current_slot_projector, "vector_store", None),
                getattr(getattr(self.unified_context_migration, "projector", None), "vector_store", None),
            )
            if store is not None
        )
        if not candidates:
            return None
        first = candidates[0]
        if any(candidate is not first for candidate in candidates[1:]):
            raise RuntimeError("derived serving rebuild components use different VectorStore instances")
        return first

    def _serving_embedding_provider(self) -> Any | None:
        for owner in (
            getattr(self.unified_context_migration, "projector", None),
            self.canonical_projector,
            self.current_slot_projector,
        ):
            provider = getattr(owner, "embedding_provider", None)
            if provider is not None:
                return provider
        return None

    def _restore_ready_after_rebuild(self) -> None:
        snapshot = getattr(self.readiness, "snapshot", None)
        transition = getattr(self.readiness, "transition", None)
        if not callable(snapshot) or not callable(transition):
            return
        raw = snapshot()
        if not isinstance(raw, dict) or str(raw.get("state") or "") != "NOT_READY":
            return
        details = raw.get("details")
        if not isinstance(details, dict) or str(details.get("artifact") or "") != "derived_serving_rebuild":
            return
        from memoryos.core.readiness import RuntimeReadinessState

        transition(
            RuntimeReadinessState.READY,
            details={"recovered_by": "derived_serving_rebuild"},
        )

    def verify_consistency(self, *, owner_user_id: str | None = None) -> dict:
        self._require_ready()
        authoritative, projection_worker = self._canonical_preflight()
        result = IndexConsistencyService(
            self.source_store,
            self.index_store,
            self.relation_store,
            domain_overlay=self.domain_overlay,
            index_policy=self.index_policy,
            relation_domain_policy=self.relation_domain_policy,
        ).verify()
        payload = self._consistency_payload(result)
        projection, projection_error = self._verify_canonical_projection(projection_worker)
        payload["canonical_authoritative"] = authoritative
        payload["canonical_projection_validation"] = projection
        payload["canonical_projection_error"] = projection_error
        if projection_error:
            payload["consistent"] = False
        if owner_user_id is None:
            return payload
        source_uris = {
            obj.uri
            for obj in self.source_store.list_objects()
            if obj.owner_user_id == owner_user_id and not self._canonical_object(obj)
        }
        indexed_uris = set(self.index_store.indexed_uris())
        payload["source_count"] = len(source_uris)
        payload["indexed_count"] = len(source_uris & indexed_uris)
        payload["missing_index"] = [uri for uri in payload["missing_index"] if uri in source_uris]
        payload["dangling_index"] = [
            uri for uri in payload["dangling_index"] if uri.startswith(f"memoryos://user/{owner_user_id}/")
        ]
        payload["broken_relations"] = [
            relation
            for relation in payload["broken_relations"]
            if relation.get("source_uri") in source_uris or relation.get("target_uri") in source_uris
        ]
        payload["consistent"] = not projection_error and not (
            payload["missing_index"]
            or payload["dangling_index"]
            or payload["deleted_or_archived_in_default_search"]
            or payload["broken_relations"]
        )
        return payload

    def _consistency_payload(self, result) -> dict:
        return {
            "source_count": result.source_count,
            "indexed_count": result.index_count,
            "missing_index": result.missing_in_index,
            "dangling_index": result.orphan_index,
            "deleted_or_archived_in_default_search": result.deleted_or_archived_in_default_search,
            "broken_relations": result.broken_relations,
            "consistent": result.consistent,
        }


    def _canonical_object(self, obj: Any) -> bool:
        return self.domain_overlay.owns_object(obj)

    def _canonical_uri(self, uri: str) -> bool:
        return self.domain_overlay.owns_uri(uri)


__all__ = ["DerivedServingMaintenanceService"]
