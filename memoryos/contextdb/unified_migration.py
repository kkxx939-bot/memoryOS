"""Resumable cutover state machine for the rebuildable Unified Context Catalog.

This module migrates only serving projections.  It never writes Canonical
Source objects, receipts, current heads, or immutable SessionArchive evidence.
Filesystem enumeration is therefore intentionally confined to this offline
migration/repair path and is streamed one archive at a time.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Protocol

from memoryos.contextdb.catalog import CatalogRecord
from memoryos.contextdb.projection_equivalence import (
    ProjectionEquivalenceProof,
    build_projection_equivalence_proof,
)
from memoryos.contextdb.session.context_projector import (
    SessionContextProjector,
    SessionProjectionResult,
)
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.security.context_projection import ContextProjectionSanitizer

_REQUIRED_CATALOG_SCHEMA_VERSION = 10
DERIVED_SERVING_REBUILD_NAME = "unified-context-derived-serving-rebuild-v1"
_PROJECTION_FENCE_TTL_SECONDS = 3_600
_PROJECTION_FENCE_MAX_RENEW_INTERVAL_SECONDS = 300.0


class _RenewingProjectionFence:
    """Keep one durable lease alive and expose explicit mutation checkpoints."""

    def __init__(self, lock_store: Any, token: Any, *, ttl_seconds: int) -> None:
        self.lock_store = lock_store
        self.token = token
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._stop = Event()
        self._mutex = Lock()
        self._renewal_error: Exception | None = None
        self._interval_seconds = max(
            0.05,
            min(
                float(self.ttl_seconds) / 3.0,
                _PROJECTION_FENCE_MAX_RENEW_INTERVAL_SECONDS,
            ),
        )
        self._thread = Thread(
            target=self._renew_loop,
            name=f"memoryos-projection-fence-{getattr(token, 'fence', 'lease')}",
            daemon=True,
        )
        self._thread.start()

    def _renew_loop(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self._renew()
            except Exception as exc:  # The owner observes this at its next checkpoint.
                self._renewal_error = exc
                return

    def _renew(self) -> None:
        renew = getattr(self.lock_store, "renew", None)
        if not callable(renew):
            raise RuntimeError("migration projection fence cannot be renewed")
        with self._mutex:
            if self._renewal_error is not None:
                raise RuntimeError("migration projection fence renewal previously failed") from self._renewal_error
            try:
                renew(self.token, ttl_seconds=self.ttl_seconds)
            except Exception as exc:
                self._renewal_error = exc
                raise RuntimeError("migration projection fence renewal failed") from exc

    def checkpoint(self) -> None:
        """Renew and re-prove ownership immediately before a durable boundary."""

        if self._renewal_error is not None:
            raise RuntimeError("migration projection fence renewal failed") from self._renewal_error
        self._renew()
        checker = getattr(self.lock_store, "assert_owned", None)
        if not callable(checker):
            raise RuntimeError("migration projection fence ownership cannot be verified")
        checker(self.token)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


class MigrationState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    SCHEMA_READY = "SCHEMA_READY"
    BACKFILLING = "BACKFILLING"
    DUAL_WRITE = "DUAL_WRITE"
    SHADOW_VALIDATING = "SHADOW_VALIDATING"
    READY_TO_CUTOVER = "READY_TO_CUTOVER"
    CUTOVER = "CUTOVER"
    ROLLBACK = "ROLLBACK"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ReadRoute(str, Enum):
    LEGACY = "LEGACY"
    SHADOW = "SHADOW"
    UNIFIED = "UNIFIED"


_ALLOWED_TRANSITIONS: Mapping[MigrationState, frozenset[MigrationState]] = {
    MigrationState.NOT_STARTED: frozenset({MigrationState.SCHEMA_READY, MigrationState.FAILED}),
    MigrationState.SCHEMA_READY: frozenset({MigrationState.BACKFILLING, MigrationState.FAILED}),
    MigrationState.BACKFILLING: frozenset({MigrationState.DUAL_WRITE, MigrationState.ROLLBACK, MigrationState.FAILED}),
    MigrationState.DUAL_WRITE: frozenset(
        {MigrationState.SHADOW_VALIDATING, MigrationState.ROLLBACK, MigrationState.FAILED}
    ),
    MigrationState.SHADOW_VALIDATING: frozenset(
        {MigrationState.READY_TO_CUTOVER, MigrationState.ROLLBACK, MigrationState.FAILED}
    ),
    MigrationState.READY_TO_CUTOVER: frozenset(
        {
            MigrationState.SHADOW_VALIDATING,
            MigrationState.CUTOVER,
            MigrationState.ROLLBACK,
            MigrationState.FAILED,
        }
    ),
    MigrationState.CUTOVER: frozenset({MigrationState.COMPLETED, MigrationState.ROLLBACK, MigrationState.FAILED}),
    MigrationState.ROLLBACK: frozenset({MigrationState.BACKFILLING, MigrationState.DUAL_WRITE, MigrationState.FAILED}),
    MigrationState.COMPLETED: frozenset({MigrationState.ROLLBACK, MigrationState.FAILED}),
    MigrationState.FAILED: frozenset(
        {
            MigrationState.SCHEMA_READY,
            MigrationState.BACKFILLING,
            MigrationState.DUAL_WRITE,
            MigrationState.SHADOW_VALIDATING,
            MigrationState.READY_TO_CUTOVER,
            MigrationState.CUTOVER,
            MigrationState.ROLLBACK,
        }
    ),
}


class MigrationStateStore(Protocol):
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
    ) -> dict[str, Any]: ...

    def get_migration_state(
        self,
        migration_name: str,
        *,
        tenant_id: str = "",
    ) -> dict[str, Any] | None: ...

    def catalog_schema_version(self) -> int: ...

    def record_migration_equivalence_proof(
        self,
        migration_name: str,
        proof: Mapping[str, Any],
        *,
        tenant_id: str = "",
    ) -> dict[str, Any]: ...

    def get_migration_equivalence_summary(
        self,
        migration_name: str,
        *,
        tenant_id: str = "",
        validation_epoch: str,
    ) -> dict[str, int]: ...

    def record_migration_shadow_read(
        self,
        migration_name: str,
        comparison: Mapping[str, Any],
        *,
        tenant_id: str = "",
    ) -> dict[str, Any]: ...

    def get_migration_shadow_read_summary(
        self,
        migration_name: str,
        *,
        tenant_id: str = "",
        validation_epoch: str,
    ) -> dict[str, int]: ...


@dataclass(frozen=True)
class CanonicalBackfillBatchResult:
    """One bounded offline CurrentSlot backfill batch."""

    processed_slots: int
    projected_records: int
    checkpoint: str
    complete: bool
    equivalence_proofs: tuple[ProjectionEquivalenceProof, ...] = ()


CanonicalCurrentBackfill = Callable[[str, int], CanonicalBackfillBatchResult]


class CurrentSlotMigrationBackfill:
    """Offline bounded rebuild of CurrentSlot rows from committed Slot heads."""

    def __init__(self, source_store: Any, projector: Any) -> None:
        self.source_store = source_store
        self.projector = projector

    def __call__(self, after_slot_uri: str, limit: int) -> CanonicalBackfillBatchResult:
        selected, has_more = self._select_slot_uris(after_slot_uri, limit)
        projected = 0
        proofs: list[ProjectionEquivalenceProof] = []
        checkpoint = after_slot_uri
        for slot_uri in selected:
            result = self.projector.project(slot_uri)
            checkpoint = slot_uri
            if str(getattr(result, "status", "")) == "projected":
                projected += 1
            record = getattr(result, "record", None)
            catalog_store = getattr(self.projector, "catalog_store", None)
            getter = getattr(catalog_store, "get_catalog", None)
            if record is not None and callable(getter):
                actual = getter(result.record_key, tenant_id=record.tenant_id)
                if actual is not None and not isinstance(actual, CatalogRecord):
                    raise TypeError("CurrentSlot proof lookup returned an invalid Catalog record")
                proofs.append(
                    build_projection_equivalence_proof(
                        plane="canonical_current_slot",
                        source_identity=slot_uri,
                        evidence_digest=record.receipt_digest,
                        expected_records=(record,),
                        actual_records=((actual,) if actual is not None else ()),
                    )
                )
        return CanonicalBackfillBatchResult(
            processed_slots=len(selected),
            projected_records=projected,
            checkpoint=checkpoint,
            complete=not has_more,
            equivalence_proofs=tuple(proofs),
        )

    def prove(self, after_slot_uri: str, limit: int) -> CanonicalBackfillBatchResult:
        """Compare Source-derived CurrentSlot rows without repairing Catalog."""

        selected, has_more = self._select_slot_uris(after_slot_uri, limit)
        expected = getattr(self.projector, "expected_projection", None)
        catalog_store = getattr(self.projector, "catalog_store", None)
        getter = getattr(catalog_store, "get_catalog", None)
        if not callable(expected) or not callable(getter):
            raise RuntimeError("CurrentSlot shadow proof requires non-mutating Source and exact Catalog reads")
        proofs: list[ProjectionEquivalenceProof] = []
        checkpoint = after_slot_uri
        for slot_uri in selected:
            result: Any = expected(slot_uri)
            checkpoint = slot_uri
            tenant_id = str(
                getattr(result.record, "tenant_id", "")
                or getattr(self.source_store, "tenant_id", "default")
                or "default"
            )
            actual = getter(result.record_key, tenant_id=tenant_id)
            if actual is not None and not isinstance(actual, CatalogRecord):
                raise TypeError("CurrentSlot proof lookup returned an invalid Catalog record")
            expected_records = (result.record,) if isinstance(result.record, CatalogRecord) else ()
            proofs.append(
                build_projection_equivalence_proof(
                    plane="canonical_current_slot",
                    source_identity=slot_uri,
                    evidence_digest=str(result.evidence_digest),
                    expected_records=expected_records,
                    actual_records=((actual,) if actual is not None else ()),
                )
            )
        return CanonicalBackfillBatchResult(
            processed_slots=len(selected),
            projected_records=0,
            checkpoint=checkpoint,
            complete=not has_more,
            equivalence_proofs=tuple(proofs),
        )

    def _select_slot_uris(self, after_slot_uri: str, limit: int) -> tuple[list[str], bool]:
        if not 1 <= int(limit) <= 1_000:
            raise ValueError("CurrentSlot backfill limit must be between 1 and 1000")
        from memoryos.memory.canonical.current_head import artifact_root_for, iter_current_head_uris

        artifact_root = artifact_root_for(self.source_store)
        if artifact_root is None:
            return [], False
        slot_uris = (
            uri for uri in sorted(iter_current_head_uris(artifact_root, kinds=("slot",))) if uri > after_slot_uri
        )
        selected: list[str] = []
        has_more = False
        for slot_uri in slot_uris:
            if len(selected) >= int(limit):
                has_more = True
                break
            selected.append(slot_uri)
        return selected, has_more

    def source_snapshot(self) -> tuple[str, int]:
        """Hash the receipt-proved CurrentSlot source set for cutover fencing."""

        from memoryos.memory.canonical.current_head import artifact_root_for, iter_current_head_uris

        artifact_root = artifact_root_for(self.source_store)
        if artifact_root is None:
            return hashlib.sha256(b"").hexdigest(), 0
        expected = getattr(self.projector, "expected_projection", None)
        if not callable(expected):
            raise RuntimeError("CurrentSlot cutover snapshot requires receipt-proved expected projections")
        digest = hashlib.sha256()
        count = 0
        for slot_uri in sorted(iter_current_head_uris(artifact_root, kinds=("slot",))):
            result: Any = expected(slot_uri)
            for value in (slot_uri, str(result.evidence_digest), str(result.record_key)):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
            count += 1
        return digest.hexdigest(), count


@dataclass(frozen=True)
class MigrationFeatureGate:
    state: MigrationState

    @property
    def dual_write_enabled(self) -> bool:
        return self.state in {
            # Enable dual-write before taking the first checkpoint.  An
            # online source whose sort key falls behind an in-flight batch is
            # therefore projected even when the next keyset pass cannot see
            # it.  FAILED keeps this protection while an interrupted
            # backfill awaits resume.
            MigrationState.BACKFILLING,
            MigrationState.FAILED,
            MigrationState.DUAL_WRITE,
            MigrationState.SHADOW_VALIDATING,
            MigrationState.READY_TO_CUTOVER,
            MigrationState.CUTOVER,
            MigrationState.COMPLETED,
            # Rollback changes the read route, not the durability of derived
            # compatibility writes.  The Catalog evolves the existing index
            # in place, so keeping it current is what makes old public search
            # wrappers usable for sessions created after rollback.
            MigrationState.ROLLBACK,
        }

    @property
    def read_route(self) -> ReadRoute:
        if self.state in {MigrationState.CUTOVER, MigrationState.COMPLETED}:
            return ReadRoute.UNIFIED
        if self.state in {
            MigrationState.DUAL_WRITE,
            MigrationState.SHADOW_VALIDATING,
            MigrationState.READY_TO_CUTOVER,
        }:
            return ReadRoute.SHADOW
        return ReadRoute.LEGACY

    @property
    def legacy_fallback_enabled(self) -> bool:
        # Cutover remains safely reversible; only an explicitly completed
        # migration retires the compatibility read as the primary fallback.
        return self.state is not MigrationState.COMPLETED


class RuntimeMigrationCoordinator:
    """Dynamically bind durable migration state to online reads and writes.

    Absence of a migration row means a greenfield v10 catalog, so Unified
    serving is enabled.  As soon as an administrator initializes a migration,
    every runtime component observes its durable state on the next operation.
    """

    def __init__(
        self,
        state_store: MigrationStateStore,
        *,
        tenant_id: str,
        migration_name: str = "unified-context-catalog-v1",
        lock_store: Any | None = None,
        sanitizer: ContextProjectionSanitizer | None = None,
    ) -> None:
        self.state_store = state_store
        self.tenant_id = tenant_id
        self.migration_name = migration_name
        self.lock_store = lock_store
        self.sanitizer = sanitizer or ContextProjectionSanitizer()
        self._projection_fence_depth: ContextVar[int] = ContextVar(
            f"memoryos_runtime_migration_fence_depth_{id(self)}",
            default=0,
        )
        self._projection_fence_lease: ContextVar[_RenewingProjectionFence | None] = ContextVar(
            f"memoryos_runtime_migration_fence_lease_{id(self)}",
            default=None,
        )
        self._bind_schema_upgrade_state()

    def _bind_schema_upgrade_state(self, *, batch_size: int = 0) -> dict[str, Any] | None:
        """Bind durable pre-v10 provenance to this runtime tenant, if present."""

        binder = getattr(self.state_store, "bind_migration_tenant_from_schema_upgrade", None)
        if not callable(binder):
            return None
        bound: Any = binder(
            self.migration_name,
            tenant_id=self.tenant_id,
            batch_size=max(0, int(batch_size)),
        )
        if bound is not None and not isinstance(bound, Mapping):
            raise TypeError("schema-upgrade migration binding returned an invalid result")
        return dict(bound) if isinstance(bound, Mapping) else None

    def require_backfill(self, *, reason: str, schema_version: int) -> dict[str, Any]:
        """Install a fail-closed tenant gate when durable evidence predates Catalog state."""

        initializer = getattr(self.state_store, "initialize_migration_state_if_absent", None)
        if not callable(initializer):
            raise RuntimeError("migration state store cannot durably initialize an evidence backfill gate")
        raw: Any = initializer(
            self.migration_name,
            MigrationState.SCHEMA_READY.value,
            {
                "schema_version": int(schema_version),
                "backfill_reason": self.sanitizer.sanitize_trace(str(reason)),
                "requires_backfill": True,
                "session_backfill_complete": False,
                "backfill_complete": False,
            },
            tenant_id=self.tenant_id,
        )
        if not isinstance(raw, Mapping):
            raise TypeError("evidence backfill migration initialization returned an invalid result")
        return dict(raw)

    def record_greenfield_catalog_origin(self) -> dict[str, Any] | None:
        """Persist why a no-row runtime may safely default to Unified reads."""

        getter = getattr(self.state_store, "get_migration_state", None)
        if callable(getter) and getter(self.migration_name, tenant_id=self.tenant_id) is not None:
            return None
        recorder = getattr(self.state_store, "record_greenfield_catalog_origin", None)
        if not callable(recorder):
            return None
        raw: Any = recorder(tenant_id=self.tenant_id)
        if not isinstance(raw, Mapping):
            raise TypeError("greenfield Catalog origin returned an invalid result")
        return dict(raw)

    @property
    def greenfield_catalog_origin_exists(self) -> bool:
        checker = getattr(self.state_store, "has_greenfield_catalog_origin", None)
        return bool(checker(tenant_id=self.tenant_id)) if callable(checker) else False

    @property
    def projection_fence_key(self) -> str:
        return f"migration:{self.tenant_id}:{self.migration_name}:projection"

    def acquire_projection_fence(self) -> Any | None:
        """Serialize every online derived-input write with tenant rebuilds.

        The lock must also be acquired while the feature gate is stable.  A
        state check followed by an unlocked write has a check/lock race: a
        rebuild can acquire its fence and publish ``BACKFILLING`` between the
        two operations.  Always taking the tenant-qualified durable fence
        makes the state transition and all Source/queue/projection mutations
        mutually exclusive across runtime processes.
        """

        depth = self._projection_fence_depth.get()
        active = self._projection_fence_lease.get()
        if depth:
            if active is None:
                raise RuntimeError("nested migration projection fence has no active durable lease")
            self._projection_fence_depth.set(depth + 1)
            return active
        acquire = getattr(self.lock_store, "acquire", None)
        if not callable(acquire):
            raise RuntimeError("online projection writes require a durable migration fence")
        raw_token = acquire(
            self.projection_fence_key,
            ttl_seconds=_PROJECTION_FENCE_TTL_SECONDS,
        )
        lease = _RenewingProjectionFence(
            self.lock_store,
            raw_token,
            ttl_seconds=_PROJECTION_FENCE_TTL_SECONDS,
        )
        try:
            lease.checkpoint()
        except Exception:
            lease.stop()
            releaser = getattr(self.lock_store, "release", None)
            if callable(releaser):
                releaser(raw_token)
            raise
        self._projection_fence_lease.set(lease)
        self._projection_fence_depth.set(1)
        return lease

    def release_projection_fence(self, token: Any | None) -> None:
        if token is None:
            return
        depth = self._projection_fence_depth.get()
        active = self._projection_fence_lease.get()
        if active is not None:
            if token is not active or depth < 1:
                raise RuntimeError("migration projection fence release does not match the active lease")
            if depth > 1:
                self._projection_fence_depth.set(depth - 1)
                return
        releaser = getattr(self.lock_store, "release", None)
        if not callable(releaser):
            raise RuntimeError("migration projection fence cannot be released")
        raw_token = token.token if isinstance(token, _RenewingProjectionFence) else token
        try:
            if isinstance(token, _RenewingProjectionFence):
                token.checkpoint()
        finally:
            try:
                if isinstance(token, _RenewingProjectionFence):
                    token.stop()
                releaser(raw_token)
            finally:
                if active is not None:
                    self._projection_fence_lease.set(None)
                    self._projection_fence_depth.set(0)

    @property
    def feature_gate(self) -> MigrationFeatureGate:
        getter = getattr(self.state_store, "get_migration_state", None)
        if not callable(getter):
            return MigrationFeatureGate(MigrationState.COMPLETED)
        rebuild = self._derived_rebuild_row()
        if rebuild is not None:
            rebuild_state = MigrationState(str(rebuild["state"]))
            if rebuild_state is not MigrationState.COMPLETED:
                # An interrupted destructive serving rebuild must never route
                # online reads through the partially repopulated Catalog.
                return MigrationFeatureGate(MigrationState.ROLLBACK)
        raw_row: Any = getter(self.migration_name, tenant_id=self.tenant_id)
        if not isinstance(raw_row, Mapping):
            return MigrationFeatureGate(MigrationState.COMPLETED)
        row = dict(raw_row)
        return MigrationFeatureGate(MigrationState(str(row["state"])))

    @property
    def derived_rebuild_requires_unavailable(self) -> bool:
        """Block every read while a destructive tenant repair is incomplete.

        LEGACY and Unified readers share the evolved ``contexts`` table, so a
        partially repopulated Catalog cannot provide a sound compatibility
        result even when it happens to return non-empty candidates.
        """

        row = self._derived_rebuild_row()
        return bool(
            row is not None
            and MigrationState(str(row["state"])) is not MigrationState.COMPLETED
        )

    @property
    def serving_generation_token(self) -> str:
        """Return a payload-free token that changes for every repair epoch."""

        row = self._derived_rebuild_row()
        if row is None:
            return self.sanitizer.digest(
                {
                    "migration_name": DERIVED_SERVING_REBUILD_NAME,
                    "tenant_id": self.tenant_id,
                    "state": "ABSENT",
                    "rebuild_epoch": "",
                }
            )
        details = UnifiedContextMigration._details(row)
        return self.sanitizer.digest(
            {
                "migration_name": DERIVED_SERVING_REBUILD_NAME,
                "tenant_id": self.tenant_id,
                "state": str(row["state"]),
                "rebuild_epoch": str(details.get("rebuild_epoch") or ""),
            }
        )

    def record_projection_equivalence(
        self,
        proof: ProjectionEquivalenceProof,
    ) -> dict[str, Any] | None:
        """Append an immutable-source projection proof during dual-write."""

        getter = getattr(self.state_store, "get_migration_state", None)
        recorder = getattr(self.state_store, "record_migration_equivalence_proof", None)
        raw = getter(self.migration_name, tenant_id=self.tenant_id) if callable(getter) else None
        # No row is the greenfield COMPLETED default.  There is no migration
        # validation journal to update and creating one from a normal write
        # would incorrectly switch the runtime into migration mode.
        if not isinstance(raw, Mapping):
            return None
        state = MigrationState(str(raw["state"]))
        if state in {MigrationState.NOT_STARTED, MigrationState.SCHEMA_READY}:
            return None
        if not callable(recorder):
            if state is MigrationState.SHADOW_VALIDATING:
                raise RuntimeError("shadow validation requires a durable equivalence journal")
            return None
        recorded: Any = recorder(
            self.migration_name,
            proof.to_journal_entry(),
            tenant_id=self.tenant_id,
        )
        if not isinstance(recorded, Mapping):
            raise TypeError("migration equivalence journal returned an invalid result")
        return dict(recorded)

    def record_shadow_read_comparison(
        self,
        comparison: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Append one sanitized, payload-free old/new online result diff."""

        if self.feature_gate.state is not MigrationState.SHADOW_VALIDATING:
            return None
        recorder = getattr(self.state_store, "record_migration_shadow_read", None)
        if not callable(recorder):
            raise RuntimeError("shadow read requires a durable result-diff journal")
        recorded: Any = recorder(
            self.migration_name,
            comparison,
            tenant_id=self.tenant_id,
        )
        if not isinstance(recorded, Mapping):
            raise TypeError("shadow read journal returned an invalid result")
        return dict(recorded)

    @property
    def empty_result_requires_unavailable(self) -> bool:
        """Whether an empty compatibility read could hide unbackfilled data."""

        getter = getattr(self.state_store, "get_migration_state", None)
        if not callable(getter):
            return False
        rebuild = self._derived_rebuild_row()
        if rebuild is not None and MigrationState(str(rebuild["state"])) is not MigrationState.COMPLETED:
            return True
        raw = getter(self.migration_name, tenant_id=self.tenant_id)
        if not isinstance(raw, Mapping):
            return False
        gate = MigrationFeatureGate(MigrationState(str(raw["state"])))
        if gate.read_route is not ReadRoute.LEGACY:
            return False
        return not bool(UnifiedContextMigration._details(raw).get("backfill_complete", False))

    def _derived_rebuild_row(self) -> dict[str, Any] | None:
        getter = getattr(self.state_store, "get_migration_state", None)
        if not callable(getter):
            return None
        raw: Any = getter(
            DERIVED_SERVING_REBUILD_NAME,
            tenant_id=self.tenant_id,
        )
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise TypeError("derived serving rebuild state is invalid")
        return dict(raw)


def has_existing_session_archive_evidence(
    archive_store: SessionArchiveStore,
    *,
    tenant_id: str,
) -> bool:
    """Detect pre-runtime Session evidence during startup, stopping at the first archive.

    This filesystem walk belongs only to startup migration gating.  Online
    retrieval never calls it and continues to use bounded Catalog queries.
    """

    if archive_store.tenant_id != tenant_id:
        raise ValueError("archive store must be bound to the migration tenant")
    root = archive_store.root.resolve()
    tenant_root = root / "tenants" / tenant_id / "users"
    if tenant_root.is_symlink():
        raise ValueError("session archive tenant root cannot be a symbolic link")
    if not tenant_root.exists():
        return False
    for directory, directory_names, file_names in os.walk(tenant_root, followlinks=False):
        base = Path(directory)
        if base.is_symlink():
            raise ValueError("session archive directory cannot be a symbolic link")
        for name in directory_names:
            if (base / name).is_symlink():
                raise ValueError("session archive directory cannot be a symbolic link")
        directory_names[:] = sorted(directory_names)
        if "commit_head.json" not in file_names:
            continue
        head_path = base / "commit_head.json"
        if head_path.is_symlink():
            raise ValueError("session archive head cannot be a symbolic link")
        parts = head_path.relative_to(tenant_root).parts
        if len(parts) >= 5 and parts[1:3] == ("sessions", "history"):
            return True
    return False


@dataclass(frozen=True)
class BackfillBatchResult:
    state: MigrationState
    processed_archives: int
    projected_records: int
    checkpoint: str
    complete: bool


@dataclass(frozen=True)
class ShadowValidationBatchResult:
    processed_archives: int
    processed_canonical_slots: int
    sample_count: int
    mismatch_count: int
    checkpoint: str
    complete: bool


@dataclass(frozen=True)
class SessionCatalogRebuildBatchResult:
    """One bounded offline SessionArchive-to-Catalog rebuild batch."""

    processed_archives: int
    projected_records: int
    vectors_projected: int
    tombstoned_records: int
    checkpoint: str
    complete: bool


class UnifiedContextMigration:
    """Idempotent SessionArchive backfill plus shadow/cutover orchestration."""

    def __init__(
        self,
        state_store: MigrationStateStore,
        archive_store: SessionArchiveStore,
        projector: SessionContextProjector,
        *,
        tenant_id: str,
        migration_name: str = "unified-context-catalog-v1",
        batch_size: int = 256,
        minimum_shadow_samples: int = 20,
        maximum_shadow_mismatch_ratio: float = 0.0,
        canonical_current_backfill: CanonicalCurrentBackfill | None = None,
        queue_store: Any | None = None,
        lock_store: Any | None = None,
        sanitizer: ContextProjectionSanitizer | None = None,
    ) -> None:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if not 1 <= int(batch_size) <= 1_000:
            raise ValueError("batch_size must be between 1 and 1000")
        if minimum_shadow_samples < 1:
            raise ValueError("minimum_shadow_samples must be positive")
        if not 0.0 <= maximum_shadow_mismatch_ratio <= 1.0:
            raise ValueError("maximum_shadow_mismatch_ratio must be between zero and one")
        if archive_store.tenant_id != tenant_id:
            raise ValueError("archive store must be bound to the migration tenant")
        self.state_store = state_store
        self.archive_store = archive_store
        self.projector = projector
        self.tenant_id = tenant_id
        self.migration_name = migration_name
        self.batch_size = int(batch_size)
        self.minimum_shadow_samples = int(minimum_shadow_samples)
        self.maximum_shadow_mismatch_ratio = float(maximum_shadow_mismatch_ratio)
        self.canonical_current_backfill = canonical_current_backfill
        self.queue_store = queue_store
        self.lock_store = lock_store
        self.sanitizer = sanitizer or ContextProjectionSanitizer()

    @property
    def projection_fence_key(self) -> str:
        return f"migration:{self.tenant_id}:{self.migration_name}:projection"

    @property
    def state(self) -> MigrationState:
        row = self._row()
        return MigrationState(str(row["state"]))

    @property
    def feature_gate(self) -> MigrationFeatureGate:
        return MigrationFeatureGate(self.state)

    @contextmanager
    def derived_rebuild_fence(self) -> Iterator[Any]:
        """Fence cross-process Session publication during destructive repair."""

        with self._cutover_fence() as token:
            yield token

    def initialize(self) -> dict[str, Any]:
        binder = getattr(self.state_store, "bind_migration_tenant_from_schema_upgrade", None)
        if callable(binder):
            bound: Any = binder(
                self.migration_name,
                tenant_id=self.tenant_id,
                batch_size=self.batch_size,
            )
            if bound is not None:
                if not isinstance(bound, Mapping):
                    raise TypeError("schema-upgrade migration binding returned an invalid result")
                return dict(bound)
        current = self.state_store.get_migration_state(
            self.migration_name,
            tenant_id=self.tenant_id,
        )
        if current is not None:
            return current
        return self.state_store.set_migration_state(
            self.migration_name,
            MigrationState.NOT_STARTED.value,
            tenant_id=self.tenant_id,
            batch_size=self.batch_size,
            details={"schema_version": 0},
        )

    def prepare_schema(self) -> dict[str, Any]:
        row = self._row()
        state = MigrationState(str(row["state"]))
        if state is MigrationState.SCHEMA_READY:
            return row
        if state is not MigrationState.NOT_STARTED:
            raise ValueError(f"cannot prepare schema from {state.value}")
        version = int(self.state_store.catalog_schema_version())
        if version < _REQUIRED_CATALOG_SCHEMA_VERSION:
            return self.fail(f"catalog schema v{_REQUIRED_CATALOG_SCHEMA_VERSION} is required; found v{version}")
        return self._transition(
            MigrationState.SCHEMA_READY,
            details={**self._details(row), "schema_version": version},
        )

    def start_backfill(self) -> dict[str, Any]:
        row = self._row()
        state = MigrationState(str(row["state"]))
        if state is MigrationState.BACKFILLING:
            return row
        if state not in {MigrationState.SCHEMA_READY, MigrationState.ROLLBACK}:
            raise ValueError(f"cannot start backfill from {state.value}")
        details = self._details(row)
        if state is MigrationState.ROLLBACK:
            # A rollback begins a new repair epoch.  The former checkpoints
            # only prove that the old derived Catalog was visited; keeping
            # them can skip a damaged or deliberately cleared Catalog and
            # return to DUAL_WRITE without rebuilding it.
            details.update(
                {
                    "backfill_epoch": uuid.uuid4().hex,
                    "backfilled_archives": 0,
                    "backfilled_canonical_slots": 0,
                    "projected_records": 0,
                    "session_checkpoint": "",
                    "canonical_checkpoint": "",
                    "session_backfill_complete": False,
                    "canonical_backfill_complete": self.canonical_current_backfill is None,
                    "backfill_complete": False,
                }
            )
        else:
            details.setdefault("backfill_epoch", uuid.uuid4().hex)
            details.setdefault("backfilled_archives", 0)
            details.setdefault("backfilled_canonical_slots", 0)
            details.setdefault("projected_records", 0)
        details["backfill_complete"] = False
        details.setdefault("session_backfill_complete", False)
        details.setdefault("canonical_backfill_complete", self.canonical_current_backfill is None)
        return self._transition(
            MigrationState.BACKFILLING,
            checkpoint="" if state is MigrationState.ROLLBACK else None,
            details=details,
        )

    def backfill_next_batch(self) -> BackfillBatchResult:
        row = self._row()
        state = MigrationState(str(row["state"]))
        if state in {
            MigrationState.DUAL_WRITE,
            MigrationState.SHADOW_VALIDATING,
            MigrationState.READY_TO_CUTOVER,
            MigrationState.CUTOVER,
            MigrationState.COMPLETED,
        }:
            return BackfillBatchResult(state, 0, 0, str(row.get("checkpoint") or ""), True)
        if state is not MigrationState.BACKFILLING:
            raise ValueError(f"cannot backfill from {state.value}")
        details = self._details(row)
        checkpoint = (
            str(details.get("session_checkpoint") or "")
            if "session_checkpoint" in details
            else str(row.get("checkpoint") or "")
        )
        session_complete = bool(details.get("session_backfill_complete", False))
        iterator = self._iter_verified_archives(after_checkpoint=checkpoint) if not session_complete else iter(())
        selected: list[tuple[str, SessionArchive]] = []
        has_more = False
        try:
            for item in iterator:
                if len(selected) >= self.batch_size:
                    has_more = True
                    break
                selected.append(item)
        except Exception as exc:
            self.fail(str(exc), failed_from=MigrationState.BACKFILLING)
            raise

        projected = 0
        last_checkpoint = checkpoint
        try:
            for archive_checkpoint, archive in selected:
                result: SessionProjectionResult = self.projector.project(archive)
                if result.equivalence_proof is None:
                    raise RuntimeError("Session backfill projection has no independent equivalence proof")
                self.record_projection_equivalence(result.equivalence_proof)
                projected += result.projected
                last_checkpoint = archive_checkpoint
        except Exception as exc:
            self.fail(str(exc), failed_from=MigrationState.BACKFILLING)
            raise

        details = self._details(row)
        details["backfilled_archives"] = int(details.get("backfilled_archives") or 0) + len(selected)
        details["projected_records"] = int(details.get("projected_records") or 0) + projected
        session_complete = session_complete or not has_more
        details["session_backfill_complete"] = session_complete
        details["session_checkpoint"] = last_checkpoint

        canonical_result: CanonicalBackfillBatchResult | None = None
        if session_complete and self.canonical_current_backfill is not None:
            canonical_checkpoint = str(details.get("canonical_checkpoint") or "")
            try:
                canonical_result = self.canonical_current_backfill(
                    canonical_checkpoint,
                    self.batch_size,
                )
            except Exception as exc:
                self.fail(str(exc), failed_from=MigrationState.BACKFILLING)
                raise
            if canonical_result.processed_slots < 0 or canonical_result.projected_records < 0:
                raise ValueError("canonical CurrentSlot backfill returned invalid counts")
            details["backfilled_canonical_slots"] = (
                int(details.get("backfilled_canonical_slots") or 0) + canonical_result.processed_slots
            )
            details["projected_records"] = int(details.get("projected_records") or 0) + (
                canonical_result.projected_records
            )
            details["canonical_checkpoint"] = canonical_result.checkpoint
            details["canonical_backfill_complete"] = canonical_result.complete
            for proof in canonical_result.equivalence_proofs:
                self.record_projection_equivalence(proof)
            projected += canonical_result.projected_records
        complete = session_complete and bool(details.get("canonical_backfill_complete", False))
        details["backfill_complete"] = complete
        journal_details = self._details(self._row())
        for key in (
            "equivalence_proof_count",
            "equivalence_mismatch_count",
            "last_equivalence_proof",
        ):
            if key in journal_details:
                details[key] = journal_details[key]
        durable_checkpoint = (
            f"canonical:{details.get('canonical_checkpoint', '')}"
            if session_complete and self.canonical_current_backfill is not None
            else last_checkpoint
        )
        if complete:
            updated = self._transition(
                MigrationState.DUAL_WRITE,
                checkpoint=durable_checkpoint,
                details=details,
            )
        else:
            updated = self._persist(
                MigrationState.BACKFILLING,
                checkpoint=durable_checkpoint,
                details=details,
            )
        return BackfillBatchResult(
            state=MigrationState(str(updated["state"])),
            processed_archives=len(selected),
            projected_records=projected,
            checkpoint=durable_checkpoint,
            complete=complete,
        )

    def rebuild_session_catalog_next_batch(
        self,
        checkpoint: str = "",
        *,
        batch_size: int | None = None,
    ) -> SessionCatalogRebuildBatchResult:
        """Rebuild a bounded Session Catalog batch from immutable evidence.

        This is an administrative repair primitive, not an online retrieval
        fallback and not a migration-state transition. Each archive is read
        through the same verified commit-head scanner used by migration,
        projected idempotently, and read back through its exact evidence
        identity before the checkpoint may advance.
        """

        bounded_batch_size = self.batch_size if batch_size is None else int(batch_size)
        if not 1 <= bounded_batch_size <= 1_000:
            raise ValueError("Session Catalog rebuild batch_size must be between 1 and 1000")
        iterator = self._iter_verified_archives(after_checkpoint=str(checkpoint or ""))
        selected: list[tuple[str, SessionArchive]] = []
        has_more = False
        for item in iterator:
            if len(selected) >= bounded_batch_size:
                has_more = True
                break
            selected.append(item)

        projected_records = 0
        vectors_projected = 0
        tombstoned_records = 0
        last_checkpoint = str(checkpoint or "")
        for archive_checkpoint, archive in selected:
            result = self.projector.project(
                archive,
                respect_applied_tombstones=True,
            )
            proof = result.equivalence_proof
            if proof is None or proof.overflow or not proof.matched:
                raise RuntimeError("Session Catalog rebuild equivalence proof failed")
            projected_records += result.projected
            vectors_projected += result.vectors_projected
            tombstoned_records += result.tombstoned_records
            last_checkpoint = archive_checkpoint
        return SessionCatalogRebuildBatchResult(
            processed_archives=len(selected),
            projected_records=projected_records,
            vectors_projected=vectors_projected,
            tombstoned_records=tombstoned_records,
            checkpoint=last_checkpoint,
            complete=not has_more,
        )

    def start_shadow_validation(self) -> dict[str, Any]:
        row = self._row()
        state = MigrationState(str(row["state"]))
        if state is MigrationState.SHADOW_VALIDATING:
            return row
        if state not in {MigrationState.DUAL_WRITE, MigrationState.READY_TO_CUTOVER}:
            raise ValueError(f"cannot start shadow validation from {state.value}")
        details = self._details(row)
        if not bool(details.get("backfill_complete", False)):
            raise ValueError("all Session and CurrentSlot backfills must complete before cutover")
        # Every validation round has an independent durable journal epoch.
        # Old samples cannot silently satisfy a later cutover attempt.
        details["shadow_validation_epoch"] = uuid.uuid4().hex
        details["shadow_sample_count"] = 0
        details["shadow_mismatch_count"] = 0
        details["shadow_read_sample_count"] = 0
        details["shadow_read_mismatch_count"] = 0
        details["shadow_archive_checkpoint"] = ""
        details["shadow_archive_validation_complete"] = False
        details["shadow_canonical_checkpoint"] = ""
        details["shadow_canonical_validation_complete"] = self.canonical_current_backfill is None
        details.update(self._shadow_source_snapshot())
        return self._transition(MigrationState.SHADOW_VALIDATING, details=details)

    def record_projection_equivalence(
        self,
        proof: ProjectionEquivalenceProof,
    ) -> dict[str, Any]:
        recorder = getattr(self.state_store, "record_migration_equivalence_proof", None)
        if not callable(recorder):
            raise RuntimeError("migration state store has no durable equivalence journal")
        recorded: Any = recorder(
            self.migration_name,
            proof.to_journal_entry(),
            tenant_id=self.tenant_id,
        )
        if not isinstance(recorded, Mapping):
            raise TypeError("migration equivalence journal returned an invalid result")
        return dict(recorded)

    def validate_next_shadow_batch(self) -> ShadowValidationBatchResult:
        """Validate a bounded source-evidence batch without running search twice."""

        row = self._row()
        if MigrationState(str(row["state"])) is not MigrationState.SHADOW_VALIDATING:
            raise ValueError("shadow projection validation requires SHADOW_VALIDATING state")
        details = self._details(row)
        current_snapshot = self._shadow_source_snapshot()
        if not self._shadow_snapshot_matches(details, current_snapshot):
            details = self._reset_shadow_validation(details, current_snapshot)
            self._persist(
                MigrationState.SHADOW_VALIDATING,
                checkpoint=str(row.get("checkpoint") or ""),
                details=details,
            )
            raise RuntimeError("shadow source set changed; validation epoch restarted")
        checkpoint = str(details.get("shadow_archive_checkpoint") or "")
        iterator = self._iter_verified_archives(after_checkpoint=checkpoint)
        selected: list[tuple[str, SessionArchive]] = []
        has_more = False
        for item in iterator:
            if len(selected) >= self.batch_size:
                has_more = True
                break
            selected.append(item)
        last_checkpoint = checkpoint
        for archive_checkpoint, archive in selected:
            proof = self.projector.prove_projection(archive)
            if proof is None:
                raise RuntimeError("shadow Session validation has no independent equivalence proof")
            self.record_projection_equivalence(proof)
            last_checkpoint = archive_checkpoint
        archive_complete = not has_more
        canonical_processed = 0
        canonical_complete = bool(details.get("shadow_canonical_validation_complete", False))
        canonical_checkpoint = str(details.get("shadow_canonical_checkpoint") or "")
        if archive_complete and self.canonical_current_backfill is not None and not canonical_complete:
            canonical_prover = getattr(self.canonical_current_backfill, "prove", None)
            if not callable(canonical_prover):
                raise RuntimeError("shadow CurrentSlot validation cannot use a mutating backfill callback")
            canonical_result = canonical_prover(canonical_checkpoint, self.batch_size)
            if not isinstance(canonical_result, CanonicalBackfillBatchResult):
                raise TypeError("CurrentSlot shadow proof returned an invalid batch result")
            canonical_processed = canonical_result.processed_slots
            canonical_checkpoint = canonical_result.checkpoint
            canonical_complete = canonical_result.complete
            for proof in canonical_result.equivalence_proofs:
                self.record_projection_equivalence(proof)
        # Reload after journal appends so checkpoint persistence cannot erase
        # concurrently recorded live dual-write proof counters.
        current = self._row()
        details = self._details(current)
        details["shadow_archive_checkpoint"] = last_checkpoint
        details["shadow_archive_validation_complete"] = archive_complete
        details["shadow_canonical_checkpoint"] = canonical_checkpoint
        details["shadow_canonical_validation_complete"] = canonical_complete
        updated = self._persist(
            MigrationState.SHADOW_VALIDATING,
            checkpoint=str(current.get("checkpoint") or ""),
            details=details,
        )
        updated_details = self._details(updated)
        return ShadowValidationBatchResult(
            processed_archives=len(selected),
            processed_canonical_slots=canonical_processed,
            sample_count=int(updated_details.get("shadow_sample_count") or 0),
            mismatch_count=int(updated_details.get("shadow_mismatch_count") or 0),
            checkpoint=last_checkpoint,
            complete=archive_complete and canonical_complete,
        )

    def mark_ready_to_cutover(self) -> dict[str, Any]:
        with self._cutover_fence() as token:
            return self._mark_ready_to_cutover_locked(token)

    def _mark_ready_to_cutover_locked(self, token: Any) -> dict[str, Any]:
        row = self._row()
        state = MigrationState(str(row["state"]))
        if state is MigrationState.READY_TO_CUTOVER:
            return row
        if state is not MigrationState.SHADOW_VALIDATING:
            raise ValueError(f"cannot mark ready from {state.value}")
        details = self._details(row)
        current_snapshot = self._shadow_source_snapshot()
        if not self._shadow_snapshot_matches(details, current_snapshot):
            details = self._reset_shadow_validation(details, current_snapshot)
            self._persist(
                MigrationState.SHADOW_VALIDATING,
                checkpoint=str(row.get("checkpoint") or ""),
                details=details,
            )
            raise ValueError("shadow source set changed; validation epoch restarted")
        self._require_projection_queues_quiescent()
        if not bool(details.get("shadow_archive_validation_complete", False)) or not bool(
            details.get("shadow_canonical_validation_complete", False)
        ):
            raise ValueError("shadow source projection validation has not completed")
        epoch = str(details.get("shadow_validation_epoch") or "")
        summary_reader = getattr(self.state_store, "get_migration_equivalence_summary", None)
        if not callable(summary_reader) or not epoch:
            raise ValueError("shadow validation has no durable equivalence journal epoch")
        raw_summary: Any = summary_reader(
            self.migration_name,
            tenant_id=self.tenant_id,
            validation_epoch=epoch,
        )
        if not isinstance(raw_summary, Mapping):
            raise ValueError("shadow validation journal summary is invalid")
        summary = dict(raw_summary)
        samples = int(summary.get("sample_count") or 0)
        mismatches = int(summary.get("mismatch_count") or 0)
        details["shadow_sample_count"] = samples
        details["shadow_mismatch_count"] = mismatches
        ratio = mismatches / samples if samples else 1.0
        if samples < self.minimum_shadow_samples:
            raise ValueError("shadow validation sample threshold is not met")
        if ratio > self.maximum_shadow_mismatch_ratio:
            raise ValueError("shadow validation mismatch threshold is exceeded")
        details["shadow_mismatch_ratio"] = ratio
        shadow_reader = getattr(self.state_store, "get_migration_shadow_read_summary", None)
        if not callable(shadow_reader):
            raise ValueError("shadow validation has no durable query-result journal")
        raw_shadow_summary: Any = shadow_reader(
            self.migration_name,
            tenant_id=self.tenant_id,
            validation_epoch=epoch,
        )
        if not isinstance(raw_shadow_summary, Mapping):
            raise ValueError("shadow query-result journal summary is invalid")
        shadow_samples = int(raw_shadow_summary.get("sample_count") or 0)
        shadow_mismatches = int(raw_shadow_summary.get("mismatch_count") or 0)
        shadow_ratio = shadow_mismatches / shadow_samples if shadow_samples else 1.0
        if shadow_samples < self.minimum_shadow_samples:
            raise ValueError("shadow read sample threshold is not met")
        if shadow_ratio > self.maximum_shadow_mismatch_ratio:
            raise ValueError("shadow read mismatch threshold is exceeded")
        details["shadow_read_sample_count"] = shadow_samples
        details["shadow_read_mismatch_count"] = shadow_mismatches
        details["shadow_read_mismatch_ratio"] = shadow_ratio
        self._assert_cutover_fence(token)
        return self._transition(MigrationState.READY_TO_CUTOVER, details=details)

    def cutover(self) -> dict[str, Any]:
        with self._cutover_fence() as token:
            return self._cutover_locked(token)

    def _cutover_locked(self, token: Any) -> dict[str, Any]:
        row = self._row()
        if MigrationState(str(row["state"])) is MigrationState.CUTOVER:
            return row
        if MigrationState(str(row["state"])) is not MigrationState.READY_TO_CUTOVER:
            raise ValueError(f"cannot cut over from {row['state']}")
        details = self._details(row)
        current_snapshot = self._shadow_source_snapshot()
        if not self._shadow_snapshot_matches(details, current_snapshot):
            reset = self._reset_shadow_validation(details, current_snapshot)
            self._transition(MigrationState.SHADOW_VALIDATING, details=reset)
            raise ValueError("cutover source set changed; shadow validation restarted")
        self._require_projection_queues_quiescent()
        self._assert_cutover_fence(token)
        return self._transition(MigrationState.CUTOVER)

    def complete(self) -> dict[str, Any]:
        row = self._row()
        if MigrationState(str(row["state"])) is MigrationState.COMPLETED:
            return row
        self._require_projection_queues_quiescent()
        return self._transition(MigrationState.COMPLETED)

    def _shadow_source_snapshot(self) -> dict[str, Any]:
        archive_digest = hashlib.sha256()
        archive_count = 0
        for checkpoint, archive in self._iter_verified_archives(after_checkpoint=""):
            for value in (checkpoint, archive.archive_digest, archive.manifest_digest):
                encoded = str(value).encode("utf-8")
                archive_digest.update(len(encoded).to_bytes(8, "big"))
                archive_digest.update(encoded)
            archive_count += 1
        canonical_digest = hashlib.sha256(b"").hexdigest()
        canonical_count = 0
        if self.canonical_current_backfill is not None:
            snapshot = getattr(self.canonical_current_backfill, "source_snapshot", None)
            if not callable(snapshot):
                raise RuntimeError("CurrentSlot migration has no non-mutating source snapshot")
            raw_snapshot: Any = snapshot()
            if not isinstance(raw_snapshot, tuple) or len(raw_snapshot) != 2:
                raise TypeError("CurrentSlot migration source snapshot is invalid")
            canonical_digest, canonical_count = raw_snapshot
        return {
            "shadow_archive_source_digest": archive_digest.hexdigest(),
            "shadow_archive_source_count": archive_count,
            "shadow_canonical_source_digest": str(canonical_digest),
            "shadow_canonical_source_count": int(canonical_count),
        }

    @staticmethod
    def _shadow_snapshot_matches(details: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
        return all(details.get(key) == value for key, value in current.items())

    def _reset_shadow_validation(
        self,
        details: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        reset = dict(details)
        reset.update(snapshot)
        reset.update(
            {
                "shadow_validation_epoch": uuid.uuid4().hex,
                "shadow_sample_count": 0,
                "shadow_mismatch_count": 0,
                "shadow_read_sample_count": 0,
                "shadow_read_mismatch_count": 0,
                "shadow_archive_checkpoint": "",
                "shadow_archive_validation_complete": False,
                "shadow_canonical_checkpoint": "",
                "shadow_canonical_validation_complete": self.canonical_current_backfill is None,
            }
        )
        return reset

    def _require_projection_queues_quiescent(self) -> None:
        frontier_reader = getattr(self.state_store, "get_session_projection_frontier_summary", None)
        if callable(frontier_reader):
            raw_frontier: Any = frontier_reader(tenant_id=self.tenant_id)
            if not isinstance(raw_frontier, Mapping):
                raise ValueError("cutover Session projection frontier is invalid")
            frontier_blockers = {
                status: int(raw_frontier.get(status, 0) or 0)
                for status in ("PENDING", "FAILED")
                if int(raw_frontier.get(status, 0) or 0) > 0
            }
            if frontier_blockers:
                raise ValueError(f"cutover has unresolved Session Catalog projection: {frontier_blockers}")
        if self.queue_store is None:
            return
        stats = getattr(self.queue_store, "stats", None)
        if not callable(stats):
            raise ValueError("cutover queue health is unavailable")
        blockers: dict[str, dict[str, int]] = {}
        for queue_name in ("session_commit", "memory_projection"):
            raw: Any = stats(queue_name=queue_name)
            if not isinstance(raw, Mapping):
                raise ValueError("cutover queue health is invalid")
            blocked = {
                state: int(raw.get(state, 0) or 0)
                for state in ("pending", "leased", "dead_letter", "quarantine")
                if int(raw.get(state, 0) or 0) > 0
            }
            if blocked:
                blockers[queue_name] = blocked
        if blockers:
            raise ValueError(f"cutover has outstanding projection work: {blockers}")

    @contextmanager
    def _cutover_fence(self) -> Iterator[Any]:
        acquire = getattr(self.lock_store, "acquire", None)
        release = getattr(self.lock_store, "release", None)
        if not callable(acquire) or not callable(release):
            raise ValueError("migration cutover requires a durable cross-process projection fence")
        raw_token = acquire(
            self.projection_fence_key,
            ttl_seconds=_PROJECTION_FENCE_TTL_SECONDS,
        )
        token = _RenewingProjectionFence(
            self.lock_store,
            raw_token,
            ttl_seconds=_PROJECTION_FENCE_TTL_SECONDS,
        )
        try:
            token.checkpoint()
            yield token
        finally:
            try:
                token.checkpoint()
            finally:
                token.stop()
                release(raw_token)

    def _assert_cutover_fence(self, token: Any) -> None:
        if isinstance(token, _RenewingProjectionFence):
            token.checkpoint()
            return
        checker = getattr(self.lock_store, "assert_owned", None)
        if not callable(checker):
            raise ValueError("migration cutover fence ownership cannot be verified")
        checker(token)

    def rollback(self, reason: str) -> dict[str, Any]:
        row = self._row()
        state = MigrationState(str(row["state"]))
        if state is MigrationState.ROLLBACK:
            return row
        if state in {MigrationState.NOT_STARTED, MigrationState.SCHEMA_READY}:
            raise ValueError(f"cannot rollback from {state.value}")
        details = self._details(row)
        details["rollback_from"] = state.value
        details["rollback_reason"] = self.sanitizer.sanitize_trace(str(reason))
        return self._transition(MigrationState.ROLLBACK, details=details)

    def fail(
        self,
        error: str,
        *,
        failed_from: MigrationState | None = None,
    ) -> dict[str, Any]:
        row = self._row()
        state = failed_from or MigrationState(str(row["state"]))
        details = self._details(row)
        details["failed_from"] = state.value
        return self._persist(
            MigrationState.FAILED,
            checkpoint=str(row.get("checkpoint") or ""),
            details=details,
            error=str(error),
        )

    def resume_failed(self) -> dict[str, Any]:
        row = self._row()
        if MigrationState(str(row["state"])) is not MigrationState.FAILED:
            return row
        details = self._details(row)
        target = MigrationState(str(details.get("failed_from") or MigrationState.BACKFILLING.value))
        if target in {MigrationState.NOT_STARTED, MigrationState.COMPLETED, MigrationState.FAILED}:
            raise ValueError(f"cannot resume failed migration at {target.value}")
        details["resumed_from_failure"] = True
        return self._transition(target, details=details)

    def _iter_verified_archives(
        self,
        *,
        after_checkpoint: str,
    ) -> Iterator[tuple[str, SessionArchive]]:
        root = self.archive_store.root.resolve()
        tenant_root = root / "tenants" / self.tenant_id / "users"
        if tenant_root.is_symlink():
            raise ValueError("session archive tenant root cannot be a symbolic link")
        if not tenant_root.exists():
            if after_checkpoint:
                raise ValueError("backfill checkpoint no longer exists")
            return
        checkpoint_path = root / after_checkpoint if after_checkpoint else None
        if checkpoint_path is not None and (
            checkpoint_path.is_symlink()
            or not checkpoint_path.is_file()
            or root not in checkpoint_path.resolve().parents
        ):
            raise ValueError("backfill checkpoint is invalid or no longer exists")
        resume_seen = not after_checkpoint
        for directory, directory_names, file_names in os.walk(tenant_root, followlinks=False):
            base = Path(directory)
            directory_names[:] = sorted(name for name in directory_names if not (base / name).is_symlink())
            for file_name in sorted(file_names):
                if file_name != "commit_head.json":
                    continue
                head_path = base / file_name
                relative = head_path.relative_to(root).as_posix()
                if not resume_seen:
                    if relative == after_checkpoint:
                        resume_seen = True
                    continue
                parts = head_path.relative_to(tenant_root).parts
                if len(parts) < 5 or parts[1:3] != ("sessions", "history"):
                    continue
                user_id = str(parts[0])
                archive = self.archive_store.read_archive_from_commit_head(
                    head_path,
                    tenant_id=self.tenant_id,
                    user_id=user_id,
                )
                yield relative, archive
        if after_checkpoint and not resume_seen:
            raise ValueError("backfill checkpoint was not found during archive enumeration")

    def _row(self) -> dict[str, Any]:
        return self.initialize()

    @staticmethod
    def _details(row: Mapping[str, Any]) -> dict[str, Any]:
        details = row.get("details_json")
        return dict(details) if isinstance(details, Mapping) else {}

    def _transition(
        self,
        target: MigrationState,
        *,
        checkpoint: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = self._row()
        current = MigrationState(str(row["state"]))
        if current is target:
            return row
        if target not in _ALLOWED_TRANSITIONS[current]:
            raise ValueError(f"invalid migration transition: {current.value} -> {target.value}")
        return self._persist(
            target,
            checkpoint=(str(row.get("checkpoint") or "") if checkpoint is None else checkpoint),
            details=self._details(row) if details is None else details,
        )

    def _persist(
        self,
        state: MigrationState,
        *,
        checkpoint: str,
        details: Mapping[str, Any],
        error: str = "",
    ) -> dict[str, Any]:
        return self.state_store.set_migration_state(
            self.migration_name,
            state.value,
            checkpoint,
            details,
            tenant_id=self.tenant_id,
            batch_size=self.batch_size,
            error=error,
        )


__all__ = [
    "BackfillBatchResult",
    "CanonicalBackfillBatchResult",
    "CanonicalCurrentBackfill",
    "CurrentSlotMigrationBackfill",
    "DERIVED_SERVING_REBUILD_NAME",
    "has_existing_session_archive_evidence",
    "MigrationFeatureGate",
    "MigrationState",
    "ReadRoute",
    "RuntimeMigrationCoordinator",
    "SessionCatalogRebuildBatchResult",
    "ShadowValidationBatchResult",
    "UnifiedContextMigration",
]
