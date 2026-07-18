"""Stable full-scan reconciliation for external Markdown edits."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from memoryos.memory.documents.control_store import MemoryDocumentControlStore
from memoryos.memory.documents.model import ManagedDocument, ScanGeneration
from memoryos.memory.documents.store import MemoryDocumentStore


class ExternalChangeKind(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RENAME = "rename"


@dataclass(frozen=True)
class ExternalDocumentChange:
    change_kind: ExternalChangeKind
    tenant_id: str
    owner_user_id: str
    document_id: str
    old_relative_path: str
    new_relative_path: str
    before_raw_digest: str
    after_raw_digest: str
    scan_generation_id: str


@dataclass(frozen=True)
class ScanReconciliation:
    generation: ScanGeneration
    confirmed_changes: tuple[ExternalDocumentChange, ...]
    pending_change_count: int
    deletions_paused: bool
    pause_reason: str = ""


class MemoryDocumentScanner:
    """Treat watcher events as hints and publish only stable full-scan facts."""

    def __init__(
        self,
        store: MemoryDocumentStore,
        *,
        control_store: MemoryDocumentControlStore | None = None,
        stability_seconds: float = 1.0,
        mass_delete_threshold: int = 50,
        clock: Callable[[], float] | None = None,
        change_publisher: Callable[[ExternalDocumentChange], None] | None = None,
    ) -> None:
        if stability_seconds < 0 or mass_delete_threshold <= 0:
            raise ValueError("invalid scanner stability or mass-delete limits")
        self.store = store
        self.control_store = control_store
        self.stability_seconds = stability_seconds
        self.mass_delete_threshold = mass_delete_threshold
        self.clock = clock or time.monotonic
        self.change_publisher = change_publisher
        self._known: dict[tuple[str, str], dict[str, ManagedDocument]] = {}
        self._root_identity: dict[tuple[str, str], str] = {}
        self._identity_blocked: set[tuple[str, str]] = set()
        self._pending: dict[tuple[str, str, str, str], tuple[tuple[str, ...], float]] = {}
        self._dirty: set[tuple[str, str]] = set()
        self._overflow: set[tuple[str, str]] = set()

    def notify(self, tenant_id: str, owner_user_id: str, *, overflow: bool = False) -> None:
        key = (str(tenant_id), str(owner_user_id))
        self._dirty.add(key)
        if overflow:
            self._overflow.add(key)

    def scan(
        self,
        tenant_id: str,
        owner_user_id: str,
        *,
        force_stable: bool = False,
    ) -> ScanReconciliation:
        key = (str(tenant_id), str(owner_user_id))
        control_store = self.control_store
        seeded_from_control = key not in self._known and control_store is not None
        durable_root_identity = ""
        has_durable_controls = False
        root_identity_blockers: tuple[str, ...] = ()
        if seeded_from_control:
            assert control_store is not None
            root_record = control_store.load_root_identity(*key)
            durable_root_identity = root_record.root_identity if root_record is not None else ""
            if not durable_root_identity:
                root_identity_blockers = control_store.root_identity_blockers(*key)
            control_records = control_store.controls(*key)
            has_durable_controls = bool(control_records)
            seed_registration = getattr(self.store, "seed_registration", None)
            if callable(seed_registration):
                for record in control_records:
                    if record.status == "present":
                        seed_registration(
                            key[0],
                            key[1],
                            record.document_id,
                            record.relative_path,
                        )
            previous = {
                record.document_id: ManagedDocument(
                    relative_path=record.relative_path,
                    document_id=record.document_id,
                    raw_sha256=record.raw_sha256,
                    size=record.size,
                )
                for record in control_records
                if record.status == "present"
            }
        else:
            previous = self._known.get(key, {})
        generation = self.store.full_scan(*key)
        now = self.clock()
        current = {item.document_id: item for item in generation.managed}
        durable_root_changed = bool(
            seeded_from_control
            and durable_root_identity
            and durable_root_identity != generation.root_identity
        )
        in_memory_root_changed = bool(
            not seeded_from_control
            and key in self._root_identity
            and self._root_identity.get(key) != generation.root_identity
        )
        root_changed = durable_root_changed or in_memory_root_changed
        missing_durable_identity = bool(
            seeded_from_control
            and (has_durable_controls or root_identity_blockers)
            and not durable_root_identity
        )
        if missing_durable_identity:
            # With no durable inode baseline there is no fact that can safely
            # distinguish the original tree from a byte-for-byte replacement.
            # Keep this scanner instance blocked until an operator restores the
            # durable artifact and restarts recovery.
            self._identity_blocked.add(key)
        identity_reconciliation_paused = key in self._identity_blocked
        unsafe_registration = bool(generation.unsafe_paths) or any(
            getattr(item, "status", "") != "managed" for item in generation.registrations
        )
        complete = generation.complete and not root_changed
        raw_changes = self._changes(key, previous=previous, current=current, generation=generation)
        delete_changes = tuple(change for change in raw_changes if change.change_kind is ExternalChangeKind.DELETE)
        deletions_paused = False
        pause_reason = ""
        if not complete:
            deletions_paused = True
            pause_reason = "scan is incomplete or root identity changed"
        elif identity_reconciliation_paused:
            deletions_paused = True
            pause_reason = "durable document root identity is missing for existing authority"
        elif unsafe_registration and delete_changes:
            deletions_paused = True
            pause_reason = "unsafe/unmanaged/quarantined paths make deletion ambiguous"
        elif len(delete_changes) >= self.mass_delete_threshold:
            deletions_paused = True
            pause_reason = "mass-delete threshold reached"

        confirmed: list[ExternalDocumentChange] = []
        for change in raw_changes:
            if identity_reconciliation_paused or root_changed:
                continue
            if change.change_kind is ExternalChangeKind.DELETE and deletions_paused:
                continue
            signature = (
                change.change_kind.value,
                change.old_relative_path,
                change.new_relative_path,
                change.before_raw_digest,
                change.after_raw_digest,
            )
            pending_key = (*key, change.document_id, change.change_kind.value)
            observed = self._pending.get(pending_key)
            if change.change_kind is ExternalChangeKind.DELETE:
                # A restart/full-scan override is never delete authority. Even
                # with a zero-second window, absence must be observed twice.
                if observed is None or observed[0] != signature:
                    self._pending[pending_key] = (signature, now)
                    stable = False
                else:
                    stable = now - observed[1] >= self.stability_seconds
            elif force_stable or self.stability_seconds == 0:
                stable = True
            elif observed is None or observed[0] != signature:
                self._pending[pending_key] = (signature, now)
                stable = False
            else:
                stable = now - observed[1] >= self.stability_seconds
            if not stable:
                continue
            confirmed.append(change)
            self._pending.pop(pending_key, None)
            if self.change_publisher is not None:
                self.change_publisher(change)

        # Advance only facts that were either unchanged or safely confirmed.
        next_known = dict(previous)
        changed_ids = {change.document_id for change in raw_changes}
        for document_id, item in current.items():
            if document_id not in changed_ids or any(change.document_id == document_id for change in confirmed):
                next_known[document_id] = item
        for change in confirmed:
            if change.change_kind is ExternalChangeKind.DELETE:
                next_known.pop(change.document_id, None)
        if not previous and not raw_changes and generation.complete:
            next_known = current
        self._known[key] = next_known
        safe_identity_advance = bool(
            generation.complete
            and generation.root_identity
            and not root_changed
            and not identity_reconciliation_paused
            and not deletions_paused
            and not unsafe_registration
        )
        if safe_identity_advance:
            if control_store is not None:
                control_store.ensure_root_identity(*key, generation.root_identity)
            self._root_identity[key] = generation.root_identity
        elif seeded_from_control and durable_root_identity:
            self._root_identity[key] = durable_root_identity
        self._dirty.discard(key)
        self._overflow.discard(key)
        active_pending_keys = {
            (*key, change.document_id, change.change_kind.value)
            for change in raw_changes
            if not identity_reconciliation_paused
            and not root_changed
            and not (change.change_kind is ExternalChangeKind.DELETE and deletions_paused)
        }
        for pending_key in tuple(self._pending):
            if pending_key[:2] == key and pending_key not in active_pending_keys:
                self._pending.pop(pending_key, None)
        pending_count = sum(1 for pending_key in self._pending if pending_key[:2] == key)
        return ScanReconciliation(
            generation=generation,
            confirmed_changes=tuple(confirmed),
            pending_change_count=pending_count,
            deletions_paused=deletions_paused,
            pause_reason=pause_reason,
        )

    @staticmethod
    def _changes(
        key: tuple[str, str],
        *,
        previous: dict[str, ManagedDocument],
        current: dict[str, ManagedDocument],
        generation: ScanGeneration,
    ) -> tuple[ExternalDocumentChange, ...]:
        tenant, owner = key
        result: list[ExternalDocumentChange] = []
        for document_id in sorted(previous.keys() | current.keys()):
            before = previous.get(document_id)
            after = current.get(document_id)
            if before == after:
                continue
            if before is None and after is not None:
                kind = ExternalChangeKind.CREATE
            elif before is not None and after is None:
                kind = ExternalChangeKind.DELETE
            elif before is not None and after is not None and before.relative_path != after.relative_path:
                kind = ExternalChangeKind.RENAME
            else:
                kind = ExternalChangeKind.UPDATE
            result.append(
                ExternalDocumentChange(
                    change_kind=kind,
                    tenant_id=tenant,
                    owner_user_id=owner,
                    document_id=document_id,
                    old_relative_path=before.relative_path if before else "",
                    new_relative_path=after.relative_path if after else "",
                    before_raw_digest=before.raw_sha256 if before else "",
                    after_raw_digest=after.raw_sha256 if after else "",
                    scan_generation_id=generation.generation_id,
                )
            )
        return tuple(result)


__all__ = [
    "ExternalChangeKind",
    "ExternalDocumentChange",
    "MemoryDocumentScanner",
    "ScanReconciliation",
]
