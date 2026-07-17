"""Outbox responsibilities for canonical projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memoryos.contextdb.catalog import (
    CatalogRecord,
)
from memoryos.contextdb.projection_equivalence import build_projection_equivalence_proof
from memoryos.contextdb.store.queue_store import (
    QueueIdempotencyConflictError,
    QueueJob,
)
from memoryos.core.durable_io.quarantine import quarantine_control_file
from memoryos.memory.canonical.slot_projection import CurrentSlotProjectionResult
from memoryos.operations.commit.outbox_envelope import (
    OUTBOX_EVENT_TYPE,
    OutboxIntegrityError,
    projection_workspace_id,
    validate_outbox,
)

from .models import (
    ProjectionOutboxIntegrityError,
    _CurrentSlotProjectionTarget,
)

if TYPE_CHECKING:
    from .worker import MemoryProjectionWorker


def _read_outbox(self: MemoryProjectionWorker, path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink():
            raise OutboxIntegrityError("canonical outbox path cannot be a symbolic link")
        return validate_outbox(
            json.loads(path.read_text(encoding="utf-8")),
            allowed_statuses={"committed"},
        )
    except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
        if path.exists():
            quarantine_control_file(
                self.projector.root,
                path,
                kind="outbox",
                error=exc,
                identifiers={"transaction_id": path.stem},
            )
        raise ProjectionOutboxIntegrityError("projection job references an invalid committed outbox event") from exc


def _load_projection_job_outbox(
    self: MemoryProjectionWorker,
    job: QueueJob,
    *,
    expected_transaction_id: str = "",
) -> dict[str, Any]:
    """Bind a durable queue identity to exactly one committed outbox."""

    self._assert_projection_job_identity_unchanged(job)
    declared_transaction = str(job.payload.get("transaction_id") or "")
    if expected_transaction_id and declared_transaction != expected_transaction_id:
        raise ProjectionOutboxIntegrityError("projection queue transaction identity does not match completion request")
    if (
        not declared_transaction
        or job.job_id != f"outbox_{declared_transaction}"
        or job.queue_name != "memory_projection"
        or job.action != "project_memory_committed"
    ):
        raise ProjectionOutboxIntegrityError("projection queue job identity is invalid")
    expected_candidate = self.projector.root / "system" / "outbox" / f"{declared_transaction}.json"
    expected_path = expected_candidate.resolve()
    raw_path = job.payload.get("outbox_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ProjectionOutboxIntegrityError("projection queue job has no outbox path")
    try:
        raw_candidate = Path(raw_path)
        if raw_candidate.is_symlink() or expected_candidate.is_symlink():
            raise ProjectionOutboxIntegrityError("projection queue outbox path cannot be a symbolic link")
        actual_path = raw_candidate.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProjectionOutboxIntegrityError("projection queue outbox path is invalid") from exc
    if actual_path != expected_path:
        raise ProjectionOutboxIntegrityError("projection queue job is detached from its tenant outbox path")
    outbox = self._read_outbox(actual_path)
    operation_ids = job.payload.get("operation_ids")
    if (
        outbox.get("transaction_id") != declared_transaction
        or not isinstance(operation_ids, list)
        or operation_ids != outbox.get("operation_ids")
    ):
        raise ProjectionOutboxIntegrityError("projection queue job is detached from its immutable operation set")
    return outbox


def _project_event(self: MemoryProjectionWorker, outbox: dict[str, Any], job_id: str, stale: list[str]) -> None:
    for item in outbox.get("claim_revisions", []) or []:
        if not isinstance(item, dict) or not item.get("uri") or item.get("revision") is None:
            raise ValueError("projection outbox contains an invalid claim revision")
        result = self.projector.project(str(item["uri"]), int(item["revision"]))
        if result.status == "skipped_stale":
            stale.append(job_id)
    if self.current_slot_projector is None:
        return
    if self.migration_gate is not None:
        feature_gate = getattr(self.migration_gate, "feature_gate", None)
        if feature_gate is None or not bool(getattr(feature_gate, "dual_write_enabled", False)):
            # Claim revision projection is the compatibility serving path.
            # CurrentSlot rows are rebuilt in bounded migration batches
            # before the feature gate can reach cutover.
            return
    for target in self._current_slot_projection_targets(outbox):
        if (
            target.previous_active_claim_id is not None
            and target.active_claim_id is not None
            and target.previous_active_claim_id != target.active_claim_id
        ):
            if target.previous_source_revision is None:
                raise ValueError("active Claim switch has no previous Slot revision")
            self.current_slot_projector.tombstone_active_claim_switch(
                slot_id=target.slot_id,
                slot_uri=target.slot_uri,
                tenant_id=target.tenant_id,
                previous_active_claim_id=target.previous_active_claim_id,
                active_claim_id=target.active_claim_id,
                previous_source_revision=target.previous_source_revision,
                replacement_source_revision=target.source_revision,
            )
        slot_result = self.current_slot_projector.project(target.slot_uri)
        self._record_current_slot_equivalence(outbox, target, slot_result)


def _record_current_slot_equivalence(
    self: MemoryProjectionWorker,
    outbox: dict[str, Any],
    target: _CurrentSlotProjectionTarget,
    result: CurrentSlotProjectionResult,
) -> None:
    """Journal exact CurrentSlot identity derived from validated outbox work."""

    recorder = getattr(self.migration_gate, "record_projection_equivalence", None)
    if not callable(recorder):
        return
    catalog_store = getattr(self.current_slot_projector, "catalog_store", None)
    getter = getattr(catalog_store, "get_catalog", None)
    state = str(
        getattr(
            getattr(getattr(self.migration_gate, "feature_gate", None), "state", None),
            "value",
            "",
        )
    )
    if not callable(getter):
        if state == "SHADOW_VALIDATING":
            raise RuntimeError("shadow CurrentSlot projection has no exact Catalog proof lookup")
        return
    actual = getter(result.record_key, tenant_id=target.tenant_id)
    if actual is not None and not isinstance(actual, CatalogRecord):
        raise TypeError("CurrentSlot proof lookup returned an invalid Catalog record")
    expected_records = (result.record,) if result.record is not None else ()
    actual_records = (actual,) if actual is not None else ()
    receipt_digest = str(outbox.get("receipt_digest") or "")
    if not receipt_digest:
        raise ProjectionOutboxIntegrityError("projection outbox has no receipt evidence digest")
    proof = build_projection_equivalence_proof(
        plane="canonical_current_slot",
        source_identity=target.slot_uri,
        evidence_digest=receipt_digest,
        expected_records=expected_records,
        actual_records=actual_records,
    )
    recorder(proof)


def _current_slot_projection_targets(
    outbox: dict[str, Any],
) -> tuple[_CurrentSlotProjectionTarget, ...]:
    """Derive exact Slot work only from the already validated durable intent."""

    tenant_id = outbox.get("tenant_id")
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("projection outbox has no tenant identity")
    before_by_uri: dict[str, dict[str, Any]] = {}
    for snapshot in outbox.get("before_images", []) or []:
        if not isinstance(snapshot, dict):
            raise ValueError("projection outbox contains an invalid before image")
        uri = snapshot.get("uri")
        if not isinstance(uri, str) or not uri:
            raise ValueError("projection outbox before image has no URI")
        if snapshot.get("exists") is True:
            before = snapshot.get("object")
            if not isinstance(before, dict):
                raise ValueError("projection outbox existing before image has no object")
            before_by_uri[uri] = before

    targets: list[_CurrentSlotProjectionTarget] = []
    seen: set[str] = set()
    for raw_operation in outbox.get("operations", []) or []:
        if not isinstance(raw_operation, dict):
            raise ValueError("projection outbox contains an invalid operation")
        payload = raw_operation.get("payload")
        context_object = payload.get("context_object") if isinstance(payload, dict) else None
        if not isinstance(context_object, dict):
            continue
        metadata_value = context_object.get("metadata")
        metadata = dict(metadata_value) if isinstance(metadata_value, dict) else {}
        if metadata.get("canonical_kind") != "slot":
            continue
        slot_uri = context_object.get("uri")
        slot_id = metadata.get("slot_id")
        source_revision = metadata.get("revision")
        object_tenant_id = str(context_object.get("tenant_id") or "default")
        if (
            not isinstance(slot_uri, str)
            or not slot_uri
            or not isinstance(slot_id, str)
            or not slot_id
            or slot_uri.rsplit("/", 1)[-1] != slot_id
            or isinstance(source_revision, bool)
            or not isinstance(source_revision, int)
            or source_revision < 1
            or object_tenant_id != tenant_id
        ):
            raise ValueError("projection outbox Slot operation has an invalid revision identity")
        if slot_uri in seen:
            raise ValueError("projection outbox contains duplicate Slot projection work")
        seen.add(slot_uri)
        active_claim_id = _optional_claim_id(
            metadata.get("active_claim_id"),
            label="Slot active_claim_id",
        )

        previous_source_revision: int | None = None
        previous_active_claim_id: str | None = None
        before = before_by_uri.get(slot_uri)
        if before is not None:
            before_metadata_value = before.get("metadata")
            before_metadata = dict(before_metadata_value) if isinstance(before_metadata_value, dict) else {}
            before_revision = before_metadata.get("revision")
            if (
                before_metadata.get("canonical_kind") != "slot"
                or before_metadata.get("slot_id") != slot_id
                or str(before.get("tenant_id") or "default") != tenant_id
                or isinstance(before_revision, bool)
                or not isinstance(before_revision, int)
                or before_revision < 1
                or before_revision >= source_revision
            ):
                raise ValueError("projection outbox Slot before image is detached from its replacement")
            previous_source_revision = before_revision
            previous_active_claim_id = _optional_claim_id(
                before_metadata.get("active_claim_id"),
                label="previous Slot active_claim_id",
            )
        targets.append(
            _CurrentSlotProjectionTarget(
                slot_uri=slot_uri,
                slot_id=slot_id,
                tenant_id=tenant_id,
                source_revision=source_revision,
                active_claim_id=active_claim_id,
                previous_source_revision=previous_source_revision,
                previous_active_claim_id=previous_active_claim_id,
            )
        )
    return tuple(targets)


def _optional_claim_id(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"projection outbox {label} is invalid")
    return value


def dispatch_outbox(self: MemoryProjectionWorker) -> list[str]:
    with self._migration_projection_fence():
        return self._dispatch_outbox_unfenced()


def _dispatch_outbox_unfenced(self: MemoryProjectionWorker) -> list[str]:
    outbox_root = self.projector.root / "system" / "outbox"
    if not outbox_root.exists():
        return []
    validated: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(outbox_root.glob("*.json")):
        try:
            if path.is_symlink():
                raise OutboxIntegrityError("canonical outbox path cannot be a symbolic link")
            event = validate_outbox(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
            quarantine_control_file(
                self.projector.root,
                path,
                kind="outbox",
                error=exc,
                identifiers={"transaction_id": path.stem},
            )
            self.last_quarantined.append(path.stem)
            self._mark_authoritative_integrity_failure(
                exc,
                artifact="committed_outbox",
                identifiers={"transaction_id": path.stem},
            )
            raise ProjectionOutboxIntegrityError("authoritative outbox scan failed before projection dispatch") from exc
        validated.append((path, event))

    pending_jobs: list[tuple[str, QueueJob]] = []
    for path, event in validated:
        if event.get("event_type") != OUTBOX_EVENT_TYPE or event.get("status") != "committed":
            continue
        transaction_id = str(event.get("transaction_id", ""))
        if not transaction_id or path.stem != transaction_id:
            failure = ProjectionOutboxIntegrityError("committed outbox path is detached from its transaction identity")
            self._mark_authoritative_integrity_failure(
                failure,
                artifact="committed_outbox",
                identifiers={"transaction_id": transaction_id or path.stem},
            )
            raise failure
        claim_revisions = event.get("claim_revisions", []) or []
        operations = [item for item in event.get("operations", []) or [] if isinstance(item, dict)]
        target_uri = next(
            (
                str(payload.get("uri", ""))
                for item in operations
                if isinstance((payload := item.get("payload", {}).get("context_object")), dict)
                and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "slot"
            ),
            str(claim_revisions[0].get("uri", "")).rsplit("/claims/", 1)[0] if claim_revisions else transaction_id,
        )
        pending_jobs.append(
            (
                transaction_id,
                QueueJob(
                    job_id=f"outbox_{transaction_id}",
                    queue_name="memory_projection",
                    action="project_memory_committed",
                    target_uri=target_uri,
                    payload={
                        "transaction_id": transaction_id,
                        "outbox_path": str(path),
                        "operation_ids": [str(item) for item in event.get("operation_ids", []) or []],
                        "tenant_id": str(event.get("tenant_id") or "default"),
                        "owner_user_id": str(event.get("user_id") or ""),
                        "workspace_id": projection_workspace_id(operations),
                    },
                ),
            )
        )

    # Validate every existing queue identity before publishing any new
    # job.  A corrupt member cannot allow later valid work in this scan to
    # reach lease or derived projection writes.
    for transaction_id, expected in pending_jobs:
        existing = self.queue_store.get(expected.job_id)
        if existing is None:
            continue
        legacy_payload = {
            "transaction_id": expected.payload["transaction_id"],
            "outbox_path": expected.payload["outbox_path"],
            "operation_ids": expected.payload["operation_ids"],
        }
        if (
            existing.queue_name != expected.queue_name
            or existing.action != expected.action
            or existing.target_uri != expected.target_uri
            or (existing.payload != expected.payload and existing.payload != legacy_payload)
        ):
            queue_conflict = QueueIdempotencyConflictError(
                "projection queue identity conflicts with its committed outbox"
            )
            self._mark_authoritative_integrity_failure(
                queue_conflict,
                artifact="projection_queue",
                identifiers={"transaction_id": transaction_id},
            )
            raise ProjectionOutboxIntegrityError(str(queue_conflict)) from queue_conflict
        if existing.status in {"dead_letter", "quarantine"}:
            terminal_failure = ProjectionOutboxIntegrityError(
                f"projection queue is terminal before publication: {existing.status}"
            )
            self._mark_authoritative_integrity_failure(
                terminal_failure,
                artifact=f"projection_queue_{existing.status}",
                identifiers={
                    "transaction_id": transaction_id,
                    "job_id": existing.job_id,
                },
            )
            raise terminal_failure

    dispatched: list[str] = []
    for transaction_id, expected in pending_jobs:
        if self.queue_store.get(expected.job_id) is not None:
            dispatched.append(transaction_id)
            continue
        try:
            self.queue_store.enqueue(expected)
        except QueueIdempotencyConflictError as exc:
            self._mark_authoritative_integrity_failure(
                exc,
                artifact="projection_queue",
                identifiers={"transaction_id": transaction_id},
            )
            raise ProjectionOutboxIntegrityError(
                "projection queue identity conflicts with its committed outbox"
            ) from exc
        dispatched.append(transaction_id)
    return dispatched
