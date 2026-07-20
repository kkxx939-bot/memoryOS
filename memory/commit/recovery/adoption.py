"""恢复已持久化收据授权、但尚未完整发布的文档接管。"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

from infrastructure.store.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from infrastructure.store.memory import (
    DocumentAdoptionReceipt,
    DocumentDeletionStatus,
    ExternalChangeKind,
    MemoryDocumentBootstrapper,
    MemoryDocumentControlStore,
)
from infrastructure.store.memory.scanner import ExternalDocumentChange
from memory.commit import MemoryDocumentCommitter
from memory.core import DocumentEditKind, ManagedDocument, PresentPath, matches_adopted_source
from memory.execute.external_change import publish_external_change as publish_external_memory_change


class AdoptionRecoveryServices(Protocol):
    """接管恢复实际需要的最小 Memory 服务集合。"""

    @property
    def document_store(self) -> FileSystemMemoryDocumentStore: ...

    @property
    def control_store(self) -> MemoryDocumentControlStore: ...

    @property
    def bootstrapper(self) -> MemoryDocumentBootstrapper: ...

    @property
    def committer(self) -> MemoryDocumentCommitter: ...


def recover_adoption_receipts(
    memory: AdoptionRecoveryServices,
    *,
    tenant_id: str,
    owners: tuple[str, ...],
) -> dict[str, Any]:
    """在普通扫描之前恢复收据授权的 Source CAS 和 CREATE 发布。"""

    totals = {
        "receipts": 0,
        "already_committed": 0,
        "bootstrap_resumed": 0,
        "erasure_blocked": 0,
        "resumed_unmanaged": 0,
        "resumed_managed": 0,
        "published": 0,
    }
    per_owner: dict[str, dict[str, int]] = {}
    for owner in owners:
        owner_counts = {key: 0 for key in totals}
        receipts = memory.control_store.adoption_receipts(tenant_id, owner)
        owner_counts["receipts"] = len(receipts)
        active: list[DocumentAdoptionReceipt] = []
        active_paths: set[str] = set()
        active_document_ids: set[str] = set()
        for receipt in receipts:
            if receipt.tenant_id != tenant_id or receipt.owner_user_id != owner:
                raise RuntimeError("adoption receipt enumeration crossed its exact scope")
            indexed_receipt = memory.control_store.load_adoption_receipt_for_document(
                tenant_id,
                owner,
                receipt.document_id,
            )
            if indexed_receipt is not None and indexed_receipt != receipt:
                raise RuntimeError("adoption receipt identity index changed its exact authority")
            erase_record = memory.committer.erasure_store.load(
                tenant_id,
                owner,
                receipt.document_id,
            )
            barrier = memory.control_store.load_publication_barrier(
                tenant_id,
                owner,
                receipt.document_id,
            )
            if erase_record is not None:
                if (
                    indexed_receipt != receipt
                    or barrier is None
                    or barrier.status is not DocumentDeletionStatus.HARD_ERASED
                ):
                    raise RuntimeError("adoption erasure is detached from its durable identity barrier")
                owner_counts["erasure_blocked"] += 1
                continue
            if barrier is not None and barrier.status is DocumentDeletionStatus.HARD_ERASED:
                raise RuntimeError("hard-erased adoption identity is missing its durable erasure epoch")
            control = memory.control_store.load_control(
                tenant_id,
                owner,
                receipt.document_id,
            )
            if control is not None:
                if indexed_receipt != receipt:
                    raise RuntimeError("committed adoption is missing its exact receipt identity index")
                owner_counts["already_committed"] += 1
                bootstrap_status = memory.bootstrapper.status(tenant_id, owner)
                if bootstrap_status == "COMPLETED":
                    continue
                binding = memory.control_store.load_event_binding(
                    tenant_id,
                    owner,
                    receipt.document_id,
                    control.last_event_id,
                )
                if binding is None:
                    raise RuntimeError("committed adoption is missing its durable CREATE event")
                intent, event = binding
                if (
                    control.status != "present"
                    or control.relative_path != receipt.relative_path
                    or event.edit_kind is not DocumentEditKind.CREATE
                    or event.old_relative_path
                    or event.new_relative_path != receipt.relative_path
                    or event.after_raw_digest != control.raw_sha256
                    or event.logical_revision != control.logical_revision
                    or event.projection_generation != control.projection_generation
                    or event.actor_binding != receipt.actor_binding
                    or event.evidence_reference != receipt.evidence_reference
                    or event.evidence_digest != receipt.evidence_digest
                    or event.edit_summary != receipt.edit_summary
                    or intent.idempotency_digest != hashlib.sha256(receipt.idempotency_key.encode()).hexdigest()
                ):
                    raise RuntimeError("committed adoption is detached from its exact receipt lineage")
                live = memory.document_store.read_state(
                    tenant_id,
                    owner,
                    receipt.relative_path,
                )
                if not isinstance(live, PresentPath) or live.raw_sha256 != control.raw_sha256:
                    raise RuntimeError("unbootstrapped adoption no longer matches its durable control")
                raw = memory.document_store.read_raw(
                    tenant_id,
                    owner,
                    relative_path=receipt.relative_path,
                )
                if hashlib.sha256(raw).hexdigest() != control.raw_sha256 or not matches_adopted_source(
                    raw,
                    receipt.document_id,
                    receipt.expected_raw_sha256,
                ):
                    raise RuntimeError("unbootstrapped adoption is not the exact receipt rewrite")
                scan = memory.document_store.full_scan(tenant_id, owner)
                exact = [
                    registration
                    for registration in scan.registrations
                    if isinstance(registration, ManagedDocument)
                    and registration.document_id == receipt.document_id
                    and registration.relative_path == receipt.relative_path
                    and registration.raw_sha256 == control.raw_sha256
                ]
                if not scan.complete or scan.errors or len(exact) != 1:
                    raise RuntimeError("unbootstrapped adoption is unsafe or duplicated")
                memory.bootstrapper.ensure_adopted_user(
                    tenant_id,
                    owner,
                    receipt.relative_path,
                    document_id=receipt.document_id,
                    adopted_raw_sha256=control.raw_sha256,
                )
                if memory.bootstrapper.status(tenant_id, owner) != "COMPLETED":
                    raise RuntimeError("committed adoption bootstrap did not reach COMPLETED")
                owner_counts["bootstrap_resumed"] += 1
                continue
            if receipt.relative_path in active_paths:
                raise RuntimeError("active adoption receipts duplicate one relative path")
            if receipt.document_id in active_document_ids:
                raise RuntimeError("active adoption receipts duplicate one document identity")
            active_paths.add(receipt.relative_path)
            active_document_ids.add(receipt.document_id)

            memory.committer.verify_adoption_root(
                tenant_id,
                owner,
                receipt.document_id,
            )
            durable = memory.control_store.prepare_adoption_receipt(
                tenant_id,
                owner,
                receipt.relative_path,
                receipt.expected_raw_sha256,
                actor_binding=receipt.actor_binding,
            )
            if durable != receipt:
                raise RuntimeError("adoption receipt replay changed its durable authority")
            active.append(receipt)

        for receipt in active:
            state = memory.document_store.read_state(
                tenant_id,
                owner,
                receipt.relative_path,
            )
            if not isinstance(state, PresentPath):
                raise RuntimeError("adoption receipt target is absent or unsafe during startup")
            raw = memory.document_store.read_raw(
                tenant_id,
                owner,
                relative_path=receipt.relative_path,
            )
            raw_digest = hashlib.sha256(raw).hexdigest()
            if raw_digest != state.raw_sha256:
                raise RuntimeError("adoption receipt target changed during startup classification")
            if raw_digest == receipt.expected_raw_sha256:
                adopted = memory.document_store.adopt(
                    tenant_id,
                    owner,
                    receipt.relative_path,
                    expected_raw_sha256=receipt.expected_raw_sha256,
                    assigned_document_id=receipt.document_id,
                    operation_id=receipt.receipt_id,
                )
                if (
                    adopted.document_id != receipt.document_id
                    or adopted.relative_path != receipt.relative_path
                    or not matches_adopted_source(
                        adopted.raw_bytes,
                        receipt.document_id,
                        receipt.expected_raw_sha256,
                    )
                ):
                    raise RuntimeError("resumed adoption produced bytes detached from its receipt")
                owner_counts["resumed_unmanaged"] += 1
            elif matches_adopted_source(raw, receipt.document_id, receipt.expected_raw_sha256):
                owner_counts["resumed_managed"] += 1
            else:
                raise RuntimeError("adoption receipt target is a third source state")

        if active:
            scan = memory.document_store.full_scan(tenant_id, owner)
            if not scan.complete or scan.errors:
                raise RuntimeError("resumed adoption requires one complete registration scan")
            for receipt in active:
                registrations = [
                    registration
                    for registration in scan.registrations
                    if isinstance(registration, ManagedDocument)
                    and registration.document_id == receipt.document_id
                    and registration.relative_path == receipt.relative_path
                ]
                if len(registrations) != 1:
                    raise RuntimeError("resumed adoption is unsafe, duplicated, or unregistered")
                managed = registrations[0]
                raw = memory.document_store.read_raw(
                    tenant_id,
                    owner,
                    document_id=receipt.document_id,
                )
                if hashlib.sha256(raw).hexdigest() != managed.raw_sha256 or not matches_adopted_source(
                    raw,
                    receipt.document_id,
                    receipt.expected_raw_sha256,
                ):
                    raise RuntimeError("resumed adoption no longer matches its exact source receipt")
                publish_external_memory_change(
                    ExternalDocumentChange(
                        change_kind=ExternalChangeKind.CREATE,
                        tenant_id=tenant_id,
                        owner_user_id=owner,
                        document_id=receipt.document_id,
                        old_relative_path="",
                        new_relative_path=receipt.relative_path,
                        before_raw_digest="",
                        after_raw_digest=managed.raw_sha256,
                        scan_generation_id=scan.generation_id,
                    ),
                    committer=memory.committer,
                    control_store=memory.control_store,
                    document_store=memory.document_store,
                    bootstrapper=memory.bootstrapper,
                )
                control = memory.control_store.load_control(
                    tenant_id,
                    owner,
                    receipt.document_id,
                )
                if (
                    control is None
                    or control.status != "present"
                    or control.relative_path != receipt.relative_path
                    or control.raw_sha256 != managed.raw_sha256
                    or memory.control_store.load_event_binding(
                        tenant_id,
                        owner,
                        receipt.document_id,
                        control.last_event_id,
                    )
                    is None
                ):
                    raise RuntimeError("resumed adoption did not publish its exact control and event")
                owner_counts["published"] += 1

        per_owner[owner] = owner_counts
        for key in totals:
            totals[key] += owner_counts[key]
    return {**totals, "owners": per_owner}


__all__ = ["recover_adoption_receipts"]
