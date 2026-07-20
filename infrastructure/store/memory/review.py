"""记忆编辑审核记录和受控内容 Blob 的文件仓储。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from foundation.clock import utc_now
from foundation.integrity import canonical_json
from infrastructure.store.filesystem.durable_io import (
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_create_json,
    atomic_write_json,
)
from infrastructure.store.filesystem.durable_io.atomic_file import _open_control_parent
from infrastructure.store.filesystem.file_lock import open_private_lock
from infrastructure.store.memory.erasure_store import MemoryDocumentEraseStore
from infrastructure.store.memory.layout import tenant_control_root
from infrastructure.store.memory.review_model import (
    _MAX_INDEPENDENT_EVIDENCE_REFERENCES,
    MemoryEditReviewIntegrityError,
    MemoryEditReviewRecord,
    MemoryEditReviewStatus,
    MemoryEditReviewWorkflow,
    ReviewConsolidationSource,
    _independent_evidence_reference,
    _is_sha256,
    _mapping,
    _validate_proposal_id,
)
from memory.core.model import DocumentEditKind, DocumentEditPlan, raw_state_to_dict
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy

_MAX_REVIEW_METADATA_BYTES = 1024 * 1024
_MAX_OWNER_REVIEW_RECORDS = 10_000


class MemoryEditReviewStore:
    """耐久封存审核元数据，并让所有含正文 Blob 都可枚举。"""

    def __init__(
        self,
        root: str | Path,
        *,
        max_blob_bytes: int = 2 * 1024 * 1024,
        max_owner_records: int = _MAX_OWNER_REVIEW_RECORDS,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        if max_blob_bytes <= 0:
            raise ValueError("review blob size limit must be positive")
        if max_owner_records <= 0:
            raise ValueError("review owner record limit must be positive")
        self.max_blob_bytes = max_blob_bytes
        self.max_owner_records = max_owner_records
        self.clock = clock
        self.erasure_store = MemoryDocumentEraseStore(self.root)

    def seal(
        self,
        plan: DocumentEditPlan,
        *,
        proposed_diff: str | bytes,
        independent_evidence_references: tuple[str, ...] = (),
        workflow_kind: MemoryEditReviewWorkflow = MemoryEditReviewWorkflow.DOCUMENT_EDIT,
        consolidation_sources: tuple[ReviewConsolidationSource, ...] = (),
    ) -> MemoryEditReviewRecord:
        diff_bytes = proposed_diff.encode() if isinstance(proposed_diff, str) else bytes(proposed_diff)
        if not diff_bytes or len(diff_bytes) > self.max_blob_bytes:
            raise ValueError("proposed diff must be non-empty and within the review blob limit")
        after = bytes(plan.after_bytes) if plan.after_bytes is not None else b""
        if len(after) > self.max_blob_bytes:
            raise ValueError("proposed after bytes exceed the review blob limit")
        tenant = MemoryDocumentPathPolicy.trusted_segment(plan.tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(plan.owner_user_id, "owner_user_id")
        identifier = validate_document_id(plan.document_id)
        relative = MemoryDocumentPathPolicy.normalize_relative_path(plan.relative_path)
        request_digest = hashlib.sha256(str(plan.idempotency_key).encode()).hexdigest()
        after_digest = hashlib.sha256(after).hexdigest() if plan.after_bytes is not None else ""
        diff_digest = hashlib.sha256(diff_bytes).hexdigest()
        references = tuple(
            sorted(
                {_independent_evidence_reference(item, owner_user_id=owner) for item in independent_evidence_references}
            )
        )
        if len(references) > _MAX_INDEPENDENT_EVIDENCE_REFERENCES:
            raise ValueError("review independent evidence reference count exceeds its bound")
        workflow = MemoryEditReviewWorkflow(workflow_kind)
        sources = tuple(consolidation_sources)
        now = self.clock()
        immutable_payload = {
            "tenant_id": tenant,
            "owner_user_id": owner,
            "document_id": identifier,
            "edit_kind": DocumentEditKind(plan.edit_kind).value,
            "expected_state": raw_state_to_dict(plan.expected_state),
            "expected_new_state": raw_state_to_dict(plan.expected_new_state),
            "relative_path": relative,
            "new_relative_path": plan.new_relative_path,
            "expected_registration_document_id": plan.expected_registration_document_id,
            "request_id_digest": request_digest,
            "evidence_digest": str(plan.evidence_digest),
            "edit_summary": str(plan.edit_summary),
            "after_blob_digest": after_digest,
            "proposed_diff_blob_digest": diff_digest,
            "independent_evidence_references": list(references),
            "workflow_kind": workflow.value,
            "consolidation_sources": [source.to_dict() for source in sources],
        }
        seal = hashlib.sha256(canonical_json(immutable_payload).encode()).hexdigest()
        record = MemoryEditReviewRecord(
            proposal_id=f"mdreview_{seal}",
            tenant_id=tenant,
            owner_user_id=owner,
            document_id=identifier,
            edit_kind=DocumentEditKind(plan.edit_kind),
            expected_state=plan.expected_state,
            expected_new_state=plan.expected_new_state,
            relative_path=relative,
            new_relative_path=plan.new_relative_path,
            expected_registration_document_id=plan.expected_registration_document_id,
            request_id_digest=request_digest,
            evidence_digest=str(plan.evidence_digest),
            edit_summary=str(plan.edit_summary),
            after_blob_digest=after_digest,
            proposed_diff_blob_digest=diff_digest,
            independent_evidence_references=references,
            workflow_kind=workflow,
            consolidation_sources=sources,
            sealed_digest=seal,
            status=MemoryEditReviewStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        with self.erasure_store.document_lock(tenant, owner, identifier):
            with self._owner_review_lock(tenant, owner):
                self.erasure_store.assert_mutation_allowed(tenant, owner, identifier)
                for source in sources:
                    self.erasure_store.assert_mutation_allowed(
                        tenant,
                        owner,
                        source.document_id,
                    )
                path = self._record_path(tenant, owner, record.proposal_id)
                try:
                    atomic_create_json(path, record.to_dict(), artifact_root=self._artifact_root(tenant))
                except ImmutableArtifactConflictError:
                    pass
                # 在任何正文 Blob 之前先发布无正文的来源和摘要绑定。这样崩溃
                # 最多留下可修复的悬空记录，不会留下无法枚举的含正文 Blob。
                if after_digest:
                    self._stage_blob(tenant, owner, identifier, after_digest, after)
                self._stage_blob(tenant, owner, identifier, diff_digest, diff_bytes)
                durable = self.load(tenant, owner, record.proposal_id)
                if durable is None or durable.sealed_digest != record.sealed_digest:
                    raise MemoryEditReviewIntegrityError("sealed review disappeared or conflicts after publication")
                return durable

    def load(
        self,
        tenant_id: str,
        owner_user_id: str,
        proposal_id: str,
    ) -> MemoryEditReviewRecord | None:
        payload = self._read_json(self._record_path(tenant_id, owner_user_id, proposal_id), tenant_id)
        if payload is None:
            return None
        record = MemoryEditReviewRecord.from_dict(payload)
        if (record.tenant_id, record.owner_user_id, record.proposal_id) != (
            tenant_id,
            owner_user_id,
            proposal_id,
        ):
            raise MemoryEditReviewIntegrityError("review path identity does not match its payload")
        return record

    def load_after_blob(self, record: MemoryEditReviewRecord) -> bytes | None:
        if not record.after_blob_digest:
            return None
        return self._read_blob(record, record.after_blob_digest)

    def load_proposed_diff(self, record: MemoryEditReviewRecord) -> bytes:
        return self._read_blob(record, record.proposed_diff_blob_digest)

    def to_plan(self, record: MemoryEditReviewRecord) -> DocumentEditPlan:
        return DocumentEditPlan(
            idempotency_key=f"review:{record.proposal_id}",
            tenant_id=record.tenant_id,
            owner_user_id=record.owner_user_id,
            edit_kind=record.edit_kind,
            expected_state=record.expected_state,
            evidence_digest=record.evidence_digest,
            edit_summary=record.edit_summary,
            document_id=record.document_id,
            relative_path=record.relative_path,
            after_bytes=self.load_after_blob(record),
            new_relative_path=record.new_relative_path,
            expected_new_state=record.expected_new_state,
            expected_registration_document_id=record.expected_registration_document_id,
        )

    def transition(
        self,
        record: MemoryEditReviewRecord,
        status: MemoryEditReviewStatus,
        *,
        commit_intent_id: str = "",
        replacement_proposal_id: str = "",
        consolidation_saga_id: str = "",
    ) -> MemoryEditReviewRecord:
        with self.erasure_store.document_lock(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        ):
            with self._owner_review_lock(record.tenant_id, record.owner_user_id):
                self.erasure_store.assert_mutation_allowed(
                    record.tenant_id,
                    record.owner_user_id,
                    record.document_id,
                )
                current = self.load(record.tenant_id, record.owner_user_id, record.proposal_id)
                if current is None or current.sealed_digest != record.sealed_digest:
                    raise MemoryEditReviewIntegrityError("review transition is detached from its sealed proposal")
                if current.status != MemoryEditReviewStatus.PENDING:
                    if (
                        current.status == status
                        and current.commit_intent_id == commit_intent_id
                        and current.replacement_proposal_id == replacement_proposal_id
                        and current.consolidation_saga_id == consolidation_saga_id
                    ):
                        return current
                    raise ValueError("memory edit review already has a terminal decision")
                updated = replace(
                    current,
                    status=status,
                    updated_at=self.clock(),
                    commit_intent_id=commit_intent_id,
                    replacement_proposal_id=replacement_proposal_id,
                    consolidation_saga_id=consolidation_saga_id,
                )
                atomic_write_json(
                    self._record_path(updated.tenant_id, updated.owner_user_id, updated.proposal_id),
                    updated.to_dict(),
                    artifact_root=self._artifact_root(updated.tenant_id),
                )
                return updated

    def purge_document(self, tenant_id: str, owner_user_id: str, document_id: str) -> int:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        with self._owner_review_lock(tenant, owner):
            records = self._owner_records(tenant, owner)
            doomed = {
                proposal_id
                for proposal_id, record in records.items()
                if record.document_id == identifier
                or identifier in {source.document_id for source in record.consolidation_sources}
            }
            blob_keys = self._document_blob_keys(tenant, owner, identifier)
            for proposal_id in doomed:
                blob_keys.update(_record_blob_keys(records[proposal_id]))

            # 正文 Blob 在目标文档内按内容寻址。如果其他提案共享将被删除的
            # 摘要，保留其记录会形成悬空引用。因此同时使它的完整正文集合失效，
            # 并持续处理直到引用闭包稳定。
            changed = True
            while changed:
                changed = False
                for proposal_id, record in records.items():
                    if proposal_id in doomed:
                        continue
                    record_keys = _record_blob_keys(record)
                    if record_keys.intersection(blob_keys):
                        doomed.add(proposal_id)
                        blob_keys.update(record_keys)
                        changed = True

            removed = 0
            for target_document_id, digest in sorted(blob_keys):
                removed += self._unlink_blob_key_if_present(
                    tenant,
                    owner,
                    target_document_id,
                    digest,
                )
            for proposal_id in sorted(doomed):
                removed += self._unlink_record(records[proposal_id])
            return removed

    def _owner_records(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> dict[str, MemoryEditReviewRecord]:
        record_directory = self._owner_root(tenant_id, owner_user_id) / "reviews"
        if not record_directory.exists():
            return {}
        descriptor = _open_control_parent(
            record_directory / ".scan",
            self._artifact_root(tenant_id),
        )
        try:
            names = _bounded_directory_names(
                descriptor,
                maximum=self.max_owner_records,
                label="review owner record",
            )
        finally:
            os.close(descriptor)
        records: dict[str, MemoryEditReviewRecord] = {}
        for name in sorted(names):
            if not name.endswith(".json"):
                raise MemoryEditReviewIntegrityError("review purge encountered an unexpected artifact")
            proposal_id = name.removesuffix(".json")
            _validate_proposal_id(proposal_id)
            record = self.load(tenant_id, owner_user_id, proposal_id)
            if record is None:
                raise MemoryEditReviewIntegrityError("review record disappeared during bounded owner enumeration")
            records[proposal_id] = record
        return records

    def _document_blob_keys(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> set[tuple[str, str]]:
        directory = self._blob_directory(tenant_id, owner_user_id, document_id)
        if not directory.exists():
            return set()
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant_id))
        try:
            names = _bounded_directory_names(
                descriptor,
                maximum=self.max_owner_records * 2,
                label="review document blob",
            )
            digests: set[str] = set()
            for name in names:
                digest = name.removesuffix(".blob")
                if not name.endswith(".blob") or not _is_sha256(digest):
                    raise MemoryEditReviewIntegrityError("review purge encountered an unexpected blob artifact")
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise MemoryEditReviewIntegrityError("review purge encountered a non-regular blob artifact")
                digests.add(digest)
        finally:
            os.close(descriptor)
        return {(document_id, digest) for digest in digests}

    def _stage_blob(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        digest: str,
        raw: bytes,
    ) -> None:
        atomic_create_bytes(
            self._blob_path(tenant_id, owner_user_id, document_id, digest),
            raw,
            artifact_root=self._artifact_root(tenant_id),
        )

    def _read_blob(self, record: MemoryEditReviewRecord, digest: str) -> bytes:
        path = self._blob_path(record.tenant_id, record.owner_user_id, record.document_id, digest)
        parent_descriptor = _open_control_parent(path, self._artifact_root(record.tenant_id))
        try:
            try:
                descriptor = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_descriptor)
            except FileNotFoundError as exc:
                raise MemoryEditReviewIntegrityError("sealed review blob is missing") from exc
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise MemoryEditReviewIntegrityError("sealed review blob is not one regular file")
                if metadata.st_size > self.max_blob_bytes:
                    raise MemoryEditReviewIntegrityError("sealed review blob exceeds its size bound")
                raw = _read_bounded(descriptor, self.max_blob_bytes)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_descriptor)
        if len(raw) > self.max_blob_bytes or hashlib.sha256(raw).hexdigest() != digest:
            raise MemoryEditReviewIntegrityError("sealed review blob digest does not match")
        return raw

    def _unlink_record(self, record: MemoryEditReviewRecord) -> int:
        path = self._record_path(record.tenant_id, record.owner_user_id, record.proposal_id)
        parent_descriptor = _open_control_parent(path, self._artifact_root(record.tenant_id))
        try:
            try:
                metadata = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return 0
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise MemoryEditReviewIntegrityError("review purge encountered a non-regular record")
            os.unlink(path.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            return 1
        finally:
            os.close(parent_descriptor)

    def _unlink_blob_key_if_present(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        digest: str,
    ) -> int:
        path = self._blob_path(
            tenant_id,
            owner_user_id,
            document_id,
            digest,
        )
        parent_descriptor = _open_control_parent(path, self._artifact_root(tenant_id))
        try:
            try:
                metadata = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return 0
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise MemoryEditReviewIntegrityError("review purge encountered a non-regular blob artifact")
            os.unlink(path.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            return 1
        finally:
            os.close(parent_descriptor)

    @contextmanager
    def _owner_review_lock(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> Iterator[None]:
        lock_path = self._owner_root(tenant_id, owner_user_id) / "locks" / "reviews.lock"
        descriptor = open_private_lock(lock_path, root=self._artifact_root(tenant_id))
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _artifact_root(self, tenant_id: str) -> Path:
        return tenant_control_root(self.root, tenant_id)

    def _owner_root(self, tenant_id: str, owner_user_id: str) -> Path:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        return self._artifact_root(tenant) / "system" / "memory-documents" / owner

    def _record_path(self, tenant_id: str, owner_user_id: str, proposal_id: str) -> Path:
        _validate_proposal_id(proposal_id)
        return self._owner_root(tenant_id, owner_user_id) / "reviews" / f"{proposal_id}.json"

    def _blob_directory(self, tenant_id: str, owner_user_id: str, document_id: str) -> Path:
        return self._owner_root(tenant_id, owner_user_id) / "review-blobs" / validate_document_id(document_id)

    def _blob_path(self, tenant_id: str, owner_user_id: str, document_id: str, digest: str) -> Path:
        if not _is_sha256(digest):
            raise ValueError("review blob key must be a SHA-256 digest")
        return self._blob_directory(tenant_id, owner_user_id, document_id) / f"{digest}.blob"

    def _read_json(self, path: Path, tenant_id: str) -> dict[str, Any] | None:
        parent_descriptor = _open_control_parent(path, self._artifact_root(tenant_id))
        try:
            try:
                descriptor = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_descriptor)
            except FileNotFoundError:
                return None
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise MemoryEditReviewIntegrityError("review metadata is not one regular file")
                if metadata.st_size > _MAX_REVIEW_METADATA_BYTES:
                    raise MemoryEditReviewIntegrityError("review metadata exceeds its size bound")
                raw = _read_bounded(descriptor, _MAX_REVIEW_METADATA_BYTES)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_descriptor)
        if len(raw) > _MAX_REVIEW_METADATA_BYTES:
            raise MemoryEditReviewIntegrityError("review metadata exceeds its size bound")
        try:
            payload = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MemoryEditReviewIntegrityError("review metadata is invalid JSON") from exc
        return _mapping(payload)


def _record_blob_keys(record: MemoryEditReviewRecord) -> set[tuple[str, str]]:
    return {
        (record.document_id, digest)
        for digest in (record.after_blob_digest, record.proposed_diff_blob_digest)
        if digest
    }


def _bounded_directory_names(
    descriptor: int,
    *,
    maximum: int,
    label: str,
) -> tuple[str, ...]:
    names: list[str] = []
    iterator = os.scandir(descriptor)
    try:
        for entry in iterator:
            if len(names) >= maximum:
                raise MemoryEditReviewIntegrityError(f"{label} enumeration exceeds its hard limit")
            names.append(entry.name)
    finally:
        iterator.close()
    return tuple(names)


def _read_bounded(descriptor: int, maximum: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__ = [
    "MemoryEditReviewIntegrityError",
    "MemoryEditReviewRecord",
    "MemoryEditReviewStatus",
    "MemoryEditReviewStore",
    "MemoryEditReviewWorkflow",
    "ReviewConsolidationSource",
]
