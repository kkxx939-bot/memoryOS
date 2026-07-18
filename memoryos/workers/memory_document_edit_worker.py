"""Content-free queue worker for durable Markdown document edits."""

from __future__ import annotations

import os
import uuid
from typing import Any

from memoryos.contextdb.store.queue_store import QueueJob, QueueStore
from memoryos.memory.documents import (
    DocumentCommitConflict,
    DocumentCommitResult,
    DocumentConflictError,
    DocumentControlIntegrityError,
    DocumentErasedError,
    DocumentIntentStatus,
    DocumentNotFoundError,
    DocumentUnsafeError,
    MemoryDocumentCommitter,
    MemoryDocumentPathPolicy,
    validate_document_id,
)
from memoryos.workers.tenant_boundary import require_bound_job_tenant

_INTENT_KEYS = frozenset({"tenant_id", "owner_user_id", "document_id", "intent_id"})


class MemoryDocumentEditWorker:
    """Recover one already-authorized durable document intent.

    Queue metadata contains only the durable intent identity. Markdown, review
    decisions, diffs and edit plans remain outside the generic queue. Review
    approval is a synchronous trusted-user CAS and has no queue variant.
    """

    def __init__(
        self,
        committer: MemoryDocumentCommitter,
        queue_store: QueueStore,
        *,
        tenant_id: str,
        readiness: Any | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.committer = committer
        self.queue_store = queue_store
        self.tenant_id = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        self.readiness = readiness
        self.worker_id = worker_id or f"memory-document-edit:{os.getpid()}:{uuid.uuid4().hex}"

    def process_pending(
        self,
        *,
        batch_size: int = 10,
        lease_seconds: int = 60,
        max_retries: int = 3,
    ) -> dict[str, object]:
        self._require_ready()
        self.queue_store.recover_expired_leases(queue_name="memory_document_edit")
        jobs = self.queue_store.lease(
            "memory_document_edit",
            lease_owner=self.worker_id,
            limit=batch_size,
            lease_seconds=lease_seconds,
        )
        committed = failed = dead_letter = 0
        released: list[str] = []
        for position, job in enumerate(jobs):
            try:
                self._require_ready()
            except RuntimeError:
                if self._is_ready():
                    raise
                released.extend(self._release_unattempted(jobs[position:]))
                break
            try:
                result = self._process_job(job)
                if result.status is not DocumentIntentStatus.COMPLETED:
                    raise RuntimeError("memory document edit did not complete its durable intent")
                if not self._is_ready():
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                self.queue_store.ack(job)
                committed += 1
                continue
            except (
                DocumentCommitConflict,
                DocumentErasedError,
                DocumentControlIntegrityError,
                DocumentNotFoundError,
                DocumentUnsafeError,
                PermissionError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                failure_name = type(exc).__name__
                retryable = False
            except DocumentConflictError as exc:
                failure_name = type(exc).__name__
                retryable = True
            except OSError as exc:
                failure_name = type(exc).__name__
                retryable = True
            except Exception as exc:
                failure_name = type(exc).__name__
                explicit = getattr(exc, "retryable", None)
                retryable = explicit if isinstance(explicit, bool) else False
            if not self._is_ready():
                failed += 1
                released.extend(self._release_unattempted(jobs[position:]))
                break
            settled = self.queue_store.retry(
                job,
                failure_name,
                max_retries=max_retries,
                retryable=retryable,
            )
            failed += 1
            dead_letter += int(settled.status == "dead_letter")

        summary: dict[str, object] = {
            "claimed": len(jobs),
            "committed": committed,
            "failed": failed,
            "dead_letter": dead_letter,
        }
        if released:
            summary["released"] = released
        if not self._is_ready():
            summary["status"] = "not_ready"
        return summary

    def _process_job(self, job: QueueJob) -> DocumentCommitResult:
        keys = frozenset(job.payload)
        if job.queue_name != "memory_document_edit":
            raise ValueError("memory document edit queue identity is invalid")
        tenant_id = require_bound_job_tenant(
            job.payload,
            bound_tenant_id=self.tenant_id,
        )
        owner_user_id = MemoryDocumentPathPolicy.trusted_segment(
            self._required_string(job.payload, "owner_user_id"),
            "owner_user_id",
        )

        if keys == _INTENT_KEYS:
            if job.action != "recover_document_intent":
                raise ValueError("memory document intent action is unsupported")
            document_id = validate_document_id(self._required_string(job.payload, "document_id"))
            intent_id = self._intent_id(self._required_string(job.payload, "intent_id"))
            expected_target = MemoryDocumentPathPolicy.document_uri(owner_user_id, document_id)
            if job.target_uri != expected_target:
                raise ValueError("memory document intent target differs from its stable document URI")
            intent = self.committer.control_store.load_intent(tenant_id, owner_user_id, intent_id)
            if intent is None:
                raise DocumentNotFoundError("memory document intent does not exist")
            if intent.document_id != document_id:
                raise DocumentControlIntegrityError("queued document identity differs from its durable intent")
            result = self.committer.recover_intent(tenant_id, owner_user_id, intent_id)
            if result.intent_id != intent_id:
                raise DocumentControlIntegrityError("recovered document result changed intent identity")
            self._validate_result_owner(result, tenant_id, owner_user_id, document_id)
            return result

        raise ValueError("memory document edit payload has unsupported or content-bearing fields")

    @staticmethod
    def _validate_result_owner(
        result: DocumentCommitResult,
        tenant_id: str,
        owner_user_id: str,
        document_id: str = "",
    ) -> None:
        event = result.event
        control = result.control
        actual_tenant = event.tenant_id if event is not None else control.tenant_id if control is not None else ""
        actual_owner = (
            event.owner_user_id if event is not None else control.owner_user_id if control is not None else ""
        )
        actual_document = event.document_id if event is not None else control.document_id if control is not None else ""
        if (actual_tenant, actual_owner) != (tenant_id, owner_user_id):
            raise DocumentControlIntegrityError("document edit result crosses its queued owner boundary")
        if document_id and actual_document != document_id:
            raise DocumentControlIntegrityError("document edit result changed its queued document identity")

    def _release_unattempted(self, jobs: list[QueueJob]) -> list[str]:
        released: list[str] = []
        for job in jobs:
            settled = self.queue_store.release(job)
            if (
                settled.status != "pending"
                or settled.retry_count != job.retry_count
                or settled.lease_token
                or settled.lease_owner
            ):
                raise RuntimeError("memory document edit release did not preserve queue state")
            released.append(job.job_id)
        return released

    def _require_ready(self) -> None:
        require_ready = getattr(self.readiness, "require_ready", None)
        if callable(require_ready):
            require_ready()

    def _is_ready(self) -> bool:
        if self.readiness is None:
            return True
        snapshot = getattr(self.readiness, "snapshot", None)
        if callable(snapshot):
            payload = snapshot()
            return isinstance(payload, dict) and bool(payload.get("ready"))
        state = getattr(self.readiness, "state", None)
        return str(getattr(state, "value", state or "")) == "READY"

    @staticmethod
    def _required_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise TypeError(f"memory document edit {key} must be a non-empty string")
        return value

    @staticmethod
    def _intent_id(value: str) -> str:
        suffix = value.removeprefix("mdintent_")
        if value != f"mdintent_{suffix}" or not MemoryDocumentEditWorker._is_hex(suffix, 64):
            raise ValueError("memory document intent ID is invalid")
        return value

    @staticmethod
    def _is_hex(value: str, length: int) -> bool:
        return len(value) == length and all(character in "0123456789abcdef" for character in value)


__all__ = ["MemoryDocumentEditWorker"]
