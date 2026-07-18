"""Content-free worker for retrieval-triggered Markdown rescans."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from memoryos.contextdb.store.queue_store import QueueJob, QueueStore
from memoryos.memory.documents import (
    DocumentConflictError,
    DocumentNotFoundError,
    DocumentUnsafeError,
    MemoryDocumentPathPolicy,
    MemoryDocumentScanner,
    validate_document_id,
)
from memoryos.workers.tenant_boundary import require_bound_job_tenant

_SCAN_KEYS = frozenset(
    {"tenant_id", "owner_user_id", "document_id", "observed_source_digest"}
)


class MemoryDocumentScanWorker:
    """Consume bounded rescan hints and publish only scanner-confirmed facts."""

    queue_name = "memory_document_scan"

    def __init__(
        self,
        scanner: MemoryDocumentScanner,
        queue_store: QueueStore,
        *,
        tenant_id: str,
        owner_user_ids: Callable[[str, int], Sequence[str]] | None = None,
        max_owners_per_run: int = 10,
        owner_enumeration_limit: int = 1_000,
        readiness: Any | None = None,
        worker_id: str | None = None,
    ) -> None:
        if not 1 <= max_owners_per_run <= 1_000:
            raise ValueError("max_owners_per_run must be between 1 and 1000")
        if not max_owners_per_run <= owner_enumeration_limit <= 10_000:
            raise ValueError(
                "owner_enumeration_limit must cover each run and be at most 10000"
            )
        self.scanner = scanner
        self.queue_store = queue_store
        self.tenant_id = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        self.owner_user_ids = owner_user_ids
        self.max_owners_per_run = max_owners_per_run
        self.owner_enumeration_limit = owner_enumeration_limit
        self.readiness = readiness
        self.worker_id = worker_id or f"memory-document-scan:{os.getpid()}:{uuid.uuid4().hex}"
        self._owner_cursor = 0

    def process_pending(
        self,
        *,
        batch_size: int = 10,
        lease_seconds: int = 60,
        max_retries: int = 3,
    ) -> dict[str, object]:
        self._require_ready()
        self.queue_store.recover_expired_leases(queue_name=self.queue_name)
        jobs = self.queue_store.lease(
            self.queue_name,
            lease_owner=self.worker_id,
            limit=max(1, int(batch_size)),
            lease_seconds=max(1, int(lease_seconds)),
        )
        processed = failed = dead_letter = pending = 0
        released: list[str] = []
        scan_results: dict[str, Any] = {}
        for position, job in enumerate(jobs):
            try:
                self._require_ready()
            except RuntimeError:
                if self._is_ready():
                    raise
                released.extend(self._release_unattempted(jobs[position:]))
                break
            try:
                owner_user_id = self._validate_job(job)
                result = scan_results.get(owner_user_id)
                if result is None:
                    if len(scan_results) >= self.max_owners_per_run:
                        released.append(
                            self._release_job(
                                job,
                                "memory document scan owner budget is exhausted",
                            )
                        )
                        continue
                    result = self._scan_owner(owner_user_id, notified=True)
                    scan_results[owner_user_id] = result
                if result.deletions_paused or result.pending_change_count:
                    reason = result.pause_reason or (
                        "memory document scan is awaiting a stable second observation"
                    )
                    released.append(self._release_job(job, reason))
                    pending += 1
                    continue
                if not self._is_ready():
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                self.queue_store.ack(job)
                processed += 1
                continue
            except (
                DocumentNotFoundError,
                DocumentUnsafeError,
                PermissionError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                failure_name = type(exc).__name__
                retryable = False
            except (DocumentConflictError, OSError) as exc:
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
                max_retries=max(1, int(max_retries)),
                retryable=retryable,
            )
            failed += 1
            dead_letter += int(settled.status == "dead_letter")

        periodic_scanned = periodic_confirmed = periodic_pending = periodic_paused = 0
        periodic_failed = 0
        periodic_error = ""
        if self._is_ready():
            remaining = self.max_owners_per_run - len(scan_results)
            if remaining > 0:
                for owner_user_id in self._periodic_owners(
                    exclude=frozenset(scan_results),
                    limit=remaining,
                ):
                    try:
                        self._require_ready()
                        result = self._scan_owner(owner_user_id, notified=False)
                    except Exception as exc:
                        periodic_failed += 1
                        failed += 1
                        periodic_error = type(exc).__name__
                        continue
                    scan_results[owner_user_id] = result
                    periodic_scanned += 1
                    periodic_confirmed += len(result.confirmed_changes)
                    periodic_pending += int(result.pending_change_count)
                    periodic_paused += int(result.deletions_paused)
                    if not self._is_ready():
                        break

        summary: dict[str, object] = {
            "claimed": len(jobs),
            "processed": processed,
            "failed": failed,
            "dead_letter": dead_letter,
            "pending": pending,
            "scanned_owners": len(scan_results),
            "periodic_scanned": periodic_scanned,
            "periodic_confirmed": periodic_confirmed,
            "periodic_pending": periodic_pending,
            "periodic_paused": periodic_paused,
            "periodic_failed": periodic_failed,
        }
        if periodic_failed:
            summary["last_error"] = periodic_error
        if released:
            summary["released"] = released
        if not self._is_ready():
            summary["status"] = "not_ready"
        return summary

    def _validate_job(self, job: QueueJob) -> str:
        if job.queue_name != self.queue_name or job.action != "rescan":
            raise ValueError("memory document scan queue identity is invalid")
        if frozenset(job.payload) != _SCAN_KEYS:
            raise ValueError("memory document scan payload has unsupported or content-bearing fields")
        require_bound_job_tenant(
            job.payload,
            bound_tenant_id=self.tenant_id,
        )
        owner_user_id = MemoryDocumentPathPolicy.trusted_segment(
            self._required_string(job.payload, "owner_user_id"),
            "owner_user_id",
        )
        document_id = validate_document_id(
            self._required_string(job.payload, "document_id")
        )
        self._digest(
            self._required_string(job.payload, "observed_source_digest")
        )
        if job.target_uri != MemoryDocumentPathPolicy.document_uri(owner_user_id, document_id):
            raise ValueError("memory document scan target differs from its stable document URI")
        return owner_user_id

    def _scan_owner(self, owner_user_id: str, *, notified: bool) -> Any:
        if notified:
            self.scanner.notify(self.tenant_id, owner_user_id)
        return self.scanner.scan(
            self.tenant_id,
            owner_user_id,
            force_stable=True,
        )

    def _periodic_owners(
        self,
        *,
        exclude: frozenset[str],
        limit: int,
    ) -> tuple[str, ...]:
        if self.owner_user_ids is None or limit <= 0:
            return ()
        raw = self.owner_user_ids(self.tenant_id, self.owner_enumeration_limit)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise TypeError("memory document owner provider returned an invalid result")
        if len(raw) > self.owner_enumeration_limit:
            raise RuntimeError("memory document owner enumeration exceeded its bound")
        owners = tuple(
            sorted(
                MemoryDocumentPathPolicy.trusted_segment(owner, "owner_user_id")
                for owner in raw
            )
        )
        if len(set(owners)) != len(owners):
            raise RuntimeError("memory document owner enumeration contains duplicates")
        if not owners:
            self._owner_cursor = 0
            return ()
        start = self._owner_cursor % len(owners)
        selected: list[str] = []
        examined = 0
        for offset in range(len(owners)):
            owner = owners[(start + offset) % len(owners)]
            examined += 1
            if owner in exclude:
                continue
            selected.append(owner)
            if len(selected) >= limit:
                break
        self._owner_cursor = (start + examined) % len(owners)
        return tuple(selected)

    def _release_job(self, job: QueueJob, reason: str) -> str:
        settled = self.queue_store.release(job, reason)
        if (
            settled.status != "pending"
            or settled.retry_count != job.retry_count
            or settled.lease_token
            or settled.lease_owner
        ):
            raise RuntimeError("memory document scan release did not preserve queue state")
        return job.job_id

    def _release_unattempted(self, jobs: list[QueueJob]) -> list[str]:
        released: list[str] = []
        for job in jobs:
            released.append(self._release_job(job, ""))
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
            raise TypeError(f"memory document scan {key} must be a non-empty string")
        return value

    @staticmethod
    def _digest(value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("observed source digest must be a lowercase SHA-256 digest")
        return value


__all__ = ["MemoryDocumentScanWorker"]
