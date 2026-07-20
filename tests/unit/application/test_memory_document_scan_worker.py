from __future__ import annotations

from dataclasses import dataclass

from infrastructure.store.contracts.queue import QueueJob
from memory.core import MemoryDocumentPathPolicy, new_document_id
from memory.worker.document_scan import MemoryDocumentScanWorker
from tests.support.persistence.in_memory import InMemoryQueueStore


@dataclass(frozen=True)
class _ScanResult:
    confirmed_changes: tuple[object, ...] = ()
    pending_change_count: int = 0
    deletions_paused: bool = False
    pause_reason: str = ""


class _RecordingScanner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.notifications: list[tuple[str, str]] = []

    def notify(self, tenant_id: str, owner_user_id: str) -> None:
        self.notifications.append((tenant_id, owner_user_id))

    def scan(
        self,
        tenant_id: str,
        owner_user_id: str,
        *,
        force_stable: bool,
    ) -> _ScanResult:
        assert force_stable is True
        self.calls.append((tenant_id, owner_user_id))
        return _ScanResult()


def _scan_job(owner_user_id: str) -> QueueJob:
    document_id = new_document_id()
    return QueueJob(
        job_id=f"memory_document_scan_{owner_user_id}",
        queue_name="memory_document_scan",
        action="rescan",
        target_uri=MemoryDocumentPathPolicy.document_uri(owner_user_id, document_id),
        payload={
            "tenant_id": "default",
            "owner_user_id": owner_user_id,
            "document_id": document_id,
            "observed_source_digest": "a" * 64,
        },
    )


def test_periodic_owner_scan_is_bounded_and_rotates_without_jobs() -> None:
    scanner = _RecordingScanner()
    worker = MemoryDocumentScanWorker(
        scanner,  # type: ignore[arg-type]
        InMemoryQueueStore(),
        owner_user_ids=lambda _tenant, _limit: ("u1", "u2", "u3"),
        max_owners_per_run=2,
        owner_enumeration_limit=3,
    )

    first = worker.process_pending()
    second = worker.process_pending()

    assert first["claimed"] == 0
    assert first["periodic_scanned"] == 2
    assert second["periodic_scanned"] == 2
    assert scanner.calls == [
        ("default", "u1"),
        ("default", "u2"),
        ("default", "u3"),
        ("default", "u1"),
    ]


def test_queue_hints_and_periodic_scan_share_one_owner_budget() -> None:
    scanner = _RecordingScanner()
    queue = InMemoryQueueStore()
    queue.enqueue(_scan_job("u2"))
    worker = MemoryDocumentScanWorker(
        scanner,  # type: ignore[arg-type]
        queue,
        owner_user_ids=lambda _tenant, _limit: ("u1", "u2", "u3"),
        max_owners_per_run=2,
        owner_enumeration_limit=3,
    )

    result = worker.process_pending()

    assert result["claimed"] == 1
    assert result["processed"] == 1
    assert result["periodic_scanned"] == 1
    assert result["scanned_owners"] == 2
    assert scanner.calls == [("default", "u2"), ("default", "u1")]
    assert scanner.notifications == [("default", "u2")]
