from __future__ import annotations

import os
import time
from pathlib import Path
from typing import cast

import pytest

from foundation.identity import LocalUserContext
from infrastructure.context.trace import RecallTraceService
from infrastructure.store.memory import DocumentIntentStatus
from infrastructure.store.memory.layout import user_memory_root
from infrastructure.store.model.catalog import CatalogRecordKind
from infrastructure.store.sqlite import SQLiteIndexStore
from infrastructure.store.trace import RecallTraceRepository, recall_trace_root
from openApi.sdk.client import MemoryOSClient
from runtime.worker.runner import WorkerRunner


def _caller() -> LocalUserContext:
    return LocalUserContext(
        user_id="u1",
    )


def _project_all(client: MemoryOSClient) -> None:
    while client.runtime.stores.queue.stats(queue_name="memory_projection").get("pending", 0):
        result = client.runtime.memory.projection_worker.process_pending(limit=10)
        assert not result.failed


def _document_records(client: MemoryOSClient, document_id: str):  # noqa: ANN202
    return cast(SQLiteIndexStore, client.runtime.stores.index).list_catalog(
        tenant_id="default",
        filters={
            "owner_user_id": "u1",
            "document_ids": (document_id,),
            "include_inactive": True,
        },
        limit=100,
    )


def test_maintenance_prunes_expired_recall_traces(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    repository = RecallTraceRepository(recall_trace_root(client.root, client.tenant_id))
    trace_id = RecallTraceService(repository).record("expired maintenance trace")
    trace_path = repository.trace_root / f"{trace_id}.json"
    expired_at = time.time() - (31 * 24 * 60 * 60)
    os.utime(trace_path, (expired_at, expired_at))

    result = WorkerRunner(client).run_once("maintenance")

    retention = result["maintenance"]["recall_trace_retention"]
    assert retention["deleted"] == 1
    assert not trace_path.exists()


def test_retrieval_rescan_job_is_consumed_and_reprojects_live_markdown(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    caller = _caller()
    old_marker = "stalerescanoldmarker"
    new_marker = "stalerescannewmarker"
    remembered = client.remember(
        f"Prefer {old_marker}.",
        target_hint="topic:rescan chain",
        caller=caller,
    )
    _project_all(client)
    document_path = user_memory_root(tmp_path, "default", "u1") / remembered["relative_path"]
    document_path.write_bytes(document_path.read_bytes().replace(old_marker.encode(), new_marker.encode()))

    stale_results = client.archive_search(old_marker, user_id="u1", caller=caller)

    assert all(item.get("document_id") != remembered["document_id"] for item in stale_results)
    assert client.runtime.stores.queue.stats(queue_name="memory_document_scan").get("pending") == 1

    run = WorkerRunner(client, batch_size=10).run_once("all")

    assert run["memory_document_scan"]["processed"] == 1
    assert client.runtime.stores.queue.stats(queue_name="memory_document_scan").get("done") == 1
    records = _document_records(client, remembered["document_id"])
    document = next(
        record for record in records if record.record_kind == CatalogRecordKind.MEMORY_DOCUMENT.value
    )
    assert new_marker in f"{document.l0_text}\n{document.l1_text}"
    assert old_marker not in f"{document.l0_text}\n{document.l1_text}"


def test_single_missing_scan_does_not_create_deletion_authority(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    caller = _caller()
    remembered = client.remember(
        "Prefer a stable deletion barrier marker.",
        target_hint="topic:deletion stability",
        caller=caller,
    )
    _project_all(client)
    document_path = user_memory_root(tmp_path, "default", "u1") / remembered["relative_path"]
    document_path.unlink()
    assert client.archive_search(
        "stable deletion barrier marker",
        user_id="u1",
        caller=caller,
    ) == []
    assert client.runtime.stores.queue.stats(queue_name="memory_document_scan").get("pending") == 1

    rebuilt = client.runtime.memory.projection_worker.rebuild_owner("default", "u1")
    verified = client.runtime.memory.projection_worker.verify_owner("default", "u1")

    assert rebuilt["pending_missing"] == 1
    assert rebuilt["deleted"] == 0
    assert verified["pending_missing"] == 1
    assert verified["degraded"] == 1
    assert client.runtime.memory.control_store.load_publication_barrier(
        "default", "u1", remembered["document_id"]
    ) is None
    assert _document_records(client, remembered["document_id"])

    client.runtime.memory.scanner.stability_seconds = 0
    runner = WorkerRunner(client, batch_size=10)
    first = runner.run_once("memory-document-scan")["memory_document_scan"]

    assert first["claimed"] == 1
    assert first["processed"] == 0
    assert first["pending"] == 1
    assert first["periodic_scanned"] == 0
    scan_job_id = first["released"][0]
    scan_job = client.runtime.stores.queue.get(scan_job_id)
    assert scan_job is not None and scan_job.status == "pending"
    assert scan_job.retry_count == 0
    assert client.runtime.memory.control_store.load_publication_barrier(
        "default", "u1", remembered["document_id"]
    ) is None

    second = runner.run_once("memory-document-scan")["memory_document_scan"]

    assert second["claimed"] == 1
    assert second["processed"] == 1
    assert second["pending"] == 0
    scan_job = client.runtime.stores.queue.get(scan_job_id)
    assert scan_job is not None and scan_job.status == "done"
    assert scan_job.retry_count == 0
    barrier = client.runtime.memory.control_store.load_publication_barrier(
        "default", "u1", remembered["document_id"]
    )
    assert barrier is not None
    _project_all(client)
    active = cast(SQLiteIndexStore, client.runtime.stores.index).list_catalog(
        tenant_id="default",
        filters={
            "owner_user_id": "u1",
            "document_ids": (remembered["document_id"],),
        },
        limit=100,
    )
    assert active == []


def test_periodic_runner_projects_external_edit_without_retrieval_hint(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    caller = _caller()
    old_marker = "periodicscanoldmarker"
    new_marker = "periodicscannewmarker"
    remembered = client.remember(
        f"Prefer {old_marker}.",
        target_hint="topic:periodic production scan",
        caller=caller,
    )
    _project_all(client)
    document_path = user_memory_root(tmp_path, "default", "u1") / remembered["relative_path"]
    document_path.write_bytes(
        document_path.read_bytes().replace(old_marker.encode(), new_marker.encode())
    )
    assert client.runtime.stores.queue.stats(queue_name="memory_document_scan").get("pending", 0) == 0

    run = WorkerRunner(client, batch_size=10).run_once("all")

    assert run["memory_document_scan"]["claimed"] == 0
    assert run["memory_document_scan"]["periodic_scanned"] == 1
    assert run["memory_document_scan"]["periodic_confirmed"] >= 1
    assert run["memory_projection"].processed
    records = _document_records(client, remembered["document_id"])
    document = next(
        record for record in records if record.record_kind == CatalogRecordKind.MEMORY_DOCUMENT.value
    )
    assert new_marker in f"{document.l0_text}\n{document.l1_text}"
    assert old_marker not in f"{document.l0_text}\n{document.l1_text}"


def test_runtime_runner_recovers_real_document_edit_producer_job(tmp_path: Path) -> None:
    class SimulatedRetryableInterruption(RuntimeError):
        retryable = True

    client = MemoryOSClient(str(tmp_path))
    caller = _caller()

    def interrupt_after_durable_intent(stage, _intent) -> None:  # noqa: ANN001
        if stage == "intent_prepared":
            raise SimulatedRetryableInterruption

    client.runtime.memory.committer.test_hook = interrupt_after_durable_intent
    with pytest.raises(SimulatedRetryableInterruption):
        client.remember(
            "Runtime producer must recover through the edit worker.",
            target_hint="topic:runtime edit recovery",
            caller=caller,
        )
    client.runtime.memory.committer.test_hook = None

    intents = client.runtime.memory.control_store.incomplete_intents("default", "u1")
    assert len(intents) == 1
    intent = intents[0]
    job_id = f"memory_document_edit_{intent.intent_id}"
    job = client.runtime.stores.queue.get(job_id)
    assert job is not None and job.status == "pending"
    assert frozenset(job.payload) == {
        "tenant_id",
        "owner_user_id",
        "document_id",
        "intent_id",
    }
    assert client.runtime.memory.edit_worker.committer is client.runtime.memory.committer
    assert client.runtime.memory.scan_worker.scanner is client.runtime.memory.scanner

    result = WorkerRunner(client, batch_size=10).run_once("memory-document-edit")

    assert result["memory_document_edit"] == {
        "claimed": 1,
        "committed": 1,
        "failed": 0,
        "dead_letter": 0,
    }
    settled = client.runtime.stores.queue.get(job_id)
    assert settled is not None and settled.status == "done"
    durable = client.runtime.memory.control_store.load_intent(
        "default",
        "u1",
        intent.intent_id,
    )
    assert durable is not None and durable.status is DocumentIntentStatus.COMPLETED
