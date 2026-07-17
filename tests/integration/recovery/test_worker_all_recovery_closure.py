from __future__ import annotations

import ast
import importlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.source_store import LeaseLostError, QueueJob, is_canonical_memory_object
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.runtime.readiness import RuntimeNotReadyError, RuntimeReadinessState
from memoryos.workers.embedding_worker import EmbeddingWorker
from memoryos.workers.memory_proposal_worker import MemoryProposalWorker
from memoryos.workers.reindex_worker import ReindexWorker
from memoryos.workers.runner import WorkerRunner
from memoryos.workers.semantic_worker import SemanticWorker
from memoryos.workers.session_commit_worker import SessionCommitWorker


def _runner(client: MemoryOSClient, *, max_retries: int = 3) -> WorkerRunner:
    return WorkerRunner(
        client,
        poll_interval=0.05,
        batch_size=20,
        lease_seconds=30,
        max_retries=max_retries,
    )


def test_every_default_queue_producer_has_a_real_worker_consumer() -> None:
    repository = Path(__file__).resolve().parents[3]
    produced: set[str] = set()
    for path in (repository / "memoryos").glob("**/*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or node.func.attr != "enqueue"
                or not node.args
                or not isinstance(node.args[0], ast.Call)
            ):
                continue
            job_call = node.args[0]
            function_name = job_call.func.id if isinstance(job_call.func, ast.Name) else ""
            if function_name != "QueueJob":
                continue
            queue_keyword = next(
                (keyword.value for keyword in job_call.keywords if keyword.arg == "queue_name"),
                None,
            )
            assert isinstance(queue_keyword, ast.Constant) and isinstance(queue_keyword.value, str)
            produced.add(queue_keyword.value)

    assert produced == {"session_commit", "memory_proposal", "memory_projection"}
    worker_consumers = {"session_commit", "memory_proposal", "memory_projection", "semantic", "embedding"}
    assert produced <= worker_consumers


def test_worker_all_recovers_regular_source_written_and_is_idempotent(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    obj = ContextObject(
        uri="memoryos://user/u1/memories/recovery/regular",
        context_type=ContextType.MEMORY,
        title="regular recovery",
        owner_user_id="u1",
    )
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.ADD,
        target_uri=obj.uri,
        payload={"context_object": obj.to_dict(), "content": "durable regular effect"},
        operation_id="regular-source-written",
    )
    relation_manifest = client.committer._build_regular_relation_manifest(operation)
    client.committer.redo.begin(
        operation,
        phase="started",
        relation_manifest=relation_manifest,
    )
    client.committer._apply_source(operation)
    client.committer._apply_regular_relation_manifest(operation, relation_manifest)
    source_effect = client.committer._capture_regular_source_effect(operation, relation_manifest)
    client.committer.redo.advance(
        operation,
        phase="source_written",
        source_effect=source_effect,
        relation_manifest=relation_manifest,
    )

    runner = _runner(client)
    first = runner.run("all", once=True)
    assert first["recovery"]["operation_ids"] == [operation.operation_id]
    assert client.committer.redo.pending_entries() == []
    assert client.index_store.search("durable regular effect", limit=10)[0].uri == obj.uri
    marker = client.committer._operation_marker(operation.operation_id)
    client.committer._validate_operation_marker(marker, operation)
    assert (tmp_path / "system" / "diffs" / f"diff_{operation.operation_id}.json").exists()
    audit = tmp_path / "system" / "audit" / "u1.jsonl"
    audit_before = audit.read_bytes()
    marker_before = marker.read_bytes()
    queue_path = getattr(client.queue_store, "path", None)
    assert isinstance(queue_path, Path)
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    for directory in (audit.parent, marker.parent, queue_path.parent):
        assert directory.stat().st_mode & 0o777 == 0o700
    for sensitive_file in (audit, marker, queue_path):
        assert sensitive_file.stat().st_mode & 0o777 == 0o600

    second = runner.run("all", once=True)
    assert second["recovery"]["recovered_count"] == 0
    assert marker.read_bytes() == marker_before
    assert audit.read_bytes() == audit_before
    assert client.queue_store.stats().get("pending", 0) == 0
    health = json.loads(runner.heartbeat.read_text(encoding="utf-8"))
    assert health["status"] == "ready"


def test_worker_all_recovers_canonical_crash_then_consumes_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    original_marker = client.committer._write_transaction_marker

    def crash_before_marker(*_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
        raise SystemExit("injected marker crash")

    monkeypatch.setattr(client.committer, "_write_transaction_marker", crash_before_marker)
    with pytest.raises(SystemExit, match="marker crash"):
        client.remember(
            user_id="u1",
            title="primary database",
            content="MemoryOS uses SQLite as the primary database",
            memory_type="project_decision",
            project_id="memoryos",
        )
    entries = client.committer.redo.pending_entries()
    claim_operation = next(
        entry.operation
        for entry in entries
        if dict(entry.operation.payload["context_object"]["metadata"]).get("canonical_kind") == "claim"
    )
    claim_uri = str(claim_operation.target_uri)
    transaction_id = str(claim_operation.payload["transaction_id"])
    idempotency_key = str(claim_operation.payload["idempotency_key"])
    outbox = client.committer._outbox_path(transaction_id)
    assert json.loads(outbox.read_text(encoding="utf-8"))["status"] == "source_committed"
    monkeypatch.setattr(client.committer, "_write_transaction_marker", original_marker)

    runner = _runner(client)
    first = runner.run("all", once=True)
    assert first["recovery"]["recovered_count"] == len(entries)
    assert client.committer.redo.pending_entries() == []
    assert json.loads(outbox.read_text(encoding="utf-8"))["status"] == "committed"
    marker = client.committer._transaction_marker(idempotency_key)
    assert marker.exists()
    projection_store = client.context_db.projection_store
    assert projection_store is not None
    projection = projection_store.load_current(claim_uri)
    assert projection is not None and projection.current and projection.usable
    assert projection.source_revision == int(client.source_store.read_object(claim_uri).metadata["revision"])
    assert client.queue_store.stats().get("pending", 0) == 0
    assert client.queue_store.stats().get("leased", 0) == 0
    recalled = client.search_context(
        "SQLite",
        user_id="u1",
        project_id="memoryos",
        query_intent="CURRENT",
    )
    assert recalled[0]["uri"] == claim_uri
    assert recalled[0]["projection_record"]["projection_attempt_id"] == projection.projection_attempt_id

    stable = {
        "outbox": outbox.read_bytes(),
        "marker": marker.read_bytes(),
        "projection": projection_store.current_path(claim_uri).read_bytes(),
        "audit": (tmp_path / "system" / "audit" / "u1.jsonl").read_bytes(),
        "queue": client.queue_store.stats(),
    }
    second = runner.run("all", once=True)
    assert second["recovery"]["recovered_count"] == 0
    assert second["memory_projection"]["processed"] == []
    assert outbox.read_bytes() == stable["outbox"]
    assert marker.read_bytes() == stable["marker"]
    assert projection_store.current_path(claim_uri).read_bytes() == stable["projection"]
    assert (tmp_path / "system" / "audit" / "u1.jsonl").read_bytes() == stable["audit"]
    assert client.queue_store.stats() == stable["queue"]


def test_worker_all_permanent_oserror_reaches_dead_letter_and_stays_degraded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    queued = client.commit_agent_session(
        user_id="u1",
        session_id="dead-letter",
        messages=[{"role": "user", "content": "archive me"}],
        async_commit=False,
    )

    def unavailable(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise OSError("permanent storage outage with secret=hidden")

    monkeypatch.setattr(client.session_archive_store, "read_archive", unavailable)
    runner = _runner(client, max_retries=2)
    first = runner.run("all", once=True)
    assert first["session_commit"]["failed"] == 1
    pending_job = client.queue_store.get(queued.task_id)
    assert pending_job is not None and pending_job.status == "pending"
    second = runner.run("all", once=True)
    job = client.queue_store.get(queued.task_id)
    assert job is not None and job.status == "dead_letter" and job.retry_count == 2
    assert job.last_error == "OSError"
    assert second["session_commit"]["dead_letter"] == 1
    heartbeat = json.loads(runner.heartbeat.read_text(encoding="utf-8"))
    assert heartbeat["status"] == "degraded"
    assert heartbeat["dead_letter"] >= 1

    third = runner.run("all", once=True)
    assert third["session_commit"]["claimed"] == 0
    assert json.loads(runner.heartbeat.read_text(encoding="utf-8"))["status"] == "degraded"
    assert client.health()["worker"] == "degraded"
    assert client.health()["status"] == "degraded"


def test_worker_crash_before_ack_replays_completed_task_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    queued = client.commit_agent_session(
        user_id="u1",
        session_id="ack-crash",
        messages=[{"role": "user", "content": "archive this completed task"}],
        async_commit=False,
    )
    queue_path = getattr(client.queue_store, "path", None)
    assert isinstance(queue_path, Path)
    original_ack = client.queue_store.ack

    def crash_before_ack(_job):  # noqa: ANN001, ANN202
        raise SystemExit("injected crash before worker ack")

    monkeypatch.setattr(client.queue_store, "ack", crash_before_ack)
    with pytest.raises(SystemExit, match="before worker ack"):
        _runner(client).run("session-commit", once=True)

    leased = client.queue_store.get(queued.task_id)
    assert leased is not None and leased.status == "leased" and leased.lease_generation == 1
    archive = client.session_archive_store.read_archive(queued.archive_uri)
    output_head = client.session_archive_store._dir(archive.archive_uri) / "async_outputs" / "current.json"
    assert client.session_archive_store.async_outputs_done_for_task(archive) is True
    head_before = output_head.read_bytes()
    group_files = sorted((tmp_path / "system" / "commit_groups").glob("*.json"))
    assert len(group_files) == 1
    group_before = group_files[0].read_bytes()

    monkeypatch.setattr(client.queue_store, "ack", original_ack)
    with sqlite3.connect(queue_path) as connection:
        connection.execute(
            "UPDATE queue_jobs SET leased_until = ? WHERE job_id = ?",
            ("1970-01-01T00:00:00+00:00", queued.task_id),
        )
    replay = _runner(client).run("all", once=True)
    settled = client.queue_store.get(queued.task_id)
    assert settled is not None and settled.status == "done" and settled.lease_generation == 2
    assert replay["session_commit"]["committed"] == 1
    assert output_head.read_bytes() == head_before
    assert group_files[0].read_bytes() == group_before

    stable = (output_head.read_bytes(), group_files[0].read_bytes(), client.queue_store.stats())
    again = _runner(client).run("all", once=True)
    assert again["session_commit"]["claimed"] == 0
    assert (output_head.read_bytes(), group_files[0].read_bytes(), client.queue_store.stats()) == stable


def test_worker_all_quarantines_corrupt_outbox_and_stops_not_ready(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    outbox = tmp_path / "system" / "outbox" / "corrupt-transaction.json"
    outbox.parent.mkdir(parents=True)
    outbox.write_text("{broken", encoding="utf-8")
    runner = _runner(client)

    first = runner.run("all", once=True)
    assert first["status"] == "not_ready"
    assert set(first) >= {"recovery", "runtime", "queue_stats", "quarantine_records"}
    assert not set(first) & {
        "session_commit",
        "memory_projection",
        "memory_proposal",
        "semantic",
        "embedding",
        "maintenance",
    }
    assert first["recovery"]["quarantine_count"] == 1
    assert not outbox.exists()
    assert client.readiness.state == RuntimeReadinessState.NOT_READY
    assert json.loads(runner.heartbeat.read_text(encoding="utf-8"))["status"] == "not_ready"
    records = list((tmp_path / "system" / "quarantine" / "outbox").glob("*.json"))
    assert len(records) == 1

    second = runner.run("all", once=True)
    assert second["status"] == "not_ready"
    assert second["recovery"]["quarantine_count"] == 0
    assert len(list((tmp_path / "system" / "quarantine" / "outbox").glob("*.json"))) == 1
    heartbeat = json.loads(runner.heartbeat.read_text(encoding="utf-8"))
    assert heartbeat["status"] == "not_ready"
    assert heartbeat["quarantine"] >= 1
    assert client.health()["worker_health"]["quarantine"] >= 1
    assert client.health()["status"] == "not_ready"


class _CountingExtractor:
    semantic_proposal_backend = True
    llm_semantic_backend = True

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, *_args, **_kwargs) -> list:  # noqa: ANN002, ANN003
        self.calls += 1
        return []


def test_not_ready_session_worker_never_leases_calls_model_or_writes_canonical(
    tmp_path: Path,
) -> None:
    extractor = _CountingExtractor()
    client = MemoryOSClient(str(tmp_path), memory_extractor=extractor)
    queued = client.commit_agent_session(
        user_id="u1",
        session_id="blocked-session",
        messages=[{"role": "user", "content": "Remember that the primary database is PostgreSQL."}],
        async_commit=False,
        project_id="memoryos",
    )
    client.readiness.transition(
        RuntimeReadinessState.NOT_READY,
        reasons=("authoritative receipt mismatch",),
    )

    with pytest.raises(RuntimeNotReadyError, match="runtime is NOT_READY"):
        _runner(client).run("session-commit", once=True)
    with pytest.raises(RuntimeNotReadyError, match="runtime is NOT_READY"):
        SessionCommitWorker(client.session_commit_service).process_pending()
    with pytest.raises(RuntimeNotReadyError, match="runtime is NOT_READY"):
        MemoryProposalWorker(client.session_commit_service).process_pending()

    job = client.queue_store.get(queued.task_id)
    assert job is not None and job.status == "pending" and job.lease_generation == 0
    assert extractor.calls == 0
    assert not any(is_canonical_memory_object(obj) for obj in client.source_store.list_objects())
    assert not list((tmp_path / "system" / "transactions").glob("*.json"))
    assert not list((tmp_path / "system" / "current-heads").glob("*.json"))


@pytest.mark.parametrize(
    ("queue_name", "worker_type"),
    [
        ("session_commit", SessionCommitWorker),
        ("memory_proposal", MemoryProposalWorker),
    ],
)
def test_live_authoritative_failure_releases_entire_leased_session_batch_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    queue_name: str,
    worker_type: type[SessionCommitWorker] | type[MemoryProposalWorker],
) -> None:
    extractor = _CountingExtractor()
    client = MemoryOSClient(str(tmp_path), memory_extractor=extractor)
    committed = client.remember(
        user_id="u1",
        content="Always run tests before release",
        title="Release validation",
        memory_type="project_rule",
        project_id="memoryos",
        constraint_polarity="REQUIRE",
        identity_fields={"rule_topic": "release_validation"},
    )
    claim_uri = str(committed["uri"])
    archives = [
        SessionArchive(
            user_id="u1",
            session_id=f"leased-authoritative-{index}",
            archive_uri=(
                f"memoryos://user/u1/sessions/history/leased-authoritative-{index}"
            ),
            messages=[
                {
                    "id": f"m{index}",
                    "role": "user",
                    "content": "Remember this durable coding-agent release rule.",
                }
            ],
            metadata={
                "tenant_id": "default",
                "project_id": "memoryos",
                "connect": {"adapter_id": "codex"},
            },
            task_id=f"leased-authoritative-task-{index}",
        )
        for index in range(2)
    ]
    job_ids: list[str] = []
    for archive in archives:
        client.session_commit_service.sync_archive(
            archive,
            enqueue_commit_job=queue_name == "session_commit",
        )
        if queue_name == "session_commit":
            job_id = archive.task_id
        else:
            job_id = f"memory_proposal_{archive.task_id}"
            client.queue_store.enqueue(
                QueueJob(
                    job_id=job_id,
                    queue_name="memory_proposal",
                    action="extract_memory_proposals",
                    target_uri=archive.archive_uri,
                    payload={
                        "tenant_id": "default",
                        "manifest_digest": archive.manifest_digest,
                    },
                )
            )
        job_ids.append(job_id)

    tampered = False

    def tamper_current_head() -> None:
        nonlocal tampered
        if tampered:
            return
        for path in (tmp_path / "system" / "current-heads").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            heads = dict(payload.get("heads", {}) or {})
            if claim_uri not in heads:
                continue
            heads[claim_uri] = {**dict(heads[claim_uri]), "head_digest": "0" * 64}
            path.write_text(
                json.dumps({**payload, "heads": heads}, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            tampered = True
            return
        raise AssertionError("committed Claim current head was not found")

    # Inject only after the worker owns the batch and has passed its per-job
    # READY check.  Runner(all) intentionally performs authoritative recovery
    # first; pre-tampering here would test the earlier before-lease gate instead
    # of the live mid-batch release path.
    original_read_archive = client.session_archive_store.read_archive
    target_archives = {archive.archive_uri for archive in archives}

    def read_archive_then_tamper(archive_uri, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        archive = original_read_archive(archive_uri, *args, **kwargs)
        if str(archive_uri) in target_archives:
            tamper_current_head()
        return archive

    monkeypatch.setattr(client.session_archive_store, "read_archive", read_archive_then_tamper)

    leased_jobs: list[QueueJob] = []
    original_lease = client.queue_store.lease

    def capture_lease(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        jobs = original_lease(*args, **kwargs)
        leased_jobs.extend(jobs)
        return jobs

    monkeypatch.setattr(client.queue_store, "lease", capture_lease)
    if queue_name == "session_commit":
        worker_result = _runner(client).run("all", once=True)
        assert worker_result["status"] == "not_ready"
        assert "session_commit" in worker_result
        assert not set(worker_result) & {
            "memory_projection",
            "memory_proposal",
            "semantic",
            "embedding",
            "maintenance",
        }
        result = worker_result["session_commit"]
    else:
        result = worker_type(client.session_commit_service).process_pending(
            batch_size=2,
            max_retries=3,
        )

    assert tampered
    assert result["status"] == "not_ready"
    assert result["failed"] == 1
    assert result["dead_letter"] == 0
    assert set(result["released"]) == set(job_ids)
    assert [job.job_id for job in leased_jobs] == job_ids
    assert extractor.calls == 0
    assert client.readiness.state == RuntimeReadinessState.NOT_READY
    for job_id in job_ids:
        job = client.queue_store.get(job_id)
        assert job is not None and job.status == "pending"
        assert job.retry_count == 0
        assert job.lease_token == job.lease_owner == ""
    second_group = tmp_path / "system" / "commit_groups" / f"commit_group_{archives[1].task_id}.json"
    second_envelope = tmp_path / "system" / "planning-envelopes" / f"{archives[1].task_id}.json"
    assert not second_group.exists()
    assert not second_envelope.exists()

    for stale in leased_jobs:
        with pytest.raises(LeaseLostError):
            client.queue_store.ack(stale)
        with pytest.raises(LeaseLostError):
            client.queue_store.release(stale)


@pytest.mark.parametrize(
    ("queue_name", "worker_type"),
    [
        ("session_commit", SessionCommitWorker),
        ("memory_proposal", MemoryProposalWorker),
    ],
)
def test_worker_releases_current_job_when_success_result_also_flips_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    queue_name: str,
    worker_type: type[SessionCommitWorker] | type[MemoryProposalWorker],
) -> None:
    client = MemoryOSClient(str(tmp_path))
    archives = [
        SessionArchive(
            user_id="u1",
            session_id=f"success-not-ready-{index}",
            archive_uri=f"memoryos://user/u1/sessions/history/success-not-ready-{index}",
            messages=[{"id": f"m{index}", "role": "user", "content": "Remember this."}],
            metadata={"tenant_id": "default", "project_id": "memoryos"},
            task_id=f"success-not-ready-task-{index}",
        )
        for index in range(2)
    ]
    job_ids: list[str] = []
    for archive in archives:
        client.session_commit_service.sync_archive(archive, enqueue_commit_job=False)
        job_id = (
            archive.task_id
            if queue_name == "session_commit"
            else f"memory_proposal_{archive.task_id}"
        )
        client.queue_store.enqueue(
            QueueJob(
                job_id=job_id,
                queue_name=queue_name,
                action=("commit_session" if queue_name == "session_commit" else "extract_memory_proposals"),
                target_uri=archive.archive_uri,
                payload={
                    "tenant_id": "default",
                    "manifest_digest": archive.manifest_digest,
                },
            )
        )
        job_ids.append(job_id)

    calls = 0

    def success_then_not_ready(archive):  # noqa: ANN001, ANN202
        nonlocal calls
        del archive
        calls += 1
        client.readiness.transition(
            RuntimeReadinessState.NOT_READY,
            reasons=("authoritative proof changed after result",),
        )
        return SimpleNamespace(done=True, canonical_committed=True)

    monkeypatch.setattr(client.session_commit_service, "async_commit", success_then_not_ready)

    result = worker_type(client.session_commit_service).process_pending(batch_size=2)

    assert result["status"] == "not_ready"
    assert result["committed"] == 0
    assert result["failed"] == 1
    assert set(result["released"]) == set(job_ids)
    assert calls == 1
    for job_id in job_ids:
        job = client.queue_store.get(job_id)
        assert job is not None and job.status == "pending"
        assert job.retry_count == 0


@pytest.mark.parametrize(
    "kind",
    ["session-commit", "memory-proposal", "memory-projection", "semantic", "embedding"],
)
def test_not_ready_runner_rejects_every_ordinary_worker_before_queue_lease(
    tmp_path: Path,
    kind: str,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    client.queue_store.enqueue(
        QueueJob(
            job_id=f"blocked-{kind}",
            queue_name="memory_projection",
            action="project_memory_committed",
            target_uri="memoryos://user/u1/memories/blocked",
        )
    )
    client.readiness.transition(RuntimeReadinessState.NOT_READY, reasons=("startup proof failed",))

    with pytest.raises(RuntimeNotReadyError, match="runtime is NOT_READY"):
        _runner(client).run(kind, once=True)

    job = client.queue_store.get(f"blocked-{kind}")
    assert job is not None and job.status == "pending" and job.lease_generation == 0


def test_not_ready_projection_entry_does_not_dispatch_or_lease(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    client.queue_store.enqueue(
        QueueJob(
            job_id="blocked-projection",
            queue_name="memory_projection",
            action="project_memory_committed",
            target_uri="memoryos://user/u1/memories/blocked",
        )
    )
    client.readiness.transition(RuntimeReadinessState.NOT_READY, reasons=("projection proof mismatch",))

    with pytest.raises(RuntimeNotReadyError, match="runtime is NOT_READY"):
        client.memory_projection_worker.process_pending()
    with pytest.raises(RuntimeNotReadyError, match="runtime is NOT_READY"):
        SemanticWorker(client.source_store, client.queue_store).process_pending()
    with pytest.raises(RuntimeNotReadyError, match="runtime is NOT_READY"):
        EmbeddingWorker(
            client.source_store,
            client.queue_store,
            InMemoryVectorStore(),
        ).process_pending()

    job = client.queue_store.get("blocked-projection")
    assert job is not None and job.status == "pending" and job.lease_generation == 0


def test_not_ready_reindex_worker_does_not_mutate_production_index(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    indexed = ContextObject(
        uri="memoryos://user/u1/memories/indexed",
        context_type=ContextType.MEMORY,
        title="indexed",
        owner_user_id="u1",
    )
    missing = ContextObject(
        uri="memoryos://user/u1/memories/not-yet-indexed",
        context_type=ContextType.MEMORY,
        title="not yet indexed",
        owner_user_id="u1",
    )
    client.source_store.write_object(indexed, content="stable index row")
    client.source_store.write_object(missing, content="must not be indexed while not ready")
    client.index_store.upsert_index(indexed, content="stable index row")
    before = set(client.index_store.indexed_uris())
    client.readiness.transition(RuntimeReadinessState.NOT_READY, reasons=("head proof mismatch",))

    with pytest.raises(RuntimeNotReadyError, match="runtime is NOT_READY"):
        ReindexWorker(client.source_store, client.index_store).rebuild()

    assert set(client.index_store.indexed_uris()) == before == {indexed.uri}
    assert missing.uri not in {hit.uri for hit in client.index_store.search("must not be indexed", limit=10)}


@pytest.mark.parametrize("worker_kind", ["all", "session-commit"])
def test_cli_worker_exits_not_ready_without_consuming_real_fs_queue(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    worker_kind: str,
) -> None:
    cli_module = importlib.import_module("memoryos.api.cli.main")
    client = MemoryOSClient(str(tmp_path))
    queued = client.commit_agent_session(
        user_id="u1",
        session_id="cli-blocked",
        messages=[{"role": "user", "content": "Remember the release rule."}],
        async_commit=False,
        project_id="memoryos",
    )
    migration_receipt = tmp_path / "system" / "migrations" / "memory-closure-v1.json"
    migration_receipt.write_text("{}", encoding="utf-8")

    exit_code = cli_module.main(["worker", worker_kind, "--root", str(tmp_path), "--once"])

    assert exit_code == 2
    error = json.loads(capsys.readouterr().err)
    assert {key: error[key] for key in ("kind", "status", "runtime_state")} == {
        "kind": worker_kind,
        "status": "not_ready",
        "runtime_state": "NOT_READY",
    }
    assert error["reasons"] == ["MemoryClosureMigrationError: migration receipt schema is unsupported"]
    job = client.queue_store.get(queued.task_id)
    assert job is not None and job.status == "pending" and job.lease_generation == 0


def test_health_fails_closed_for_malformed_worker_counters(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    heartbeat = tmp_path / "system" / "worker-health.json"
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.write_text(
        json.dumps(
            {
                "status": "ready",
                "dead_letter": "not-a-count",
                "quarantine": -1,
            }
        ),
        encoding="utf-8",
    )

    health = client.health()

    assert health["status"] == "degraded"
    assert health["runtime"]["ready"] is True
