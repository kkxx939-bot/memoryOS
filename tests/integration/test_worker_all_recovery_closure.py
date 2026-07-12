from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.workers.runner import WorkerRunner


def _runner(client: MemoryOSClient, *, max_retries: int = 3) -> WorkerRunner:
    return WorkerRunner(
        client,
        poll_interval=0.05,
        batch_size=20,
        lease_seconds=30,
        max_retries=max_retries,
    )


def test_every_default_queue_producer_has_a_real_worker_consumer() -> None:
    repository = Path(__file__).resolve().parents[2]
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
    assert projection.source_revision == int(
        client.source_store.read_object(claim_uri).metadata["revision"]
    )
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
    output_head = (
        client.session_archive_store._dir(archive.archive_uri)
        / "async_outputs"
        / "current.json"
    )
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


def test_worker_all_quarantines_corrupt_outbox_and_health_remains_degraded(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    outbox = tmp_path / "system" / "outbox" / "corrupt-transaction.json"
    outbox.parent.mkdir(parents=True)
    outbox.write_text("{broken", encoding="utf-8")
    runner = _runner(client)

    first = runner.run("all", once=True)
    assert set(first) >= {
        "recovery",
        "session_commit",
        "memory_projection",
        "memory_proposal",
        "semantic",
        "embedding",
        "maintenance",
        "queue_stats",
        "quarantine_records",
    }
    assert first["recovery"]["quarantine_count"] == 1
    assert not outbox.exists()
    assert json.loads(runner.heartbeat.read_text(encoding="utf-8"))["status"] == "degraded"
    records = list((tmp_path / "system" / "quarantine" / "outbox").glob("*.json"))
    assert len(records) == 1

    second = runner.run("all", once=True)
    assert second["recovery"]["quarantine_count"] == 0
    assert len(list((tmp_path / "system" / "quarantine" / "outbox").glob("*.json"))) == 1
    heartbeat = json.loads(runner.heartbeat.read_text(encoding="utf-8"))
    assert heartbeat["status"] == "degraded"
    assert heartbeat["quarantine"] >= 1
    assert client.health()["worker_health"]["quarantine"] >= 1
