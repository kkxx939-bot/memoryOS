from __future__ import annotations

import json
import multiprocessing as mp
import queue
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from infrastructure.store.contracts.queue import (
    LeaseLostError,
    QueueIdempotencyConflictError,
    QueueJob,
    QueueLeaseIdentityError,
)
from infrastructure.store.sqlite.queue_store import SQLiteQueueStore
from tests.support.persistence import InMemoryQueueStore


def _lease_once(
    path: str,
    owner: str,
    barrier: Any,
    results: Any,
    job_id: str,
) -> None:
    store = SQLiteQueueStore(path)
    barrier.wait()
    leased = store.lease(
        "race",
        lease_owner=owner,
        limit=1,
        lease_seconds=60,
        job_ids=[job_id],
    )
    if leased:
        results.put((owner, leased[0].lease_token, leased[0].lease_generation))


def _lease_rounds(
    path: str,
    owner: str,
    rounds: int,
    start: Any,
    finished: Any,
    results: Any,
) -> None:
    store = SQLiteQueueStore(path)
    for index in range(rounds):
        start.wait()
        leased = store.lease(
            "race",
            lease_owner=owner,
            limit=1,
            lease_seconds=60,
            job_ids=[f"race-{index}"],
        )
        if leased:
            item = leased[0]
            results.put((index, owner, item.lease_token, item.lease_generation))
        finished.wait()


def _crash_after_lease(path: str, ready: Any) -> None:
    store = SQLiteQueueStore(path)
    leased = store.lease("crash", lease_owner="crashed-worker", limit=1, lease_seconds=60)
    ready.put(leased[0].lease_generation if leased else 0)


def _retry_at_barrier(path: str, job: QueueJob, barrier: Any, results: Any) -> None:
    store = SQLiteQueueStore(path)
    barrier.wait()
    try:
        settled = store.retry(job, "retry", max_retries=5, retryable=True)
    except LeaseLostError:
        results.put("lease_lost")
    else:
        results.put((settled.status, settled.retry_count))


def _collect(result_queue: Any, count: int) -> list[Any]:
    return [result_queue.get(timeout=5) for _ in range(count)]


def _expire(path: Path, job_id: str) -> None:
    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE queue_jobs SET leased_until = ? WHERE job_id = ?", (expired, job_id))


def _job(job_id: str, queue_name: str = "race", *, payload: dict | None = None) -> QueueJob:
    return QueueJob(
        job_id=job_id,
        queue_name=queue_name,
        action="run",
        target_uri=f"memoryos://user/u1/jobs/{job_id}",
        payload=dict(payload or {}),
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_hard_erase_target_purge_removes_every_job_state_and_semantic_path(
    tmp_path: Path,
    store_kind: str,
) -> None:
    path = tmp_path / "queue-purge.sqlite3"
    store = InMemoryQueueStore() if store_kind == "memory" else SQLiteQueueStore(path)
    target = "memoryos://user/u1/memory/documents/memdoc_AAAAAAAAAAAAAAAA"
    secret_path = "knowledge/topics/semantic-secret-path.md"
    jobs = []
    for index in range(3):
        jobs.append(
            store.enqueue(
                QueueJob(
                    job_id=f"purge-{index}",
                    queue_name="memory_projection",
                    action="project_memory_document",
                    target_uri=target,
                    payload={
                        "tenant_id": "tenant-a",
                        "owner_user_id": "u1",
                        "old_relative_path": secret_path,
                    },
                )
            )
        )
    leased = store.lease(
        "memory_projection",
        lease_owner="worker",
        limit=1,
        job_ids=(jobs[1].job_id,),
    )[0]
    store.ack(leased)
    unrelated = store.enqueue(
        QueueJob(
            job_id="unrelated",
            queue_name="memory_projection",
            action="project_memory_document",
            target_uri="memoryos://user/u1/memory/documents/memdoc_BBBBBBBBBBBBBBBB",
            payload={"tenant_id": "tenant-a", "owner_user_id": "u1"},
        )
    )

    assert store.purge_target_jobs(
        queue_name="memory_projection",
        target_uri=target,
        tenant_id="tenant-a",
        owner_user_id="u1",
    ) == 3
    assert all(store.get(job.job_id) is None for job in jobs)
    assert store.get(unrelated.job_id) is not None
    assert store.purge_target_jobs(
        queue_name="memory_projection",
        target_uri=target,
        tenant_id="tenant-a",
        owner_user_id="u1",
    ) == 0
    if store_kind == "sqlite":
        persisted = b"".join(
            candidate.read_bytes()
            for candidate in path.parent.glob(f"{path.name}*")
            if candidate.is_file()
        )
        assert secret_path.encode() not in persisted


def test_two_processes_claim_one_job_exactly_once(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"
    SQLiteQueueStore(path).enqueue(_job("one"))
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    results = ctx.Queue()
    processes = [
        ctx.Process(target=_lease_once, args=(str(path), f"worker-{index}", barrier, results, "one"))
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0

    winners = _collect(results, 1)
    assert len(winners) == 1
    with pytest.raises(queue.Empty):
        results.get(timeout=0.1)
    assert winners[0][2] == 1
    assert winners[0][1]


def test_two_processes_have_no_double_lease_across_200_rounds(tmp_path: Path) -> None:
    rounds = 200
    path = tmp_path / "queue.sqlite3"
    store = SQLiteQueueStore(path)
    for index in range(rounds):
        store.enqueue(_job(f"race-{index}"))

    ctx = mp.get_context("spawn")
    start = ctx.Barrier(3)
    finished = ctx.Barrier(3)
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_lease_rounds,
            args=(str(path), f"worker-{index}", rounds, start, finished, results),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for _ in range(rounds):
        start.wait()
        finished.wait()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0

    winners = _collect(results, rounds)
    with pytest.raises(queue.Empty):
        results.get(timeout=0.1)
    by_round: dict[int, list[Any]] = {}
    for item in winners:
        by_round.setdefault(int(item[0]), []).append(item)
    assert set(by_round) == set(range(rounds))
    assert all(len(items) == 1 for items in by_round.values())
    assert all(items[0][3] == 1 and items[0][2] for items in by_round.values())


def test_expired_lease_increments_generation_and_fences_stale_ack_and_fail(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"
    store = SQLiteQueueStore(path)
    store.enqueue(_job("stale"))
    first = store.lease("race", lease_owner="worker-a", limit=1)[0]
    _expire(path, first.job_id)
    second = store.lease("race", lease_owner="worker-b", limit=1)[0]

    assert second.lease_generation == first.lease_generation + 1
    assert second.lease_token != first.lease_token
    with pytest.raises(LeaseLostError):
        store.ack(first)
    store.ack(second)
    with pytest.raises(LeaseLostError):
        store.fail(first, "late failure")
    stored = store.get("stale")
    assert stored is not None and stored.status == "done"


def test_concurrent_retry_only_changes_current_generation_and_keeps_count(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"
    store = SQLiteQueueStore(path)
    store.enqueue(_job("retry"))
    stale = store.lease("race", lease_owner="worker-a", limit=1)[0]
    _expire(path, stale.job_id)
    current = store.lease("race", lease_owner="worker-b", limit=1)[0]

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    results = ctx.Queue()
    processes = [
        ctx.Process(target=_retry_at_barrier, args=(str(path), item, barrier, results)) for item in (stale, current)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0

    outcomes = _collect(results, 2)
    assert "lease_lost" in outcomes
    assert ("pending", 1) in outcomes
    stored = store.get("retry")
    assert stored is not None and stored.retry_count == 1


def test_crashed_worker_job_is_reclaimed_after_expiry(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"
    store = SQLiteQueueStore(path)
    store.enqueue(_job("crash", "crash"))
    ctx = mp.get_context("spawn")
    ready = ctx.Queue()
    process = ctx.Process(target=_crash_after_lease, args=(str(path), ready))
    process.start()
    process.join(10)
    assert process.exitcode == 0
    assert ready.get(timeout=2) == 1

    _expire(path, "crash")
    recovered = store.lease("crash", lease_owner="recovery-worker", limit=1)
    assert len(recovered) == 1
    assert recovered[0].lease_generation == 2
    store.ack(recovered[0])
    stored = store.get("crash")
    assert stored is not None and stored.status == "done"


def test_fresh_queue_schema_is_complete_and_exact(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"

    SQLiteQueueStore(path)

    with sqlite3.connect(path) as conn:
        layout = tuple(
            (row[1], row[2], row[3], row[4], row[5])
            for row in conn.execute("PRAGMA table_info(queue_jobs)")
        )
    assert layout == (
        ("job_id", "TEXT", 0, None, 1),
        ("queue_name", "TEXT", 1, None, 0),
        ("action", "TEXT", 1, None, 0),
        ("target_uri", "TEXT", 1, None, 0),
        ("payload_json", "TEXT", 1, None, 0),
        ("tenant_id", "TEXT", 1, "''", 0),
        ("owner_user_id", "TEXT", 1, "''", 0),
        ("workspace_id", "TEXT", 1, "''", 0),
        ("status", "TEXT", 1, None, 0),
        ("leased_until", "TEXT", 0, None, 0),
        ("lease_token", "TEXT", 1, "''", 0),
        ("lease_generation", "INTEGER", 1, "0", 0),
        ("lease_owner", "TEXT", 1, "''", 0),
        ("retry_count", "INTEGER", 1, "0", 0),
        ("created_at", "TEXT", 1, None, 0),
        ("updated_at", "TEXT", 1, None, 0),
        ("last_error", "TEXT", 1, "''", 0),
    )


def test_old_queue_schema_fails_fast_without_mutating_database(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE queue_jobs (
              job_id TEXT PRIMARY KEY, queue_name TEXT NOT NULL, action TEXT NOT NULL,
              target_uri TEXT NOT NULL, payload_json TEXT NOT NULL, status TEXT NOT NULL,
              leased_until TEXT, retry_count INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              last_error TEXT NOT NULL DEFAULT ''
            )
            """
        )
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO queue_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("legacy", "race", "run", "memoryos://user/u1/jobs/legacy", "{}", "pending", None, 0, now, now, ""),
        )

    with pytest.raises(RuntimeError, match="unsupported QueueStore layout"):
        SQLiteQueueStore(path)

    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(queue_jobs)")}
        row = conn.execute(
            "SELECT status, payload_json FROM queue_jobs WHERE job_id = 'legacy'"
        ).fetchone()
    assert "lease_token" not in columns
    assert "tenant_id" not in columns
    assert row == ("pending", "{}")


def test_queue_open_rejects_legacy_data_without_status_or_scope_backfill(
    tmp_path: Path,
) -> None:
    path = tmp_path / "queue.sqlite3"
    store = SQLiteQueueStore(path)
    store.enqueue(_job("legacy-data"))
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE queue_jobs SET status = 'failed', tenant_id = '' WHERE job_id = ?",
            ("legacy-data",),
        )

    with pytest.raises(RuntimeError, match="unsupported QueueStore data"):
        SQLiteQueueStore(path)

    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT status, tenant_id FROM queue_jobs WHERE job_id = ?",
            ("legacy-data",),
        ).fetchone()
    assert row == ("failed", "")


def test_subject_uri_projection_queue_scope_is_indexed_and_workspace_bounded(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"
    store = SQLiteQueueStore(path)
    store.enqueue(
        QueueJob(
            job_id="outbox-subject",
            queue_name="memory_projection",
            action="memory_committed",
            target_uri="memoryos://user/subject_hash/memory/documents/memdoc_queue_scope",
            payload={
                "transaction_id": "subject",
                "tenant_id": "tenant-a",
                "owner_user_id": "u1",
                "workspace_id": "workspace-a",
            },
        )
    )

    assert store.stats_for_scope(
        queue_name="memory_projection",
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_ids=("", "workspace-a"),
    ) == {"pending": 1}
    assert store.stats_for_scope(
        queue_name="memory_projection",
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_ids=("", "workspace-b"),
    ) == {}
    assert store.stats_for_scope(
        queue_name="memory_projection",
        tenant_id="tenant-a",
        owner_user_id="u2",
        workspace_ids=("", "workspace-a"),
    ) == {}
    with sqlite3.connect(path) as conn:
        plan = " ".join(
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT status, COUNT(*) FROM queue_jobs "
                "WHERE queue_name = ? AND tenant_id = ? AND owner_user_id = ? "
                "AND workspace_id = ? GROUP BY status",
                ("memory_projection", "tenant-a", "u1", "workspace-a"),
            )
        )
    assert "queue_jobs_scope_status_idx" in plan


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_unresolved_subject_job_blocks_only_its_own_tenant(tmp_path: Path, store_kind: str) -> None:
    store = (
        InMemoryQueueStore()
        if store_kind == "memory"
        else SQLiteQueueStore(tmp_path / "legacy-subject-queue.sqlite3")
    )
    store.enqueue(
        QueueJob(
            job_id="legacy-subject",
            queue_name="memory_projection",
            action="memory_committed",
            target_uri="memoryos://user/subject_hash/memory/documents/memdoc_queue_scope",
            payload={"transaction_id": "legacy", "tenant_id": "tenant-a"},
        )
    )

    assert store.stats_for_scope(
        queue_name="memory_projection",
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_ids=("", "workspace-a"),
    ) == {"pending": 1}
    assert store.stats_for_scope(
        queue_name="memory_projection",
        tenant_id="tenant-a",
        owner_user_id="u2",
        workspace_ids=("", "workspace-b"),
    ) == {"pending": 1}
    assert store.stats_for_scope(
        queue_name="memory_projection",
        tenant_id="tenant-b",
        owner_user_id="u2",
        workspace_ids=("", "workspace-b"),
    ) == {}


def test_scope_health_real_queries_use_scope_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "queue-plan.sqlite3"
    store = SQLiteQueueStore(path)
    store.enqueue(
        QueueJob(
            job_id="legacy-subject",
            queue_name="memory_projection",
            action="memory_committed",
            target_uri="memoryos://user/subject_hash/memory/documents/memdoc_queue_scope",
            payload={"transaction_id": "legacy", "tenant_id": "tenant-a"},
        )
    )
    traced_sql: list[str] = []
    original_connect = store._connect

    def traced_connect() -> sqlite3.Connection:
        conn = original_connect()
        conn.set_trace_callback(traced_sql.append)
        return conn

    monkeypatch.setattr(store, "_connect", traced_connect)
    assert store.stats_for_scope(
        queue_name="memory_projection",
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_ids=("", "workspace-a"),
    ) == {"pending": 1}
    health_queries = [
        sql
        for sql in traced_sql
        if sql.startswith("SELECT status, COUNT(*) AS count FROM queue_jobs")
    ]
    assert len(health_queries) == 2
    with sqlite3.connect(path) as conn:
        plans = [
            " ".join(str(row[3]) for row in conn.execute(f"EXPLAIN QUERY PLAN {sql}"))
            for sql in health_queries
        ]
    assert all("queue_jobs_scope_status_idx" in plan for plan in plans)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_duplicate_enqueue_never_revives_done_or_dead_letter(tmp_path: Path, store_kind: str) -> None:
    store = InMemoryQueueStore() if store_kind == "memory" else SQLiteQueueStore(tmp_path / "queue.sqlite3")
    done_job = _job("done")
    store.enqueue(done_job)
    leased_done = store.lease("race", lease_owner="worker", limit=1)[0]
    store.ack(leased_done)
    assert store.enqueue(done_job).status == "done"
    assert store.lease("race", lease_owner="other", limit=10) == []

    dead_job = _job("dead")
    store.enqueue(dead_job)
    leased_dead = store.lease("race", lease_owner="worker", limit=1)[0]
    store.retry(leased_dead, "permanent", max_retries=1, retryable=False)
    assert store.enqueue(dead_job).status == "dead_letter"
    assert store.lease("race", lease_owner="other", limit=10) == []


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_release_returns_unattempted_lease_without_retry_cost(
    tmp_path: Path,
    store_kind: str,
) -> None:
    store = InMemoryQueueStore() if store_kind == "memory" else SQLiteQueueStore(tmp_path / "queue.sqlite3")
    store.enqueue(_job("released"))
    first = store.lease("race", lease_owner="worker-a", limit=1)[0]

    released = store.release(first, "batch aborted before attempt")

    assert released.status == "pending"
    assert released.retry_count == 0
    assert released.lease_token == released.lease_owner == ""
    second = store.lease("race", lease_owner="worker-b", limit=1)[0]
    assert second.lease_generation == first.lease_generation + 1
    with pytest.raises(LeaseLostError):
        store.release(first, "stale worker")
    store.ack(second)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_expired_lease_recovery_and_stats_are_queue_scoped(
    tmp_path: Path,
    store_kind: str,
) -> None:
    store = InMemoryQueueStore() if store_kind == "memory" else SQLiteQueueStore(tmp_path / "queue.sqlite3")
    projection = store.enqueue(_job("projection-expired", queue_name="memory_projection"))
    unrelated = store.enqueue(_job("session-expired", queue_name="commit"))
    projection_lease = store.lease(
        projection.queue_name,
        lease_owner="projection-worker",
        job_ids=(projection.job_id,),
    )[0]
    unrelated_lease = store.lease(
        unrelated.queue_name,
        lease_owner="session-worker",
        job_ids=(unrelated.job_id,),
    )[0]
    if isinstance(store, InMemoryQueueStore):
        for lease in (projection_lease, unrelated_lease):
            store.jobs[lease.job_id] = QueueJob(
                **{
                    **lease.__dict__,
                    "leased_until": "1970-01-01T00:00:00+00:00",
                }
            )
    else:
        _expire(store.path, projection_lease.job_id)
        _expire(store.path, unrelated_lease.job_id)

    assert store.recover_expired_leases(queue_name="memory_projection") == 1

    recovered = store.get(projection.job_id)
    untouched = store.get(unrelated.job_id)
    assert recovered is not None and recovered.status == "pending"
    assert recovered.retry_count == 0
    assert recovered.lease_token == recovered.lease_owner == ""
    assert untouched is not None and untouched.status == "leased"
    assert store.stats(queue_name="memory_projection") == {"pending": 1}
    assert store.stats(queue_name="commit") == {"leased": 1}
    assert store.stats() == {"pending": 1, "leased": 1}
    with pytest.raises(LeaseLostError):
        store.ack(projection_lease)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_leased_identity_tamper_is_fenced_and_quarantined(
    tmp_path: Path,
    store_kind: str,
) -> None:
    store = InMemoryQueueStore() if store_kind == "memory" else SQLiteQueueStore(tmp_path / "queue.sqlite3")
    store.enqueue(_job("identity-tamper", payload={"version": 1}))
    leased = store.lease("race", lease_owner="worker-a", limit=1)[0]
    forged_payload = {"version": 2}
    if isinstance(store, InMemoryQueueStore):
        store.jobs[leased.job_id] = QueueJob(
            **{
                **leased.__dict__,
                "payload": forged_payload,
            }
        )
    else:
        with sqlite3.connect(store.path) as connection:
            connection.execute(
                "UPDATE queue_jobs SET payload_json = ? WHERE job_id = ?",
                (json.dumps(forged_payload, sort_keys=True), leased.job_id),
            )

    with pytest.raises(QueueLeaseIdentityError):
        store.ack(leased)
    quarantined = store.quarantine_identity_conflict(
        leased,
        "immutable identity changed",
    )
    assert quarantined.status == "quarantine"
    assert quarantined.retry_count == 1
    assert quarantined.lease_token == quarantined.lease_owner == ""


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_duplicate_job_id_with_different_payload_is_conflict(tmp_path: Path, store_kind: str) -> None:
    store = InMemoryQueueStore() if store_kind == "memory" else SQLiteQueueStore(tmp_path / "queue.sqlite3")
    store.enqueue(_job("same", payload={"version": 1}))
    with pytest.raises(QueueIdempotencyConflictError):
        store.enqueue(_job("same", payload={"version": 2}))


def test_retry_is_finite_and_fail_is_fenced_terminal(tmp_path: Path) -> None:
    store = SQLiteQueueStore(tmp_path / "queue.sqlite3")
    store.enqueue(_job("finite"))
    first = store.lease("race", lease_owner="worker-a", limit=1)[0]
    assert store.retry(first, "one", max_retries=2).status == "pending"
    second = store.lease("race", lease_owner="worker-b", limit=1)[0]
    terminal = store.retry(second, "two", max_retries=2)
    assert terminal.status == "dead_letter"
    assert terminal.retry_count == 2

    store.enqueue(_job("failed"))
    leased = store.lease("race", lease_owner="worker-c", limit=1)[0]
    failed = store.fail(leased, "boom")
    assert failed.status == "dead_letter"
    assert failed.retry_count == 1
