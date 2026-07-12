from __future__ import annotations

import multiprocessing as mp
import queue
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from memoryos.contextdb.store.local_stores import InMemoryQueueStore
from memoryos.contextdb.store.source_store import (
    LeaseLostError,
    QueueIdempotencyConflictError,
    QueueJob,
)
from memoryos.contextdb.store.sqlite_queue_store import SQLiteQueueStore


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
        ctx.Process(target=_retry_at_barrier, args=(str(path), item, barrier, results))
        for item in (stale, current)
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


def test_old_queue_schema_is_migrated_without_deleting_database(tmp_path: Path) -> None:
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

    store = SQLiteQueueStore(path)
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(queue_jobs)")}
    assert {"lease_token", "lease_generation", "lease_owner"}.issubset(columns)
    leased = store.lease("race", lease_owner="migrated", limit=1)
    assert leased[0].job_id == "legacy"
    assert leased[0].lease_generation == 1


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
