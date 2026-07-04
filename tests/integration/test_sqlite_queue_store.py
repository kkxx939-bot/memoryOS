from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from memoryos.contextdb.store.source_store import QueueJob
from memoryos.contextdb.store.sqlite_queue_store import SQLiteQueueStore


def test_sqlite_queue_store_lease_ack_and_expired_retry(tmp_path) -> None:
    store = SQLiteQueueStore(tmp_path / "queue.sqlite3")
    store.enqueue(QueueJob(job_id="j1", queue_name="semantic", action="refresh", target_uri="memoryos://user/u1/sessions/s1"))

    first = store.lease("semantic", limit=1)
    assert [job.job_id for job in first] == ["j1"]
    assert store.lease("semantic", limit=1) == []

    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        conn.execute("UPDATE queue_jobs SET leased_until = ? WHERE job_id = ?", (expired, "j1"))

    assert [job.job_id for job in store.lease("semantic", limit=1)] == ["j1"]
    store.ack("j1")
    assert store.lease("semantic", limit=1) == []


def test_sqlite_queue_store_fail_records_retry(tmp_path) -> None:
    store = SQLiteQueueStore(tmp_path / "queue.sqlite3")
    store.enqueue(QueueJob(job_id="j2", queue_name="embedding", action="embed", target_uri="memoryos://user/u1/memories/a"))
    store.fail("j2", "boom")

    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        status, retry_count, last_error = conn.execute("SELECT status, retry_count, last_error FROM queue_jobs WHERE job_id = 'j2'").fetchone()
    assert (status, retry_count, last_error) == ("failed", 1, "boom")

