from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from infrastructure.store.contracts.lock import LockLostError, LockToken
from infrastructure.store.sqlite.lock_store import SQLiteLockStore


def test_sqlite_lock_store_acquire_release_and_expiry(tmp_path) -> None:
    store = SQLiteLockStore(tmp_path / "locks.sqlite3")
    token = store.acquire("user:u1", ttl_seconds=30)
    with pytest.raises(TimeoutError):
        store.acquire("user:u1", ttl_seconds=30)

    store.release(LockToken(lock_key="user:u1", token="wrong"))
    with pytest.raises(TimeoutError):
        store.acquire("user:u1", ttl_seconds=30)

    store.release(token)
    token2 = store.acquire("user:u1", ttl_seconds=30)
    assert token2.token != token.token

    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(tmp_path / "locks.sqlite3") as conn:
        conn.execute("UPDATE locks SET expires_at = ? WHERE lock_key = ?", (expired, "user:u1"))
    token3 = store.acquire("user:u1", ttl_seconds=30)
    assert token3.token != token2.token
    assert token3.fence > token2.fence > token.fence

    store.release(token2)
    with pytest.raises(TimeoutError):
        store.acquire("user:u1", ttl_seconds=30)
    with pytest.raises(LockLostError):
        store.renew(token2)


def test_sqlite_lock_store_concurrent_acquire_has_one_winner(tmp_path) -> None:
    path = tmp_path / "locks.sqlite3"
    stores = [SQLiteLockStore(path, owner=f"worker-{index}") for index in range(8)]
    barrier = threading.Barrier(len(stores))

    def acquire(store: SQLiteLockStore):  # noqa: ANN202
        barrier.wait(timeout=5)
        try:
            return store.acquire("shared", ttl_seconds=30)
        except TimeoutError:
            return None

    with ThreadPoolExecutor(max_workers=len(stores)) as pool:
        tokens = list(pool.map(acquire, stores))

    winners = [token for token in tokens if token is not None]
    assert len(winners) == 1
    stores[0].release(winners[0])


def test_sqlite_lock_store_fresh_schema_is_complete_and_exact(tmp_path) -> None:
    path = tmp_path / "locks.sqlite3"

    SQLiteLockStore(path)

    with sqlite3.connect(path) as conn:
        layout = tuple(
            (row[1], row[2], row[3], row[4], row[5])
            for row in conn.execute("PRAGMA table_info(locks)")
        )
    assert layout == (
        ("lock_key", "TEXT", 0, None, 1),
        ("token", "TEXT", 1, None, 0),
        ("expires_at", "TEXT", 1, None, 0),
        ("owner", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
        ("fence", "INTEGER", 1, "0", 0),
    )


def test_sqlite_lock_store_rejects_old_schema_without_alter(tmp_path) -> None:
    path = tmp_path / "locks.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE locks (
              lock_key TEXT PRIMARY KEY,
              token TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              owner TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
    with pytest.raises(RuntimeError, match="unsupported LockStore layout"):
        SQLiteLockStore(path)

    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(locks)").fetchall()}
    assert "fence" not in columns


def test_sqlite_lock_store_rejects_stale_token_from_fenced_section(tmp_path) -> None:
    path = tmp_path / "locks.sqlite3"
    store = SQLiteLockStore(path)
    stale = store.acquire("fenced-stale")
    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE locks SET expires_at = ? WHERE lock_key = ?",
            (expired, stale.lock_key),
        )
    current = store.acquire("fenced-stale")

    with pytest.raises(LockLostError):
        with store.fenced((stale,)):
            pytest.fail("stale fence must be rejected before entering the critical section")
    with store.fenced((current,)):
        store.assert_owned(current)


def test_sqlite_fenced_section_prevents_parallel_takeover(tmp_path) -> None:
    path = tmp_path / "locks.sqlite3"
    owner = SQLiteLockStore(path, owner="owner")
    competitor = SQLiteLockStore(path, owner="competitor")
    token = owner.acquire("fenced-critical", ttl_seconds=30)
    attempting = threading.Event()

    def compete() -> str:
        attempting.set()
        try:
            competitor.acquire("fenced-critical", ttl_seconds=30)
        except TimeoutError:
            return "blocked"
        return "acquired"

    with ThreadPoolExecutor(max_workers=1) as pool:
        with owner.fenced((token,), ttl_seconds=30):
            future = pool.submit(compete)
            assert attempting.wait(timeout=5)
            time.sleep(0.05)
            assert not future.done()
        assert future.result(timeout=5) == "blocked"


def test_sqlite_fenced_contention_is_reported_as_retryable_lock_timeout(tmp_path) -> None:
    path = tmp_path / "locks.sqlite3"
    owner = SQLiteLockStore(path, owner="owner", sqlite_timeout_seconds=0.05)
    competitor = SQLiteLockStore(path, owner="competitor", sqlite_timeout_seconds=0.05)
    token = owner.acquire("long-fenced-critical", ttl_seconds=30)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with owner.fenced((token,), ttl_seconds=30):
            future = pool.submit(competitor.acquire, "another-lock", 30)
            with pytest.raises(TimeoutError) as caught:
                future.result(timeout=1)
            assert isinstance(caught.value.__cause__, sqlite3.OperationalError)


def test_sqlite_release_contention_is_reported_as_retryable_timeout(tmp_path) -> None:
    path = tmp_path / "locks.sqlite3"
    first = SQLiteLockStore(path, owner="first", sqlite_timeout_seconds=0.05)
    second = SQLiteLockStore(path, owner="second", sqlite_timeout_seconds=0.05)
    first_token = first.acquire("first-key")
    second_token = second.acquire("second-key")

    with second.fenced((second_token,)):
        with pytest.raises(TimeoutError) as caught:
            first.release(first_token)
        assert isinstance(caught.value.__cause__, sqlite3.OperationalError)

    first.release(first_token)


def test_sqlite_renew_parses_legacy_z_expiry_instead_of_comparing_text(tmp_path) -> None:
    path = tmp_path / "locks.sqlite3"
    store = SQLiteLockStore(path)
    token = store.acquire("legacy-z")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE locks SET expires_at = ? WHERE lock_key = ?",
            ("2000-01-01T00:00:00Z", token.lock_key),
        )

    with pytest.raises(LockLostError):
        store.renew(token)
