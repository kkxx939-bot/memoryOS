from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from memoryos.contextdb.store.source_store import LockToken
from memoryos.contextdb.store.sqlite_lock_store import SQLiteLockStore


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

