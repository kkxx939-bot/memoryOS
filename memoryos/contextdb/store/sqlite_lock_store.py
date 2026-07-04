from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memoryos.contextdb.store.source_store import LockToken


class SQLiteLockStore:
    def __init__(self, path: str | Path, owner: str = "memoryos") -> None:
        self.path = Path(path)
        self.owner = owner
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def acquire(self, lock_key: str, ttl_seconds: int = 30) -> LockToken:
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
        token = uuid.uuid4().hex
        with self._connect() as conn:
            existing = conn.execute("SELECT * FROM locks WHERE lock_key = ?", (lock_key,)).fetchone()
            if existing and str(existing["expires_at"]) > now.isoformat():
                raise TimeoutError(f"Lock already held: {lock_key}")
            conn.execute(
                """
                INSERT INTO locks(lock_key, token, expires_at, owner, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(lock_key) DO UPDATE SET
                  token=excluded.token,
                  expires_at=excluded.expires_at,
                  owner=excluded.owner,
                  created_at=excluded.created_at
                """,
                (lock_key, token, expires_at, self.owner, now.isoformat()),
            )
        return LockToken(lock_key=lock_key, token=token)

    def release(self, token: LockToken) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM locks WHERE lock_key = ? AND token = ?", (token.lock_key, token.token))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS locks (
                  lock_key TEXT PRIMARY KEY,
                  token TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  owner TEXT NOT NULL,
                  created_at TEXT NOT NULL
                )
                """
            )


SqliteLockStore = SQLiteLockStore

__all__ = ["SQLiteLockStore", "SqliteLockStore"]
