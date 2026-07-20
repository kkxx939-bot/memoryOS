"""上下文数据库里的SQLite锁存储。"""

from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from infrastructure.store.contracts.lock import LockLostError, LockToken

_LOCK_TABLE_LAYOUT = (
    ("lock_key", "TEXT", 0, None, 1),
    ("token", "TEXT", 1, None, 0),
    ("expires_at", "TEXT", 1, None, 0),
    ("owner", "TEXT", 1, None, 0),
    ("created_at", "TEXT", 1, None, 0),
    ("fence", "INTEGER", 1, "0", 0),
)


class SQLiteLockStore:
    def __init__(
        self,
        path: str | Path,
        owner: str = "memoryos",
        *,
        sqlite_timeout_seconds: float = 5.0,
    ) -> None:
        self.path = Path(path)
        self.owner = owner
        self.sqlite_timeout_seconds = max(0.001, float(sqlite_timeout_seconds))
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self._init_db()
        os.chmod(self.path, 0o600)

    def acquire(self, lock_key: str, ttl_seconds: int = 30) -> LockToken:
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=max(1, ttl_seconds))).isoformat()
        token = uuid.uuid4().hex
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM locks WHERE lock_key = ?", (lock_key,)).fetchone()
            if existing and self._lease_active(str(existing["expires_at"]), now):
                conn.rollback()
                raise TimeoutError(f"Lock already held: {lock_key}")
            if existing is None:
                fence = 1
                conn.execute(
                    """
                    INSERT INTO locks(lock_key, token, expires_at, owner, created_at, fence)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (lock_key, token, expires_at, self.owner, now.isoformat(), fence),
                )
            else:
                fence = int(existing["fence"] or 0) + 1
                conn.execute(
                    """
                    UPDATE locks
                    SET token = ?, expires_at = ?, owner = ?, created_at = ?, fence = ?
                    WHERE lock_key = ?
                    """,
                    (token, expires_at, self.owner, now.isoformat(), fence, lock_key),
                )
            conn.commit()
        except sqlite3.OperationalError as exc:
            if conn.in_transaction:
                conn.rollback()
            if self._is_contention(exc):
                raise TimeoutError(f"Lock store busy while acquiring: {lock_key}") from exc
            raise
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()
        return LockToken(lock_key=lock_key, token=token, fence=fence)

    def renew(self, token: LockToken, ttl_seconds: int = 30) -> LockToken:
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=max(1, ttl_seconds))).isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT token, fence, expires_at FROM locks WHERE lock_key = ?",
                (token.lock_key,),
            ).fetchone()
            if (
                row is None
                or str(row["token"]) != token.token
                or int(row["fence"] or 0) != token.fence
                or not self._lease_active(str(row["expires_at"]), now)
            ):
                raise LockLostError(f"Lock lease lost: {token.lock_key}")
            conn.execute(
                "UPDATE locks SET expires_at = ? WHERE lock_key = ? AND token = ? AND fence = ?",
                (expires_at, token.lock_key, token.token, token.fence),
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            if conn.in_transaction:
                conn.rollback()
            if self._is_contention(exc):
                raise LockLostError(f"Lock store busy while renewing: {token.lock_key}") from exc
            raise
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()
        return token

    def assert_owned(self, token: LockToken) -> None:
        now = datetime.now(timezone.utc)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT token, fence, expires_at FROM locks WHERE lock_key = ?",
                    (token.lock_key,),
                ).fetchone()
        except sqlite3.OperationalError as exc:
            if self._is_contention(exc):
                raise LockLostError(f"Lock store busy while checking: {token.lock_key}") from exc
            raise
        if (
            row is None
            or str(row["token"]) != token.token
            or int(row["fence"] or 0) != token.fence
            or not self._lease_active(str(row["expires_at"]), now)
        ):
            raise LockLostError(f"Lock lease lost: {token.lock_key}")

    @contextmanager
    def fenced(self, tokens: Sequence[LockToken], ttl_seconds: int = 30) -> Iterator[None]:
        unique = {(token.lock_key, token.token, token.fence): token for token in tokens}
        if not unique:
            yield
            return
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = datetime.now(timezone.utc)
            for token in unique.values():
                row = conn.execute(
                    "SELECT token, fence, expires_at FROM locks WHERE lock_key = ?",
                    (token.lock_key,),
                ).fetchone()
                if (
                    row is None
                    or str(row["token"]) != token.token
                    or int(row["fence"] or 0) != token.fence
                    or not self._lease_active(str(row["expires_at"]), now)
                ):
                    raise LockLostError(f"Lock lease lost: {token.lock_key}")
            renewed_until = (now + timedelta(seconds=max(1, ttl_seconds))).isoformat()
            for token in unique.values():
                conn.execute(
                    "UPDATE locks SET expires_at = ? WHERE lock_key = ? AND token = ? AND fence = ?",
                    (renewed_until, token.lock_key, token.token, token.fence),
                )
            yield
            renewed_until = (datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds))).isoformat()
            for token in unique.values():
                conn.execute(
                    "UPDATE locks SET expires_at = ? WHERE lock_key = ? AND token = ? AND fence = ?",
                    (renewed_until, token.lock_key, token.token, token.fence),
                )
            conn.commit()
        except sqlite3.OperationalError as exc:
            if conn.in_transaction:
                conn.rollback()
            if self._is_contention(exc):
                keys = ",".join(sorted(token.lock_key for token in unique.values()))
                raise LockLostError(f"Lock store busy while fencing: {keys}") from exc
            raise
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

    def release(self, token: LockToken) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE locks SET token = '', owner = '', expires_at = ?
                WHERE lock_key = ? AND token = ? AND fence = ?
                """,
                (datetime.now(timezone.utc).isoformat(), token.lock_key, token.token, token.fence),
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            if conn.in_transaction:
                conn.rollback()
            if self._is_contention(exc):
                raise TimeoutError(f"Lock store busy while releasing: {token.lock_key}") from exc
            raise
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=self.sqlite_timeout_seconds)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            existing = conn.execute("SELECT type FROM sqlite_master WHERE name = 'locks'").fetchone()
            if existing is None:
                self._create_lock_table(conn)
            elif str(existing["type"]) != "table":
                raise RuntimeError("unsupported LockStore layout; reset the greenfield runtime")
            self._require_exact_lock_layout(conn)

    @staticmethod
    def _create_lock_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE locks (
              lock_key TEXT PRIMARY KEY,
              token TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              owner TEXT NOT NULL,
              created_at TEXT NOT NULL,
              fence INTEGER NOT NULL DEFAULT 0
            )
            """
        )

    @staticmethod
    def _require_exact_lock_layout(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(locks)").fetchall()
        layout = tuple(
            (
                str(row["name"]),
                str(row["type"]).upper(),
                int(row["notnull"]),
                row["dflt_value"],
                int(row["pk"]),
            )
            for row in rows
        )
        if layout != _LOCK_TABLE_LAYOUT:
            raise RuntimeError("unsupported LockStore layout; reset the greenfield runtime")

    def _lease_active(self, value: str, now: datetime) -> bool:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc) > now

    def _is_contention(self, exc: sqlite3.OperationalError) -> bool:
        code = getattr(exc, "sqlite_errorcode", None)
        busy_codes = {
            getattr(sqlite3, "SQLITE_BUSY", 5),
            getattr(sqlite3, "SQLITE_LOCKED", 6),
        }
        message = str(exc).casefold()
        return code in busy_codes or "database is locked" in message or "database table is locked" in message


SqliteLockStore = SQLiteLockStore

__all__ = ["SQLiteLockStore", "SqliteLockStore"]
