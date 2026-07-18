"""SQLite catalog SQLiteConnectionManager responsibility component."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _ONLINE_PROGRESS_GRANULARITY,
    Any,
    CatalogCandidateBoundExceeded,
    Sequence,
    sqlite3,
)


class SQLiteConnectionManager:
    """Own one stable subset of SQLite catalog behavior."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._store.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _online_fetchall(
        self,
        conn: sqlite3.Connection,
        sql: str,
        params: Sequence[Any],
    ) -> list[sqlite3.Row]:
        """Execute one serving query under a fixed SQLite VM-step ceiling.

        Indexes make normal Top-K queries stop early.  This final guard covers
        adversarial combinations (for example, a long run of newer expired
        records before an older valid record) without returning a truncated
        result as an empty success. Repair, audit, and keyset GC deliberately
        use their separate unguarded administrative methods.
        """

        limit = max(_ONLINE_PROGRESS_GRANULARITY, int(self._store.online_vm_step_limit))
        ticks = 0
        interrupted = False

        def progress() -> int:
            nonlocal ticks, interrupted
            ticks += _ONLINE_PROGRESS_GRANULARITY
            if ticks >= limit:
                interrupted = True
                return 1
            return 0

        conn.set_progress_handler(progress, _ONLINE_PROGRESS_GRANULARITY)
        try:
            return conn.execute(sql, list(params)).fetchall()
        except sqlite3.OperationalError as exc:
            if interrupted and "interrupted" in str(exc).casefold():
                raise CatalogCandidateBoundExceeded(f"online Catalog query exceeded {limit} SQLite VM steps") from exc
            raise
        finally:
            conn.set_progress_handler(None, 0)

__all__ = ["SQLiteConnectionManager"]
