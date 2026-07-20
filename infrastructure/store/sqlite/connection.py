"""SQLite Catalog 的连接管理组件。"""

from __future__ import annotations

from infrastructure.store.sqlite._common import (
    _ONLINE_PROGRESS_GRANULARITY,
    Any,
    CatalogCandidateBoundExceeded,
    Sequence,
    sqlite3,
)


class SQLiteConnectionManager:
    """集中管理 SQLite Catalog 的连接和在线查询执行边界。"""

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
        """在固定 SQLite 虚拟机步数上限内执行一次在线服务查询。

        普通 Top-K 查询会依靠索引提前停止；该保护用于阻断恶意过滤组合，且不会把
        被中断的结果伪装成空成功。修复、审计和 keyset GC 使用各自不受此上限影响
        的离线管理入口。
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
