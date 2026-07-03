from __future__ import annotations

import sqlite3
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.source_store import IndexHit


class SqliteIndexStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def upsert_index(self, obj: ContextObject, content: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO context_index(uri, owner_user_id, context_type, title, content, score_hint)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(uri) DO UPDATE SET
                  owner_user_id=excluded.owner_user_id,
                  context_type=excluded.context_type,
                  title=excluded.title,
                  content=excluded.content,
                  score_hint=excluded.score_hint
                """,
                (obj.uri, obj.owner_user_id or "", obj.context_type.value, obj.title, content, obj.hotness),
            )

    def delete_index(self, uri: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM context_index WHERE uri = ?", (uri,))

    def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
        filters = filters or {}
        terms = [term.lower() for term in str(query).split() if term.strip()]
        sql = "SELECT * FROM context_index WHERE 1=1"
        params: list[str] = []
        if filters.get("owner_user_id"):
            sql += " AND owner_user_id = ?"
            params.append(str(filters["owner_user_id"]))
        if filters.get("context_type"):
            sql += " AND context_type = ?"
            params.append(str(filters["context_type"]))
        rows = []
        with self._connect() as conn:
            for row in conn.execute(sql, params):
                text = " ".join([row["title"], row["content"]]).lower()
                score = sum(1.0 for term in terms if term in text) + float(row["score_hint"])
                if not terms:
                    score = 0.1 + float(row["score_hint"])
                if score > 0:
                    rows.append(
                        IndexHit(
                            uri=row["uri"],
                            score=score,
                            context_type=row["context_type"],
                            title=row["title"],
                        )
                    )
        rows.sort(key=lambda item: item.score, reverse=True)
        return rows[:limit]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS context_index (
                    uri TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    context_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    score_hint REAL NOT NULL DEFAULT 0.0
                )
                """
            )


__all__ = ["SqliteIndexStore"]
