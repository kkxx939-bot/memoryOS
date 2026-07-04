from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.source_store import IndexHit


class SQLiteIndexStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def upsert_index(self, obj: ContextObject, content: str = "") -> None:
        metadata_json = json.dumps(obj.metadata, ensure_ascii=False)
        metadata_text = " ".join(str(value) for value in obj.metadata.values())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO contexts(
                  uri, tenant_id, owner_user_id, context_type, title, lifecycle_state,
                  hotness, semantic_hotness, behavior_support_hotness,
                  metadata_json, content_text, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(uri) DO UPDATE SET
                  tenant_id=excluded.tenant_id,
                  owner_user_id=excluded.owner_user_id,
                  context_type=excluded.context_type,
                  title=excluded.title,
                  lifecycle_state=excluded.lifecycle_state,
                  hotness=excluded.hotness,
                  semantic_hotness=excluded.semantic_hotness,
                  behavior_support_hotness=excluded.behavior_support_hotness,
                  metadata_json=excluded.metadata_json,
                  content_text=excluded.content_text,
                  updated_at=excluded.updated_at
                """,
                (
                    obj.uri,
                    obj.tenant_id or "",
                    obj.owner_user_id or "",
                    obj.context_type.value,
                    obj.title,
                    obj.lifecycle_state.value,
                    obj.hotness,
                    obj.semantic_hotness,
                    obj.behavior_support_hotness,
                    metadata_json,
                    content,
                    obj.updated_at,
                ),
            )
            conn.execute("DELETE FROM contexts_fts WHERE uri = ?", (obj.uri,))
            conn.execute(
                "INSERT INTO contexts_fts(uri, title, content_text, metadata_text) VALUES (?, ?, ?, ?)",
                (obj.uri, obj.title, content, metadata_text),
            )

    def delete_index(self, uri: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM contexts WHERE uri = ?", (uri,))
            conn.execute("DELETE FROM contexts_fts WHERE uri = ?", (uri,))

    def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
        filters = filters or {}
        terms = [term.lower() for term in str(query).split() if term.strip()]
        sql = "SELECT c.*, f.title AS fts_title, f.content_text AS fts_content, f.metadata_text FROM contexts c JOIN contexts_fts f ON c.uri = f.uri WHERE 1=1"
        params: list[str] = []
        for field in ("tenant_id", "owner_user_id", "context_type", "lifecycle_state"):
            if filters.get(field) is not None:
                sql += f" AND c.{field} = ?"
                params.append(str(filters[field]))
        hits: list[IndexHit] = []
        with self._connect() as conn:
            for row in conn.execute(sql, params):
                haystack = " ".join([row["fts_title"], row["fts_content"], row["metadata_text"]]).lower()
                lexical = sum(1.0 for term in terms if term in haystack)
                if not terms:
                    lexical = 0.1
                score = lexical + float(row["hotness"]) + float(row["semantic_hotness"]) + float(row["behavior_support_hotness"])
                if score <= 0:
                    continue
                hits.append(
                    IndexHit(
                        uri=row["uri"],
                        score=score,
                        context_type=row["context_type"],
                        title=row["title"],
                        metadata=json.loads(row["metadata_json"] or "{}"),
                    )
                )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contexts (
                  uri TEXT PRIMARY KEY,
                  tenant_id TEXT NOT NULL,
                  owner_user_id TEXT NOT NULL,
                  context_type TEXT NOT NULL,
                  title TEXT NOT NULL,
                  lifecycle_state TEXT NOT NULL,
                  hotness REAL NOT NULL,
                  semantic_hotness REAL NOT NULL,
                  behavior_support_hotness REAL NOT NULL,
                  metadata_json TEXT NOT NULL,
                  content_text TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contexts_fts (
                  uri TEXT PRIMARY KEY,
                  title TEXT NOT NULL,
                  content_text TEXT NOT NULL,
                  metadata_text TEXT NOT NULL
                )
                """
            )


SqliteIndexStore = SQLiteIndexStore

__all__ = ["SQLiteIndexStore", "SqliteIndexStore"]
