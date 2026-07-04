from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.source_store import IndexHit


class SQLiteIndexStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fts_enabled = True
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
            self._delete_fts(conn, obj.uri)
            conn.execute("INSERT INTO contexts_fts(uri, title, content_text, metadata_text) VALUES (?, ?, ?, ?)", (obj.uri, obj.title, content, metadata_text))

    def delete_index(self, uri: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM contexts WHERE uri = ?", (uri,))
            self._delete_fts(conn, uri)

    def indexed_uris(self) -> list[str]:
        with self._connect() as conn:
            return [str(row["uri"]) for row in conn.execute("SELECT uri FROM contexts").fetchall()]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM contexts")
            conn.execute("DELETE FROM contexts_fts")

    def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
        filters = filters or {}
        hits = self._search_fts(query, filters, limit) if self.fts_enabled and str(query).strip() else []
        if hits:
            return hits[:limit]
        return self._search_contains(query, filters, limit)

    def _base_filter_sql(self, filters: dict) -> tuple[str, list[str]]:
        sql = ""
        params: list[str] = []
        for field in ("tenant_id", "owner_user_id", "context_type", "lifecycle_state"):
            if filters.get(field) is not None:
                sql += f" AND c.{field} = ?"
                params.append(str(filters[field]))
        if filters.get("lifecycle_state") is None:
            sql += " AND c.lifecycle_state != ?"
            params.append("deleted")
        return sql, params

    def _search_fts(self, query: str, filters: dict, limit: int) -> list[IndexHit]:
        filter_sql, params = self._base_filter_sql(filters)
        match_query = self._match_query(query)
        sql = f"""
            SELECT c.*, contexts_fts.title AS fts_title, contexts_fts.content_text AS fts_content, contexts_fts.metadata_text, bm25(contexts_fts) AS rank
            FROM contexts_fts
            JOIN contexts c ON c.uri = contexts_fts.uri
            WHERE contexts_fts MATCH ? {filter_sql}
            ORDER BY rank
            LIMIT ?
        """
        with self._connect() as conn:
            try:
                rows = conn.execute(sql, [match_query, *params, limit]).fetchall()
            except sqlite3.OperationalError:
                return []
        return [self._hit_from_row(row, lexical=max(0.0, -float(row["rank"]))) for row in rows]

    def _search_contains(self, query: str, filters: dict, limit: int) -> list[IndexHit]:
        terms = [term.lower() for term in str(query).split() if term.strip()]
        filter_sql, params = self._base_filter_sql(filters)
        sql = f"SELECT c.*, f.title AS fts_title, f.content_text AS fts_content, f.metadata_text FROM contexts c JOIN contexts_fts f ON c.uri = f.uri WHERE 1=1 {filter_sql}"
        hits: list[IndexHit] = []
        with self._connect() as conn:
            for row in conn.execute(sql, params):
                haystack = " ".join([row["fts_title"], row["fts_content"], row["metadata_text"]]).lower()
                lexical = sum(1.0 for term in terms if term in haystack)
                if not terms:
                    lexical = 0.1
                score = self._score(row, lexical)
                if score <= 0:
                    continue
                hits.append(self._hit_from_row(row, lexical=lexical))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _hit_from_row(self, row: sqlite3.Row, lexical: float) -> IndexHit:
        return IndexHit(
            uri=row["uri"],
            score=self._score(row, lexical),
            context_type=row["context_type"],
            title=row["title"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def _score(self, row: sqlite3.Row, lexical: float) -> float:
        return lexical + float(row["hotness"]) + float(row["semantic_hotness"]) + float(row["behavior_support_hotness"])

    def _match_query(self, query: str) -> str:
        terms = [term.replace('"', "") for term in str(query).split() if term.strip()]
        return " OR ".join(terms) if terms else str(query).replace('"', "")

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
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS contexts_fts USING fts5(
                      uri UNINDEXED,
                      title,
                      content_text,
                      metadata_text
                    )
                    """
                )
                self.fts_enabled = True
            except sqlite3.OperationalError:
                self.fts_enabled = False
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

    def _delete_fts(self, conn: sqlite3.Connection, uri: str) -> None:
        conn.execute("DELETE FROM contexts_fts WHERE uri = ?", (uri,))


SqliteIndexStore = SQLiteIndexStore

__all__ = ["SQLiteIndexStore", "SqliteIndexStore"]
