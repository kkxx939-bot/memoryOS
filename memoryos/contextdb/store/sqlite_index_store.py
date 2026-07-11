"""上下文数据库里的SQLite索引存储。"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.source_store import IndexHit


class SQLiteIndexStore:
    """保存可检索元数据，并在截断结果前完成结构化过滤。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fts_enabled = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def upsert_index(self, obj: ContextObject, content: str = "") -> None:
        """写入检索文本以及租户、用户、状态和作用域等过滤字段。"""

        metadata_json = json.dumps(obj.metadata, ensure_ascii=False)
        metadata_text = " ".join(str(value) for value in obj.metadata.values())
        scope = dict(obj.metadata.get("scope", {}) or {})
        fields = dict(obj.metadata.get("fields", {}) or {})
        connect = dict(obj.metadata.get("connect", {}) or {})
        admission = dict(obj.metadata.get("admission", {}) or {})
        applicability = dict(scope.get("applicability", {}) or {})
        scope_keys = [
            f"{item.get('namespace', 'memoryos')}:{item.get('kind')}:{item.get('id')}"
            for item in applicability.get("all_of", []) or []
            if isinstance(item, dict) and item.get("kind") and item.get("id")
        ]
        workspace = next(
            (str(item.get("id")) for item in applicability.get("all_of", []) or [] if isinstance(item, dict) and item.get("kind") == "workspace"),
            "",
        )
        project_id = str(scope.get("project_id") or fields.get("project_id") or obj.metadata.get("project_id") or workspace)
        adapter_id = str(connect.get("adapter_id") or obj.metadata.get("source_adapter_id") or "")
        admission_status = str(admission.get("decision") or "")
        claim_state = str(obj.metadata.get("state") or obj.metadata.get("claim_state") or "")
        slot_id = str(obj.metadata.get("slot_id") or "")
        memory_type = str(obj.metadata.get("memory_type") or "")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO contexts(
                  uri, tenant_id, owner_user_id, context_type, project_id, adapter_id, admission_status, claim_state,
                  slot_id, memory_type, scope_keys, title, lifecycle_state,
                  hotness, semantic_hotness, behavior_support_hotness,
                  metadata_json, content_text, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(uri) DO UPDATE SET
                  tenant_id=excluded.tenant_id,
                  owner_user_id=excluded.owner_user_id,
                  context_type=excluded.context_type,
                  project_id=excluded.project_id,
                  adapter_id=excluded.adapter_id,
                  admission_status=excluded.admission_status,
                  claim_state=excluded.claim_state,
                  slot_id=excluded.slot_id,
                  memory_type=excluded.memory_type,
                  scope_keys=excluded.scope_keys,
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
                    project_id,
                    adapter_id,
                    admission_status,
                    claim_state,
                    slot_id,
                    memory_type,
                    json.dumps(scope_keys, ensure_ascii=False),
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
        """删除可重建的索引记录，不碰权威源数据。"""

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
        """按给定条件查找匹配结果。"""

        filters = filters or {}
        exact_hits = self._search_metadata_exact(query, filters, limit)
        if len(exact_hits) >= limit:
            return exact_hits[:limit]
        hits = self._search_fts(query, filters, limit) if self.fts_enabled and str(query).strip() else []
        if not hits:
            hits = self._search_contains(query, filters, limit)
        merged: dict[str, IndexHit] = {hit.uri: hit for hit in exact_hits}
        for hit in hits:
            if hit.uri not in merged:
                merged[hit.uri] = hit
        return list(merged.values())[:limit]

    def _base_filter_sql(self, filters: dict) -> tuple[str, list[str]]:
        sql = ""
        params: list[str] = []
        for field in (
            "tenant_id",
            "owner_user_id",
            "context_type",
            "adapter_id",
            "admission_status",
            "lifecycle_state",
            "claim_state",
            "slot_id",
            "memory_type",
        ):
            value = filters.get(field)
            if value is None:
                continue
            values = list(value) if isinstance(value, (list, tuple, set, frozenset)) else [value]
            sql += f" AND c.{field} IN ({','.join('?' for _ in values)})"
            params.extend(str(item) for item in values)
        if filters.get("project_id") is not None:
            sql += " AND (c.project_id = ? OR (c.project_id = '' AND c.memory_type NOT IN (?, ?, ?)))"
            params.append(str(filters["project_id"]))
            params.extend(["project_rule", "project_decision", "agent_experience"])
        allowed_uris = list(filters.get("allowed_uris", []) or [])
        if allowed_uris:
            sql += f" AND c.uri IN ({','.join('?' for _ in allowed_uris)})"
            params.extend(str(uri) for uri in allowed_uris)
        elif "allowed_uris" in filters:
            sql += " AND 1 = 0"
        if filters.get("admission_status") is None:
            sql += " AND c.admission_status NOT IN (?, ?, ?, ?)"
            params.extend(["pending", "restricted", "archive_only", "reject"])
        if filters.get("lifecycle_state") is None:
            sql += " AND c.lifecycle_state NOT IN (?, ?, ?)"
            params.extend(["deleted", "archived", "obsolete"])
        available_scopes = list(filters.get("applicability_scope_keys", []) or [])
        if available_scopes:
            sql += (
                " AND NOT EXISTS (SELECT 1 FROM json_each(CASE WHEN json_valid(c.scope_keys) THEN c.scope_keys ELSE '[]' END) "
                f"WHERE value NOT IN ({','.join('?' for _ in available_scopes)}))"
            )
            params.extend(str(scope_key) for scope_key in available_scopes)
        return sql, params

    def _search_fts(self, query: str, filters: dict, limit: int) -> list[IndexHit]:
        filter_sql, params = self._base_filter_sql(filters)
        match_query = self._match_query(query)
        if not match_query:
            return []
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

    def _search_metadata_exact(self, query: str, filters: dict, limit: int) -> list[IndexHit]:
        value = str(query).strip()
        if not value:
            return []
        filter_sql, params = self._base_filter_sql(filters)
        sql = f"SELECT c.*, f.title AS fts_title, f.content_text AS fts_content, f.metadata_text FROM contexts c JOIN contexts_fts f ON c.uri = f.uri WHERE 1=1 {filter_sql} AND (c.metadata_json LIKE ? OR f.metadata_text LIKE ?)"
        like = f"%{value}%"
        hits: list[IndexHit] = []
        with self._connect() as conn:
            rows = conn.execute(sql, [*params, like, like]).fetchall()
        exact_fields = {"scene_key", "action", "memory_anchor_uri"}
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            if any(str(metadata.get(field, "")) == value for field in exact_fields):
                hits.append(self._hit_from_row(row, lexical=10.0))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

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
        terms = re.findall(r"[\w\u4e00-\u9fff]+", str(query), flags=re.UNICODE)
        escaped = [f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in terms if term.strip()]
        return " OR ".join(escaped)

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
                  project_id TEXT NOT NULL DEFAULT '',
                  adapter_id TEXT NOT NULL DEFAULT '',
                  admission_status TEXT NOT NULL DEFAULT '',
                  claim_state TEXT NOT NULL DEFAULT '',
                  slot_id TEXT NOT NULL DEFAULT '',
                  memory_type TEXT NOT NULL DEFAULT '',
                  scope_keys TEXT NOT NULL DEFAULT '[]',
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
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(contexts)").fetchall()}
            for name in ("project_id", "adapter_id", "admission_status", "claim_state", "slot_id", "memory_type", "scope_keys"):
                if name not in columns:
                    default = "'[]'" if name == "scope_keys" else "''"
                    conn.execute(f"ALTER TABLE contexts ADD COLUMN {name} TEXT NOT NULL DEFAULT {default}")
            conn.execute("UPDATE contexts SET scope_keys = '[]' WHERE NOT json_valid(scope_keys)")
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
