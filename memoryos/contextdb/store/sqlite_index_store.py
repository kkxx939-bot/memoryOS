"""上下文数据库里的SQLite索引存储。"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.source_store import IndexHit

_SCOPE_KEY_SCHEMA_VERSION = 2
_INVALID_SCOPE_KEY = "__memoryos_invalid_scope__"


def lexical_terms(text: object) -> tuple[str, ...]:
    """Return deterministic complete Latin tokens and CJK character bigrams."""

    normalized = str(text).casefold()
    terms = re.findall(r"[a-z0-9_]+", normalized)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(sequence) == 1:
            terms.append(sequence)
        else:
            terms.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tuple(dict.fromkeys(term for term in terms if term))


def lexical_match_count(query: object, haystack: object) -> int:
    query_terms = lexical_terms(query)
    if not query_terms:
        return 0
    haystack_terms = set(lexical_terms(haystack))
    return sum(1 for term in query_terms if term in haystack_terms)


def lexical_relevance(query: object, haystack: object) -> float:
    query_terms = lexical_terms(query)
    if not query_terms:
        return 0.0
    return lexical_match_count(query, haystack) / len(query_terms)


class SQLiteIndexStore:
    """保存可检索元数据，并在截断结果前完成结构化过滤。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fts_enabled = True
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self._init_db()
        os.chmod(self.path, 0o600)

    def upsert_index(self, obj: ContextObject, content: str = "") -> None:
        """写入检索文本以及租户、用户、状态和作用域等过滤字段。"""

        metadata_json = json.dumps(obj.metadata, ensure_ascii=False)
        metadata_text = " ".join(str(value) for value in obj.metadata.values())
        scope = self._mapping(obj.metadata.get("scope", {}))
        fields = self._mapping(obj.metadata.get("fields", {}))
        connect = self._mapping(obj.metadata.get("connect", {}))
        admission = self._mapping(obj.metadata.get("admission", {}))
        applicability = self._mapping(scope.get("applicability", {}))
        scope_keys = self._scope_keys_from_metadata(obj.metadata)
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
        sql = (
            " AND NOT EXISTS (SELECT 1 FROM json_each("
            "CASE WHEN json_valid(c.scope_keys) THEN "
            "CASE WHEN json_type(c.scope_keys) = 'array' THEN c.scope_keys "
            "ELSE '[\"__memoryos_invalid_scope__\"]' END "
            "ELSE '[\"__memoryos_invalid_scope__\"]' END"
            ") WHERE value = ?)"
        )
        params: list[str] = [_INVALID_SCOPE_KEY]
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
            values = list(value) if isinstance(value, list | tuple | set | frozenset) else [value]
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
        hits = []
        for row in rows:
            haystack = " ".join([row["fts_title"], row["fts_content"], row["metadata_text"]])
            lexical = self._lexical_relevance(
                query,
                haystack,
            )
            if lexical > 0:
                hits.append(
                    self._hit_from_row(
                        row,
                        lexical=lexical,
                        lexical_rank=self._lexical_match_count(query, haystack),
                    )
                )
        return hits

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
            metadata = self._mapping(json.loads(row["metadata_json"] or "{}"))
            if any(str(metadata.get(field, "")) == value for field in exact_fields):
                hits.append(self._hit_from_row(row, identity=1.0, identity_rank=10.0))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _search_contains(self, query: str, filters: dict, limit: int) -> list[IndexHit]:
        filter_sql, params = self._base_filter_sql(filters)
        sql = f"SELECT c.*, f.title AS fts_title, f.content_text AS fts_content, f.metadata_text FROM contexts c JOIN contexts_fts f ON c.uri = f.uri WHERE 1=1 {filter_sql}"
        hits: list[IndexHit] = []
        with self._connect() as conn:
            for row in conn.execute(sql, params):
                haystack = " ".join([row["fts_title"], row["fts_content"], row["metadata_text"]]).lower()
                lexical = self._lexical_relevance(query, haystack)
                if lexical <= 0:
                    continue
                hits.append(
                    self._hit_from_row(
                        row,
                        lexical=lexical,
                        lexical_rank=self._lexical_match_count(query, haystack),
                    )
                )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _hit_from_row(
        self,
        row: sqlite3.Row,
        lexical: float = 0.0,
        lexical_rank: float | None = None,
        vector: float = 0.0,
        identity: float = 0.0,
        identity_rank: float | None = None,
    ) -> IndexHit:
        metadata = self._mapping(json.loads(row["metadata_json"] or "{}"))
        metadata["retrieval_scores"] = self._score_components(
            row,
            lexical=lexical,
            lexical_rank=lexical_rank,
            vector=vector,
            identity=identity,
            identity_rank=identity_rank,
        )
        return IndexHit(
            uri=row["uri"],
            score=float(metadata["retrieval_scores"]["score"]),
            context_type=row["context_type"],
            title=row["title"],
            metadata=metadata,
        )

    def _score_components(
        self,
        row: sqlite3.Row,
        *,
        lexical: float = 0.0,
        lexical_rank: float | None = None,
        vector: float = 0.0,
        identity: float = 0.0,
        identity_rank: float | None = None,
    ) -> dict[str, float]:
        lexical = self._bounded(lexical)
        vector = self._bounded(vector)
        identity = self._bounded(identity)
        base_relevance = max(lexical, vector, identity)
        hotness = (
            self._bounded(row["hotness"])
            + self._bounded(row["semantic_hotness"])
            + self._bounded(row["behavior_support_hotness"])
        ) / 3.0
        # Hotness is deliberately a small tie-breaker after a real relevance
        # signal. It can never turn a zero-relevance row into a hit.
        ranking_relevance = max(
            self._finite_rank(lexical_rank if lexical_rank is not None else lexical),
            vector,
            self._finite_rank(identity_rank if identity_rank is not None else identity),
        )
        score = ranking_relevance + (0.05 * hotness if base_relevance > 0 else 0.0)
        return {
            "lexical": lexical,
            "vector": vector,
            "identity": identity,
            "base_relevance": base_relevance,
            "hotness": hotness,
            "score": score,
        }

    def _lexical_relevance(self, query: str, haystack: str) -> float:
        return lexical_relevance(query, haystack)

    def _lexical_match_count(self, query: str, haystack: str) -> float:
        return float(lexical_match_count(query, haystack))

    def _lexical_terms(self, query: str) -> tuple[str, ...]:
        return lexical_terms(query)

    def _bounded(self, value: Any) -> float:
        if isinstance(value, bool):
            return 0.0
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(number):
            return 0.0
        return max(0.0, min(1.0, number))

    def _finite_rank(self, value: Any) -> float:
        if isinstance(value, bool):
            return 0.0
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(number) or number < 0:
            return 0.0
        return number

    def _mapping(self, value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    def _scope_keys_from_metadata(self, metadata: Mapping[str, Any]) -> list[str]:
        # Imported lazily because context stores are loaded before the canonical
        # memory package during SDK initialization.
        from memoryos.memory.canonical.scope import MemoryScope, scope_key_from_payload

        if metadata.get("canonical_kind") in {"claim", "slot", "pending_proposal"}:
            raw_scope = metadata.get("scope")
            if not isinstance(raw_scope, Mapping):
                raise ValueError("canonical scope must be an object")
            canonical_scope = MemoryScope.from_dict(raw_scope)
            if canonical_scope.canonical_subject is None:
                raise ValueError("canonical scope requires a subject")
            return [scope.key for scope in canonical_scope.applicability.all_of]

        raw_scope = metadata.get("scope")
        if raw_scope is None:
            return []
        if not isinstance(raw_scope, Mapping):
            raise ValueError("scope must be an object")
        scope = dict(raw_scope)
        raw_applicability = scope.get("applicability")
        if raw_applicability is None:
            return []
        if not isinstance(raw_applicability, Mapping):
            raise ValueError("scope applicability must be an object")
        applicability = dict(raw_applicability)
        items = applicability.get("all_of", [])
        if not isinstance(items, list | tuple) or any(not isinstance(item, Mapping) for item in items):
            raise ValueError("scope applicability must contain scope objects")
        return list(dict.fromkeys(scope_key_from_payload(item) for item in items))

    def _migrate_scope_keys(self, conn: sqlite3.Connection) -> None:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version >= _SCOPE_KEY_SCHEMA_VERSION:
            return
        for row in conn.execute("SELECT uri, metadata_json FROM contexts").fetchall():
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
                if not isinstance(metadata, Mapping):
                    raise ValueError("metadata must be an object")
                keys = self._scope_keys_from_metadata(metadata)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                keys = [_INVALID_SCOPE_KEY]
            conn.execute(
                "UPDATE contexts SET scope_keys = ? WHERE uri = ?",
                (json.dumps(keys, ensure_ascii=False), str(row["uri"])),
            )
        conn.execute(f"PRAGMA user_version = {_SCOPE_KEY_SCHEMA_VERSION}")

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
            conn.execute(
                "UPDATE contexts SET scope_keys = '[\"__memoryos_invalid_scope__\"]' "
                "WHERE NOT json_valid(scope_keys) OR json_type(scope_keys) != 'array'"
            )
            self._migrate_scope_keys(conn)
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
