from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memoryos.contextdb.model.context_relation import ContextRelation


class SQLiteRelationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_relation(self, relation: ContextRelation) -> None:
        tenant_id = str(relation.metadata.get("tenant_id", "default"))
        owner_user_id = str(relation.metadata.get("owner_user_id", ""))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO relations(source_uri, relation_type, target_uri, tenant_id, owner_user_id, weight, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_uri, relation_type, target_uri) DO UPDATE SET
                  tenant_id=excluded.tenant_id,
                  owner_user_id=excluded.owner_user_id,
                  weight=excluded.weight,
                  metadata_json=excluded.metadata_json
                """,
                (
                    relation.source_uri,
                    relation.relation_type,
                    relation.target_uri,
                    tenant_id,
                    owner_user_id,
                    relation.weight,
                    json.dumps(relation.metadata, ensure_ascii=False),
                    relation.created_at,
                ),
            )

    def relations_of(
        self,
        uri: str,
        *,
        tenant_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> list[ContextRelation]:
        sql = "SELECT * FROM relations WHERE (source_uri = ? OR target_uri = ?)"
        params: list[str] = [uri, uri]
        if tenant_id is not None:
            sql += " AND tenant_id = ?"
            params.append(tenant_id)
        if owner_user_id is not None:
            sql += " AND (owner_user_id = ? OR owner_user_id = '')"
            params.append(owner_user_id)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            ContextRelation(
                source_uri=row["source_uri"],
                relation_type=row["relation_type"],
                target_uri=row["target_uri"],
                weight=float(row["weight"]),
                metadata=json.loads(row["metadata_json"] or "{}"),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete_relation(self, source_uri: str, relation_type: str, target_uri: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM relations WHERE source_uri = ? AND relation_type = ? AND target_uri = ?",
                (source_uri, relation_type, target_uri),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS relations (
                  source_uri TEXT NOT NULL,
                  relation_type TEXT NOT NULL,
                  target_uri TEXT NOT NULL,
                  tenant_id TEXT NOT NULL DEFAULT 'default',
                  owner_user_id TEXT NOT NULL DEFAULT '',
                  weight REAL NOT NULL,
                  metadata_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY(source_uri, relation_type, target_uri)
                )
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(relations)").fetchall()}
            if "tenant_id" not in columns:
                conn.execute("ALTER TABLE relations ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")
            if "owner_user_id" not in columns:
                conn.execute("ALTER TABLE relations ADD COLUMN owner_user_id TEXT NOT NULL DEFAULT ''")


SqliteRelationStore = SQLiteRelationStore

__all__ = ["SQLiteRelationStore", "SqliteRelationStore"]
