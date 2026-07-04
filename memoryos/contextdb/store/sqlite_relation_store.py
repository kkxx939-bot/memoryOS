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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO relations(source_uri, relation_type, target_uri, weight, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_uri, relation_type, target_uri) DO UPDATE SET
                  weight=excluded.weight,
                  metadata_json=excluded.metadata_json
                """,
                (
                    relation.source_uri,
                    relation.relation_type,
                    relation.target_uri,
                    relation.weight,
                    json.dumps(relation.metadata, ensure_ascii=False),
                    relation.created_at,
                ),
            )

    def relations_of(self, uri: str) -> list[ContextRelation]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM relations WHERE source_uri = ? OR target_uri = ?",
                (uri, uri),
            ).fetchall()
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
                  weight REAL NOT NULL,
                  metadata_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY(source_uri, relation_type, target_uri)
                )
                """
            )


SqliteRelationStore = SQLiteRelationStore

__all__ = ["SQLiteRelationStore", "SqliteRelationStore"]
