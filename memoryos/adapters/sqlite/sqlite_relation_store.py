from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memoryos.contextdb.model.context_relation import ContextRelation


class SqliteRelationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_relation(self, relation: ContextRelation) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO context_relations(source_uri, relation_type, target_uri, weight, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    relation.source_uri,
                    relation.relation_type,
                    relation.target_uri,
                    relation.weight,
                    json.dumps(relation.metadata, ensure_ascii=False),
                ),
            )

    def relations_of(self, uri: str) -> list[ContextRelation]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM context_relations WHERE source_uri = ? OR target_uri = ?",
                (uri, uri),
            ).fetchall()
        return [
            ContextRelation(
                source_uri=row["source_uri"],
                relation_type=row["relation_type"],
                target_uri=row["target_uri"],
                weight=float(row["weight"]),
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS context_relations (
                    source_uri TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    target_uri TEXT NOT NULL,
                    weight REAL NOT NULL,
                    metadata TEXT NOT NULL,
                    PRIMARY KEY(source_uri, relation_type, target_uri)
                )
                """
            )


__all__ = ["SqliteRelationStore"]
