"""上下文数据库里的SQLite关系存储。"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from memoryos.contextdb.extensions import ContextDomainClassifier, NoDomainOverlay
from memoryos.contextdb.model.context_relation import ContextRelation


class SQLiteRelationStore:
    def __init__(
        self,
        path: str | Path,
        *,
        domain_classifier: ContextDomainClassifier | None = None,
    ) -> None:
        self.path = Path(path)
        self.domain_classifier = domain_classifier or NoDomainOverlay()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self._init_db()
        os.chmod(self.path, 0o600)

    def add_relation(self, relation: ContextRelation) -> None:
        tenant_id = str(relation.metadata.get("tenant_id", "default"))
        owner_user_id = str(relation.metadata.get("owner_user_id", ""))
        catalog_record_key = str(relation.metadata.get("catalog_record_key", ""))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO relations(
                  source_uri, relation_type, target_uri, tenant_id, owner_user_id,
                  catalog_record_key, weight, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, source_uri, relation_type, target_uri) DO UPDATE SET
                  owner_user_id=excluded.owner_user_id,
                  catalog_record_key=excluded.catalog_record_key,
                  weight=excluded.weight,
                  metadata_json=excluded.metadata_json
                """,
                (
                    relation.source_uri,
                    relation.relation_type,
                    relation.target_uri,
                    tenant_id,
                    owner_user_id,
                    catalog_record_key,
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
        limit: int | None = None,
    ) -> list[ContextRelation]:
        sql = "SELECT * FROM relations WHERE (source_uri = ? OR target_uri = ?)"
        params: list[str] = [uri, uri]
        if tenant_id is not None:
            sql += " AND tenant_id = ?"
            params.append(tenant_id)
        if owner_user_id is not None:
            sql += " AND (owner_user_id = ? OR owner_user_id = '')"
            params.append(owner_user_id)
        sql += " ORDER BY weight DESC, created_at DESC, source_uri, target_uri"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(str(max(0, int(limit))))
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

    def delete_relation(
        self,
        source_uri: str,
        relation_type: str,
        target_uri: str,
        *,
        tenant_id: str | None = None,
    ) -> None:
        with self._connect() as conn:
            if tenant_id is None:
                tenants = conn.execute(
                    "SELECT DISTINCT tenant_id FROM relations WHERE source_uri = ? "
                    "AND relation_type = ? AND target_uri = ? LIMIT 2",
                    (source_uri, relation_type, target_uri),
                ).fetchall()
                if len(tenants) > 1:
                    raise ValueError("tenant_id is required for an ambiguous relation identity")
                tenant_id = str(tenants[0]["tenant_id"]) if tenants else None
            if tenant_id is None:
                return
            conn.execute(
                "DELETE FROM relations WHERE tenant_id = ? AND source_uri = ? AND relation_type = ? AND target_uri = ?",
                (tenant_id, source_uri, relation_type, target_uri),
            )

    def delete_projection_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        catalog_record_key: str,
        limit: int,
    ) -> int:
        """Delete a bounded, ownership-filtered batch without relation overfetch.

        Legacy relations with no Catalog owner remain eligible so upgrades keep
        the previous cleanup behavior. Relations explicitly owned by another
        Catalog record are excluded in SQL before LIMIT, preventing a large
        unrelated prefix from hiding the projection's own edges.
        """

        maximum = max(1, min(int(limit), 1_000))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_uri, relation_type, target_uri
                FROM relations
                WHERE tenant_id = ?
                  AND (catalog_record_key = '' OR catalog_record_key = ?)
                  AND (source_uri = ? OR target_uri = ?)
                ORDER BY source_uri, relation_type, target_uri
                LIMIT ?
                """,
                (tenant_id, catalog_record_key, uri, uri, maximum),
            ).fetchall()
            conn.executemany(
                "DELETE FROM relations WHERE tenant_id = ? AND source_uri = ? AND relation_type = ? AND target_uri = ?",
                (
                    (tenant_id, str(row["source_uri"]), str(row["relation_type"]), str(row["target_uri"]))
                    for row in rows
                ),
            )
        return len(rows)

    def delete_uri_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        limit: int,
    ) -> int:
        """Delete a bounded tenant+URI batch after Catalog ownership is lost."""

        maximum = max(1, min(int(limit), 1_000))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_uri, relation_type, target_uri
                FROM relations
                WHERE tenant_id = ?
                  AND (source_uri = ? OR target_uri = ?)
                ORDER BY source_uri, relation_type, target_uri
                LIMIT ?
                """,
                (tenant_id, uri, uri, maximum),
            ).fetchall()
            conn.executemany(
                """
                DELETE FROM relations
                WHERE tenant_id = ? AND source_uri = ?
                  AND relation_type = ? AND target_uri = ?
                """,
                (
                    (tenant_id, str(row["source_uri"]), str(row["relation_type"]), str(row["target_uri"]))
                    for row in rows
                ),
            )
        return len(rows)

    def clear_ordinary_relations(self, *, tenant_id: str, limit: int) -> int:
        """Delete one bounded tenant batch while preserving domain-owned edges."""

        maximum = max(1, min(int(limit), 1_000))
        with self._connect() as conn:
            selected: list[sqlite3.Row] = []
            after = ("", "", "")
            page_size = max(64, maximum)
            while len(selected) < maximum:
                rows = conn.execute(
                    """
                    SELECT source_uri, relation_type, target_uri
                    FROM relations
                    WHERE tenant_id = ?
                      AND (source_uri, relation_type, target_uri) > (?, ?, ?)
                    ORDER BY source_uri, relation_type, target_uri
                    LIMIT ?
                    """,
                    (tenant_id, *after, page_size),
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    after = (
                        str(row["source_uri"]),
                        str(row["relation_type"]),
                        str(row["target_uri"]),
                    )
                    if self.domain_classifier.owns_uri(after[0]):
                        continue
                    selected.append(row)
                    if len(selected) >= maximum:
                        break
            conn.executemany(
                "DELETE FROM relations WHERE tenant_id = ? AND source_uri = ? AND relation_type = ? AND target_uri = ?",
                (
                    (tenant_id, str(row["source_uri"]), str(row["relation_type"]), str(row["target_uri"]))
                    for row in selected
                ),
            )
        return len(selected)

    def reconcile_ordinary_relations(
        self,
        relations: Sequence[ContextRelation],
        *,
        tenant_id: str,
    ) -> dict[str, int]:
        """Idempotently upsert one bounded, tenant-owned ordinary batch."""

        values = tuple(relations)
        if len(values) > 1_000:
            raise ValueError("ordinary relation reconcile batch exceeds 1000")
        prepared: dict[tuple[str, str, str], ContextRelation] = {}
        for relation in values:
            relation_tenant = str(relation.metadata.get("tenant_id") or "default")
            if relation_tenant != tenant_id:
                raise ValueError("ordinary relation tenant differs from reconcile tenant")
            if self.domain_classifier.owns_uri(relation.source_uri):
                raise ValueError("ordinary relation reconcile cannot mutate a canonical Source")
            if str(relation.metadata.get("catalog_record_key") or ""):
                raise ValueError("ordinary Source relation cannot claim Catalog projection ownership")
            identity = (relation.source_uri, relation.relation_type, relation.target_uri)
            existing = prepared.get(identity)
            if existing is not None and not self._ordinary_projection_equal(existing, relation):
                raise ValueError("ordinary relation batch contains a conflicting identity")
            prepared[identity] = relation

        written = 0
        skipped = 0
        with self._connect() as conn:
            for identity in sorted(prepared):
                relation = prepared[identity]
                existing = conn.execute(
                    "SELECT owner_user_id, catalog_record_key, weight, metadata_json "
                    "FROM relations WHERE tenant_id = ? AND source_uri = ? "
                    "AND relation_type = ? AND target_uri = ?",
                    (tenant_id, *identity),
                ).fetchone()
                if existing is not None:
                    try:
                        existing_metadata = json.loads(str(existing["metadata_json"] or "{}"))
                    except (TypeError, ValueError) as exc:
                        raise ValueError("ordinary RelationStore metadata is invalid") from exc
                    if (
                        str(existing["owner_user_id"] or "") == str(relation.metadata.get("owner_user_id") or "")
                        and not str(existing["catalog_record_key"] or "")
                        and float(existing["weight"]) == relation.weight
                        and existing_metadata == dict(relation.metadata or {})
                    ):
                        skipped += 1
                        continue
                conn.execute(
                    """
                    INSERT INTO relations(
                      source_uri, relation_type, target_uri, tenant_id, owner_user_id,
                      catalog_record_key, weight, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, '', ?, ?, ?)
                    ON CONFLICT(tenant_id, source_uri, relation_type, target_uri) DO UPDATE SET
                      owner_user_id=excluded.owner_user_id,
                      catalog_record_key='',
                      weight=excluded.weight,
                      metadata_json=excluded.metadata_json
                    """,
                    (
                        relation.source_uri,
                        relation.relation_type,
                        relation.target_uri,
                        tenant_id,
                        str(relation.metadata.get("owner_user_id") or ""),
                        relation.weight,
                        json.dumps(relation.metadata, ensure_ascii=False, sort_keys=True),
                        relation.created_at,
                    ),
                )
                written += 1
        return {"processed": len(prepared), "written": written, "skipped": skipped}

    @staticmethod
    def _ordinary_projection_equal(left: ContextRelation, right: ContextRelation) -> bool:
        return (
            left.source_uri == right.source_uri
            and left.relation_type == right.relation_type
            and left.target_uri == right.target_uri
            and left.weight == right.weight
            and dict(left.metadata or {}) == dict(right.metadata or {})
        )

    def all_relations(self) -> list[ContextRelation]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM relations").fetchall()
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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            self._create_relations_table(conn)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(relations)").fetchall()}
            if "tenant_id" not in columns:
                conn.execute("ALTER TABLE relations ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")
            if "owner_user_id" not in columns:
                conn.execute("ALTER TABLE relations ADD COLUMN owner_user_id TEXT NOT NULL DEFAULT ''")
            if "catalog_record_key" not in columns:
                conn.execute("ALTER TABLE relations ADD COLUMN catalog_record_key TEXT NOT NULL DEFAULT ''")
            primary_key = tuple(
                str(row["name"])
                for row in sorted(
                    (row for row in conn.execute("PRAGMA table_info(relations)").fetchall() if int(row["pk"]) > 0),
                    key=lambda row: int(row["pk"]),
                )
            )
            expected_primary_key = ("tenant_id", "source_uri", "relation_type", "target_uri")
            if primary_key != expected_primary_key:
                conn.execute("ALTER TABLE relations RENAME TO relations_legacy_identity")
                self._create_relations_table(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO relations(tenant_id, source_uri, relation_type, target_uri, "
                    "owner_user_id, catalog_record_key, weight, metadata_json, created_at) "
                    "SELECT tenant_id, source_uri, relation_type, target_uri, owner_user_id, "
                    "catalog_record_key, weight, metadata_json, created_at FROM relations_legacy_identity"
                )
                conn.execute("DROP TABLE relations_legacy_identity")
            self._migrate_catalog_record_keys(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relations_source_scope "
                "ON relations(source_uri, tenant_id, owner_user_id, weight DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relations_target_scope "
                "ON relations(target_uri, tenant_id, owner_user_id, weight DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relations_source_projection "
                "ON relations(source_uri, tenant_id, catalog_record_key)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relations_target_projection "
                "ON relations(target_uri, tenant_id, catalog_record_key)"
            )

    @staticmethod
    def _create_relations_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS relations (
                  source_uri TEXT NOT NULL,
                  relation_type TEXT NOT NULL,
                  target_uri TEXT NOT NULL,
                  tenant_id TEXT NOT NULL DEFAULT 'default',
                  owner_user_id TEXT NOT NULL DEFAULT '',
                  catalog_record_key TEXT NOT NULL DEFAULT '',
                  weight REAL NOT NULL,
                  metadata_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY(tenant_id, source_uri, relation_type, target_uri)
                )
            """
        )

    @staticmethod
    def _migrate_catalog_record_keys(conn: sqlite3.Connection) -> None:
        """Backfill the normalized owner from legacy metadata in bounded batches."""

        after_rowid = 0
        while True:
            rows = conn.execute(
                """
                SELECT rowid, metadata_json
                FROM relations
                WHERE rowid > ? AND catalog_record_key = ''
                ORDER BY rowid
                LIMIT 1000
                """,
                (after_rowid,),
            ).fetchall()
            if not rows:
                return
            updates: list[tuple[str, int]] = []
            for row in rows:
                after_rowid = int(row["rowid"])
                try:
                    metadata = json.loads(str(row["metadata_json"] or "{}"))
                except (TypeError, ValueError):
                    continue
                if not isinstance(metadata, dict):
                    continue
                catalog_record_key = str(metadata.get("catalog_record_key") or "")
                if catalog_record_key:
                    updates.append((catalog_record_key, after_rowid))
            conn.executemany(
                "UPDATE relations SET catalog_record_key = ? WHERE rowid = ? AND catalog_record_key = ''",
                updates,
            )


SqliteRelationStore = SQLiteRelationStore

__all__ = ["SQLiteRelationStore", "SqliteRelationStore"]
