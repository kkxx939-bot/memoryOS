"""Greenfield tenant-qualified SQLite relation serving store."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

from memoryos.contextdb.extensions import ContextDomainClassifier, NoDomainOverlay
from memoryos.contextdb.model.context_relation import ContextRelation

_RELATION_COLUMNS = frozenset(
    {
        "tenant_id",
        "source_uri",
        "relation_type",
        "target_uri",
        "owner_user_id",
        "catalog_record_key",
        "weight",
        "metadata_json",
        "created_at",
    }
)
_RELATION_PRIMARY_KEY = ("tenant_id", "source_uri", "relation_type", "target_uri")


class SQLiteRelationStore:
    """Store rebuildable relation edges without any tenant-free identity."""

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

    @staticmethod
    def _require_tenant(tenant_id: str) -> str:
        resolved = str(tenant_id or "").strip()
        if not resolved:
            raise ValueError("tenant_id is required")
        return resolved

    def add_relation(self, relation: ContextRelation, *, tenant_id: str) -> None:
        resolved_tenant = self._require_tenant(tenant_id)
        metadata = self._metadata_for_tenant(relation.metadata, resolved_tenant)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO relations(
                  tenant_id, source_uri, relation_type, target_uri, owner_user_id,
                  catalog_record_key, weight, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, source_uri, relation_type, target_uri) DO UPDATE SET
                  owner_user_id=excluded.owner_user_id,
                  catalog_record_key=excluded.catalog_record_key,
                  weight=excluded.weight,
                  metadata_json=excluded.metadata_json,
                  created_at=excluded.created_at
                """,
                (
                    resolved_tenant,
                    relation.source_uri,
                    relation.relation_type,
                    relation.target_uri,
                    str(metadata.get("owner_user_id") or ""),
                    str(metadata.get("catalog_record_key") or ""),
                    relation.weight,
                    self._json_dump(metadata),
                    relation.created_at,
                ),
            )

    def relations_of(
        self,
        uri: str,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
        limit: int | None = None,
    ) -> list[ContextRelation]:
        resolved_tenant = self._require_tenant(tenant_id)
        sql = (
            "SELECT * FROM relations WHERE tenant_id = ? "
            "AND (source_uri = ? OR target_uri = ?)"
        )
        params: list[object] = [resolved_tenant, str(uri), str(uri)]
        if owner_user_id is not None:
            sql += " AND (owner_user_id = ? OR owner_user_id = '')"
            params.append(str(owner_user_id))
        sql += " ORDER BY weight DESC, created_at DESC, source_uri, relation_type, target_uri"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, min(int(limit), 1_000)))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._relation_from_row(row) for row in rows]

    def delete_relation(
        self,
        source_uri: str,
        relation_type: str,
        target_uri: str,
        *,
        tenant_id: str,
    ) -> None:
        resolved_tenant = self._require_tenant(tenant_id)
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM relations WHERE tenant_id = ? AND source_uri = ? "
                "AND relation_type = ? AND target_uri = ?",
                (resolved_tenant, str(source_uri), str(relation_type), str(target_uri)),
            )

    def delete_projection_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        catalog_record_key: str,
        limit: int,
    ) -> int:
        """Delete a bounded batch owned by one exact Catalog projection."""

        resolved_tenant = self._require_tenant(tenant_id)
        resolved_record_key = str(catalog_record_key or "").strip()
        if not resolved_record_key:
            raise ValueError("catalog_record_key is required")
        maximum = max(1, min(int(limit), 1_000))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT source_uri, relation_type, target_uri
                FROM relations
                WHERE tenant_id = ? AND catalog_record_key = ?
                  AND (source_uri = ? OR target_uri = ?)
                ORDER BY source_uri, relation_type, target_uri
                LIMIT ?
                """,
                (resolved_tenant, resolved_record_key, str(uri), str(uri), maximum),
            ).fetchall()
            self._delete_rows(conn, resolved_tenant, rows)
        return len(rows)

    def delete_memory_document_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        owner_user_id: str,
        limit: int,
    ) -> int:
        """Delete one bounded exact-owner batch touching a memory document URI."""

        resolved_tenant = self._require_tenant(tenant_id)
        resolved_owner = str(owner_user_id or "").strip()
        if not resolved_owner:
            raise ValueError("owner_user_id is required")
        maximum = max(1, min(int(limit), 1_000))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT source_uri, relation_type, target_uri
                FROM relations
                WHERE tenant_id = ? AND owner_user_id = ?
                  AND (source_uri = ? OR target_uri = ?)
                ORDER BY source_uri, relation_type, target_uri
                LIMIT ?
                """,
                (resolved_tenant, resolved_owner, str(uri), str(uri), maximum),
            ).fetchall()
            self._delete_rows(conn, resolved_tenant, rows)
        return len(rows)

    def delete_uri_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        limit: int,
    ) -> int:
        """Delete a bounded tenant-local URI batch after ownership is gone."""

        resolved_tenant = self._require_tenant(tenant_id)
        maximum = max(1, min(int(limit), 1_000))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT source_uri, relation_type, target_uri
                FROM relations
                WHERE tenant_id = ? AND (source_uri = ? OR target_uri = ?)
                ORDER BY source_uri, relation_type, target_uri
                LIMIT ?
                """,
                (resolved_tenant, str(uri), str(uri), maximum),
            ).fetchall()
            self._delete_rows(conn, resolved_tenant, rows)
        return len(rows)

    def clear_ordinary_relations(self, *, tenant_id: str, limit: int) -> int:
        """Delete one bounded tenant batch while preserving domain-owned edges."""

        resolved_tenant = self._require_tenant(tenant_id)
        maximum = max(1, min(int(limit), 1_000))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
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
                    (resolved_tenant, *after, page_size),
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
            self._delete_rows(conn, resolved_tenant, selected)
        return len(selected)

    def reconcile_ordinary_relations(
        self,
        relations: Sequence[ContextRelation],
        *,
        tenant_id: str,
    ) -> dict[str, int]:
        """Idempotently upsert one bounded tenant-owned ordinary batch."""

        resolved_tenant = self._require_tenant(tenant_id)
        values = tuple(relations)
        if len(values) > 1_000:
            raise ValueError("ordinary relation reconcile batch exceeds 1000")
        prepared: dict[tuple[str, str, str], tuple[ContextRelation, dict[str, object]]] = {}
        for relation in values:
            metadata = self._metadata_for_tenant(relation.metadata, resolved_tenant)
            if self.domain_classifier.owns_uri(relation.source_uri):
                raise ValueError("ordinary relation reconcile cannot mutate a domain-owned Source")
            if str(metadata.get("catalog_record_key") or ""):
                raise ValueError("ordinary Source relation cannot claim Catalog projection ownership")
            identity = (relation.source_uri, relation.relation_type, relation.target_uri)
            prior = prepared.get(identity)
            if prior is not None and not self._ordinary_projection_equal(
                prior[0],
                relation,
                left_metadata=prior[1],
                right_metadata=metadata,
            ):
                raise ValueError("ordinary relation batch contains a conflicting identity")
            prepared[identity] = (relation, metadata)

        written = 0
        skipped = 0
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for identity in sorted(prepared):
                relation, metadata = prepared[identity]
                existing = conn.execute(
                    "SELECT owner_user_id, catalog_record_key, weight, metadata_json, created_at "
                    "FROM relations WHERE tenant_id = ? AND source_uri = ? "
                    "AND relation_type = ? AND target_uri = ?",
                    (resolved_tenant, *identity),
                ).fetchone()
                if existing is not None:
                    existing_metadata = self._json_mapping(existing["metadata_json"])
                    if (
                        str(existing["owner_user_id"] or "") == str(metadata.get("owner_user_id") or "")
                        and not str(existing["catalog_record_key"] or "")
                        and float(existing["weight"]) == relation.weight
                        and existing_metadata == metadata
                        and str(existing["created_at"]) == relation.created_at
                    ):
                        skipped += 1
                        continue
                conn.execute(
                    """
                    INSERT INTO relations(
                      tenant_id, source_uri, relation_type, target_uri, owner_user_id,
                      catalog_record_key, weight, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, '', ?, ?, ?)
                    ON CONFLICT(tenant_id, source_uri, relation_type, target_uri) DO UPDATE SET
                      owner_user_id=excluded.owner_user_id,
                      catalog_record_key='',
                      weight=excluded.weight,
                      metadata_json=excluded.metadata_json,
                      created_at=excluded.created_at
                    """,
                    (
                        resolved_tenant,
                        *identity,
                        str(metadata.get("owner_user_id") or ""),
                        relation.weight,
                        self._json_dump(metadata),
                        relation.created_at,
                    ),
                )
                written += 1
        return {"processed": len(prepared), "written": written, "skipped": skipped}

    def all_relations(self, *, tenant_id: str) -> list[ContextRelation]:
        resolved_tenant = self._require_tenant(tenant_id)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM relations WHERE tenant_id = ? "
                "ORDER BY source_uri, relation_type, target_uri",
                (resolved_tenant,),
            ).fetchall()
        return [self._relation_from_row(row) for row in rows]

    @staticmethod
    def _ordinary_projection_equal(
        left: ContextRelation,
        right: ContextRelation,
        *,
        left_metadata: Mapping[str, object] | None = None,
        right_metadata: Mapping[str, object] | None = None,
    ) -> bool:
        return (
            left.source_uri == right.source_uri
            and left.relation_type == right.relation_type
            and left.target_uri == right.target_uri
            and left.weight == right.weight
            and dict(left_metadata or left.metadata or {}) == dict(right_metadata or right.metadata or {})
            and left.created_at == right.created_at
        )

    @staticmethod
    def _metadata_for_tenant(metadata: Mapping[str, object], tenant_id: str) -> dict[str, object]:
        result = dict(metadata or {})
        supplied = str(result.get("tenant_id") or "")
        if supplied and supplied != tenant_id:
            raise ValueError("relation tenant differs from explicit tenant_id")
        result["tenant_id"] = tenant_id
        return result

    @staticmethod
    def _delete_rows(
        conn: sqlite3.Connection,
        tenant_id: str,
        rows: Sequence[sqlite3.Row],
    ) -> None:
        conn.executemany(
            "DELETE FROM relations WHERE tenant_id = ? AND source_uri = ? "
            "AND relation_type = ? AND target_uri = ?",
            (
                (
                    tenant_id,
                    str(row["source_uri"]),
                    str(row["relation_type"]),
                    str(row["target_uri"]),
                )
                for row in rows
            ),
        )

    def _relation_from_row(self, row: sqlite3.Row) -> ContextRelation:
        return ContextRelation(
            source_uri=str(row["source_uri"]),
            relation_type=str(row["relation_type"]),
            target_uri=str(row["target_uri"]),
            weight=float(row["weight"]),
            metadata=self._json_mapping(row["metadata_json"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _json_dump(value: Mapping[str, object]) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _json_mapping(value: object) -> dict[str, object]:
        try:
            decoded = json.loads(str(value or "{}"))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("RelationStore metadata is invalid") from exc
        if not isinstance(decoded, dict):
            raise ValueError("RelationStore metadata is invalid")
        return decoded

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'relations'"
            ).fetchone()
            if row is None:
                self._create_relations_table(conn)
            else:
                info = conn.execute("PRAGMA table_info(relations)").fetchall()
                columns = {str(item["name"]) for item in info}
                primary_key = tuple(
                    str(item["name"])
                    for item in sorted(info, key=lambda candidate: int(candidate["pk"]))
                    if int(item["pk"]) > 0
                )
                if columns != _RELATION_COLUMNS or primary_key != _RELATION_PRIMARY_KEY:
                    raise RuntimeError("unsupported RelationStore layout; reset the greenfield runtime")
            for statement in (
                "CREATE INDEX IF NOT EXISTS idx_relations_tenant_source "
                "ON relations(tenant_id, source_uri, owner_user_id, weight DESC, target_uri)",
                "CREATE INDEX IF NOT EXISTS idx_relations_tenant_target "
                "ON relations(tenant_id, target_uri, owner_user_id, weight DESC, source_uri)",
                "CREATE INDEX IF NOT EXISTS idx_relations_tenant_source_projection "
                "ON relations(tenant_id, source_uri, catalog_record_key)",
                "CREATE INDEX IF NOT EXISTS idx_relations_tenant_target_projection "
                "ON relations(tenant_id, target_uri, catalog_record_key)",
            ):
                conn.execute(statement)

    @staticmethod
    def _create_relations_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE relations (
              tenant_id TEXT NOT NULL,
              source_uri TEXT NOT NULL,
              relation_type TEXT NOT NULL,
              target_uri TEXT NOT NULL,
              owner_user_id TEXT NOT NULL DEFAULT '',
              catalog_record_key TEXT NOT NULL DEFAULT '',
              weight REAL NOT NULL,
              metadata_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              PRIMARY KEY (tenant_id, source_uri, relation_type, target_uri)
            )
            """
        )


SqliteRelationStore = SQLiteRelationStore

__all__ = ["SQLiteRelationStore", "SqliteRelationStore"]
