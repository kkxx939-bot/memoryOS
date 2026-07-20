"""可重建统一 Context Catalog 使用的 SQLite Schema。"""

from __future__ import annotations

from infrastructure.store.sqlite._common import (
    _CATALOG_SCHEMA_VERSION,
    _CONTEXT_COLUMNS,
    Any,
    sqlite3,
)
from infrastructure.store.sqlite.schema_definitions import (
    _AUXILIARY_TABLE_COLUMNS,
    _AUXILIARY_TABLE_PRIMARY_KEYS,
    _AUXILIARY_TABLE_UNIQUE_IDENTITIES,
)


class SchemaManager:
    """创建并校验唯一受支持的 Catalog 布局。"""

    def __init__(self, store: Any) -> None:
        self._store = store

    def _init_db(self) -> None:
        with self._store._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            existing = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'contexts'").fetchone()
            if existing is None:
                self._create_contexts_table(conn, "contexts")
            else:
                self._validate_contexts_table(conn)
            self._create_auxiliary_tables(conn)
            self._validate_auxiliary_tables(conn)
            fts_created = self._ensure_fts_table(conn)
            self._create_indexes(conn)
            if not fts_created and not self._fts_row_map_is_consistent(conn):
                self._rebuild_fts(conn)
            conn.execute(f"PRAGMA user_version = {_CATALOG_SCHEMA_VERSION}")

    @staticmethod
    def _validate_contexts_table(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(contexts)").fetchall()
        columns = {str(row[1]) for row in rows}
        primary = tuple(str(row[1]) for row in sorted(rows, key=lambda item: int(item[5])) if int(row[5]) > 0)
        if columns != set(_CONTEXT_COLUMNS) or primary != ("tenant_id", "record_key"):
            raise RuntimeError("unsupported Catalog layout; reset the greenfield runtime")

    def _create_contexts_table(self, conn: sqlite3.Connection, table_name: str) -> None:
        conn.execute(
            f"""
            CREATE TABLE {table_name} (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              uri TEXT NOT NULL,
              owner_user_id TEXT NOT NULL DEFAULT '',
              project_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              workspace_shared INTEGER NOT NULL DEFAULT 0 CHECK(workspace_shared IN (0, 1)),
              session_id TEXT NOT NULL DEFAULT '',
              adapter_id TEXT NOT NULL DEFAULT '',
              context_type TEXT NOT NULL DEFAULT '',
              source_kind TEXT NOT NULL DEFAULT '',
              record_kind TEXT NOT NULL DEFAULT 'context',
              lifecycle_state TEXT NOT NULL DEFAULT 'active',
              scope_keys TEXT NOT NULL DEFAULT '[]',
              scope_signature TEXT NOT NULL DEFAULT '',
              parent_uri TEXT NOT NULL DEFAULT '',
              primary_tree_path TEXT NOT NULL DEFAULT '',
              path_depth INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL DEFAULT '',
              event_time TEXT NOT NULL DEFAULT '',
              ingested_at TEXT NOT NULL DEFAULT '',
              transaction_time TEXT NOT NULL DEFAULT '',
              title TEXT NOT NULL DEFAULT '',
              l0_text TEXT NOT NULL DEFAULT '',
              l1_text TEXT NOT NULL DEFAULT '',
              l2_uri TEXT NOT NULL DEFAULT '',
              source_uri TEXT NOT NULL DEFAULT '',
              source_digest TEXT NOT NULL DEFAULT '',
              source_revision INTEGER NOT NULL DEFAULT 0,
              document_id TEXT NOT NULL DEFAULT '',
              block_id TEXT NOT NULL DEFAULT '',
              document_kind TEXT NOT NULL DEFAULT '',
              document_revision INTEGER NOT NULL DEFAULT 0,
              projection_generation INTEGER NOT NULL DEFAULT 0,
              projection_effect_hash TEXT NOT NULL DEFAULT '',
              hotness REAL NOT NULL DEFAULT 0,
              semantic_hotness REAL NOT NULL DEFAULT 0,
              behavior_support_hotness REAL NOT NULL DEFAULT 0,
              serving_tier TEXT NOT NULL DEFAULT 'HOT',
              projection_status TEXT NOT NULL DEFAULT 'PROJECTED',
              metadata_json TEXT NOT NULL DEFAULT '{{}}',
              content_digest TEXT NOT NULL DEFAULT '',
              stored_content_digest TEXT NOT NULL DEFAULT '',
              content_text TEXT NOT NULL DEFAULT '',
              scene_key TEXT NOT NULL DEFAULT '',
              action TEXT NOT NULL DEFAULT '',
              support_anchor_uri TEXT NOT NULL DEFAULT '',
              PRIMARY KEY (tenant_id, record_key)
            )
            """
        )

    def _create_auxiliary_tables(self, conn: sqlite3.Connection) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS context_paths (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              uri TEXT NOT NULL,
              owner_user_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              workspace_shared INTEGER NOT NULL DEFAULT 0 CHECK(workspace_shared IN (0, 1)),
              context_type TEXT NOT NULL DEFAULT '',
              record_kind TEXT NOT NULL DEFAULT 'context',
              document_id TEXT NOT NULL DEFAULT '',
              document_kind TEXT NOT NULL DEFAULT '',
              event_time TEXT NOT NULL DEFAULT '',
              transaction_time TEXT NOT NULL DEFAULT '',
              path TEXT NOT NULL,
              path_kind TEXT NOT NULL,
              depth INTEGER NOT NULL,
              is_primary INTEGER NOT NULL CHECK(is_primary IN (0, 1)),
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (tenant_id, record_key, path)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS context_acl_grants (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              grant_kind TEXT NOT NULL,
              grant_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              scope_signature TEXT NOT NULL DEFAULT '',
              uri TEXT NOT NULL DEFAULT '',
              context_type TEXT NOT NULL DEFAULT '',
              source_kind TEXT NOT NULL DEFAULT '',
              record_kind TEXT NOT NULL DEFAULT 'context',
              adapter_id TEXT NOT NULL DEFAULT '',
              adapter_access_id TEXT NOT NULL DEFAULT '',
              session_id TEXT NOT NULL DEFAULT '',
              event_time TEXT NOT NULL DEFAULT '',
              transaction_time TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL,
              PRIMARY KEY (tenant_id, record_key, grant_kind, grant_id, workspace_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS context_path_closure (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              path TEXT NOT NULL,
              ancestor_path TEXT NOT NULL,
              owner_user_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              workspace_shared INTEGER NOT NULL DEFAULT 0 CHECK(workspace_shared IN (0, 1)),
              scope_signature TEXT NOT NULL DEFAULT '',
              uri TEXT NOT NULL DEFAULT '',
              context_type TEXT NOT NULL DEFAULT '',
              source_kind TEXT NOT NULL DEFAULT '',
              record_kind TEXT NOT NULL DEFAULT 'context',
              adapter_id TEXT NOT NULL DEFAULT '',
              adapter_access_id TEXT NOT NULL DEFAULT '',
              session_id TEXT NOT NULL DEFAULT '',
              document_id TEXT NOT NULL DEFAULT '',
              document_kind TEXT NOT NULL DEFAULT '',
              event_time TEXT NOT NULL DEFAULT '',
              transaction_time TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL DEFAULT '',
              PRIMARY KEY (tenant_id, record_key, path, ancestor_path)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS context_path_acl (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              path TEXT NOT NULL,
              ancestor_path TEXT NOT NULL,
              grant_kind TEXT NOT NULL,
              grant_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              owner_user_id TEXT NOT NULL DEFAULT '',
              scope_signature TEXT NOT NULL DEFAULT '',
              uri TEXT NOT NULL DEFAULT '',
              context_type TEXT NOT NULL DEFAULT '',
              source_kind TEXT NOT NULL DEFAULT '',
              record_kind TEXT NOT NULL DEFAULT 'context',
              adapter_id TEXT NOT NULL DEFAULT '',
              adapter_access_id TEXT NOT NULL DEFAULT '',
              session_id TEXT NOT NULL DEFAULT '',
              event_time TEXT NOT NULL DEFAULT '',
              transaction_time TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL DEFAULT '',
              PRIMARY KEY (
                tenant_id, record_key, path, ancestor_path,
                grant_kind, grant_id, workspace_id
              )
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS context_fts_map (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              fts_rowid INTEGER NOT NULL UNIQUE,
              PRIMARY KEY (tenant_id, record_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS context_links (
              tenant_id TEXT NOT NULL,
              link_key TEXT NOT NULL,
              source_record_key TEXT NOT NULL,
              source_uri TEXT NOT NULL,
              relation_type TEXT NOT NULL,
              target_record_key TEXT NOT NULL DEFAULT '',
              target_uri TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (tenant_id, link_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS context_projection_state (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              source_revision INTEGER NOT NULL DEFAULT 0,
              projection_status TEXT NOT NULL,
              projection_effect_hash TEXT NOT NULL DEFAULT '',
              retry_count INTEGER NOT NULL DEFAULT 0,
              last_error TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL,
              PRIMARY KEY (tenant_id, record_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS context_tombstones (
              tenant_id TEXT NOT NULL,
              tombstone_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              uri TEXT NOT NULL DEFAULT '',
              reason TEXT NOT NULL,
              source_revision INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL,
              payload_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              retry_count INTEGER NOT NULL DEFAULT 0,
              last_error TEXT NOT NULL DEFAULT '',
              PRIMARY KEY (tenant_id, tombstone_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS context_projection_journal (
              tenant_id TEXT NOT NULL,
              projector_kind TEXT NOT NULL,
              source_uri TEXT NOT NULL,
              owner_user_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              source_id TEXT NOT NULL DEFAULT '',
              source_digest TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL,
              last_error TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (tenant_id, projector_kind, source_uri)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_document_projection_state (
              tenant_id TEXT NOT NULL,
              owner_user_id TEXT NOT NULL,
              document_id TEXT NOT NULL,
              relative_path TEXT NOT NULL,
              source_digest TEXT NOT NULL,
              projection_generation INTEGER NOT NULL,
              projection_status TEXT NOT NULL,
              projected_at TEXT NOT NULL,
              last_error TEXT NOT NULL DEFAULT '',
              deletion_generation INTEGER NOT NULL DEFAULT 0,
              deletion_event_digest TEXT NOT NULL DEFAULT '',
              deletion_status TEXT NOT NULL DEFAULT '',
              PRIMARY KEY (tenant_id, owner_user_id, document_id),
              UNIQUE (tenant_id, owner_user_id, relative_path)
            )
            """,
        )
        for statement in statements:
            conn.execute(statement)

    @staticmethod
    def _validate_auxiliary_tables(conn: sqlite3.Connection) -> None:
        for table_name, expected_columns in _AUXILIARY_TABLE_COLUMNS.items():
            rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            columns = frozenset(str(row[1]) for row in rows)
            primary = tuple(str(row[1]) for row in sorted(rows, key=lambda item: int(item[5])) if int(row[5]) > 0)
            unique_identities = frozenset(
                tuple(
                    str(column[0])
                    for column in conn.execute(
                        "SELECT name FROM pragma_index_info(?) ORDER BY seqno",
                        (str(index[0]),),
                    ).fetchall()
                )
                for index in conn.execute(
                    "SELECT name FROM pragma_index_list(?) WHERE [unique] = 1 AND origin = 'u'",
                    (table_name,),
                ).fetchall()
            )
            expected_unique = _AUXILIARY_TABLE_UNIQUE_IDENTITIES.get(
                table_name,
                frozenset(),
            )
            if (
                columns != expected_columns
                or primary != _AUXILIARY_TABLE_PRIMARY_KEYS[table_name]
                or unique_identities != expected_unique
            ):
                raise RuntimeError(
                    f"unsupported Catalog auxiliary layout for {table_name}; reset the greenfield runtime"
                )

    def _ensure_fts_table(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'contexts_fts' AND type = 'table'").fetchone()
        desired = {
            "tenant_id",
            "record_key",
            "uri",
            "title",
            "content_text",
            "metadata_text",
            "search_terms",
            "acl_tokens",
        }
        if row is not None:
            columns = {str(item[1]) for item in conn.execute("PRAGMA table_info(contexts_fts)").fetchall()}
            if columns != desired:
                raise RuntimeError("unsupported Catalog FTS layout; reset the greenfield runtime")
            self._store.fts_enabled = "VIRTUAL TABLE" in str(row["sql"] or "").upper()
            return False
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE contexts_fts USING fts5(
                  tenant_id UNINDEXED,
                  record_key UNINDEXED,
                  uri UNINDEXED,
                  title,
                  content_text,
                  metadata_text,
                  search_terms,
                  acl_tokens
                )
                """
            )
            self._store.fts_enabled = True
        except sqlite3.OperationalError:
            self._store.fts_enabled = False
            conn.execute(
                """
                CREATE TABLE contexts_fts (
                  tenant_id TEXT NOT NULL,
                  record_key TEXT NOT NULL,
                  uri TEXT NOT NULL,
                  title TEXT NOT NULL,
                  content_text TEXT NOT NULL,
                  metadata_text TEXT NOT NULL,
                  search_terms TEXT NOT NULL,
                  acl_tokens TEXT NOT NULL,
                  PRIMARY KEY (tenant_id, record_key)
                )
                """
            )
        return True

    @staticmethod
    def _fts_row_map_is_consistent(conn: sqlite3.Connection) -> bool:
        unmapped = conn.execute(
            "SELECT 1 FROM contexts_fts AS f LEFT JOIN context_fts_map AS m "
            "ON m.fts_rowid = f.rowid AND m.tenant_id = f.tenant_id AND m.record_key = f.record_key "
            "WHERE m.record_key IS NULL LIMIT 1"
        ).fetchone()
        if unmapped is not None:
            return False
        orphaned = conn.execute(
            "SELECT 1 FROM context_fts_map AS m LEFT JOIN contexts_fts AS f ON f.rowid = m.fts_rowid "
            "WHERE f.rowid IS NULL OR f.tenant_id != m.tenant_id OR f.record_key != m.record_key LIMIT 1"
        ).fetchone()
        if orphaned is not None:
            return False
        ineligible = conn.execute(
            "SELECT 1 FROM context_fts_map AS m LEFT JOIN contexts AS c "
            "ON c.tenant_id = m.tenant_id AND c.record_key = m.record_key "
            "WHERE c.record_key IS NULL OR c.serving_tier NOT IN ('HOT', 'WARM') "
            "OR c.projection_status NOT IN ('PROJECTED', 'DEGRADED') "
            "OR c.lifecycle_state IN ('deleted', 'archived', 'obsolete') LIMIT 1"
        ).fetchone()
        if ineligible is not None:
            return False
        missing = conn.execute(
            "SELECT 1 FROM contexts AS c LEFT JOIN context_fts_map AS m "
            "ON m.tenant_id = c.tenant_id AND m.record_key = c.record_key "
            "WHERE c.serving_tier IN ('HOT', 'WARM') "
            "AND c.projection_status IN ('PROJECTED', 'DEGRADED') "
            "AND c.lifecycle_state NOT IN ('deleted', 'archived', 'obsolete') "
            "AND m.record_key IS NULL LIMIT 1"
        ).fetchone()
        return missing is None

    def _rebuild_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM contexts_fts")
        conn.execute("DELETE FROM context_fts_map")
        rows = conn.execute("SELECT * FROM contexts ORDER BY tenant_id, record_key").fetchall()
        for row in rows:
            record = self._store._catalog_record_from_row(conn, row)
            item = self._store._prepare_record(
                record,
                value_overrides={
                    "content_text": str(row["content_text"]),
                    "content_digest": str(row["content_digest"]),
                    "stored_content_digest": str(row["stored_content_digest"]),
                },
            )
            self._store._replace_fts(conn, item)

    @staticmethod
    def _create_indexes(conn: sqlite3.Connection) -> None:
        statements = (
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_uri ON contexts(tenant_id, uri, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_owner_updated ON contexts(tenant_id, owner_user_id, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_owner_type_updated ON contexts(tenant_id, owner_user_id, context_type, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_workspace_updated ON contexts(tenant_id, workspace_id, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_session_updated ON contexts(tenant_id, session_id, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_record_kind_updated ON contexts(tenant_id, record_kind, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_document ON contexts(tenant_id, owner_user_id, document_id, projection_generation, source_digest, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_document_id ON contexts(tenant_id, document_id, record_kind, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_block_id ON contexts(tenant_id, block_id, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_document_kind ON contexts(tenant_id, owner_user_id, document_kind, record_kind, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_event_time ON contexts(tenant_id, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_transaction_time ON contexts(tenant_id, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_scene_key ON contexts(tenant_id, scene_key, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_action ON contexts(tenant_id, action, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_support_anchor ON contexts(tenant_id, support_anchor_uri, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_projection_evidence ON contexts(tenant_id, source_uri, projection_effect_hash, record_key)",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_document_projection ON contexts(tenant_id, owner_user_id, document_id) WHERE record_kind = 'memory_document'",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_block_projection ON contexts(tenant_id, owner_user_id, document_id, block_id) WHERE record_kind = 'memory_block'",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_context_paths_primary ON context_paths(tenant_id, record_key) WHERE is_primary = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_tenant_path ON context_paths(tenant_id, path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_path ON context_paths(tenant_id, owner_user_id, path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_ancestor ON context_path_closure(tenant_id, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_ancestor ON context_path_closure(tenant_id, owner_user_id, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_lookup ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_lookup ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_links_source ON context_links(tenant_id, source_record_key, relation_type)",
            "CREATE INDEX IF NOT EXISTS idx_context_links_target ON context_links(tenant_id, target_record_key, relation_type)",
            "CREATE INDEX IF NOT EXISTS idx_context_tombstones_status ON context_tombstones(tenant_id, status, updated_at, tombstone_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_tombstones_uri ON context_tombstones(tenant_id, uri, status, tombstone_id)",
            "CREATE INDEX IF NOT EXISTS idx_projection_journal_status ON context_projection_journal(tenant_id, projector_kind, status, updated_at, source_uri)",
            "CREATE INDEX IF NOT EXISTS idx_memory_document_projection_status ON memory_document_projection_state(tenant_id, owner_user_id, projection_status, projected_at, document_id)",
        )
        for statement in statements:
            conn.execute(statement)


__all__ = ["SchemaManager"]
