"""SQLite catalog SchemaManager responsibility component."""

from __future__ import annotations

from memoryos.adapters.persistence.sqlite._common import (
    _ALTER_COLUMN_DEFINITIONS,
    _CATALOG_SCHEMA_VERSION,
    Any,
    sqlite3,
)


class SchemaManager:
    """Own one stable subset of SQLite catalog behavior."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def _init_db(self) -> None:
        with self._store._connect() as conn:
            initial_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            had_existing_schema = (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
                ).fetchone()
                is not None
            )
            requires_unified_backfill = initial_version < _CATALOG_SCHEMA_VERSION and (
                initial_version > 0 or had_existing_schema
            )
            conn.execute("PRAGMA journal_mode = WAL")
            rebuilt = self._store._ensure_catalog_table(conn)
            self._store._create_auxiliary_tables(conn)
            self._store._migrate_scope_keys(conn)
            fts_recreated = self._store._ensure_fts_table(conn)
            fts_map_consistent = bool(
                initial_version >= _CATALOG_SCHEMA_VERSION
                and not fts_recreated
                and self._store._fts_row_map_is_consistent(conn)
            )
            if initial_version < _CATALOG_SCHEMA_VERSION and not rebuilt:
                self._store._sanitize_existing_rows(conn)
            self._store._create_indexes(conn)
            if rebuilt or fts_recreated or not fts_map_consistent:
                self._store._rebuild_fts(conn)
            conn.execute(f"PRAGMA user_version = {_CATALOG_SCHEMA_VERSION}")
            if requires_unified_backfill:
                self._store._record_unified_catalog_schema_upgrade(
                    conn,
                    upgraded_from_schema_version=initial_version,
                )

    def _ensure_catalog_table(self, conn: sqlite3.Connection) -> bool:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'contexts'").fetchone()
        if exists is None:
            self._store._create_contexts_table(conn, "contexts")
            return False
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(contexts)").fetchall()}
        if "record_key" not in columns:
            self._store._rebuild_legacy_contexts(conn, columns)
            return True
        for name, definition in _ALTER_COLUMN_DEFINITIONS.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE contexts ADD COLUMN {name} {definition}")
        return False

    def _create_contexts_table(self, conn: sqlite3.Connection, table_name: str) -> None:
        conn.execute(
            f"""
            CREATE TABLE {table_name} (
              record_key TEXT PRIMARY KEY,
              uri TEXT NOT NULL,
              tenant_id TEXT NOT NULL,
              owner_user_id TEXT NOT NULL DEFAULT '',
              project_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              workspace_shared INTEGER NOT NULL DEFAULT 0,
              session_id TEXT NOT NULL DEFAULT '',
              adapter_id TEXT NOT NULL DEFAULT '',
              context_type TEXT NOT NULL DEFAULT '',
              source_kind TEXT NOT NULL DEFAULT '',
              record_kind TEXT NOT NULL DEFAULT 'context',
              lifecycle_state TEXT NOT NULL DEFAULT 'active',
              admission_status TEXT NOT NULL DEFAULT '',
              claim_state TEXT NOT NULL DEFAULT '',
              slot_id TEXT NOT NULL DEFAULT '',
              memory_type TEXT NOT NULL DEFAULT '',
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
              valid_from TEXT NOT NULL DEFAULT '',
              valid_to TEXT NOT NULL DEFAULT '',
              title TEXT NOT NULL DEFAULT '',
              l0_text TEXT NOT NULL DEFAULT '',
              l1_text TEXT NOT NULL DEFAULT '',
              l2_uri TEXT NOT NULL DEFAULT '',
              source_uri TEXT NOT NULL DEFAULT '',
              source_digest TEXT NOT NULL DEFAULT '',
              source_revision INTEGER NOT NULL DEFAULT 0,
              canonical_slot_id TEXT NOT NULL DEFAULT '',
              canonical_slot_uri TEXT NOT NULL DEFAULT '',
              canonical_claim_id TEXT NOT NULL DEFAULT '',
              canonical_claim_uri TEXT NOT NULL DEFAULT '',
              canonical_revision INTEGER NOT NULL DEFAULT 0,
              canonical_state TEXT NOT NULL DEFAULT '',
              canonical_head_digest TEXT NOT NULL DEFAULT '',
              receipt_digest TEXT NOT NULL DEFAULT '',
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
              memory_anchor_uri TEXT NOT NULL DEFAULT ''
            )
            """
        )

    def _create_auxiliary_tables(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_paths (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              uri TEXT NOT NULL,
              owner_user_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              workspace_shared INTEGER NOT NULL DEFAULT 0,
              context_type TEXT NOT NULL DEFAULT '',
              record_kind TEXT NOT NULL DEFAULT 'context',
              canonical_slot_id TEXT NOT NULL DEFAULT '',
              canonical_claim_id TEXT NOT NULL DEFAULT '',
              event_time TEXT NOT NULL DEFAULT '',
              transaction_time TEXT NOT NULL DEFAULT '',
              valid_from TEXT NOT NULL DEFAULT '',
              valid_to TEXT NOT NULL DEFAULT '',
              path TEXT NOT NULL,
              path_kind TEXT NOT NULL,
              depth INTEGER NOT NULL,
              is_primary INTEGER NOT NULL CHECK(is_primary IN (0, 1)),
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(tenant_id, record_key, path)
            )
            """
        )
        path_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(context_paths)").fetchall()}
        for name, definition in {
            "workspace_id": "TEXT NOT NULL DEFAULT ''",
            "workspace_shared": "INTEGER NOT NULL DEFAULT 0",
            "record_kind": "TEXT NOT NULL DEFAULT 'context'",
            "canonical_slot_id": "TEXT NOT NULL DEFAULT ''",
            "canonical_claim_id": "TEXT NOT NULL DEFAULT ''",
            "transaction_time": "TEXT NOT NULL DEFAULT ''",
            "valid_from": "TEXT NOT NULL DEFAULT ''",
            "valid_to": "TEXT NOT NULL DEFAULT ''",
        }.items():
            if name not in path_columns:
                conn.execute(f"ALTER TABLE context_paths ADD COLUMN {name} {definition}")
        conn.execute(
            "UPDATE context_paths SET "
            "workspace_id = COALESCE((SELECT c.workspace_id FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), ''), "
            "workspace_shared = COALESCE((SELECT c.workspace_shared FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), 0), "
            "record_kind = COALESCE((SELECT c.record_kind FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), 'context'), "
            "canonical_slot_id = COALESCE((SELECT c.canonical_slot_id FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), ''), "
            "canonical_claim_id = COALESCE((SELECT c.canonical_claim_id FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), ''), "
            "transaction_time = COALESCE((SELECT c.transaction_time FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), ''), "
            "valid_from = COALESCE((SELECT c.valid_from FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), ''), "
            "valid_to = COALESCE((SELECT c.valid_to FROM contexts c "
            "WHERE c.record_key = context_paths.record_key), '')"
        )
        conn.execute(
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
              PRIMARY KEY(tenant_id, record_key, grant_kind, grant_id, workspace_id)
            )
            """
        )
        grant_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(context_acl_grants)").fetchall()}
        for name, definition in {
            "scope_signature": "TEXT NOT NULL DEFAULT ''",
            "uri": "TEXT NOT NULL DEFAULT ''",
            "context_type": "TEXT NOT NULL DEFAULT ''",
            "source_kind": "TEXT NOT NULL DEFAULT ''",
            "record_kind": "TEXT NOT NULL DEFAULT 'context'",
            "adapter_id": "TEXT NOT NULL DEFAULT ''",
            "adapter_access_id": "TEXT NOT NULL DEFAULT ''",
            "session_id": "TEXT NOT NULL DEFAULT ''",
            "event_time": "TEXT NOT NULL DEFAULT ''",
            "transaction_time": "TEXT NOT NULL DEFAULT ''",
        }.items():
            if name not in grant_columns:
                conn.execute(f"ALTER TABLE context_acl_grants ADD COLUMN {name} {definition}")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_path_closure (
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              path TEXT NOT NULL,
              ancestor_path TEXT NOT NULL,
              owner_user_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              workspace_shared INTEGER NOT NULL DEFAULT 0,
              scope_signature TEXT NOT NULL DEFAULT '',
              uri TEXT NOT NULL DEFAULT '',
              context_type TEXT NOT NULL DEFAULT '',
              source_kind TEXT NOT NULL DEFAULT '',
              record_kind TEXT NOT NULL DEFAULT 'context',
              adapter_id TEXT NOT NULL DEFAULT '',
              adapter_access_id TEXT NOT NULL DEFAULT '',
              session_id TEXT NOT NULL DEFAULT '',
              canonical_slot_id TEXT NOT NULL DEFAULT '',
              canonical_claim_id TEXT NOT NULL DEFAULT '',
              event_time TEXT NOT NULL DEFAULT '',
              transaction_time TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL DEFAULT '',
              PRIMARY KEY(tenant_id, record_key, path, ancestor_path)
            )
            """
        )
        closure_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(context_path_closure)").fetchall()}
        for name, definition in {
            "scope_signature": "TEXT NOT NULL DEFAULT ''",
            "uri": "TEXT NOT NULL DEFAULT ''",
            "source_kind": "TEXT NOT NULL DEFAULT ''",
            "adapter_id": "TEXT NOT NULL DEFAULT ''",
            "adapter_access_id": "TEXT NOT NULL DEFAULT ''",
            "session_id": "TEXT NOT NULL DEFAULT ''",
        }.items():
            if name not in closure_columns:
                conn.execute(f"ALTER TABLE context_path_closure ADD COLUMN {name} {definition}")
        conn.execute(
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
              PRIMARY KEY(
                tenant_id, record_key, path, ancestor_path,
                grant_kind, grant_id, workspace_id
              )
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_tenants (
              tenant_key INTEGER PRIMARY KEY AUTOINCREMENT,
              tenant_id TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_validity_map (
              validity_id INTEGER PRIMARY KEY AUTOINCREMENT,
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS context_validity_rtree USING rtree(
              validity_id,
              tenant_min, tenant_max,
              valid_from_min, valid_from_max,
              valid_to_min, valid_to_max
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_fts_map (
              record_key TEXT PRIMARY KEY,
              fts_rowid INTEGER NOT NULL UNIQUE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_links (
              link_key TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              source_record_key TEXT NOT NULL,
              source_uri TEXT NOT NULL,
              relation_type TEXT NOT NULL,
              target_record_key TEXT NOT NULL DEFAULT '',
              target_uri TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
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
              PRIMARY KEY(tenant_id, record_key)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_tombstones (
              tombstone_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              record_key TEXT NOT NULL,
              uri TEXT NOT NULL DEFAULT '',
              reason TEXT NOT NULL,
              source_revision INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL,
              payload_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              retry_count INTEGER NOT NULL DEFAULT 0,
              last_error TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS migration_state (
              migration_name TEXT NOT NULL,
              tenant_id TEXT NOT NULL DEFAULT '',
              state TEXT NOT NULL,
              checkpoint TEXT NOT NULL DEFAULT '',
              batch_size INTEGER NOT NULL DEFAULT 0,
              details_json TEXT NOT NULL DEFAULT '{}',
              last_error TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL,
              PRIMARY KEY(migration_name, tenant_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS migration_equivalence_journal (
              proof_id TEXT PRIMARY KEY,
              migration_name TEXT NOT NULL,
              tenant_id TEXT NOT NULL DEFAULT '',
              validation_epoch TEXT NOT NULL DEFAULT '',
              plane TEXT NOT NULL,
              source_identity_digest TEXT NOT NULL,
              evidence_digest TEXT NOT NULL,
              expected_count INTEGER NOT NULL,
              actual_count INTEGER NOT NULL,
              expected_digest TEXT NOT NULL,
              actual_digest TEXT NOT NULL,
              matched INTEGER NOT NULL CHECK(matched IN (0, 1)),
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS migration_shadow_read_journal (
              comparison_id TEXT PRIMARY KEY,
              migration_name TEXT NOT NULL,
              tenant_id TEXT NOT NULL DEFAULT '',
              validation_epoch TEXT NOT NULL,
              plan_digest TEXT NOT NULL,
              legacy_count INTEGER NOT NULL,
              unified_count INTEGER NOT NULL,
              overlap_count INTEGER NOT NULL,
              legacy_digest TEXT NOT NULL,
              unified_digest TEXT NOT NULL,
              matched INTEGER NOT NULL CHECK(matched IN (0, 1)),
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_projection_frontier (
              tenant_id TEXT NOT NULL,
              archive_uri TEXT NOT NULL,
              owner_user_id TEXT NOT NULL DEFAULT '',
              workspace_id TEXT NOT NULL DEFAULT '',
              session_id TEXT NOT NULL,
              manifest_digest TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL,
              last_error TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(tenant_id, archive_uri)
            )
            """
        )
        frontier_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(session_projection_frontier)").fetchall()
        }
        for name, definition in {
            "owner_user_id": "TEXT NOT NULL DEFAULT ''",
            "workspace_id": "TEXT NOT NULL DEFAULT ''",
        }.items():
            if name not in frontier_columns:
                conn.execute(f"ALTER TABLE session_projection_frontier ADD COLUMN {name} {definition}")
        conn.execute(
            "UPDATE session_projection_frontier SET owner_user_id = "
            "CASE WHEN owner_user_id = '' AND archive_uri LIKE 'memoryos://user/%' "
            "THEN substr(substr(archive_uri, length('memoryos://user/') + 1), 1, "
            "instr(substr(archive_uri, length('memoryos://user/') + 1), '/') - 1) "
            "ELSE owner_user_id END"
        )

    def _ensure_fts_table(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'contexts_fts' AND type IN ('table', 'view')"
        ).fetchone()
        desired = {
            "record_key",
            "uri",
            "title",
            "content_text",
            "metadata_text",
            "search_terms",
            "acl_tokens",
        }
        columns = (
            {str(item[1]) for item in conn.execute("PRAGMA table_info(contexts_fts)").fetchall()}
            if row is not None
            else set()
        )
        if row is not None and columns != desired:
            conn.execute("DROP TABLE contexts_fts")
            row = None
        if row is not None:
            self._store.fts_enabled = "VIRTUAL TABLE" in str(row["sql"] or "").upper()
            return False
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE contexts_fts USING fts5(
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
                  record_key TEXT PRIMARY KEY,
                  uri TEXT NOT NULL,
                  title TEXT NOT NULL,
                  content_text TEXT NOT NULL,
                  metadata_text TEXT NOT NULL,
                  search_terms TEXT NOT NULL,
                  acl_tokens TEXT NOT NULL
                )
                """
            )
        return True

    @staticmethod
    def _fts_row_map_is_consistent(conn: sqlite3.Connection) -> bool:
        """Validate the rebuildable FTS rowid map during startup integrity."""

        unmapped = conn.execute(
            "SELECT 1 FROM contexts_fts AS f "
            "LEFT JOIN context_fts_map AS m "
            "ON m.fts_rowid = f.rowid AND m.record_key = f.record_key "
            "WHERE m.record_key IS NULL LIMIT 1"
        ).fetchone()
        if unmapped is not None:
            return False
        orphaned = conn.execute(
            "SELECT 1 FROM context_fts_map AS m "
            "LEFT JOIN contexts_fts AS f ON f.rowid = m.fts_rowid "
            "WHERE f.rowid IS NULL OR f.record_key != m.record_key LIMIT 1"
        ).fetchone()
        return orphaned is None

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        statements = (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_contexts_record_key ON contexts(record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_updated "
            "ON contexts(tenant_id, owner_user_id, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_event "
            "ON contexts(tenant_id, owner_user_id, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_transaction "
            "ON contexts(tenant_id, owner_user_id, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_valid "
            "ON contexts(tenant_id, owner_user_id, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_uri "
            "ON contexts(tenant_id, owner_user_id, uri, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_type_updated "
            "ON contexts(tenant_id, owner_user_id, context_type, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_type_event "
            "ON contexts(tenant_id, owner_user_id, context_type, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_type_transaction "
            "ON contexts(tenant_id, owner_user_id, context_type, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_type_valid "
            "ON contexts(tenant_id, owner_user_id, context_type, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_source_updated "
            "ON contexts(tenant_id, owner_user_id, source_kind, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_source_event "
            "ON contexts(tenant_id, owner_user_id, source_kind, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_source_transaction "
            "ON contexts(tenant_id, owner_user_id, source_kind, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_source_valid "
            "ON contexts(tenant_id, owner_user_id, source_kind, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_session_updated "
            "ON contexts(tenant_id, owner_user_id, session_id, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_session_event "
            "ON contexts(tenant_id, owner_user_id, session_id, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_session_transaction "
            "ON contexts(tenant_id, owner_user_id, session_id, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_owner_session_valid "
            "ON contexts(tenant_id, owner_user_id, session_id, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_public_updated "
            "ON contexts(tenant_id, updated_at DESC, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_public_event "
            "ON contexts(tenant_id, event_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_public_transaction "
            "ON contexts(tenant_id, transaction_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_public_valid "
            "ON contexts(tenant_id, valid_from, valid_to, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_shared_workspace_updated "
            "ON contexts(tenant_id, workspace_id, updated_at DESC, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_shared_workspace_event "
            "ON contexts(tenant_id, workspace_id, event_time, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_shared_workspace_transaction "
            "ON contexts(tenant_id, workspace_id, transaction_time, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_contexts_acl_shared_workspace_valid "
            "ON contexts(tenant_id, workspace_id, valid_from, valid_to, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_uri ON contexts(tenant_id, uri, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_uri_kind_updated "
            "ON contexts(tenant_id, uri, record_kind, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_canonical_slot_uri "
            "ON contexts(tenant_id, canonical_slot_uri, record_kind, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_canonical_claim_uri "
            "ON contexts(tenant_id, canonical_claim_uri, record_kind, updated_at DESC, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_owner_type ON contexts(tenant_id, owner_user_id, context_type, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_workspace ON contexts(tenant_id, workspace_id, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_session ON contexts(tenant_id, session_id, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_record_kind ON contexts(tenant_id, record_kind, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_created_at ON contexts(tenant_id, created_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_event_time ON contexts(tenant_id, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_ingested_at ON contexts(tenant_id, ingested_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_transaction_time ON contexts(tenant_id, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_updated_at ON contexts(tenant_id, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_valid_interval ON contexts(tenant_id, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_scene_key ON contexts(tenant_id, scene_key, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_action ON contexts(tenant_id, action, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_tenant_anchor ON contexts(tenant_id, memory_anchor_uri, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_projection_evidence ON contexts(tenant_id, source_uri, projection_effect_hash, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_contexts_claim_revision ON contexts(tenant_id, canonical_claim_id, canonical_revision)",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_contexts_current_slot ON contexts(tenant_id, canonical_slot_id) "
            "WHERE record_kind = 'current_slot' AND canonical_slot_id != '' "
            "AND lifecycle_state NOT IN ('deleted', 'obsolete')",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_context_paths_primary ON context_paths(tenant_id, record_key) WHERE is_primary = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_tenant_path ON context_paths(tenant_id, path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_path ON context_paths(tenant_id, owner_user_id, path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_type_path "
            "ON context_paths(tenant_id, owner_user_id, context_type, path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_type_path_event "
            "ON context_paths(tenant_id, owner_user_id, context_type, path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_type_path_transaction "
            "ON context_paths(tenant_id, owner_user_id, context_type, path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_type_path_updated "
            "ON context_paths(tenant_id, owner_user_id, context_type, path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_type_path_valid "
            "ON context_paths(tenant_id, owner_user_id, context_type, path, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_path_event "
            "ON context_paths(tenant_id, owner_user_id, path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_path_transaction "
            "ON context_paths(tenant_id, owner_user_id, path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_path_updated "
            "ON context_paths(tenant_id, owner_user_id, path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_owner_path_valid "
            "ON context_paths(tenant_id, owner_user_id, path, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_public_path ON context_paths(tenant_id, path, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_public_path_event "
            "ON context_paths(tenant_id, path, event_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_public_path_transaction "
            "ON context_paths(tenant_id, path, transaction_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_public_path_updated "
            "ON context_paths(tenant_id, path, updated_at, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_public_path_valid "
            "ON context_paths(tenant_id, path, valid_from, valid_to, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_shared_workspace_path "
            "ON context_paths(tenant_id, workspace_id, path, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_shared_workspace_path_event "
            "ON context_paths(tenant_id, workspace_id, path, event_time, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_shared_workspace_path_transaction "
            "ON context_paths(tenant_id, workspace_id, path, transaction_time, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_shared_workspace_path_updated "
            "ON context_paths(tenant_id, workspace_id, path, updated_at, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_shared_workspace_path_valid "
            "ON context_paths(tenant_id, workspace_id, path, valid_from, valid_to, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_type_path ON context_paths(tenant_id, context_type, path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_time_path ON context_paths(tenant_id, event_time, path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_path_time ON context_paths(tenant_id, path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_path_event "
            "ON context_paths(tenant_id, path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_path_transaction "
            "ON context_paths(tenant_id, path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_path_updated "
            "ON context_paths(tenant_id, path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_path_valid "
            "ON context_paths(tenant_id, path, valid_from, valid_to, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_paths_uri ON context_paths(tenant_id, uri, path)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_tenant_ancestor "
            "ON context_path_closure(tenant_id, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_scope_kind_ancestor "
            "ON context_path_closure(tenant_id, scope_signature, record_kind, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_scope_kind_ancestor_event "
            "ON context_path_closure(tenant_id, scope_signature, record_kind, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_scope_kind_ancestor_transaction "
            "ON context_path_closure(tenant_id, scope_signature, record_kind, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_scope_kind_ancestor_updated "
            "ON context_path_closure(tenant_id, scope_signature, record_kind, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_type_ancestor "
            "ON context_path_closure(tenant_id, context_type, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_ancestor_event "
            "ON context_path_closure(tenant_id, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_ancestor_transaction "
            "ON context_path_closure(tenant_id, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_ancestor_updated "
            "ON context_path_closure(tenant_id, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_ancestor "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_ancestor "
            "ON context_path_closure(tenant_id, owner_user_id, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_type_ancestor "
            "ON context_path_closure(tenant_id, owner_user_id, context_type, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_ancestor_event "
            "ON context_path_closure(tenant_id, owner_user_id, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_ancestor_transaction "
            "ON context_path_closure(tenant_id, owner_user_id, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_ancestor_updated "
            "ON context_path_closure(tenant_id, owner_user_id, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_type_ancestor_event "
            "ON context_path_closure(tenant_id, owner_user_id, context_type, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_type_ancestor_transaction "
            "ON context_path_closure(tenant_id, owner_user_id, context_type, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_type_ancestor_updated "
            "ON context_path_closure(tenant_id, owner_user_id, context_type, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_type_ancestor "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, context_type, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_ancestor_event "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_ancestor_transaction "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_ancestor_updated "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_type_ancestor_event "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, context_type, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_type_ancestor_transaction "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, context_type, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_owner_workspace_type_ancestor_updated "
            "ON context_path_closure(tenant_id, owner_user_id, workspace_id, context_type, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_workspace_ancestor "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_ancestor "
            "ON context_path_closure(tenant_id, ancestor_path, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_ancestor_event "
            "ON context_path_closure(tenant_id, ancestor_path, event_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_ancestor_transaction "
            "ON context_path_closure(tenant_id, ancestor_path, transaction_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_ancestor_updated "
            "ON context_path_closure(tenant_id, ancestor_path, updated_at, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_workspace_ancestor_event "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, event_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_workspace_ancestor_transaction "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, transaction_time, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_public_workspace_ancestor_updated "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, updated_at, record_key) "
            "WHERE owner_user_id = '' AND context_type IN ('resource', 'skill')",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_shared_workspace_ancestor "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_shared_workspace_ancestor_event "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, event_time, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_shared_workspace_ancestor_transaction "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, transaction_time, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_path_closure_shared_workspace_ancestor_updated "
            "ON context_path_closure(tenant_id, workspace_id, ancestor_path, updated_at, record_key) "
            "WHERE context_type = 'memory' AND record_kind IN ('current_slot', 'claim_revision') "
            "AND canonical_slot_id != '' AND canonical_claim_id != '' AND workspace_shared = 1",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_updated "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_event "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_transaction "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_scope_updated "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, scope_signature, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_scope_event "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, scope_signature, ancestor_path, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_scope_transaction "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, scope_signature, ancestor_path, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_type_updated "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, context_type, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_adapter_access_updated "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_access_id, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_source_updated "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, source_kind, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_session_updated "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, record_kind, session_id, ancestor_path, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_path_acl_workspace_uri "
            "ON context_path_acl(tenant_id, grant_kind, grant_id, workspace_id, uri, ancestor_path, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_validity_map_tenant_record "
            "ON context_validity_map(tenant_id, record_key, validity_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_lookup "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_access_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, updated_at DESC, record_key DESC, workspace_id, record_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_access_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, event_time DESC, record_key DESC, workspace_id, record_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_access_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, transaction_time DESC, record_key DESC, workspace_id, record_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_access_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, updated_at DESC, record_key DESC, record_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_access_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, event_time DESC, record_key DESC, record_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_access_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, transaction_time DESC, record_key DESC, record_kind)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_kind_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, record_kind, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_kind_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, record_kind, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_kind_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, record_kind, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_scope_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, scope_signature, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_scope_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, scope_signature, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_scope_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, scope_signature, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_type_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, context_type, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_type_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, context_type, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_type_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, context_type, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_adapter_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_id, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_adapter_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_id, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_adapter_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_id, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_adapter_access_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_access_id, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_adapter_access_event "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_access_id, event_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_adapter_access_transaction "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, adapter_access_id, transaction_time, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_source_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, source_kind, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_source_type_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, source_kind, context_type, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_session_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, session_id, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_session_type_updated "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, record_kind, session_id, context_type, updated_at, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_workspace_uri "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, workspace_id, uri, record_key)",
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_record "
            "ON context_acl_grants(tenant_id, record_key, grant_kind, grant_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_links_source ON context_links(tenant_id, source_record_key, relation_type)",
            "CREATE INDEX IF NOT EXISTS idx_context_links_target ON context_links(tenant_id, target_record_key, relation_type)",
            "CREATE INDEX IF NOT EXISTS idx_context_tombstones_status ON context_tombstones(status, updated_at, tombstone_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_tombstones_tenant_uri_status "
            "ON context_tombstones(tenant_id, uri, status, updated_at, tombstone_id)",
            "CREATE INDEX IF NOT EXISTS idx_context_tombstones_tenant_uri_id "
            "ON context_tombstones(tenant_id, uri, tombstone_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_migration_state_status ON migration_state(state, updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_migration_equivalence_epoch ON migration_equivalence_journal(migration_name, tenant_id, validation_epoch, created_at, proof_id)",
            "CREATE INDEX IF NOT EXISTS idx_migration_shadow_read_epoch "
            "ON migration_shadow_read_journal(migration_name, tenant_id, validation_epoch, created_at, comparison_id)",
            "CREATE INDEX IF NOT EXISTS idx_session_projection_frontier_status "
            "ON session_projection_frontier(tenant_id, status, updated_at, archive_uri)",
            "CREATE INDEX IF NOT EXISTS idx_session_projection_frontier_scope_status "
            "ON session_projection_frontier(tenant_id, owner_user_id, workspace_id, status, updated_at, archive_uri)",
            "CREATE INDEX IF NOT EXISTS idx_session_projection_frontier_replay "
            "ON session_projection_frontier(tenant_id, archive_uri, status)",
        )
        for statement in statements:
            conn.execute(statement)
        for dimension, column in (
            ("scope", "scope_signature"),
            ("type", "context_type"),
            ("source", "source_kind"),
            ("session", "session_id"),
            ("adapter", "adapter_id"),
            ("adapter_access", "adapter_access_id"),
        ):
            for time_name, time_column in (
                ("updated", "updated_at"),
                ("event", "event_time"),
                ("transaction", "transaction_time"),
            ):
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS "
                    f"idx_context_acl_grants_access_{dimension}_{time_name} "
                    "ON context_acl_grants("
                    f"tenant_id, grant_kind, grant_id, {column}, "
                    f"{time_column} DESC, record_key DESC, record_kind)"
                )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_context_acl_grants_access_uri "
            "ON context_acl_grants(tenant_id, grant_kind, grant_id, uri, record_key)"
        )


__all__ = ["SchemaManager"]
