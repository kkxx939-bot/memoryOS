"""SQLite Catalog 辅助表的结构契约。"""

from __future__ import annotations

_AUXILIARY_TABLE_COLUMNS = {
    "context_paths": frozenset(
        {
            "tenant_id",
            "record_key",
            "uri",
            "owner_user_id",
            "workspace_id",
            "workspace_shared",
            "context_type",
            "record_kind",
            "event_time",
            "transaction_time",
            "path",
            "path_kind",
            "depth",
            "is_primary",
            "created_at",
            "updated_at",
        }
    ),
    "context_acl_grants": frozenset(
        {
            "tenant_id",
            "record_key",
            "grant_kind",
            "grant_id",
            "workspace_id",
            "scope_signature",
            "uri",
            "context_type",
            "source_kind",
            "record_kind",
            "adapter_id",
            "adapter_access_id",
            "session_id",
            "event_time",
            "transaction_time",
            "updated_at",
        }
    ),
    "context_path_closure": frozenset(
        {
            "tenant_id",
            "record_key",
            "path",
            "ancestor_path",
            "owner_user_id",
            "workspace_id",
            "workspace_shared",
            "scope_signature",
            "uri",
            "context_type",
            "source_kind",
            "record_kind",
            "adapter_id",
            "adapter_access_id",
            "session_id",
            "event_time",
            "transaction_time",
            "updated_at",
        }
    ),
    "context_path_acl": frozenset(
        {
            "tenant_id",
            "record_key",
            "path",
            "ancestor_path",
            "grant_kind",
            "grant_id",
            "workspace_id",
            "owner_user_id",
            "scope_signature",
            "uri",
            "context_type",
            "source_kind",
            "record_kind",
            "adapter_id",
            "adapter_access_id",
            "session_id",
            "event_time",
            "transaction_time",
            "updated_at",
        }
    ),
    "context_fts_map": frozenset({"tenant_id", "record_key", "fts_rowid"}),
    "context_links": frozenset(
        {
            "tenant_id",
            "link_key",
            "source_record_key",
            "source_uri",
            "relation_type",
            "target_record_key",
            "target_uri",
            "metadata_json",
            "created_at",
            "updated_at",
        }
    ),
    "context_projection_state": frozenset(
        {
            "tenant_id",
            "record_key",
            "source_revision",
            "projection_status",
            "projection_effect_hash",
            "retry_count",
            "last_error",
            "updated_at",
        }
    ),
    "context_tombstones": frozenset(
        {
            "tenant_id",
            "tombstone_id",
            "record_key",
            "uri",
            "reason",
            "source_revision",
            "status",
            "payload_json",
            "created_at",
            "updated_at",
            "retry_count",
            "last_error",
        }
    ),
    "context_projection_journal": frozenset(
        {
            "tenant_id",
            "projector_kind",
            "source_uri",
            "owner_user_id",
            "workspace_id",
            "source_id",
            "source_digest",
            "status",
            "last_error",
            "created_at",
            "updated_at",
        }
    ),
}

_AUXILIARY_TABLE_PRIMARY_KEYS = {
    "context_paths": ("tenant_id", "record_key", "path"),
    "context_acl_grants": ("tenant_id", "record_key", "grant_kind", "grant_id", "workspace_id"),
    "context_path_closure": ("tenant_id", "record_key", "path", "ancestor_path"),
    "context_path_acl": (
        "tenant_id",
        "record_key",
        "path",
        "ancestor_path",
        "grant_kind",
        "grant_id",
        "workspace_id",
    ),
    "context_fts_map": ("tenant_id", "record_key"),
    "context_links": ("tenant_id", "link_key"),
    "context_projection_state": ("tenant_id", "record_key"),
    "context_tombstones": ("tenant_id", "tombstone_id"),
    "context_projection_journal": ("tenant_id", "projector_kind", "source_uri"),
}

_AUXILIARY_TABLE_UNIQUE_IDENTITIES = {
    "context_fts_map": frozenset({("fts_rowid",)}),
}


__all__ = [
    "_AUXILIARY_TABLE_COLUMNS",
    "_AUXILIARY_TABLE_PRIMARY_KEYS",
    "_AUXILIARY_TABLE_UNIQUE_IDENTITIES",
]
