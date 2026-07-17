"""SQLite-backed, rebuildable Unified Context Catalog serving index."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from memoryos.contextdb.catalog import (
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    ServingTier,
    normalize_tree_path,
)
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.retrieval.errors import CatalogCandidateBoundExceeded
from memoryos.contextdb.retrieval.lexical import lexical_match_count, lexical_relevance, lexical_terms
from memoryos.contextdb.store.index_store import IndexHit
from memoryos.security.context_projection import ContextProjectionSanitizer
from memoryos.security.workspace_identity import normalize_workspace_id

_SCOPE_KEY_SCHEMA_VERSION = 2
_CATALOG_SCHEMA_VERSION = 10
_INVALID_SCOPE_KEY = "__memoryos_invalid_scope__"
_MAX_FILTER_VALUES = 900
_MAX_QUERY_LIMIT = 1_000
_MAX_TARGET_PATHS = 16
_BOUNDED_FTS_OVERFETCH = 256
_MIGRATION_BATCH_SIZE = 256
_UNIFIED_CATALOG_MIGRATION_NAME = "unified-context-catalog-v1"
_SCHEMA_UPGRADE_BOOTSTRAP_TENANT = "__memoryos_schema_upgrade__"
_GREENFIELD_CATALOG_ORIGIN_NAME = "__memoryos_greenfield_catalog_v10__"
_MAX_FTS_METADATA_TEXT = 4_000
_MAX_SCOPE_KEYS_PER_RECORD = 8
_MAX_SCOPE_SIGNATURE_OPTIONS = 256
_FTS_BM25 = "bm25(contexts_fts, 0.0, 0.0, 5.0, 4.0, 2.0, 1.0, 0.0)"
_FTS_RANK_CONFIG = "bm25(0.0, 0.0, 5.0, 4.0, 2.0, 1.0, 0.0)"
_ONLINE_VM_STEP_LIMIT = 1_000_000
_ONLINE_PROGRESS_GRANULARITY = 1_000
_MIGRATION_STATES = frozenset(
    {
        "NOT_STARTED",
        "SCHEMA_READY",
        "BACKFILLING",
        "DUAL_WRITE",
        "SHADOW_VALIDATING",
        "READY_TO_CUTOVER",
        "CUTOVER",
        "ROLLBACK",
        "COMPLETED",
        "FAILED",
    }
)
_SAFE_FTS_METADATA_KEYS = frozenset(
    {
        "action",
        "dimension",
        "file_name",
        "filename",
        "keywords",
        "memory_type",
        "resource_location",
        "resource_name",
        "scene_key",
        "subject",
        "summary",
        "tags",
        "topic",
    }
)

_CONTEXT_COLUMNS = (
    "record_key",
    "uri",
    "tenant_id",
    "owner_user_id",
    "project_id",
    "workspace_id",
    "workspace_shared",
    "session_id",
    "adapter_id",
    "context_type",
    "source_kind",
    "record_kind",
    "lifecycle_state",
    "admission_status",
    "claim_state",
    "slot_id",
    "memory_type",
    "scope_keys",
    "scope_signature",
    "parent_uri",
    "primary_tree_path",
    "path_depth",
    "created_at",
    "updated_at",
    "event_time",
    "ingested_at",
    "transaction_time",
    "valid_from",
    "valid_to",
    "title",
    "l0_text",
    "l1_text",
    "l2_uri",
    "source_uri",
    "source_digest",
    "source_revision",
    "canonical_slot_id",
    "canonical_slot_uri",
    "canonical_claim_id",
    "canonical_claim_uri",
    "canonical_revision",
    "canonical_state",
    "canonical_head_digest",
    "receipt_digest",
    "projection_effect_hash",
    "hotness",
    "semantic_hotness",
    "behavior_support_hotness",
    "serving_tier",
    "projection_status",
    "metadata_json",
    "content_digest",
    "stored_content_digest",
    "content_text",
    "scene_key",
    "action",
    "memory_anchor_uri",
)

_ALTER_COLUMN_DEFINITIONS = {
    "project_id": "TEXT NOT NULL DEFAULT ''",
    "workspace_id": "TEXT NOT NULL DEFAULT ''",
    "workspace_shared": "INTEGER NOT NULL DEFAULT 0",
    "session_id": "TEXT NOT NULL DEFAULT ''",
    "adapter_id": "TEXT NOT NULL DEFAULT ''",
    "source_kind": "TEXT NOT NULL DEFAULT ''",
    "record_kind": "TEXT NOT NULL DEFAULT 'context'",
    "admission_status": "TEXT NOT NULL DEFAULT ''",
    "claim_state": "TEXT NOT NULL DEFAULT ''",
    "slot_id": "TEXT NOT NULL DEFAULT ''",
    "memory_type": "TEXT NOT NULL DEFAULT ''",
    "scope_keys": "TEXT NOT NULL DEFAULT '[]'",
    "scope_signature": "TEXT NOT NULL DEFAULT ''",
    "parent_uri": "TEXT NOT NULL DEFAULT ''",
    "primary_tree_path": "TEXT NOT NULL DEFAULT ''",
    "path_depth": "INTEGER NOT NULL DEFAULT 0",
    "created_at": "TEXT NOT NULL DEFAULT ''",
    "event_time": "TEXT NOT NULL DEFAULT ''",
    "ingested_at": "TEXT NOT NULL DEFAULT ''",
    "transaction_time": "TEXT NOT NULL DEFAULT ''",
    "valid_from": "TEXT NOT NULL DEFAULT ''",
    "valid_to": "TEXT NOT NULL DEFAULT ''",
    "l0_text": "TEXT NOT NULL DEFAULT ''",
    "l1_text": "TEXT NOT NULL DEFAULT ''",
    "l2_uri": "TEXT NOT NULL DEFAULT ''",
    "source_uri": "TEXT NOT NULL DEFAULT ''",
    "source_digest": "TEXT NOT NULL DEFAULT ''",
    "source_revision": "INTEGER NOT NULL DEFAULT 0",
    "canonical_slot_id": "TEXT NOT NULL DEFAULT ''",
    "canonical_slot_uri": "TEXT NOT NULL DEFAULT ''",
    "canonical_claim_id": "TEXT NOT NULL DEFAULT ''",
    "canonical_claim_uri": "TEXT NOT NULL DEFAULT ''",
    "canonical_revision": "INTEGER NOT NULL DEFAULT 0",
    "canonical_state": "TEXT NOT NULL DEFAULT ''",
    "canonical_head_digest": "TEXT NOT NULL DEFAULT ''",
    "receipt_digest": "TEXT NOT NULL DEFAULT ''",
    "projection_effect_hash": "TEXT NOT NULL DEFAULT ''",
    "serving_tier": "TEXT NOT NULL DEFAULT 'HOT'",
    "projection_status": "TEXT NOT NULL DEFAULT 'PROJECTED'",
    "content_digest": "TEXT NOT NULL DEFAULT ''",
    "stored_content_digest": "TEXT NOT NULL DEFAULT ''",
    "scene_key": "TEXT NOT NULL DEFAULT ''",
    "action": "TEXT NOT NULL DEFAULT ''",
    "memory_anchor_uri": "TEXT NOT NULL DEFAULT ''",
}

_SIMPLE_FILTER_FIELDS = {
    "record_key": "record_key",
    "tenant_id": "tenant_id",
    "owner_user_id": "owner_user_id",
    "workspace_id": "workspace_id",
    "session_id": "session_id",
    "adapter_id": "adapter_id",
    "context_type": "context_type",
    "source_kind": "source_kind",
    "record_kind": "record_kind",
    "lifecycle_state": "lifecycle_state",
    "admission_status": "admission_status",
    "claim_state": "claim_state",
    "slot_id": "slot_id",
    "canonical_slot_id": "canonical_slot_id",
    "canonical_claim_id": "canonical_claim_id",
    "canonical_state": "canonical_state",
    "memory_type": "memory_type",
    "serving_tier": "serving_tier",
    "projection_status": "projection_status",
    "scene_key": "scene_key",
    "action": "action",
    "memory_anchor_uri": "memory_anchor_uri",
}

_PLURAL_FILTER_ALIASES = {
    "record_keys": "record_key",
    "target_uris": "uri",
    "workspace_ids": "workspace_id",
    "workspace_access_ids": "workspace_id",
    "session_ids": "session_id",
    "context_types": "context_type",
    "source_kinds": "source_kind",
    "source_uris": "source_uri",
    "record_kinds": "record_kind",
    "canonical_slot_ids": "canonical_slot_id",
    "canonical_claim_ids": "canonical_claim_id",
}


def _path_ancestors(path: str) -> tuple[str, ...]:
    """Return the bounded taxonomy closure used by online prefix queries."""

    segments = path.split("/")
    return tuple("/".join(segments[:depth]) for depth in range(1, len(segments) + 1))


@dataclass(frozen=True)
class _PreparedCatalogRecord:
    record: CatalogRecord
    values: dict[str, Any]
    scope_signature: str
    fts_metadata_text: str
    fts_search_terms: str


__all__ = [
    "Any",
    "CatalogCandidateBoundExceeded",
    "CatalogProjectionStatus",
    "CatalogRecord",
    "CatalogRecordKind",
    "ContextObject",
    "ContextProjectionSanitizer",
    "IndexHit",
    "Mapping",
    "Path",
    "Sequence",
    "ServingTier",
    "_ALTER_COLUMN_DEFINITIONS",
    "_BOUNDED_FTS_OVERFETCH",
    "_CATALOG_SCHEMA_VERSION",
    "_CONTEXT_COLUMNS",
    "_FTS_BM25",
    "_FTS_RANK_CONFIG",
    "_GREENFIELD_CATALOG_ORIGIN_NAME",
    "_INVALID_SCOPE_KEY",
    "_MAX_FILTER_VALUES",
    "_MAX_FTS_METADATA_TEXT",
    "_MAX_QUERY_LIMIT",
    "_MAX_SCOPE_KEYS_PER_RECORD",
    "_MAX_SCOPE_SIGNATURE_OPTIONS",
    "_MAX_TARGET_PATHS",
    "_MIGRATION_BATCH_SIZE",
    "_MIGRATION_STATES",
    "_ONLINE_PROGRESS_GRANULARITY",
    "_ONLINE_VM_STEP_LIMIT",
    "_PLURAL_FILTER_ALIASES",
    "_PreparedCatalogRecord",
    "_SAFE_FTS_METADATA_KEYS",
    "_SCHEMA_UPGRADE_BOOTSTRAP_TENANT",
    "_SCOPE_KEY_SCHEMA_VERSION",
    "_SIMPLE_FILTER_FIELDS",
    "_UNIFIED_CATALOG_MIGRATION_NAME",
    "_path_ancestors",
    "combinations",
    "dataclass",
    "datetime",
    "hashlib",
    "json",
    "lexical_match_count",
    "lexical_relevance",
    "lexical_terms",
    "math",
    "normalize_tree_path",
    "normalize_workspace_id",
    "os",
    "replace",
    "sqlite3",
    "timezone",
]
