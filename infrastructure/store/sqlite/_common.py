"""基于 SQLite、可重建的统一 Context Catalog 服务索引。"""

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

from foundation.identity.workspace import normalize_workspace_id
from infrastructure.store.contracts.index import IndexHit
from infrastructure.store.model.catalog import (
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    ServingTier,
    normalize_tree_path,
)
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.query import (
    CatalogCandidateBoundExceeded,
    lexical_match_count,
    lexical_relevance,
    lexical_terms,
)
from sanitization.context_projection import ContextProjectionSanitizer

_CATALOG_SCHEMA_VERSION = 1
_INVALID_SCOPE_KEY = "__memoryos_invalid_scope__"
_MAX_FILTER_VALUES = 900
_MAX_QUERY_LIMIT = 1_000
_MAX_TARGET_PATHS = 16
_BOUNDED_FTS_OVERFETCH = 256
_MAX_FTS_METADATA_TEXT = 4_000
_MAX_SCOPE_KEYS_PER_RECORD = 8
_MAX_SCOPE_SIGNATURE_OPTIONS = 256
_FTS_BM25 = "bm25(contexts_fts, 0.0, 0.0, 0.0, 5.0, 4.0, 2.0, 1.0, 0.0)"
_FTS_RANK_CONFIG = "bm25(0.0, 0.0, 0.0, 5.0, 4.0, 2.0, 1.0, 0.0)"
_ONLINE_VM_STEP_LIMIT = 1_000_000
_ONLINE_PROGRESS_GRANULARITY = 1_000
_SAFE_FTS_METADATA_KEYS = frozenset(
    {
        "action",
        "dimension",
        "file_name",
        "filename",
        "keywords",
        "block_id",
        "document_id",
        "document_kind",
        "resource_location",
        "resource_name",
        "scene_key",
        "support_anchor_uri",
        "subject",
        "summary",
        "tags",
        "topic",
    }
)

_CONTEXT_COLUMNS = (
    "tenant_id",
    "record_key",
    "uri",
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
    "title",
    "l0_text",
    "l1_text",
    "l2_uri",
    "source_uri",
    "source_digest",
    "source_revision",
    "document_id",
    "block_id",
    "document_kind",
    "document_revision",
    "projection_generation",
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
    "support_anchor_uri",
)

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
    "document_id": "document_id",
    "block_id": "block_id",
    "document_kind": "document_kind",
    "document_revision": "document_revision",
    "projection_generation": "projection_generation",
    "serving_tier": "serving_tier",
    "projection_status": "projection_status",
    "scene_key": "scene_key",
    "action": "action",
    "support_anchor_uri": "support_anchor_uri",
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
    "document_ids": "document_id",
    "block_ids": "block_id",
    "document_kinds": "document_kind",
}


def _path_ancestors(path: str) -> tuple[str, ...]:
    """返回在线前缀查询使用的有界分类路径闭包。"""

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
    "_BOUNDED_FTS_OVERFETCH",
    "_CATALOG_SCHEMA_VERSION",
    "_CONTEXT_COLUMNS",
    "_FTS_BM25",
    "_FTS_RANK_CONFIG",
    "_INVALID_SCOPE_KEY",
    "_MAX_FILTER_VALUES",
    "_MAX_FTS_METADATA_TEXT",
    "_MAX_QUERY_LIMIT",
    "_MAX_SCOPE_KEYS_PER_RECORD",
    "_MAX_SCOPE_SIGNATURE_OPTIONS",
    "_MAX_TARGET_PATHS",
    "_ONLINE_PROGRESS_GRANULARITY",
    "_ONLINE_VM_STEP_LIMIT",
    "_PLURAL_FILTER_ALIASES",
    "_PreparedCatalogRecord",
    "_SAFE_FTS_METADATA_KEYS",
    "_SIMPLE_FILTER_FIELDS",
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
