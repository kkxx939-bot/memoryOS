"""Typed records for the rebuildable Unified Context Catalog."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.core.types import scope_keys_from_payloads
from memoryos.security.context_projection import ContextProjectionSanitizer
from memoryos.security.workspace_identity import normalize_workspace_id

MAX_SECONDARY_PATHS = 7
MAX_TREE_DEPTH = 12
_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_PATH_ROOTS = frozenset({"timeline", "sessions", "projects", "resources", "memories", "skills", "agents"})
_RESOURCE_PATH_KINDS = frozenset({"desktop", "repository", "uploads", "temporary", "user", "external"})
_MEMORY_FIXED_PATHS = frozenset(
    {
        ("root",),
        ("profile",),
        ("preferences",),
        ("knowledge",),
        ("knowledge", "open-loops"),
    }
)
_MEMORY_DYNAMIC_BRANCHES = frozenset(
    {
        ("knowledge", "entities"),
        ("knowledge", "topics"),
        ("knowledge", "episodes"),
        ("experiences",),
    }
)


class CatalogRecordKind(str, Enum):
    CONTEXT = "context"
    SESSION_ROOT = "session_root"
    SESSION_L0 = "session_l0"
    SESSION_L1 = "session_l1"
    SEMANTIC_SEGMENT = "semantic_segment"
    MESSAGE = "message"
    TOOL_RESULT = "tool_result"
    RESOURCE_REFERENCE = "resource_reference"
    USED_CONTEXT = "used_context"
    USED_SKILL = "used_skill"
    OBSERVATION = "observation"
    ACTION_RESULT = "action_result"
    EVENT = "event"
    MEMORY_DOCUMENT = "memory_document"
    MEMORY_BLOCK = "memory_block"
    TREE_OVERVIEW = "tree_overview"


class ServingTier(str, Enum):
    HOT = "HOT"
    WARM = "WARM"
    COLD = "COLD"
    ARCHIVED = "ARCHIVED"


class CatalogProjectionStatus(str, Enum):
    PENDING = "PENDING"
    PROJECTED = "PROJECTED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    TOMBSTONED = "TOMBSTONED"


@dataclass(frozen=True)
class CatalogRecord:
    record_key: str
    uri: str
    tenant_id: str
    owner_user_id: str = ""
    workspace_id: str = ""
    workspace_shared: bool = False
    session_id: str = ""
    adapter_id: str = ""
    context_type: str = ""
    source_kind: str = ""
    record_kind: str = CatalogRecordKind.CONTEXT.value
    lifecycle_state: str = "active"
    parent_uri: str = ""
    primary_tree_path: str = ""
    tree_paths: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""
    event_time: str = ""
    ingested_at: str = ""
    transaction_time: str = ""
    title: str = ""
    l0_text: str = ""
    l1_text: str = ""
    l2_uri: str = ""
    source_uri: str = ""
    source_digest: str = ""
    source_revision: int = 0
    document_id: str = ""
    block_id: str = ""
    document_kind: str = ""
    document_revision: int = 0
    projection_generation: int = 0
    projection_effect_hash: str = ""
    hotness: float = 0.0
    semantic_hotness: float = 0.0
    behavior_support_hotness: float = 0.0
    serving_tier: str = ServingTier.HOT.value
    projection_status: str = CatalogProjectionStatus.PROJECTED.value
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_key or len(self.record_key) > 500:
            raise ValueError("catalog record_key must be non-empty and bounded")
        if not self.uri or not self.tenant_id:
            raise ValueError("catalog URI and tenant are required")
        for name in (
            "uri",
            "parent_uri",
            "l2_uri",
            "source_uri",
        ):
            value = str(getattr(self, name) or "")
            if value and not value.startswith("memoryos://"):
                raise ValueError(f"catalog {name} must be a logical memoryos URI")
        record_kind = (
            self.record_kind.value if isinstance(self.record_kind, CatalogRecordKind) else str(self.record_kind)
        )
        serving_tier = self.serving_tier.value if isinstance(self.serving_tier, ServingTier) else str(self.serving_tier)
        projection_status = (
            self.projection_status.value
            if isinstance(self.projection_status, CatalogProjectionStatus)
            else str(self.projection_status)
        )
        object.__setattr__(self, "record_kind", CatalogRecordKind(record_kind).value)
        object.__setattr__(self, "serving_tier", ServingTier(serving_tier.upper()).value)
        object.__setattr__(self, "projection_status", CatalogProjectionStatus(projection_status.upper()).value)
        metadata = dict(self.metadata)
        workspace_id = normalize_workspace_id(self.workspace_id)
        object.__setattr__(self, "workspace_id", workspace_id)
        object.__setattr__(
            self,
            "workspace_shared",
            _workspace_is_publicly_shared(metadata, tenant_id=self.tenant_id, workspace_id=workspace_id),
        )
        paths = validate_tree_paths(self.tree_paths, primary=self.primary_tree_path)
        object.__setattr__(self, "tree_paths", paths)
        object.__setattr__(self, "primary_tree_path", paths[0] if paths else "")
        for name in (
            "created_at",
            "updated_at",
            "event_time",
            "ingested_at",
            "transaction_time",
        ):
            timestamp_value = str(getattr(self, name) or "")
            if timestamp_value:
                object.__setattr__(self, name, normalize_timestamp(timestamp_value, name))
        object.__setattr__(self, "metadata", metadata)
        for name in ("source_revision", "document_revision", "projection_generation"):
            revision_value = int(getattr(self, name))
            if revision_value < 0:
                raise ValueError(f"catalog {name} must be non-negative")
            object.__setattr__(self, name, revision_value)
        if record_kind in {
            CatalogRecordKind.MEMORY_DOCUMENT.value,
            CatalogRecordKind.MEMORY_BLOCK.value,
        }:
            if not self.owner_user_id or not self.document_id or not self.document_kind or not self.source_digest:
                raise ValueError("memory document projections require owner, document identity, kind and digest")
            if self.primary_tree_path and not self.primary_tree_path.startswith("memories/"):
                raise ValueError("memory document projections require a memories tree path")
            if record_kind == CatalogRecordKind.MEMORY_BLOCK.value and not self.block_id:
                raise ValueError("memory block projections require block_id")
        for name in ("hotness", "semantic_hotness", "behavior_support_hotness"):
            score = float(getattr(self, name))
            if score != score or score in {float("inf"), float("-inf")}:
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, max(0.0, min(1.0, score)))

    @property
    def path_depth(self) -> int:
        return len(self.primary_tree_path.split("/")) if self.primary_tree_path else 0

    def with_sanitized_projection(self, sanitizer: ContextProjectionSanitizer | None = None) -> CatalogRecord:
        policy = sanitizer or ContextProjectionSanitizer()
        projection_metadata = dict(self.metadata)
        # Catalog columns are authoritative for logical placement.  Metadata
        # mirrors must never retain the unsanitized path supplied by a Source
        # object after ``__post_init__`` has normalized Primary and Secondary
        # paths through the controlled taxonomy.
        if "tree_paths" in projection_metadata:
            projection_metadata["tree_paths"] = list(self.tree_paths)
        if "primary_tree_path" in projection_metadata:
            projection_metadata["primary_tree_path"] = self.primary_tree_path
        safe = policy.sanitize(
            title=self.title,
            l0_text=self.l0_text,
            l1_text=self.l1_text,
            metadata=projection_metadata,
            source_kind=self.source_kind,
        )
        return replace(
            self,
            title=safe.title,
            l0_text=safe.l0_text,
            l1_text=safe.l1_text,
            metadata=safe.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "tree_paths": list(self.tree_paths),
            "metadata": dict(self.metadata),
            "path_depth": self.path_depth,
        }

    @classmethod
    def from_context_object(
        cls,
        obj: ContextObject,
        *,
        content: str = "",
        record_key: str | None = None,
        record_kind: str | None = None,
        tree_paths: Sequence[str] | None = None,
    ) -> CatalogRecord:
        metadata = dict(obj.metadata or {})
        scope = _mapping(metadata.get("scope"))
        fields = _mapping(metadata.get("fields"))
        connect = _mapping(metadata.get("connect"))
        applicability = _mapping(scope.get("applicability"))
        workspace = next(
            (
                str(item.get("id"))
                for item in applicability.get("all_of", []) or []
                if isinstance(item, Mapping) and item.get("kind") == "workspace"
            ),
            "",
        )
        workspace_id = str(
            scope.get("project_id")
            or fields.get("project_id")
            or metadata.get("workspace_id")
            or metadata.get("project_id")
            or workspace
            or ""
        )
        raw_paths = tuple(tree_paths or metadata.get("tree_paths", ()) or ())
        primary = str(metadata.get("primary_tree_path") or (raw_paths[0] if raw_paths else ""))
        projection_source_revision = int(
            metadata.get("projection_source_revision")
            or metadata.get("source_revision")
            or metadata.get("revision")
            or 0
        )
        resolved_record_kind = record_kind or str(metadata.get("record_kind") or CatalogRecordKind.CONTEXT.value)
        # Generic Context projection remains URI-keyed. Domain projectors pass
        # their stable, tenant-scoped record key explicitly.
        resolved_record_key = record_key or obj.uri
        source_uri = str(metadata.get("source_uri") or obj.uri)
        source_digest = str(
            metadata.get("source_digest") or ContextProjectionSanitizer().digest(content or obj.to_dict())
        )
        return cls(
            record_key=resolved_record_key,
            uri=obj.uri,
            tenant_id=str(obj.tenant_id or "default"),
            owner_user_id=str(obj.owner_user_id or ""),
            workspace_id=workspace_id,
            session_id=str(metadata.get("session_id") or ""),
            adapter_id=str(connect.get("adapter_id") or metadata.get("source_adapter_id") or ""),
            context_type=obj.context_type.value,
            source_kind=str(metadata.get("source_kind") or connect.get("source_kind") or "context"),
            record_kind=resolved_record_kind,
            lifecycle_state=obj.lifecycle_state.value,
            parent_uri=str(metadata.get("parent_uri") or ""),
            primary_tree_path=primary,
            tree_paths=raw_paths,
            created_at=obj.created_at,
            updated_at=obj.updated_at,
            event_time=str(metadata.get("event_time") or metadata.get("occurred_at") or obj.created_at or ""),
            ingested_at=str(metadata.get("ingested_at") or obj.created_at or ""),
            transaction_time=str(metadata.get("transaction_time") or obj.updated_at or obj.created_at or ""),
            title=obj.title,
            l0_text=str(metadata.get("l0_text") or obj.title),
            l1_text=str(metadata.get("l1_text") or metadata.get("summary") or content),
            l2_uri=str(obj.layers.l2_uri or obj.uri),
            source_uri=source_uri,
            source_digest=source_digest,
            source_revision=projection_source_revision,
            document_id=str(metadata.get("document_id") or ""),
            block_id=str(metadata.get("block_id") or ""),
            document_kind=str(metadata.get("document_kind") or ""),
            document_revision=int(metadata.get("document_revision") or 0),
            projection_generation=int(metadata.get("projection_generation") or 0),
            projection_effect_hash=str(
                metadata.get("projection_effect_hash") or metadata.get("projection_input_effect_hash") or ""
            ),
            hotness=obj.hotness,
            semantic_hotness=obj.semantic_hotness,
            behavior_support_hotness=obj.behavior_support_hotness,
            serving_tier=str(metadata.get("serving_tier") or ServingTier.HOT.value),
            projection_status=str(metadata.get("projection_status") or CatalogProjectionStatus.PROJECTED.value),
            metadata=metadata,
        )


def catalog_vector_metadata(
    record: CatalogRecord,
    *,
    sanitizer: ContextProjectionSanitizer | None = None,
) -> dict[str, Any]:
    """Return the complete, sanitized filter contract for one vector row.

    Vector databases are a rebuildable serving layer.  Every row therefore
    carries the exact Catalog identity and all trusted structured-filter
    dimensions needed to filter before Top-K. Raw projection metadata is not
    copied: only typed Catalog fields and validated scope keys are emitted.
    """

    policy = sanitizer or ContextProjectionSanitizer()
    payload = {
        "catalog_record_key": record.record_key,
        "uri": record.uri,
        "tenant_id": record.tenant_id,
        "owner_user_id": record.owner_user_id,
        "workspace_id": record.workspace_id,
        "workspace_shared": record.workspace_shared,
        "session_id": record.session_id,
        "adapter_id": record.adapter_id,
        "context_type": record.context_type,
        "source_kind": record.source_kind,
        "record_kind": record.record_kind,
        "lifecycle_state": record.lifecycle_state,
        "primary_tree_path": record.primary_tree_path,
        "tree_paths": record.tree_paths,
        "scope_keys": _catalog_scope_keys(record.metadata),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "event_time": record.event_time,
        "ingested_at": record.ingested_at,
        "transaction_time": record.transaction_time,
        "source_uri": record.source_uri,
        "source_digest": record.source_digest,
        "source_revision": record.source_revision,
        "document_id": record.document_id,
        "block_id": record.block_id,
        "document_kind": record.document_kind,
        "document_revision": record.document_revision,
        "projection_generation": record.projection_generation,
        "projection_effect_hash": record.projection_effect_hash,
        "serving_tier": record.serving_tier,
        "projection_status": record.projection_status,
    }
    safe = policy.sanitize_trace(payload)
    if not isinstance(safe, Mapping):
        raise ValueError("vector metadata sanitizer returned a non-object")
    result = dict(safe)
    # ``record_key`` is a server-owned join/CAS identity, not free-form
    # content. Generic credential redaction must not detach a trusted row
    # identity from its Catalog owner.
    result["catalog_record_key"] = record.record_key
    return result


def _catalog_scope_keys(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    explicit = metadata.get("scope_keys")
    if explicit is not None:
        if not isinstance(explicit, Sequence) or isinstance(explicit, str | bytes):
            raise ValueError("Catalog vector scope_keys must be an array")
        keys = tuple(str(item).strip() for item in explicit)
        if any(not key or "\x00" in key for key in keys):
            raise ValueError("Catalog vector scope_keys contain an invalid key")
        return tuple(dict.fromkeys(keys))
    scope = metadata.get("scope")
    if scope is None:
        return ()
    if not isinstance(scope, Mapping):
        raise ValueError("Catalog vector scope must be an object")
    applicability = scope.get("applicability")
    if not isinstance(applicability, Mapping):
        raise ValueError("Catalog vector applicability must be an object")
    return scope_keys_from_payloads(applicability.get("all_of"))


def _workspace_is_publicly_shared(
    metadata: Mapping[str, Any],
    *,
    tenant_id: str,
    workspace_id: str,
) -> bool:
    """Derive a fail-closed, indexable workspace visibility bit.

    Workspace applicability alone is not visibility: profile/preferences may
    carry a workspace applicability ref while remaining principal-private.
    Only an explicit tenant-public scope is eligible for cross-owner candidate
    generation; exact Source validation still runs after fusion.
    """

    if not workspace_id:
        return False
    scope = metadata.get("scope")
    if not isinstance(scope, Mapping):
        return False
    visibility = scope.get("visibility")
    applicability = scope.get("applicability")
    if not isinstance(visibility, Mapping) or not isinstance(applicability, Mapping):
        return False
    if (
        str(visibility.get("tenant_id") or "") != str(tenant_id)
        or visibility.get("private") is not False
        or visibility.get("allowed_principal_ids") not in ([], ())
        or visibility.get("allowed_service_ids") not in ([], ())
    ):
        return False
    all_of = applicability.get("all_of")
    if not isinstance(all_of, Sequence) or isinstance(all_of, str | bytes):
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("kind") == "workspace"
        and normalize_workspace_id(item.get("id")) == workspace_id
        for item in all_of
    )


def normalize_tree_path(path: object) -> str:
    value = str(path or "").strip().strip("/")
    if not value or "\\" in value or "//" in value:
        raise ValueError("tree path must be a normalized relative path")
    segments = value.split("/")
    if len(segments) > MAX_TREE_DEPTH or segments[0] not in _PATH_ROOTS:
        raise ValueError("tree path is outside the controlled taxonomy")
    if any(segment in {".", ".."} or "\x00" in segment for segment in segments):
        raise ValueError("tree path contains an unsafe segment")
    _validate_taxonomy_shape(segments)
    dynamic_indexes = _dynamic_tree_segment_indexes(segments)
    dynamic_values = tuple(segments[index] for index in dynamic_indexes)
    if dynamic_values:
        sanitizer = ContextProjectionSanitizer()
        safe_values = sanitizer.sanitize_tree_segments(dynamic_values)
        for index, safe_value in zip(dynamic_indexes, safe_values, strict=True):
            segments[index] = safe_value
    if any(not _PATH_SEGMENT.fullmatch(segment) for segment in segments):
        raise ValueError("tree path contains an unsafe segment")
    _validate_taxonomy_shape(segments)
    return "/".join(segments)


def _dynamic_tree_segment_indexes(segments: Sequence[str]) -> tuple[int, ...]:
    root = segments[0]
    if root in {"sessions", "projects", "skills", "agents"}:
        return tuple(range(1, len(segments)))
    if root == "memories":
        tail = tuple(segments[1:])
        if len(tail) == 2 and tail[:1] == ("experiences",):
            return (2,)
        if len(tail) == 3 and tail[:2] in {
            ("knowledge", "entities"),
            ("knowledge", "topics"),
            ("knowledge", "episodes"),
        }:
            return (3,)
        return ()
    return ()


def _validate_taxonomy_shape(segments: Sequence[str]) -> None:
    """Keep all logical paths inside the finite, schema-owned taxonomy.

    Dynamic identifiers are permitted only below roots whose shape is fixed.
    This prevents an LLM or caller from inventing an unbounded directory tree
    while still allowing root and partial timeline prefixes for querying.
    """

    root, *tail = segments
    if root == "timeline":
        if len(tail) > 3:
            raise ValueError("timeline path exceeds year/month/day")
        if tail and (len(tail[0]) != 4 or not tail[0].isdigit()):
            raise ValueError("timeline year must use YYYY")
        if len(tail) >= 2 and (len(tail[1]) != 2 or not tail[1].isdigit()):
            raise ValueError("timeline month must use MM")
        if len(tail) == 3 and (len(tail[2]) != 2 or not tail[2].isdigit()):
            raise ValueError("timeline day must use DD")
        if tail:
            year = int(tail[0])
            month = int(tail[1]) if len(tail) >= 2 else 1
            day = int(tail[2]) if len(tail) == 3 else 1
            try:
                datetime(year, month, day)
            except ValueError as exc:
                raise ValueError("timeline path is not a valid calendar date") from exc
        return
    if root in {"sessions", "projects", "skills", "agents"}:
        if len(tail) > 1:
            raise ValueError(f"{root} path accepts one controlled identifier")
        return
    if root == "resources":
        if len(tail) > 1 or (tail and tail[0] not in _RESOURCE_PATH_KINDS):
            raise ValueError("resource path kind is outside the controlled taxonomy")
        return
    if root == "memories":
        path = tuple(tail)
        if not path:
            return
        if path in _MEMORY_FIXED_PATHS:
            return
        if path[:-1] in _MEMORY_DYNAMIC_BRANCHES:
            return
        if path in _MEMORY_DYNAMIC_BRANCHES:
            return
        raise ValueError("memory path is outside the Markdown document taxonomy")
    raise ValueError("tree path is outside the controlled taxonomy")


def validate_tree_paths(paths: Sequence[object], *, primary: object = "") -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(normalize_tree_path(path) for path in paths if str(path or "").strip()))
    primary_value = normalize_tree_path(primary) if str(primary or "").strip() else ""
    if primary_value and primary_value not in normalized:
        normalized = (primary_value, *normalized)
    elif primary_value:
        normalized = (primary_value, *(path for path in normalized if path != primary_value))
    if len(normalized) > MAX_SECONDARY_PATHS + 1:
        raise ValueError("context has too many secondary tree paths")
    return normalized


def normalize_timestamp(value: object, label: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = [
    "CatalogProjectionStatus",
    "CatalogRecord",
    "CatalogRecordKind",
    "MAX_SECONDARY_PATHS",
    "ServingTier",
    "catalog_vector_metadata",
    "normalize_timestamp",
    "normalize_tree_path",
    "validate_tree_paths",
]
