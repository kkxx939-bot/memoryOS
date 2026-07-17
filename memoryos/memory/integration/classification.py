"""Memory-owned classification for canonical and pending context objects."""

from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject

CANONICAL_MEMORY_KINDS = frozenset({"slot", "claim", "pending_proposal"})
CANONICAL_MEMORY_SCHEMA_VERSIONS = frozenset(
    {"canonical_memory_v2", "canonical_pending_proposal_v1"}
)


def is_canonical_memory_uri(uri: str) -> bool:
    return "/memories/canonical/" in uri or "/memories/pending/" in uri


def is_canonical_memory_object(obj: ContextObject) -> bool:
    return (
        str(dict(obj.metadata or {}).get("canonical_kind") or "") in CANONICAL_MEMORY_KINDS
        or obj.schema_version in CANONICAL_MEMORY_SCHEMA_VERSIONS
        or is_canonical_memory_uri(obj.uri)
    )


__all__ = [
    "CANONICAL_MEMORY_KINDS",
    "CANONICAL_MEMORY_SCHEMA_VERSIONS",
    "is_canonical_memory_object",
    "is_canonical_memory_uri",
]
