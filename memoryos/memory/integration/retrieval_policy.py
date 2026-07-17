"""Canonical-memory validation hooks for generic ContextDB retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.types import scope_keys_from_payloads
from memoryos.memory.canonical.scope import MemoryScope
from memoryos.memory.canonical.visibility import (
    CommittedStateIntegrityError,
    read_committed_canonical,
)
from memoryos.memory.integration.classification import (
    is_canonical_memory_object,
    is_canonical_memory_uri,
)


class CanonicalMemoryRetrievalPolicy:
    """Apply receipt-backed reads and canonical scope/authority validation."""

    def read_serving_object(
        self,
        source_store: SourceStore,
        uri: str,
    ) -> ContextObject:
        if is_canonical_memory_uri(uri):
            return read_committed_canonical(source_store, uri).object
        obj = source_store.read_object(uri)
        if is_canonical_memory_object(obj):
            return read_committed_canonical(source_store, uri).object
        return obj

    def applicability_scope_keys(
        self,
        metadata: Mapping[str, Any],
        *,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[str, ...] | None:
        if metadata.get("canonical_kind") not in {
            "claim",
            "slot",
            "pending_proposal",
        }:
            raw_scope = metadata.get("scope", {}) or {}
            if not isinstance(raw_scope, Mapping):
                return None
            raw_applicability = raw_scope.get("applicability", {}) or {}
            if not isinstance(raw_applicability, Mapping):
                return None
            try:
                return scope_keys_from_payloads(raw_applicability.get("all_of", ()))
            except (KeyError, TypeError, ValueError):
                return None

        raw_scope = metadata.get("scope", {}) or {}
        if not isinstance(raw_scope, Mapping):
            return None
        try:
            scope = MemoryScope.from_dict(raw_scope)
        except (KeyError, TypeError, ValueError):
            return None
        if scope.canonical_subject is None:
            return None
        if scope.visibility.tenant_id != tenant_id or scope.authority.inferred:
            return None
        asserted_by = str(
            metadata.get("asserted_by")
            or (owner_user_id if metadata.get("canonical_kind") == "pending_proposal" else "")
            or ""
        )
        asserted_by_service = str(metadata.get("asserted_by_service") or "")
        if (scope.authority.principal_ids or scope.authority.service_ids) and not (
            asserted_by in set(scope.authority.principal_ids)
            or asserted_by_service in set(scope.authority.service_ids)
        ):
            return None
        return tuple(item.key for item in scope.applicability.all_of)

    def is_authoritative_integrity_error(self, error: BaseException) -> bool:
        return isinstance(error, CommittedStateIntegrityError)


__all__ = ["CanonicalMemoryRetrievalPolicy"]
