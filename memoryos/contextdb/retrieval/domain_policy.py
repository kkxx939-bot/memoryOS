"""Optional domain policy used by generic retrieval primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.types import scope_keys_from_payloads


class ContextRetrievalDomainPolicy(Protocol):
    """Narrow domain hooks required to validate a serving candidate."""

    def read_serving_object(
        self,
        source_store: SourceStore,
        uri: str,
    ) -> ContextObject: ...

    def applicability_scope_keys(
        self,
        metadata: Mapping[str, Any],
        *,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[str, ...] | None: ...

    def is_authoritative_integrity_error(self, error: BaseException) -> bool: ...


class NoRetrievalDomainPolicy:
    """Domain-neutral candidate validation."""

    def read_serving_object(
        self,
        source_store: SourceStore,
        uri: str,
    ) -> ContextObject:
        return source_store.read_object(uri)

    def applicability_scope_keys(
        self,
        metadata: Mapping[str, Any],
        *,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[str, ...] | None:
        del tenant_id, owner_user_id
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

    def is_authoritative_integrity_error(self, error: BaseException) -> bool:
        del error
        return False


__all__ = ["ContextRetrievalDomainPolicy", "NoRetrievalDomainPolicy"]
