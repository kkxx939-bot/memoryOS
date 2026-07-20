"""通用召回组件使用的可选领域策略。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from foundation.scope import scope_keys_from_payloads
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_object import ContextObject


class ContextRetrievalDomainPolicy(Protocol):
    """校验 Serving 候选所需的最小领域钩子。"""

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
    """不叠加领域规则的候选校验实现。"""

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
