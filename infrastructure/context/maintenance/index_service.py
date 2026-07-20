"""通用上下文索引校验与重建编排。"""

from __future__ import annotations

from threading import RLock
from typing import Any, Protocol

from infrastructure.context.contracts import (
    ContextDomainOverlay,
    ContextIndexPolicy,
    NoContextIndexPolicy,
    NoDomainOverlay,
)
from infrastructure.context.maintenance.index_consistency import (
    IndexConsistencyResult,
    IndexConsistencyService,
)
from infrastructure.context.relations.ordinary import (
    NoRelationDomainPolicy,
    RelationDomainPolicy,
)
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore


class ContextAdministrationService(Protocol):
    """索引重建和一致性检查的独立维护能力。"""

    def rebuild_index(self, *, owner_user_id: str | None = None) -> dict[str, Any]: ...

    def verify_consistency(self, *, owner_user_id: str | None = None) -> dict[str, Any]: ...


class GenericContextMaintenance:
    """手动组合运行时时使用的领域无关索引维护实现。"""

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore,
        *,
        tenant_id: str,
        domain_overlay: ContextDomainOverlay | None = None,
        index_policy: ContextIndexPolicy | None = None,
        readiness: Any | None = None,
        serving_lock: RLock | None = None,
        relation_domain_policy: RelationDomainPolicy | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.tenant_id = str(tenant_id or "").strip()
        if not self.tenant_id:
            raise ValueError("generic maintenance requires an explicit tenant_id")
        self.domain_overlay = domain_overlay or NoDomainOverlay()
        self.index_policy = index_policy or NoContextIndexPolicy()
        self.readiness = readiness
        self.serving_lock = serving_lock or RLock()
        self.relation_domain_policy = relation_domain_policy or NoRelationDomainPolicy()

    def _require_ready(self) -> None:
        if self.readiness is not None:
            self.readiness.require_ready()

    def _service(self) -> IndexConsistencyService:
        return IndexConsistencyService(
            self.source_store,
            self.index_store,
            self.relation_store,
            tenant_id=self.tenant_id,
            domain_overlay=self.domain_overlay,
            index_policy=self.index_policy,
        )

    def rebuild_index(self, *, owner_user_id: str | None = None) -> dict[str, Any]:
        self._require_ready()
        with self.serving_lock:
            return self._payload(self._service().rebuild(owner_user_id=owner_user_id))

    def verify_consistency(self, *, owner_user_id: str | None = None) -> dict[str, Any]:
        self._require_ready()
        result = self._service().verify(owner_user_id=owner_user_id)
        if owner_user_id is None:
            return self._payload(result)
        return self._owner_payload(result, owner_user_id=owner_user_id)

    def _owner_payload(
        self,
        result: IndexConsistencyResult,
        *,
        owner_user_id: str,
    ) -> dict[str, Any]:
        payload = self._payload(result)
        source_uris = {
            obj.uri
            for obj in self.source_store.list_objects()
            if obj.owner_user_id == owner_user_id and not self.domain_overlay.owns_object(obj)
        }
        tenant_id = self._tenant_id()
        indexed_uris = set(self.index_store.indexed_uris(tenant_id=tenant_id))
        payload["source_count"] = len(source_uris)
        payload["indexed_count"] = len(source_uris & indexed_uris)
        payload["missing_index"] = [uri for uri in payload["missing_index"] if uri in source_uris]
        payload["dangling_index"] = [
            uri for uri in payload["dangling_index"] if uri.startswith(f"memoryos://user/{owner_user_id}/")
        ]
        payload["broken_relations"] = [
            relation
            for relation in payload["broken_relations"]
            if relation.get("source_uri") in source_uris or relation.get("target_uri") in source_uris
        ]
        payload["consistent"] = not (
            payload["missing_index"]
            or payload["dangling_index"]
            or payload["deleted_or_archived_in_default_search"]
            or payload["broken_relations"]
        )
        return payload

    def _tenant_id(self) -> str:
        return self.tenant_id

    @staticmethod
    def _payload(result: IndexConsistencyResult) -> dict[str, Any]:
        return {
            "source_count": result.source_count,
            "indexed_count": result.index_count,
            "missing_index": result.missing_in_index,
            "dangling_index": result.orphan_index,
            "deleted_or_archived_in_default_search": (result.deleted_or_archived_in_default_search),
            "broken_relations": result.broken_relations,
            "consistent": result.consistent,
        }


__all__ = ["ContextAdministrationService", "GenericContextMaintenance"]
