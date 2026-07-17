"""Generic index verification and rebuild orchestration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from typing import Any, Protocol

from memoryos.contextdb.extensions import (
    ContextDomainOverlay,
    ContextIndexPolicy,
    NoContextIndexPolicy,
    NoDomainOverlay,
)
from memoryos.contextdb.ordinary_relations import (
    NoRelationDomainPolicy,
    RelationDomainPolicy,
)
from memoryos.contextdb.store.index_consistency import (
    IndexConsistencyResult,
    IndexConsistencyService,
)
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore


class ContextDBAdministration(Protocol):
    """Administrative capability injected into the ContextDB facade."""

    def rebuild_index(self, *, owner_user_id: str | None = None) -> dict[str, Any]: ...

    def resume_derived_serving_rebuild_if_needed(self) -> dict[str, Any]: ...

    def rollback_derived_serving_rebuild(self, reason: str) -> dict[str, Any]: ...

    def verify_consistency(self, *, owner_user_id: str | None = None) -> dict[str, Any]: ...


class GenericContextMaintenance:
    """Domain-neutral maintenance used by a directly assembled ContextDB."""

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore,
        *,
        domain_overlay: ContextDomainOverlay | None = None,
        index_policy: ContextIndexPolicy | None = None,
        migration_gate: Any | None = None,
        readiness: Any | None = None,
        serving_lock: RLock | None = None,
        relation_domain_policy: RelationDomainPolicy | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.domain_overlay = domain_overlay or NoDomainOverlay()
        self.index_policy = index_policy or NoContextIndexPolicy()
        self.migration_gate = migration_gate
        self.readiness = readiness
        self.serving_lock = serving_lock or RLock()
        self.relation_domain_policy = relation_domain_policy or NoRelationDomainPolicy()

    @contextmanager
    def _projection_fence(self) -> Iterator[None]:
        acquire = getattr(self.migration_gate, "acquire_projection_fence", None)
        release = getattr(self.migration_gate, "release_projection_fence", None)
        token = acquire() if callable(acquire) else None
        try:
            yield
        finally:
            if callable(release):
                release(token)

    def _require_ready(self) -> None:
        if self.readiness is not None:
            self.readiness.require_ready()

    def _service(self) -> IndexConsistencyService:
        return IndexConsistencyService(
            self.source_store,
            self.index_store,
            self.relation_store,
            migration_gate=self.migration_gate,
            domain_overlay=self.domain_overlay,
            index_policy=self.index_policy,
            relation_domain_policy=self.relation_domain_policy,
        )

    def rebuild_index(self, *, owner_user_id: str | None = None) -> dict[str, Any]:
        with self._projection_fence():
            self._require_ready()
            with self.serving_lock:
                if owner_user_id is None:
                    result = self._service().rebuild(projection_fence_held=True)
                    return self._payload(result)
                rebuilt = 0
                for obj in self.source_store.list_objects():
                    if (
                        obj.owner_user_id != owner_user_id
                        or self.domain_overlay.owns_object(obj)
                    ):
                        continue
                    try:
                        content = self.source_store.read_content(obj.uri)
                    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                        content = obj.title
                    self.index_store.upsert_index(obj, content=content)
                    rebuilt += 1
                payload = self._owner_payload(
                    self._service().verify(),
                    owner_user_id=owner_user_id,
                )
                payload["rebuilt_count"] = rebuilt
                return payload

    def resume_derived_serving_rebuild_if_needed(self) -> dict[str, Any]:
        return {"resumed": False, "state": "NOT_CONFIGURED"}

    def rollback_derived_serving_rebuild(self, reason: str) -> dict[str, Any]:
        del reason
        raise RuntimeError("generic ContextDB has no derived serving rebuild journal")

    def verify_consistency(self, *, owner_user_id: str | None = None) -> dict[str, Any]:
        self._require_ready()
        result = self._service().verify()
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
            if obj.owner_user_id == owner_user_id
            and not self.domain_overlay.owns_object(obj)
        }
        indexed_uris = set(self.index_store.indexed_uris())
        payload["source_count"] = len(source_uris)
        payload["indexed_count"] = len(source_uris & indexed_uris)
        payload["missing_index"] = [
            uri for uri in payload["missing_index"] if uri in source_uris
        ]
        payload["dangling_index"] = [
            uri
            for uri in payload["dangling_index"]
            if uri.startswith(f"memoryos://user/{owner_user_id}/")
        ]
        payload["broken_relations"] = [
            relation
            for relation in payload["broken_relations"]
            if relation.get("source_uri") in source_uris
            or relation.get("target_uri") in source_uris
        ]
        payload["consistent"] = not (
            payload["missing_index"]
            or payload["dangling_index"]
            or payload["deleted_or_archived_in_default_search"]
            or payload["broken_relations"]
        )
        return payload

    @staticmethod
    def _payload(result: IndexConsistencyResult) -> dict[str, Any]:
        return {
            "source_count": result.source_count,
            "indexed_count": result.index_count,
            "missing_index": result.missing_in_index,
            "dangling_index": result.orphan_index,
            "deleted_or_archived_in_default_search": (
                result.deleted_or_archived_in_default_search
            ),
            "broken_relations": result.broken_relations,
            "consistent": result.consistent,
        }


__all__ = ["ContextDBAdministration", "GenericContextMaintenance"]
