"""按租户校验和重建普通上下文的 Serving 索引。"""

from __future__ import annotations

from dataclasses import dataclass, field

from infrastructure.context.contracts import (
    ContextDomainOverlay,
    ContextIndexPolicy,
    NoContextIndexPolicy,
    NoDomainOverlay,
)
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.lifecycle import LifecycleState

_NON_SERVING_STATES = {
    LifecycleState.DELETED,
    LifecycleState.ARCHIVED,
    LifecycleState.OBSOLETE,
}


def _validate_tenant(tenant_id: str) -> str:
    tenant = str(tenant_id).strip()
    if not tenant or "\x00" in tenant:
        raise ValueError("index consistency requires an explicit valid tenant_id")
    return tenant


def prepare_generic_index_rebuild(
    source_store: SourceStore,
    index_store: IndexStore,
    *,
    tenant_id: str,
    owner_user_id: str | None = None,
    index_policy: ContextIndexPolicy | None = None,
) -> dict[str, int]:
    """只删除能够从 SourceStore 重建的普通索引记录。

    Session 和领域投影使用不同于公开 URI 的 Catalog 键，只有所属投影器发布替代记录时
    才能更新；限定所有者的修复不能修改其他所有者的记录。
    """

    tenant = _validate_tenant(tenant_id)
    policy = index_policy or NoContextIndexPolicy()
    indexed = tuple(index_store.indexed_uris(tenant_id=tenant))
    preserved = 0
    removed = 0
    for uri in indexed:
        metadata = index_store.get_index_metadata(uri, tenant_id=tenant)
        if owner_user_id is not None and str((metadata or {}).get("owner_user_id") or "") != owner_user_id:
            preserved += 1
            continue
        domain_owned = policy.owns_index_entry(source_store, uri, metadata)
        derived = bool(metadata and str(metadata.get("record_key") or uri) != uri)
        if derived or (
            domain_owned
            and policy.preserve_index_entry(source_store, index_store, uri, metadata)
        ):
            preserved += 1
            continue
        index_store.delete_index(uri, tenant_id=tenant)
        removed += 1
    return {"removed": removed, "derived_preserved": preserved}


@dataclass(frozen=True)
class IndexConsistencyResult:
    source_count: int
    index_count: int
    missing_in_index: list[str]
    orphan_index: list[str] = field(default_factory=list)
    deleted_or_archived_in_default_search: list[str] = field(default_factory=list)
    broken_relations: list[dict] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return not (
            self.missing_in_index
            or self.orphan_index
            or self.deleted_or_archived_in_default_search
            or self.broken_relations
        )


class IndexConsistencyService:
    """只针对一个明确租户校验或重建普通上下文索引。"""

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore | None = None,
        *,
        tenant_id: str,
        domain_overlay: ContextDomainOverlay | None = None,
        index_policy: ContextIndexPolicy | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.tenant_id = _validate_tenant(tenant_id)
        self.domain_overlay = domain_overlay or NoDomainOverlay()
        self.index_policy = index_policy or NoContextIndexPolicy()

    def verify(self, *, owner_user_id: str | None = None) -> IndexConsistencyResult:
        objects = [
            obj
            for obj in self.source_store.list_objects()
            if str(obj.tenant_id) == self.tenant_id
            and not self.domain_overlay.owns_object(obj)
            and (owner_user_id is None or obj.owner_user_id == owner_user_id)
        ]
        source_uris = {obj.uri for obj in objects}
        indexed_uris: set[str] = set()
        derived_uris: set[str] = set()
        for uri in self.index_store.indexed_uris(tenant_id=self.tenant_id):
            metadata = self.index_store.get_index_metadata(uri, tenant_id=self.tenant_id)
            if owner_user_id is not None and str((metadata or {}).get("owner_user_id") or "") != owner_user_id:
                continue
            if self.index_policy.owns_index_entry(self.source_store, uri, metadata):
                continue
            indexed_uris.add(uri)
            if metadata and str(metadata.get("record_key") or uri) != uri:
                derived_uris.add(uri)

        missing = sorted(
            obj.uri
            for obj in objects
            if obj.lifecycle_state not in _NON_SERVING_STATES and obj.uri not in indexed_uris
        )
        orphan = sorted(uri for uri in indexed_uris - derived_uris if uri not in source_uris)
        visible_retired: list[str] = []
        for obj in objects:
            if obj.lifecycle_state not in _NON_SERVING_STATES:
                continue
            hits = self.index_store.search(
                obj.title or obj.uri,
                tenant_id=self.tenant_id,
                filters={
                    "tenant_id": self.tenant_id,
                    "owner_user_id": obj.owner_user_id,
                    "context_type": obj.context_type.value,
                },
                limit=50,
            )
            if obj.uri in {hit.uri for hit in hits}:
                visible_retired.append(obj.uri)
        return IndexConsistencyResult(
            source_count=len(objects),
            index_count=len(indexed_uris),
            missing_in_index=missing,
            orphan_index=orphan,
            deleted_or_archived_in_default_search=sorted(visible_retired),
            broken_relations=self._broken_relations(source_uris),
        )

    def rebuild(self, *, owner_user_id: str | None = None) -> IndexConsistencyResult:
        prepare_generic_index_rebuild(
            self.source_store,
            self.index_store,
            tenant_id=self.tenant_id,
            owner_user_id=owner_user_id,
            index_policy=self.index_policy,
        )
        for obj in self.source_store.list_objects():
            if str(obj.tenant_id) != self.tenant_id:
                continue
            if owner_user_id is not None and obj.owner_user_id != owner_user_id:
                continue
            if self.domain_overlay.owns_object(obj):
                continue
            if obj.lifecycle_state in _NON_SERVING_STATES:
                self.index_store.delete_index(obj.uri, tenant_id=self.tenant_id)
                continue
            try:
                content = self.source_store.read_content(obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                content = ""
            self.index_store.upsert_index(
                obj,
                content=content,
                tenant_id=self.tenant_id,
            )
        return self.verify(owner_user_id=owner_user_id)

    def _broken_relations(self, source_uris: set[str]) -> list[dict]:
        if self.relation_store is None:
            return []
        broken: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for uri in source_uris:
            for relation in self.relation_store.relations_of(
                uri,
                tenant_id=self.tenant_id,
            ):
                key = (relation.source_uri, relation.relation_type, relation.target_uri)
                if key in seen:
                    continue
                seen.add(key)
                if self._global_uri(relation.source_uri) or self._global_uri(relation.target_uri):
                    continue
                if self.domain_overlay.owns_uri(
                    relation.source_uri
                ) or self.domain_overlay.owns_uri(relation.target_uri):
                    continue
                if relation.source_uri not in source_uris or relation.target_uri not in source_uris:
                    broken.append(relation.to_dict())
        return broken

    @staticmethod
    def _global_uri(uri: str) -> bool:
        return uri.startswith(("memoryos://resources/", "memoryos://skills/"))


__all__ = [
    "IndexConsistencyResult",
    "IndexConsistencyService",
    "prepare_generic_index_rebuild",
]
