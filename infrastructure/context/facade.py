"""上下文对象读取、普通检索和关系协调门面。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from infrastructure.context.commit_protocol import OrdinaryRelationCommitter
from infrastructure.context.contracts import (
    ContextDomainOverlay,
    ContextIndexPolicy,
    NoContextIndexPolicy,
    NoDomainOverlay,
)
from infrastructure.context.relations.ordinary import (
    NoRelationDomainPolicy,
    RelationDomainPolicy,
    ordinary_relation_serving_eligibility,
)
from infrastructure.store.contracts.index import IndexHit, IndexStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_relation import ContextRelation
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.model.context.context_uri import ContextURI


class ContextDB:
    """只封装上下文对象读取、普通检索和关系语义。"""

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore,
        relation_committer: OrdinaryRelationCommitter | None = None,
        readiness=None,
        tenant_id: str = "",
        serving_lock: RLock | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.relation_committer = relation_committer
        self.tenant_id = str(tenant_id or getattr(source_store, "tenant_id", "default") or "default")
        self.readiness = readiness
        # 离线重建与在线召回共享发布锁，查询只能观察重建前或重建后的完整 Catalog。
        self.serving_lock = serving_lock or RLock()
        self._configure_extensions()

    def _configure_extensions(
        self,
        *,
        domain_overlay: ContextDomainOverlay | None = None,
        index_policy: ContextIndexPolicy | None = None,
        relation_domain_policy: RelationDomainPolicy | None = None,
    ) -> None:
        """安装由组合根负责的领域扩展，不扩大公开构造函数。"""

        self.domain_overlay = domain_overlay or NoDomainOverlay()
        self.index_policy = index_policy or NoContextIndexPolicy()
        self.relation_domain_policy = relation_domain_policy or NoRelationDomainPolicy()

    def _require_ready(self) -> None:
        if self.readiness is not None:
            self.readiness.require_ready()

    def serving_generation_token(self) -> str:
        """返回底层 Catalog 可提供的跨进程服务版本。"""

        provider = getattr(self.index_store, "serving_generation_token", None)
        return str(provider() if callable(provider) else "")

    @contextmanager
    def _mutation_fence(self) -> Iterator[None]:
        """在当前组合实例内串行化事实源写入和 Serving 发布。"""

        with self.serving_lock:
            yield

    def read_object(self, uri: str) -> ContextObject:
        self._require_ready()
        if self.domain_overlay.owns_uri(uri):
            return self.domain_overlay.read_object(self.source_store, self.relation_store, uri)
        obj = self.source_store.read_object(uri)
        if self.domain_overlay.owns_object(obj):
            return self.domain_overlay.read_object(self.source_store, self.relation_store, uri)
        return obj

    def add_relation(self, relation: ContextRelation) -> None:
        with self._mutation_fence():
            self._add_relation_unfenced(relation)

    def _add_relation_unfenced(self, relation: ContextRelation) -> None:
        self._require_ready()
        if self._document_owned_uri(relation.source_uri) or self._document_owned_uri(relation.target_uri):
            raise PermissionError("Markdown document relations can only be published by the document projector")
        domain_source = self._domain_owned_uri(relation.source_uri)
        endpoint_objects: dict[str, ContextObject] = {}
        for uri in (relation.source_uri, relation.target_uri):
            try:
                endpoint = self.source_store.read_object(uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            endpoint_objects[uri] = endpoint
            if self._document_owned_object(endpoint):
                raise PermissionError("Markdown document relations can only be published by the document projector")
            if uri == relation.source_uri and self._domain_owned_object(endpoint):
                domain_source = True
        if domain_source:
            raise PermissionError("domain-owned relations can only be published by their authoritative handler")
        if not endpoint_objects:
            raise FileNotFoundError("ordinary relation requires at least one durable Source endpoint")

        identity = (relation.source_uri, relation.relation_type, relation.target_uri)
        existing_authorities = [
            obj
            for obj in endpoint_objects.values()
            if any((item.source_uri, item.relation_type, item.target_uri) == identity for item in obj.relations)
        ]
        if len(existing_authorities) > 1:
            raise RuntimeError("ordinary relation has multiple Source authorities")
        authority = (
            existing_authorities[0]
            if existing_authorities
            else endpoint_objects.get(relation.source_uri) or endpoint_objects[relation.target_uri]
        )
        if self._domain_owned_object(authority):
            raise PermissionError("ordinary relation requires a generic Source authority")
        tenant_id = str(authority.tenant_id or "default")
        declared_tenant = relation.metadata.get("tenant_id")
        if declared_tenant not in (None, "", tenant_id):
            raise PermissionError("ordinary relation metadata crosses its Source tenant")
        owner_user_id = str(authority.owner_user_id or relation.metadata.get("owner_user_id") or "")
        if not owner_user_id:
            raise ValueError("ordinary relation Source authority requires an owner_user_id")
        declared_owner = relation.metadata.get("owner_user_id")
        if declared_owner not in (None, "", owner_user_id):
            raise PermissionError("ordinary relation metadata crosses its Source owner")
        for uri, endpoint in endpoint_objects.items():
            if uri == authority.uri or ContextURI.parse(uri).authority != "user":
                continue
            if str(endpoint.tenant_id or "default") != tenant_id:
                raise PermissionError("ordinary relation endpoints cross a tenant boundary")
            endpoint_owner = str(endpoint.owner_user_id or "")
            if not self._domain_owned_object(endpoint) and endpoint_owner and endpoint_owner != owner_user_id:
                raise PermissionError("ordinary relation endpoints cross an owner boundary")
        normalized_metadata = dict(relation.metadata or {})
        normalized_metadata.pop("catalog_record_key", None)
        normalized_metadata.update({"tenant_id": tenant_id, "owner_user_id": owner_user_id})
        existing_relation = next(
            (
                item
                for item in authority.relations
                if (item.source_uri, item.relation_type, item.target_uri) == identity
            ),
            None,
        )
        desired = ContextRelation(
            source_uri=relation.source_uri,
            relation_type=relation.relation_type,
            target_uri=relation.target_uri,
            weight=relation.weight,
            metadata=normalized_metadata,
            created_at=(existing_relation.created_at if existing_relation is not None else relation.created_at),
        )
        eligibility = ordinary_relation_serving_eligibility(
            {
                "source_uri": desired.source_uri,
                "relation_type": desired.relation_type,
                "target_uri": desired.target_uri,
                "weight": desired.weight,
                "metadata": dict(desired.metadata or {}),
            },
            authority_uri=authority.uri,
            tenant_id=tenant_id,
            source_store=self.source_store,
            index_store=self.index_store,
            domain_policy=self.relation_domain_policy,
            domain_reader=lambda uri: self.domain_overlay.read_object(self.source_store, self.relation_store, uri),
        )
        if not eligibility.allowed:
            raise ValueError(
                f"ordinary relation is not serving-eligible: {eligibility.reason or 'endpoint is unavailable'}"
            )
        if existing_relation is not None and self._ordinary_relation_equal(existing_relation, desired):
            self._repair_ordinary_relation_projection(desired, tenant_id=tenant_id)
            return

        desired_authority = ContextObject.from_dict(authority.to_dict())
        updated_relations: list[ContextRelation] = []
        replaced = False
        for item in desired_authority.relations:
            if (item.source_uri, item.relation_type, item.target_uri) != identity:
                updated_relations.append(item)
                continue
            if not replaced:
                updated_relations.append(desired)
                replaced = True
        if not replaced:
            updated_relations.append(desired)
        desired_authority.relations = updated_relations
        desired_authority.updated_at = datetime.now(timezone.utc).isoformat()
        try:
            content = self.source_store.read_content(authority.layers.l2_uri or authority.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            content = ""
        committer = self._ordinary_relation_committer(tenant_id)
        # 只有发生普通关系写入时才加载操作命令，读取门面不应主动装载事务实现图。
        from infrastructure.context.operation_commit import commit_ordinary_relation_update

        commit_ordinary_relation_update(
            committer,
            owner_user_id=owner_user_id,
            desired_authority=desired_authority,
            content=content,
            tenant_id=tenant_id,
        )

    def _ordinary_relation_committer(self, tenant_id: str) -> OrdinaryRelationCommitter:
        committer = self.relation_committer
        if committer is None:
            raise RuntimeError("ContextDB.add_relation requires an injected OrdinaryRelationCommitter")
        if not callable(getattr(committer, "commit", None)):
            raise RuntimeError("ContextDB.add_relation requires a generic operation committer")
        if (
            committer.source_store is not self.source_store
            or committer.index_store is not self.index_store
            or committer.relation_store is not self.relation_store
            or committer.tenant_id != tenant_id
        ):
            raise RuntimeError("ContextDB.add_relation committer differs from its bound stores")
        return committer

    def _repair_ordinary_relation_projection(self, relation: ContextRelation, *, tenant_id: str) -> None:
        reconcile = getattr(self.relation_store, "reconcile_ordinary_relations", None)
        if callable(reconcile):
            reconcile((relation,), tenant_id=tenant_id)
            return
        self.relation_store.add_relation(relation, tenant_id=tenant_id)

    @staticmethod
    def _ordinary_relation_equal(left: ContextRelation, right: ContextRelation) -> bool:
        return (
            left.source_uri == right.source_uri
            and left.relation_type == right.relation_type
            and left.target_uri == right.target_uri
            and left.weight == right.weight
            and dict(left.metadata or {}) == dict(right.metadata or {})
        )

    def search(
        self,
        query: str,
        *,
        owner_user_id: str | None = None,
        context_type: ContextType | None = None,
        limit: int = 10,
        tenant_id: str = "",
        project_id: str = "",
        adapter_id: str = "",
        admission_status: str = "",
        allowed_uris: list[str] | tuple[str, ...] | None = None,
    ) -> list[IndexHit]:
        """按给定条件查找匹配结果。"""

        self._require_ready()
        filters: dict[str, Any] = {}
        if owner_user_id is not None:
            filters["owner_user_id"] = owner_user_id
        if tenant_id:
            filters["tenant_id"] = tenant_id
        if context_type is not None:
            filters["context_type"] = context_type.value
        if project_id:
            filters["project_id"] = project_id
        if adapter_id:
            filters["adapter_id"] = adapter_id
        if admission_status:
            filters["admission_status"] = admission_status
        if allowed_uris is not None:
            filters["allowed_uris"] = tuple(allowed_uris)
        effective_tenant = str(tenant_id or self.tenant_id)
        filters["tenant_id"] = effective_tenant
        hits = self.index_store.search(
            query,
            tenant_id=effective_tenant,
            filters=filters,
            limit=limit,
        )
        visible: list[IndexHit] = []
        for hit in hits:
            if self.index_policy.owns_index_entry(
                self.source_store,
                hit.uri,
                dict(hit.metadata or {}),
            ):
                continue
            try:
                obj = self.source_store.read_object(hit.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            if self._domain_owned_object(obj):
                continue
            visible.append(hit)
        return visible

    def relations_of(
        self,
        uri: str,
        *,
        owner_user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[ContextRelation]:
        self._require_ready()
        effective_tenant = str(tenant_id or self.tenant_id)
        domain_target = self._domain_owned_uri(uri)
        if not domain_target:
            try:
                domain_target = self._domain_owned_object(self.source_store.read_object(uri))
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                domain_target = False
        if domain_target:
            return self.domain_overlay.relations_of(
                self.source_store,
                self.relation_store,
                uri,
                owner_user_id=owner_user_id,
                tenant_id=effective_tenant,
            )
        return self.relation_store.relations_of(
            uri,
            tenant_id=effective_tenant,
            owner_user_id=owner_user_id,
        )

    def _domain_owned_object(self, obj: ContextObject) -> bool:
        return self.domain_overlay.owns_object(obj)

    def _domain_owned_uri(self, uri: str) -> bool:
        return self.domain_overlay.owns_uri(uri)

    @staticmethod
    def _document_owned_object(obj: ContextObject) -> bool:
        return obj.context_type is ContextType.MEMORY or ContextDB._document_owned_uri(obj.uri)

    @staticmethod
    def _document_owned_uri(uri: str) -> bool:
        raw = str(uri or "")
        return raw.startswith("memoryos://user/") and "/memory/documents/" in raw


__all__ = ["ContextDB"]
