"""上下文查询和维护流程使用的领域扩展协议。"""

from __future__ import annotations

from typing import Any, Protocol

from infrastructure.store.contracts.domain import ContextDomainClassifier
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_relation import ContextRelation


class ContextDomainOverlay(ContextDomainClassifier, Protocol):
    """读取由专属领域负责的已提交上下文视图。"""

    def owns_uri(self, uri: str) -> bool: ...

    def owns_object(self, obj: ContextObject) -> bool: ...

    def read_object(
        self,
        source_store: SourceStore,
        relation_store: RelationStore,
        uri: str,
    ) -> ContextObject: ...

    def relations_of(
        self,
        source_store: SourceStore,
        relation_store: RelationStore,
        uri: str,
        *,
        owner_user_id: str | None = None,
        tenant_id: str,
    ) -> list[ContextRelation]: ...


class ContextIndexPolicy(Protocol):
    """在通用索引重建期间保护专属领域的 Serving 记录。"""

    def owns_index_entry(
        self,
        source_store: SourceStore,
        uri: str,
        metadata: dict[str, Any] | None,
    ) -> bool: ...

    def preserve_index_entry(
        self,
        source_store: SourceStore,
        index_store: Any,
        uri: str,
        metadata: dict[str, Any] | None,
    ) -> bool: ...


class ContextObjectReader(Protocol):
    """精确读取服务所需的最小上下文对象能力。"""

    def read_object(self, uri: str) -> ContextObject: ...


class NoDomainOverlay:
    """未配置专属领域时使用的默认上下文覆盖层。"""

    def owns_uri(self, uri: str) -> bool:
        del uri
        return False

    def owns_object(self, obj: ContextObject) -> bool:
        del obj
        return False

    def read_object(
        self,
        source_store: SourceStore,
        relation_store: RelationStore,
        uri: str,
    ) -> ContextObject:
        del relation_store
        return source_store.read_object(uri)

    def relations_of(
        self,
        source_store: SourceStore,
        relation_store: RelationStore,
        uri: str,
        *,
        owner_user_id: str | None = None,
        tenant_id: str,
    ) -> list[ContextRelation]:
        del source_store
        return relation_store.relations_of(
            uri,
            owner_user_id=owner_user_id,
            tenant_id=tenant_id,
        )


class NoContextIndexPolicy:
    """未配置专属领域时使用的默认索引策略。"""

    def owns_index_entry(
        self,
        source_store: SourceStore,
        uri: str,
        metadata: dict[str, Any] | None,
    ) -> bool:
        del source_store, uri, metadata
        return False

    def preserve_index_entry(
        self,
        source_store: SourceStore,
        index_store: Any,
        uri: str,
        metadata: dict[str, Any] | None,
    ) -> bool:
        del source_store, index_store, uri, metadata
        return False


__all__ = [
    "ContextDomainOverlay",
    "ContextIndexPolicy",
    "ContextObjectReader",
    "NoContextIndexPolicy",
    "NoDomainOverlay",
]
