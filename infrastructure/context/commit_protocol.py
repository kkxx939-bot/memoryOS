"""普通上下文关系写入所需的最小提交协议。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore

if TYPE_CHECKING:
    from transaction.model.context_operation import ContextOperation


class OrdinaryRelationCommitter(Protocol):
    """关系服务用于耐久更新 Source 与派生关系投影的最小能力。"""

    source_store: SourceStore
    index_store: IndexStore
    relation_store: RelationStore | None
    tenant_id: str

    def commit(self, user_id: str, operations: list[ContextOperation]) -> Any: ...


__all__ = ["OrdinaryRelationCommitter"]
