"""通用操作事务层使用的领域扩展协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_object import ContextObject
from transaction.model.context_operation import ContextOperation


@dataclass(frozen=True)
class DomainConflictResolution:
    """领域扩展对同一目标操作组给出的冲突裁决。"""

    accepted: list[ContextOperation]
    rejected: list[ContextOperation]
    conflict_type: str | None
    reason: str


class OperationDomainPolicy(Protocol):
    """把领域规则接入通用事务层，但不让事务层反向依赖具体领域。"""

    def preprocess(
        self,
        operations: list[ContextOperation],
    ) -> tuple[list[ContextOperation], list[dict]]: ...

    def resolve_group(
        self,
        operations: list[ContextOperation],
    ) -> DomainConflictResolution | None: ...


class NoOperationDomainPolicy:
    """没有注册领域扩展时使用的空策略。"""

    def preprocess(
        self,
        operations: list[ContextOperation],
    ) -> tuple[list[ContextOperation], list[dict]]:
        return operations, []

    def resolve_group(
        self,
        operations: list[ContextOperation],
    ) -> DomainConflictResolution | None:
        del operations
        return None


@dataclass(frozen=True)
class RelationEligibility:
    """领域无关的关系投影资格结果。"""

    allowed: bool
    reason: str = ""


class OperationDomainHost(Protocol):
    """领域处理器可以使用的最小事务宿主能力。"""

    source_store: SourceStore
    relation_store: RelationStore | None
    tenant_id: str

    def _read_content_or_empty(self, uri: str) -> str: ...

    def _apply_relations(self, obj: ContextObject, operation: ContextOperation) -> None: ...

    def _relation_specs_for_object(self, obj: ContextObject) -> list[dict]: ...

    def _relation_spec_key(self, spec: dict) -> tuple[str, str, str]: ...


class OperationDomainHandler(Protocol):
    """领域模块向统一事务内核提供的副作用扩展。"""

    def handles(self, operation: ContextOperation) -> bool: ...

    def owns_object(self, obj: ContextObject) -> bool: ...

    def validate(self, host: OperationDomainHost, operation: ContextOperation) -> None: ...

    def apply_source(self, host: OperationDomainHost, operation: ContextOperation) -> None: ...

    def validate_postcondition(
        self,
        host: OperationDomainHost,
        operation: ContextOperation,
        effect: dict,
    ) -> None: ...

    def materialize_object(self, host: OperationDomainHost, obj: ContextObject) -> ContextObject: ...

    def allows_source_only_relation(
        self,
        host: OperationDomainHost,
        obj: ContextObject,
        spec: dict,
        eligibility: RelationEligibility,
    ) -> bool: ...


@dataclass(frozen=True)
class TransactionDomainExtensions:
    """由 Runtime 组合根显式注入的一组领域事务扩展。"""

    conflict_policy: OperationDomainPolicy = field(default_factory=NoOperationDomainPolicy)
    handlers: tuple[OperationDomainHandler, ...] = ()

    def handler_for(self, operation: ContextOperation) -> OperationDomainHandler | None:
        matches = [handler for handler in self.handlers if handler.handles(operation)]
        if len(matches) > 1:
            raise RuntimeError("multiple domain handlers claim the same operation")
        return matches[0] if matches else None

    def handler_for_object(self, obj: ContextObject) -> OperationDomainHandler | None:
        matches = [handler for handler in self.handlers if handler.owns_object(obj)]
        if len(matches) > 1:
            raise RuntimeError("multiple domain handlers claim the same object")
        return matches[0] if matches else None


class ContextOperationEffects(Protocol):
    """Context 语义层提供给通用事务层的副作用规划能力。"""

    def prepare_object(self, obj: ContextObject, content: str) -> ContextObject: ...

    def refresh_layers(
        self,
        source_store: SourceStore,
        obj: ContextObject,
        content: str,
        *,
        bullets: list[str] | None = None,
    ) -> ContextObject: ...

    def relation_specs_for_object(self, obj: ContextObject) -> list[dict]: ...

    def relation_eligibility(
        self,
        spec: dict,
        *,
        authority_uri: str,
        tenant_id: str,
        source_store: SourceStore,
        index_store: IndexStore,
        authority_object: ContextObject | None,
    ) -> RelationEligibility: ...


__all__ = [
    "ContextOperationEffects",
    "DomainConflictResolution",
    "NoOperationDomainPolicy",
    "OperationDomainHandler",
    "OperationDomainHost",
    "OperationDomainPolicy",
    "RelationEligibility",
    "TransactionDomainExtensions",
]
