"""普通对象和领域扩展对象的耐久事务提交器。

Markdown 记忆文档由 ``MemoryDocumentCommitter`` 独占；本提交器会在创建
Redo 意图前拒绝这类写入。
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import TypeGuard

from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.lock import LockStore
from infrastructure.store.contracts.path_lock import PathLock
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.model.context.context_uri import ContextURI
from transaction.commit.audit_diff import CommitAuditDiff
from transaction.commit.control import OperationControlStores
from transaction.commit.coordinator import CommitCoordinator
from transaction.commit.domain_protocols import (
    ContextOperationEffects,
    OperationDomainHandler,
    RelationEligibility,
    TransactionDomainExtensions,
)
from transaction.commit.effects.regular import RegularEffectExecutor
from transaction.commit.effects.writer import StoreEffectWriter
from transaction.commit.host import OperationTransactionHost
from transaction.commit.markers.operation import OperationMarkerStore
from transaction.commit.recovery_state_machine import CommitRecoveryStateMachine
from transaction.commit.state_machine import CommitStateMachine
from transaction.commit.validation import RegularOperationValidator
from transaction.model.context_operation import ContextOperation
from transaction.resolver.conflict_resolver import ConflictResolver
from transaction.resolver.target_resolver import TargetResolver


def _is_lock_store(candidate: object) -> TypeGuard[LockStore]:
    required = ("acquire", "renew", "assert_owned", "fenced", "release")
    return all(callable(getattr(candidate, name, None)) for name in required)


class OperationCommitter(
    CommitCoordinator,
    CommitStateMachine,
    RegularOperationValidator,
    CommitAuditDiff,
    CommitRecoveryStateMachine,
    RegularEffectExecutor,
    OperationMarkerStore,
    StoreEffectWriter,
    OperationTransactionHost,
):
    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        root: str,
        control_stores: OperationControlStores,
        lock_store: LockStore | None = None,
        relation_store: RelationStore | None = None,
        target_resolver: TargetResolver | None = None,
        context_effects: ContextOperationEffects | None = None,
        domain_extensions: TransactionDomainExtensions | None = None,
        tenant_id: str | None = None,
        test_hook=None,  # noqa: ANN001
        tombstone_service=None,  # noqa: ANN001
    ) -> None:
        source_tenant = getattr(source_store, "tenant_id", None)
        if source_tenant is not None:
            source_tenant = self._validate_tenant_id(source_tenant, "SourceStore tenant_id")
        bound = (
            self._validate_tenant_id(tenant_id, "OperationCommitter tenant_id")
            if tenant_id is not None
            else source_tenant or "default"
        )
        if source_tenant is not None and source_tenant != bound:
            raise ValueError("OperationCommitter tenant does not match SourceStore tenant")
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.context_effects = context_effects
        self.root = Path(root)
        self.artifact_root = self.root if bound == "default" else self.root / "tenants" / bound
        if control_stores.root != self.artifact_root:
            raise ValueError("OperationControlStores root does not match the bound tenant root")
        self.tenant_id = bound
        self.domain_extensions = domain_extensions or TransactionDomainExtensions()
        self.conflicts = ConflictResolver(self.domain_extensions.conflict_policy)
        self.target_resolver = target_resolver or TargetResolver(source_store=source_store)
        self.redo = control_stores.redo
        self.diff_writer = control_stores.diff
        self.audit = control_stores.audit
        self.marker_store = control_stores.marker
        candidate = lock_store
        if candidate is None:
            provider = getattr(source_store, "operation_lock_store", None)
            candidate = provider() if callable(provider) else None
        if candidate is None:
            raise RuntimeError("OperationCommitter requires an injected LockStore")
        if not _is_lock_store(candidate):
            raise TypeError("OperationCommitter received an invalid LockStore")
        self.path_lock = PathLock(candidate)
        self.test_hook = test_hook
        self.tombstone_service = tombstone_service
        self._startup_recovery_group: ContextVar[str] = ContextVar(
            f"memoryos_startup_recovery_group_{id(self)}", default=""
        )

    def _domain_handler_for(self, operation: ContextOperation) -> OperationDomainHandler | None:
        return self.domain_extensions.handler_for(operation)

    def _domain_handler_for_object(self, obj: ContextObject) -> OperationDomainHandler | None:
        return self.domain_extensions.handler_for_object(obj)

    @staticmethod
    def _is_document_owned_uri(uri: str) -> bool:
        parsed = ContextURI.parse(uri)
        return parsed.authority == "user" and parsed.segments[1:3] == ("memory", "documents")

    def _reject_document_owned_uri(self, uri: str) -> None:
        if self._is_document_owned_uri(uri):
            raise PermissionError(
                "Markdown memory documents cannot pass through OperationCommitter; use MemoryDocumentCommitter"
            )

    def _reject_document_owned_operation(self, operation: ContextOperation) -> None:
        if operation.context_type == ContextType.MEMORY:
            raise PermissionError(
                "ContextType.MEMORY is reserved for Markdown document projections; use MemoryDocumentCommitter"
            )
        if operation.target_uri:
            self._reject_document_owned_uri(operation.target_uri)
        raw = operation.payload.get("context_object")
        if not isinstance(raw, dict):
            return
        uri = raw.get("uri")
        if isinstance(uri, str) and uri:
            self._reject_document_owned_uri(uri)
        for relation in raw.get("relations", []) or []:
            if not isinstance(relation, dict):
                continue
            for key in ("source_uri", "target_uri"):
                endpoint = relation.get(key)
                if isinstance(endpoint, str) and endpoint.startswith("memoryos://"):
                    self._reject_document_owned_uri(endpoint)
        if self.context_effects is None:
            return
        try:
            obj = ContextObject.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            return
        for spec in self.context_effects.relation_specs_for_object(obj):
            for key in ("source_uri", "target_uri"):
                endpoint = str(spec.get(key) or "")
                if endpoint.startswith("memoryos://"):
                    self._reject_document_owned_uri(endpoint)

    def _validate_domain_operation(self, operation: ContextOperation) -> bool:
        handler = self._domain_handler_for(operation)
        if handler is None:
            return False
        handler.validate(self, operation)
        return True

    def _domain_allows_source_only_relation(
        self, desired: ContextObject | None, spec: dict, eligibility: RelationEligibility
    ) -> bool:
        if desired is None:
            return False
        handler = self._domain_handler_for_object(desired)
        return bool(handler is not None and handler.allows_source_only_relation(self, desired, spec, eligibility))

    def _apply_domain_source(self, operation: ContextOperation) -> bool:
        handler = self._domain_handler_for(operation)
        if handler is None:
            return False
        handler.apply_source(self, operation)
        return True

    def _validate_domain_postcondition(self, operation: ContextOperation, effect: dict) -> bool:
        handler = self._domain_handler_for(operation)
        if handler is None:
            return False
        handler.validate_postcondition(self, operation, effect)
        return True

    def _materialize_domain_object(self, obj: ContextObject) -> ContextObject:
        handler = self._domain_handler_for_object(obj)
        return handler.materialize_object(self, obj) if handler is not None else obj

    def _refresh_context_layers(
        self,
        obj: ContextObject,
        content: str,
        *,
        bullets: list[str] | None = None,
    ) -> ContextObject:
        if self.context_effects is None:
            raise RuntimeError("context layer mutation requires an injected ContextOperationEffects")
        return self.context_effects.refresh_layers(
            self.source_store,
            obj,
            content,
            bullets=bullets,
        )

    def _reject_control_symlink(self, path: Path, label: str) -> None:
        del self
        if path.is_symlink():
            raise ValueError(f"{label} cannot be a symbolic link")


__all__ = ["OperationCommitter"]
