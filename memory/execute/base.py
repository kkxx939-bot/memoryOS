"""各类记忆操作共享的依赖、本地身份一致性和 CAS 提交能力。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from foundation.identity import LocalUserContext
from infrastructure.store.memory.bootstrap import MemoryDocumentBootstrapper
from infrastructure.store.memory.control_store import document_intent_id
from infrastructure.store.memory.review import MemoryEditReviewStore
from memory.commit.consolidation import MemoryDocumentConsolidator
from memory.commit.document_commit import DocumentCommitResult, MemoryDocumentCommitter
from memory.commit.erase import MemoryDocumentEraser
from memory.core.model import ABSENT, AbsentPath, DocumentEditPlan, ManagedDocument, PresentPath
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.execute.write_planner import MemoryDocumentPlanner
from memory.ports.document_store import DocumentConflictError, DocumentNotFoundError

IndependentEvidenceLocator = Callable[[str, str, str, str], Sequence[str]]


class ReadinessGate(Protocol):
    """执行记忆命令前使用的运行时就绪检查。"""

    def require_ready(self) -> None: ...


@dataclass(frozen=True)
class _LiveDocument:
    """一次命令读取并校验后的精确 live Markdown 快照。"""

    tenant_id: str
    owner_user_id: str
    document_id: str
    relative_path: str
    state: PresentPath
    raw_bytes: bytes

    @property
    def document_uri(self) -> str:
        return MemoryDocumentPathPolicy.document_uri(self.owner_user_id, self.document_id)

    @property
    def document_kind(self) -> str:
        return MemoryDocumentPathPolicy.kind_for(self.relative_path).value


class MemoryCommandBase:
    """只提供各操作都会使用的依赖和安全原语，不承载具体用例。"""

    def __init__(
        self,
        planner: MemoryDocumentPlanner,
        committer: MemoryDocumentCommitter,
        eraser: MemoryDocumentEraser,
        *,
        bootstrapper: MemoryDocumentBootstrapper | None = None,
        independent_evidence_locator: IndependentEvidenceLocator | None = None,
        readiness: ReadinessGate | None = None,
        consolidator: MemoryDocumentConsolidator | None = None,
        review_store: MemoryEditReviewStore | None = None,
    ) -> None:
        if planner.store is not committer.document_store or eraser.document_store is not committer.document_store:
            raise ValueError("memory document command components must share one live document store")
        if eraser.control_store is not committer.control_store:
            raise ValueError("memory document command components must share one control store")
        if consolidator is not None and consolidator.committer is not committer:
            raise ValueError("memory command consolidator must share the document committer")
        self.planner = planner
        self.committer = committer
        self.eraser = eraser
        self.document_store = committer.document_store
        self.control_store = committer.control_store
        self.revision_store = committer.revision_store
        self.erase_store = eraser.erase_store
        self.bootstrapper = bootstrapper
        self.readiness = readiness
        self.consolidator = consolidator
        self.review_store = review_store
        self.independent_evidence_locator = independent_evidence_locator or (
            lambda _tenant, _owner, _document, _digest: ()
        )

    def _load_live(
        self,
        document_uri: str,
        caller: LocalUserContext,
        *,
        allow_erasure: bool = False,
    ) -> _LiveDocument:
        owner, document_id = self._bind_document_uri(document_uri, caller)
        if not allow_erasure:
            self.erase_store.assert_mutation_allowed(caller.tenant_id, owner, document_id)
        scan = self.document_store.full_scan(caller.tenant_id, owner)
        if not scan.complete or scan.errors:
            raise DocumentConflictError("memory document command requires a complete registration scan")
        matches = [
            item for item in scan.registrations if isinstance(item, ManagedDocument) and item.document_id == document_id
        ]
        if len(matches) != 1:
            raise DocumentNotFoundError("document URI is not one exact managed live document")
        registration = matches[0]
        state = self.document_store.read_state(caller.tenant_id, owner, registration.relative_path)
        if not isinstance(state, PresentPath):
            raise DocumentConflictError("registered memory document is not safely PRESENT")
        raw = self.document_store.read_raw(
            caller.tenant_id,
            owner,
            relative_path=registration.relative_path,
        )
        if hashlib.sha256(raw).hexdigest() != state.raw_sha256:
            raise DocumentConflictError("memory document changed during command read")
        return _LiveDocument(
            tenant_id=caller.tenant_id,
            owner_user_id=owner,
            document_id=document_id,
            relative_path=registration.relative_path,
            state=state,
            raw_bytes=raw,
        )

    def _commit_or_replay(
        self,
        plan: DocumentEditPlan,
        *,
        caller: LocalUserContext,
        evidence_reference: str,
    ) -> DocumentCommitResult:
        idempotency_digest = hashlib.sha256(plan.idempotency_key.encode()).hexdigest()
        intent_id = document_intent_id(
            plan.tenant_id,
            plan.owner_user_id,
            plan.document_id,
            idempotency_digest,
        )
        existing = self.control_store.load_intent(plan.tenant_id, plan.owner_user_id, intent_id)
        if existing is not None:
            if (
                existing.evidence_digest != plan.evidence_digest
                or existing.actor_binding != self._actor_binding(caller)
                or existing.evidence_reference != evidence_reference
            ):
                raise DocumentConflictError("memory command replay is detached from its durable intent")
            return self.committer.recover_intent(plan.tenant_id, plan.owner_user_id, intent_id)
        return self.committer.commit(
            plan,
            actor_binding=self._actor_binding(caller),
            evidence_reference=evidence_reference,
        )

    def _result_fields(self, plan: DocumentEditPlan, result: DocumentCommitResult) -> dict[str, Any]:
        control = result.control or self.control_store.load_control(
            plan.tenant_id,
            plan.owner_user_id,
            plan.document_id,
        )
        source_digest = control.raw_sha256 if control is not None else ""
        revision = control.logical_revision if control is not None else 0
        relative_path = control.relative_path if control is not None else plan.relative_path
        return {
            "document_uri": MemoryDocumentPathPolicy.document_uri(plan.owner_user_id, plan.document_id),
            "document_id": plan.document_id,
            "document_kind": MemoryDocumentPathPolicy.kind_for(relative_path).value,
            "relative_path": relative_path,
            "document_revision": revision,
            "source_digest": source_digest,
            "changed": result.event is not None and not result.no_op,
            "edit_summary": plan.edit_summary,
            "projection_status": "ENQUEUED" if result.event is not None else "UNCHANGED",
        }

    @staticmethod
    def _bind_document_uri(document_uri: str, caller: LocalUserContext) -> tuple[str, str]:
        owner, document_id = MemoryDocumentPathPolicy.parse_document_uri(document_uri)
        caller.assert_identity(user_id=owner)
        return owner, document_id

    @staticmethod
    def _actor_binding(caller: LocalUserContext) -> str:
        return f"local:{caller.user_id}"

    def _require_ready(self) -> None:
        if self.readiness is not None:
            self.readiness.require_ready()


def _assert_expected_digest(state: object, expected: str | None) -> None:
    if expected is None:
        return
    if expected == "":
        if state != ABSENT:
            raise DocumentConflictError("expected_document_digest asserted ABSENT but document is present")
        return
    _require_sha256(expected, "expected_document_digest")
    if not isinstance(state, PresentPath) or state.raw_sha256 != expected:
        raise DocumentConflictError("expected_document_digest does not match live Markdown")


def _assert_optional_live_digest(state: PresentPath, expected: str | None) -> None:
    if expected is None:
        return
    _require_sha256(expected, "expected_digest")
    if state.raw_sha256 != expected:
        raise DocumentConflictError("expected digest does not match live Markdown")


def _assert_restore_expected(state: object, expected: str) -> None:
    if isinstance(state, AbsentPath):
        if expected != "":
            raise DocumentConflictError("restore of a deleted document requires expected_digest='' for ABSENT")
        return
    _require_sha256(expected, "expected_digest")
    if not isinstance(state, PresentPath) or state.raw_sha256 != expected:
        raise DocumentConflictError("restore expected digest does not match live Markdown")


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


__all__ = ["IndependentEvidenceLocator", "MemoryCommandBase", "ReadinessGate"]
