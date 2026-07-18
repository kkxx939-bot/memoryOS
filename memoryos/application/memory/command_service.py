"""Trusted document-native remember, edit, forget, history and restore commands."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from difflib import unified_diff
from typing import Any, Literal, Protocol

from memoryos.core.integrity import canonical_json
from memoryos.memory.documents.bootstrap import MemoryDocumentBootstrapper
from memoryos.memory.documents.commit import DocumentCommitResult, MemoryDocumentCommitter
from memoryos.memory.documents.consolidation import (
    ConsolidationResult,
    ConsolidationSource,
    MemoryDocumentConsolidator,
)
from memoryos.memory.documents.control_store import (
    adoption_document_id,
    adoption_request_digest,
    document_intent_id,
)
from memoryos.memory.documents.erase import (
    DocumentEraseResult,
    MemoryDocumentEraser,
)
from memoryos.memory.documents.frontmatter import matches_adopted_source, parse_front_matter
from memoryos.memory.documents.model import (
    ABSENT,
    AbsentPath,
    DocumentEditKind,
    DocumentEditPlan,
    ManagedDocument,
    MemoryCandidateKind,
    MemoryEditProposal,
    PresentPath,
    UnmanagedDocument,
    UnsafePath,
)
from memoryos.memory.documents.path_policy import MemoryDocumentPathPolicy
from memoryos.memory.documents.planner import MemoryDocumentPlanner, explicit_evidence_digest
from memoryos.memory.documents.review import (
    MemoryEditReviewStore,
    MemoryEditReviewWorkflow,
    ReviewConsolidationSource,
)
from memoryos.memory.documents.revision_store import DocumentRevisionRecord
from memoryos.memory.documents.scanner import ExternalChangeKind, ExternalDocumentChange
from memoryos.memory.documents.store import DocumentConflictError, DocumentNotFoundError
from memoryos.security.trusted_context import (
    AUTHORITATIVE_FORGET,
    AUTHORITATIVE_REMEMBER,
    HARD_ERASE_MEMORY,
    READ_CONTEXT,
    TrustedRequestContext,
)

ForgetMode = Literal["SOFT_FORGET", "HARD_ERASE"]
IndependentEvidenceLocator = Callable[[str, str, str, str], Sequence[str]]
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")


class ReadinessGate(Protocol):
    def require_ready(self) -> None: ...


@dataclass(frozen=True)
class MemoryDocumentCommandResult:
    document_uri: str
    document_id: str
    document_kind: str
    relative_path: str
    document_revision: int
    source_digest: str
    changed: bool
    edit_summary: str
    projection_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RememberResult(MemoryDocumentCommandResult):
    pass


@dataclass(frozen=True)
class AdoptResult(MemoryDocumentCommandResult):
    pass


@dataclass(frozen=True)
class DocumentEditResult(MemoryDocumentCommandResult):
    pass


@dataclass(frozen=True)
class ForgetResult(MemoryDocumentCommandResult):
    mode: ForgetMode
    recoverable: bool
    erasure_status: str = ""
    erasure_epoch: str = ""
    pending_backends: tuple[str, ...] = ()
    independent_evidence_retained: tuple[str, ...] = ()
    media_disclaimer: str = ""


@dataclass(frozen=True)
class MemoryRevisionInfo:
    document_revision: int
    projection_generation: int
    edit_kind: str
    relative_path: str
    source_digest: str
    state: str
    created_at: str
    restorable: bool


@dataclass(frozen=True)
class MemoryHistoryResult:
    document_uri: str
    document_id: str
    document_kind: str
    relative_path: str
    revisions: tuple[MemoryRevisionInfo, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["revisions"] = [asdict(revision) for revision in self.revisions]
        return payload


@dataclass(frozen=True)
class MemoryConsolidationProposalResult:
    proposal_id: str
    status: str
    document_uri: str
    document_id: str
    document_kind: str
    relative_path: str
    source_digest: str
    proposed_source_digest: str
    proposed_diff_digest: str
    proposed_diff: str
    edit_summary: str
    workflow_kind: str
    consolidation_sources: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _LiveDocument:
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


@dataclass(frozen=True)
class _PreparedConsolidation:
    target: _LiveDocument
    plan: DocumentEditPlan
    sources: tuple[ConsolidationSource, ...]
    request_digest: str


class MemoryCommandService:
    """Application service whose only durable memory mutation is a document CAS."""

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
        self.planner = planner
        self.committer = committer
        self.eraser = eraser
        self.document_store = committer.document_store
        self.control_store = committer.control_store
        self.revision_store = committer.revision_store
        self.erase_store = eraser.erase_store
        self.bootstrapper = bootstrapper
        self.readiness = readiness
        if consolidator is not None and consolidator.committer is not committer:
            raise ValueError("memory command consolidator must share the document committer")
        self.consolidator = consolidator
        self.review_store = review_store
        self.independent_evidence_locator = independent_evidence_locator or (
            lambda _tenant, _owner, _document, _digest: ()
        )

    def remember(
        self,
        content: str,
        occurred_at: str | None = None,
        target_hint: str | None = None,
        expected_document_digest: str | None = None,
        *,
        caller: TrustedRequestContext,
    ) -> RememberResult:
        self._require_ready()
        self._require_trusted_user(caller, AUTHORITATIVE_REMEMBER)
        if self.bootstrapper is not None:
            self.bootstrapper.ensure_user(caller.tenant_id, caller.user_id)
        body = str(content or "").strip()
        if not body:
            raise ValueError("remember content is required")
        evidence_digest = explicit_evidence_digest(body)
        proposal = _explicit_proposal(
            body,
            occurred_at=occurred_at,
            target_hint=target_hint,
            evidence_reference=f"explicit-input:sha256:{evidence_digest}",
        )
        request_key = (
            "remember:"
            + hashlib.sha256(
                canonical_json([caller.tenant_id, caller.user_id, body, occurred_at or "", target_hint or ""]).encode()
            ).hexdigest()
        )
        plan = self.planner.plan(
            proposal,
            tenant_id=caller.tenant_id,
            owner_user_id=caller.user_id,
            idempotency_key=request_key,
            evidence_digest=evidence_digest,
        )
        _assert_expected_digest(plan.expected_state, expected_document_digest)
        self.erase_store.assert_mutation_allowed(caller.tenant_id, caller.user_id, plan.document_id)
        result = self._commit_or_replay(
            plan,
            caller=caller,
            evidence_reference=f"explicit-input:sha256:{evidence_digest}",
        )
        return RememberResult(**self._result_fields(plan, result))

    def consolidate_memory_documents(
        self,
        target_plan: DocumentEditPlan,
        sources: Sequence[ConsolidationSource],
        *,
        idempotency_key: str,
        caller: TrustedRequestContext,
    ) -> ConsolidationResult:
        """Start a trusted roll-forward consolidation from a validated plan."""

        self._require_ready()
        self._require_trusted_user(caller, AUTHORITATIVE_REMEMBER)
        caller.require(AUTHORITATIVE_FORGET)
        caller.assert_identity(
            tenant_id=target_plan.tenant_id,
            user_id=target_plan.owner_user_id,
        )
        if self.consolidator is None:
            raise RuntimeError("memory document consolidation is not configured")
        return self.consolidator.consolidate(
            target_plan,
            sources,
            idempotency_key=idempotency_key,
            actor_binding=self._actor_binding(caller),
        )

    def merge_memory_documents(
        self,
        target_document_uri: str,
        merged_edit: str,
        expected_target_digest: str,
        source_documents: Sequence[Mapping[str, str]],
        *,
        caller: TrustedRequestContext,
    ) -> ConsolidationResult:
        """Build a bounded consolidation plan from caller-owned document URIs.

        Public callers supply semantic document targets and exact digests, not
        trusted ``DocumentEditPlan`` or filesystem paths.  The service reads
        every live source and constructs the roll-forward saga itself.
        """

        self._require_ready()
        self._require_trusted_user(caller, AUTHORITATIVE_REMEMBER)
        caller.require(AUTHORITATIVE_FORGET)
        if self.consolidator is None:
            raise RuntimeError("memory document consolidation is not configured")
        prepared = self._prepare_memory_consolidation(
            target_document_uri,
            merged_edit,
            expected_target_digest,
            source_documents,
            caller=caller,
        )
        return self.consolidator.consolidate(
            prepared.plan,
            prepared.sources,
            idempotency_key=f"merge:{prepared.request_digest}",
            actor_binding=self._actor_binding(caller),
        )

    def propose_memory_consolidation(
        self,
        target_document_uri: str,
        merged_edit: str,
        expected_target_digest: str,
        source_documents: Sequence[Mapping[str, str]],
        *,
        caller: TrustedRequestContext,
    ) -> MemoryConsolidationProposalResult:
        """Seal a copy-on-write consolidation preview without mutating live sources."""

        self._require_ready()
        self._require_trusted_user(caller, AUTHORITATIVE_REMEMBER)
        caller.require(AUTHORITATIVE_FORGET)
        if self.review_store is None:
            raise RuntimeError("memory consolidation review is not configured")
        prepared = self._prepare_memory_consolidation(
            target_document_uri,
            merged_edit,
            expected_target_digest,
            source_documents,
            caller=caller,
        )
        assert prepared.plan.after_bytes is not None
        proposed_diff = _document_diff(prepared.target.raw_bytes, prepared.plan.after_bytes)
        if not proposed_diff:
            raise ValueError("consolidation proposal must change the target Markdown")
        review_sources = tuple(
            ReviewConsolidationSource(
                document_id=source.document_id,
                relative_path=source.relative_path,
                raw_sha256=source.raw_sha256,
                size=source.size,
            )
            for source in prepared.sources
        )
        record = self.review_store.seal(
            prepared.plan,
            proposed_diff=proposed_diff,
            workflow_kind=MemoryEditReviewWorkflow.CONSOLIDATION,
            consolidation_sources=review_sources,
        )
        return MemoryConsolidationProposalResult(
            proposal_id=record.proposal_id,
            status=record.status.value,
            document_uri=prepared.target.document_uri,
            document_id=prepared.target.document_id,
            document_kind=prepared.target.document_kind,
            relative_path=prepared.target.relative_path,
            source_digest=prepared.target.state.raw_sha256,
            proposed_source_digest=record.after_blob_digest,
            proposed_diff_digest=record.proposed_diff_blob_digest,
            proposed_diff=proposed_diff.decode("utf-8", errors="strict"),
            edit_summary=record.edit_summary,
            workflow_kind=record.workflow_kind.value,
            consolidation_sources=tuple(
                _review_consolidation_source_payload(prepared.target.owner_user_id, source)
                for source in review_sources
            ),
        )

    def resume_memory_consolidation(
        self,
        saga_id: str,
        *,
        caller: TrustedRequestContext,
    ) -> ConsolidationResult:
        """Resume an already sealed roll-forward merge after projection/restart."""

        self._require_ready()
        self._require_trusted_user(caller, AUTHORITATIVE_REMEMBER)
        caller.require(AUTHORITATIVE_FORGET)
        if self.consolidator is None:
            raise RuntimeError("memory document consolidation is not configured")
        return self.consolidator.resume(
            tenant_id=caller.tenant_id,
            owner_user_id=caller.user_id,
            saga_id=str(saga_id),
            actor_binding=self._actor_binding(caller),
        )

    def adopt_memory_document(
        self,
        relative_path: str,
        expected_raw_sha256: str,
        *,
        caller: TrustedRequestContext,
    ) -> AdoptResult:
        """Explicitly bind one safe caller-owned UNMANAGED Markdown file.

        Adoption changes only the exact source file under the caller-bound
        tenant/owner root.  Its durable metadata then crosses the same
        content-free external-change boundary used by the full scanner, which
        records a CREATE event and enqueues the normal projection job.
        """

        self._require_ready()
        self._require_trusted_user(caller, AUTHORITATIVE_REMEMBER)
        _require_sha256(expected_raw_sha256, "expected_raw_sha256")
        relative = MemoryDocumentPathPolicy.normalize_relative_path(relative_path)
        request_digest = adoption_request_digest(
            caller.tenant_id,
            caller.user_id,
            relative,
            expected_raw_sha256,
        )
        receipt_id = f"mdadopt_{request_digest}"
        assigned_document_id = adoption_document_id(request_digest)
        receipt = self.control_store.load_adoption_receipt(
            caller.tenant_id,
            caller.user_id,
            receipt_id,
        )
        if receipt is not None:
            # A durable receipt proves that initial root preflight already
            # completed. Never reconstruct missing authority from retry bytes.
            self.committer.verify_adoption_root(
                caller.tenant_id,
                caller.user_id,
                assigned_document_id,
            )
            # Repair a process stop between immutable receipt publication and
            # its document-ID index before any source CAS can proceed.
            receipt = self.control_store.prepare_adoption_receipt(
                caller.tenant_id,
                caller.user_id,
                relative,
                expected_raw_sha256,
                actor_binding=receipt.actor_binding,
            )
            self.erase_store.assert_mutation_allowed(
                caller.tenant_id,
                caller.user_id,
                receipt.document_id,
            )
            replay_key = receipt.idempotency_key
            replay_intent_id = document_intent_id(
                caller.tenant_id,
                caller.user_id,
                receipt.document_id,
                hashlib.sha256(replay_key.encode()).hexdigest(),
            )
            existing_intent = self.control_store.load_intent(
                caller.tenant_id,
                caller.user_id,
                replay_intent_id,
            )
            if existing_intent is not None:
                result = self.committer.recover_intent(
                    caller.tenant_id,
                    caller.user_id,
                    replay_intent_id,
                )
                return self._adopt_result(receipt.relative_path, receipt.document_id, result)

        before = self.document_store.full_scan(caller.tenant_id, caller.user_id)
        if not before.complete or before.errors:
            raise DocumentConflictError("memory document adoption requires a complete registration scan")
        if receipt is None:
            matches = [
                item
                for item in before.registrations
                if item.relative_path == relative
                and isinstance(item, UnmanagedDocument)
                and item.raw_sha256 == expected_raw_sha256
            ]
            if len(matches) != 1:
                raise DocumentConflictError(
                    "adopt target must be one safe UNMANAGED Markdown file matching expected_raw_sha256"
                )
            self.committer.preflight_adoption_create(
                caller.tenant_id,
                caller.user_id,
                assigned_document_id,
            )
            receipt = self.control_store.prepare_adoption_receipt(
                caller.tenant_id,
                caller.user_id,
                relative,
                expected_raw_sha256,
                actor_binding=self._actor_binding(caller),
            )

        # The retained content-free receipt is also an anti-reuse identity
        # mapping.  A hard-erased assigned ID must be rejected before any live
        # file can be rewritten again.
        self.erase_store.assert_mutation_allowed(
            caller.tenant_id,
            caller.user_id,
            receipt.document_id,
        )
        idempotency_key = receipt.idempotency_key
        idempotency_digest = hashlib.sha256(idempotency_key.encode()).hexdigest()
        intent_id = document_intent_id(
            caller.tenant_id,
            caller.user_id,
            receipt.document_id,
            idempotency_digest,
        )
        existing_intent = self.control_store.load_intent(
            caller.tenant_id,
            caller.user_id,
            intent_id,
        )
        if existing_intent is not None:
            result = self.committer.recover_intent(
                caller.tenant_id,
                caller.user_id,
                intent_id,
            )
            return self._adopt_result(receipt.relative_path, receipt.document_id, result)

        managed = [
            item
            for item in before.registrations
            if isinstance(item, ManagedDocument)
            and item.document_id == receipt.document_id
            and item.relative_path == relative
        ]
        unmanaged = [
            item
            for item in before.registrations
            if isinstance(item, UnmanagedDocument)
            and item.relative_path == relative
            and item.raw_sha256 == expected_raw_sha256
        ]
        if len(managed) == 1:
            self.committer.verify_adoption_root(
                caller.tenant_id,
                caller.user_id,
                receipt.document_id,
            )
            raw = self.document_store.read_raw(
                caller.tenant_id,
                caller.user_id,
                document_id=receipt.document_id,
            )
            if not matches_adopted_source(raw, receipt.document_id, expected_raw_sha256):
                raise DocumentConflictError("managed adopt retry does not match the durable source digest")
        elif len(unmanaged) == 1:
            self.committer.verify_adoption_root(
                caller.tenant_id,
                caller.user_id,
                receipt.document_id,
            )
            try:
                self.document_store.adopt(
                    caller.tenant_id,
                    caller.user_id,
                    relative,
                    expected_raw_sha256=expected_raw_sha256,
                    assigned_document_id=receipt.document_id,
                    operation_id=receipt.receipt_id,
                )
            except DocumentConflictError:
                # A concurrent identical request may have completed the exact
                # front-matter CAS.  The complete scan below proves whether it
                # installed this receipt's identity or produced a third state.
                pass
        else:
            raise DocumentConflictError("adopt retry target is detached from its durable receipt")

        after = self.document_store.full_scan(caller.tenant_id, caller.user_id)
        if not after.complete or after.errors:
            raise DocumentConflictError("adopted memory document could not be confirmed by a complete scan")
        registrations = [
            item
            for item in after.registrations
            if isinstance(item, ManagedDocument)
            and item.document_id == receipt.document_id
            and item.relative_path == relative
        ]
        if len(registrations) != 1:
            raise DocumentConflictError("adopted memory document changed before durable registration")
        adopted = registrations[0]
        raw = self.document_store.read_raw(
            caller.tenant_id,
            caller.user_id,
            document_id=receipt.document_id,
        )
        if not matches_adopted_source(raw, receipt.document_id, expected_raw_sha256):
            raise DocumentConflictError("adopted memory document is not the exact authorized rewrite")

        commit_result = self.committer.record_external_change(
            ExternalDocumentChange(
                change_kind=ExternalChangeKind.CREATE,
                tenant_id=caller.tenant_id,
                owner_user_id=caller.user_id,
                document_id=receipt.document_id,
                old_relative_path="",
                new_relative_path=relative,
                before_raw_digest="",
                after_raw_digest=adopted.raw_sha256,
                scan_generation_id=after.generation_id,
            ),
            actor_binding=receipt.actor_binding,
            evidence_reference=receipt.evidence_reference,
            evidence_digest=receipt.evidence_digest,
            idempotency_key=idempotency_key,
            edit_summary=receipt.edit_summary,
        )
        if commit_result is None:
            # A concurrent identical request may have completed between our
            # pre-lock intent lookup and the committer's controlled no-op.
            concurrent_intent = self.control_store.load_intent(
                caller.tenant_id,
                caller.user_id,
                intent_id,
            )
            if concurrent_intent is None:
                raise DocumentConflictError("adopted memory document has no durable commit intent")
            commit_result = self.committer.recover_intent(
                caller.tenant_id,
                caller.user_id,
                intent_id,
            )
        return self._adopt_result(receipt.relative_path, receipt.document_id, commit_result)

    def _adopt_result(
        self,
        relative_path: str,
        document_id: str,
        result: DocumentCommitResult,
    ) -> AdoptResult:
        event = result.event
        control = result.control
        if (
            control is None
            or event is None
            or event.edit_kind is not DocumentEditKind.CREATE
            or event.document_id != document_id
            or event.new_relative_path != relative_path
        ):
            raise DocumentConflictError("adopted memory document was not durably registered")
        if self.bootstrapper is not None:
            self.bootstrapper.ensure_adopted_user(
                event.tenant_id,
                event.owner_user_id,
                event.new_relative_path,
                document_id=event.document_id,
                adopted_raw_sha256=event.after_raw_digest,
            )
        return AdoptResult(
            document_uri=MemoryDocumentPathPolicy.document_uri(control.owner_user_id, document_id),
            document_id=document_id,
            document_kind=MemoryDocumentPathPolicy.kind_for(relative_path).value,
            relative_path=event.new_relative_path,
            document_revision=event.logical_revision,
            source_digest=event.after_raw_digest,
            changed=not result.no_op,
            edit_summary="adopt unmanaged Markdown document",
            projection_status="ENQUEUED",
        )

    def edit_memory_document(
        self,
        document_uri: str,
        edit: str,
        expected_digest: str,
        *,
        caller: TrustedRequestContext,
    ) -> DocumentEditResult:
        self._require_ready()
        self._require_trusted_user(caller, AUTHORITATIVE_REMEMBER)
        live = self._load_live(document_uri, caller)
        _require_sha256(expected_digest, "expected_digest")
        if live.state.raw_sha256 != expected_digest:
            raise DocumentConflictError("document edit expected digest does not match live Markdown")
        replacement_body = str(edit)
        if not replacement_body.strip():
            raise ValueError("document edit body is empty; use forget for deletion")
        parsed = parse_front_matter(
            live.raw_bytes,
            max_header_bytes=self.planner.max_front_matter_bytes,
            max_depth=self.planner.max_front_matter_depth,
        )
        after = _replace_document_body(parsed.header_bytes, replacement_body)
        evidence_digest = hashlib.sha256(after).hexdigest()
        plan = DocumentEditPlan(
            idempotency_key="edit:"
            + hashlib.sha256(canonical_json([document_uri, expected_digest, evidence_digest]).encode()).hexdigest(),
            tenant_id=live.tenant_id,
            owner_user_id=live.owner_user_id,
            edit_kind=DocumentEditKind.UPDATE,
            expected_state=live.state,
            evidence_digest=evidence_digest,
            edit_summary="explicit full-document edit",
            document_id=live.document_id,
            relative_path=live.relative_path,
            after_bytes=after,
            expected_registration_document_id=live.document_id,
        )
        result = self._commit_or_replay(
            plan,
            caller=caller,
            evidence_reference=f"explicit-edit:sha256:{evidence_digest}",
        )
        return DocumentEditResult(**self._result_fields(plan, result))

    def rename_memory_document(
        self,
        document_uri: str,
        new_relative_path: str,
        expected_digest: str,
        edit: str | None = None,
        *,
        caller: TrustedRequestContext,
    ) -> DocumentEditResult:
        """Rename, and optionally edit, one stable document in a single effect."""

        self._require_ready()
        self._require_trusted_user(caller, AUTHORITATIVE_REMEMBER)
        live = self._load_live(document_uri, caller)
        _require_sha256(expected_digest, "expected_digest")
        target = MemoryDocumentPathPolicy.normalize_relative_path(new_relative_path)
        # Validate the destination taxonomy before creating a durable intent.
        MemoryDocumentPathPolicy.kind_for(target)
        replacement_body = None if edit is None else str(edit)
        if replacement_body is not None and not replacement_body.strip():
            raise ValueError("rename edit body is empty; omit edit for a pure rename")
        after: bytes | None = None
        if replacement_body is not None:
            parsed = parse_front_matter(
                live.raw_bytes,
                max_header_bytes=self.planner.max_front_matter_bytes,
                max_depth=self.planner.max_front_matter_depth,
            )
            after = _replace_document_body(parsed.header_bytes, replacement_body)
        after_digest = hashlib.sha256(after).hexdigest() if after is not None else expected_digest
        request_digest = hashlib.sha256(
            canonical_json(
                [
                    "RENAME_REQUEST_V2",
                    document_uri,
                    target,
                    expected_digest,
                    hashlib.sha256(replacement_body.encode()).hexdigest()
                    if replacement_body is not None
                    else "",
                ]
            ).encode()
        ).hexdigest()
        idempotency_key = f"rename:{request_digest}"
        edit_summary = (
            "rename and edit memory document"
            if replacement_body is not None
            else "rename memory document"
        )
        evidence_digest = hashlib.sha256(
            canonical_json(
                [
                    "RENAME_EFFECT_V2",
                    document_uri,
                    target,
                    expected_digest,
                    after_digest,
                ]
            ).encode()
        ).hexdigest()
        evidence_reference = f"explicit-rename:sha256:{evidence_digest}"
        if live.state.raw_sha256 != expected_digest:
            if (
                after is None
                or live.relative_path != target
                or live.raw_bytes != after
            ):
                raise DocumentConflictError(
                    "document rename expected digest does not match live Markdown"
                )
            # A rename+edit changes the digest, so a retry after the target was
            # installed can no longer satisfy the original source CAS.  Only a
            # durable intent with the exact request identity may authorize
            # roll-forward from this installed target state.
            intent_id = document_intent_id(
                live.tenant_id,
                live.owner_user_id,
                live.document_id,
                hashlib.sha256(idempotency_key.encode()).hexdigest(),
            )
            if self.control_store.load_intent(
                live.tenant_id,
                live.owner_user_id,
                intent_id,
            ) is None:
                raise DocumentConflictError(
                    "rename and edit target matches requested bytes without its durable intent"
                )
            replay_plan = DocumentEditPlan(
                idempotency_key=idempotency_key,
                tenant_id=live.tenant_id,
                owner_user_id=live.owner_user_id,
                edit_kind=DocumentEditKind.RENAME,
                expected_state=live.state,
                expected_new_state=ABSENT,
                evidence_digest=evidence_digest,
                edit_summary=edit_summary,
                document_id=live.document_id,
                relative_path=live.relative_path,
                new_relative_path=target,
                after_bytes=after,
                expected_registration_document_id=live.document_id,
            )
            result = self._commit_or_replay(
                replay_plan,
                caller=caller,
                evidence_reference=evidence_reference,
            )
            return DocumentEditResult(**self._result_fields(replay_plan, result))
        if target == live.relative_path:
            if replacement_body is not None:
                raise ValueError("rename and edit requires a distinct new_relative_path")
            control = self.control_store.load_control(
                live.tenant_id,
                live.owner_user_id,
                live.document_id,
            )
            if control is None:
                raise DocumentConflictError("document rename is detached from its durable registration")
            return DocumentEditResult(
                document_uri=live.document_uri,
                document_id=live.document_id,
                document_kind=live.document_kind,
                relative_path=live.relative_path,
                document_revision=control.logical_revision,
                source_digest=live.state.raw_sha256,
                changed=False,
                edit_summary=edit_summary,
                projection_status="UNCHANGED",
            )
        target_state = self.document_store.read_state(
            live.tenant_id,
            live.owner_user_id,
            target,
        )
        if target_state != ABSENT:
            raise DocumentConflictError("document rename target must be ABSENT")
        plan = DocumentEditPlan(
            idempotency_key=idempotency_key,
            tenant_id=live.tenant_id,
            owner_user_id=live.owner_user_id,
            edit_kind=DocumentEditKind.RENAME,
            expected_state=live.state,
            expected_new_state=ABSENT,
            evidence_digest=evidence_digest,
            edit_summary=edit_summary,
            document_id=live.document_id,
            relative_path=live.relative_path,
            new_relative_path=target,
            after_bytes=after,
            expected_registration_document_id=live.document_id,
        )
        result = self._commit_or_replay(
            plan,
            caller=caller,
            evidence_reference=evidence_reference,
        )
        return DocumentEditResult(**self._result_fields(plan, result))

    def forget(
        self,
        document_uri: str,
        section_anchor: str | None = None,
        mode: ForgetMode = "SOFT_FORGET",
        expected_digest: str | None = None,
        *,
        caller: TrustedRequestContext,
    ) -> ForgetResult:
        self._require_ready()
        caller.require(AUTHORITATIVE_FORGET)
        owner, document_id = self._bind_document_uri(document_uri, caller)
        normalized_mode = str(mode or "").strip().upper()
        if normalized_mode not in {"SOFT_FORGET", "HARD_ERASE"}:
            raise ValueError("forget mode must be SOFT_FORGET or HARD_ERASE")
        if normalized_mode == "HARD_ERASE":
            caller.require(HARD_ERASE_MEMORY)
            if section_anchor is not None:
                raise ValueError("HARD_ERASE only accepts a whole-document target")
            return self._hard_erase(
                document_uri,
                owner,
                document_id,
                expected_digest=expected_digest,
                caller=caller,
            )

        live = self._load_live(document_uri, caller)
        _assert_optional_live_digest(live.state, expected_digest)
        evidence_digest = hashlib.sha256(
            canonical_json(["SOFT_FORGET", document_uri, section_anchor or "", live.state.raw_sha256]).encode()
        ).hexdigest()
        if section_anchor is None:
            edit_kind = DocumentEditKind.DELETE
            after = None
            summary = "soft forget whole document (recoverable)"
        else:
            edit_kind = DocumentEditKind.UPDATE
            after = _remove_markdown_section(
                live.raw_bytes,
                section_anchor,
                max_header_bytes=self.planner.max_front_matter_bytes,
                max_depth=self.planner.max_front_matter_depth,
            )
            summary = f"soft forget section: {_normalized_anchor(section_anchor)[:180]}"
        plan = DocumentEditPlan(
            idempotency_key="soft-forget:"
            + hashlib.sha256(
                canonical_json([document_uri, live.state.raw_sha256, section_anchor or ""]).encode()
            ).hexdigest(),
            tenant_id=live.tenant_id,
            owner_user_id=live.owner_user_id,
            edit_kind=edit_kind,
            expected_state=live.state,
            evidence_digest=evidence_digest,
            edit_summary=summary,
            document_id=live.document_id,
            relative_path=live.relative_path,
            after_bytes=after,
            expected_registration_document_id=live.document_id,
        )
        result = self._commit_or_replay(
            plan,
            caller=caller,
            evidence_reference=f"soft-forget:sha256:{evidence_digest}",
        )
        return ForgetResult(
            **self._result_fields(plan, result),
            mode="SOFT_FORGET",
            recoverable=True,
        )

    def list_memory_history(
        self,
        document_uri: str,
        *,
        caller: TrustedRequestContext,
    ) -> MemoryHistoryResult:
        self._require_ready()
        caller.require(READ_CONTEXT)
        owner, document_id = self._bind_document_uri(document_uri, caller)
        self.erase_store.assert_mutation_allowed(caller.tenant_id, owner, document_id)
        records = self.revision_store.list_revisions(caller.tenant_id, owner, document_id)
        if not records:
            raise DocumentNotFoundError("memory document has no retained history")
        latest = records[-1]
        return MemoryHistoryResult(
            document_uri=document_uri,
            document_id=document_id,
            document_kind=MemoryDocumentPathPolicy.kind_for(latest.relative_path).value,
            relative_path=latest.relative_path,
            revisions=tuple(_revision_info(record) for record in records),
        )

    def restore_memory_revision(
        self,
        document_uri: str,
        revision: int,
        expected_digest: str,
        *,
        caller: TrustedRequestContext,
    ) -> DocumentEditResult:
        self._require_ready()
        self._require_trusted_user(caller, AUTHORITATIVE_REMEMBER)
        owner, document_id = self._bind_document_uri(document_uri, caller)
        self.erase_store.assert_mutation_allowed(caller.tenant_id, owner, document_id)
        if revision <= 0:
            raise ValueError("revision must be positive")
        control = self.control_store.load_control(caller.tenant_id, owner, document_id)
        if control is None:
            raise DocumentNotFoundError("memory document control state does not exist")
        state = self.document_store.read_state(caller.tenant_id, owner, control.relative_path)
        if isinstance(state, UnsafePath):
            raise DocumentConflictError("memory document path is unsafe")
        edit_kind = DocumentEditKind.CREATE if isinstance(state, AbsentPath) else DocumentEditKind.UPDATE
        evidence_digest = hashlib.sha256(
            canonical_json(["RESTORE", document_uri, revision, expected_digest]).encode()
        ).hexdigest()
        plan = DocumentEditPlan(
            idempotency_key="restore:"
            + hashlib.sha256(canonical_json([document_uri, revision, expected_digest]).encode()).hexdigest(),
            tenant_id=caller.tenant_id,
            owner_user_id=owner,
            edit_kind=edit_kind,
            expected_state=state,
            evidence_digest=evidence_digest,
            edit_summary=f"restore retained document revision {revision}",
            document_id=document_id,
            relative_path=control.relative_path,
            expected_registration_document_id=document_id if isinstance(state, PresentPath) else "",
        )
        evidence_reference = f"restore-revision:{revision}:sha256:{evidence_digest}"
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
                raise DocumentConflictError("restore replay is detached from its durable intent")
            result = self.committer.recover_intent(
                plan.tenant_id,
                plan.owner_user_id,
                intent_id,
            )
        else:
            _assert_restore_expected(state, expected_digest)
            result = self.committer.restore_revision(
                plan,
                revision=revision,
                actor_binding=self._actor_binding(caller),
                evidence_reference=evidence_reference,
            )
        return DocumentEditResult(**self._result_fields(plan, result))

    def _hard_erase(
        self,
        document_uri: str,
        owner: str,
        document_id: str,
        *,
        expected_digest: str | None,
        caller: TrustedRequestContext,
    ) -> ForgetResult:
        existing = self.erase_store.load(caller.tenant_id, owner, document_id)
        live: _LiveDocument | None = None
        if existing is None:
            control = self.control_store.load_control(caller.tenant_id, owner, document_id)
            if control is not None and control.status == "deleted":
                scan = self.document_store.full_scan(caller.tenant_id, owner)
                if not scan.complete or scan.errors:
                    raise DocumentConflictError(
                        "hard erase of soft-forgotten memory requires a complete registration scan"
                    )
                if any(
                    isinstance(item, ManagedDocument) and item.document_id == document_id
                    for item in scan.registrations
                ):
                    raise DocumentConflictError(
                        "soft-forgotten memory document identity unexpectedly remains live"
                    )
                revisions = self.revision_store.list_revisions(
                    caller.tenant_id,
                    owner,
                    document_id,
                )
                latest = revisions[-1] if revisions else None
                if (
                    latest is None
                    or latest.state != "ABSENT"
                    or latest.edit_kind is not DocumentEditKind.DELETE
                    or latest.content_blob_role != "before_delete"
                    or not latest.content_blob_digest
                    or latest.relative_path != control.relative_path
                ):
                    raise DocumentConflictError(
                        "soft-forgotten memory lacks one exact retained deletion revision"
                    )
                source_digest = latest.content_blob_digest
                if expected_digest is not None and expected_digest != source_digest:
                    raise DocumentConflictError(
                        "hard erase expected digest does not match the soft-forgotten source"
                    )
                relative_path = latest.relative_path
                document_kind = MemoryDocumentPathPolicy.kind_for(relative_path).value
            else:
                live = self._load_live(document_uri, caller, allow_erasure=False)
                _assert_optional_live_digest(live.state, expected_digest)
                source_digest = live.state.raw_sha256
                relative_path = live.relative_path
                document_kind = live.document_kind
            retained = tuple(
                self.independent_evidence_locator(
                    caller.tenant_id,
                    owner,
                    document_id,
                    source_digest,
                )
            )
        else:
            if expected_digest is not None and expected_digest != existing.source_digest:
                raise DocumentConflictError("hard erase retry changed its exact expected digest")
            source_digest = existing.source_digest
            relative_path = existing.relative_path
            document_kind = existing.document_kind
            retained = existing.independent_evidence_retained
        erased = self.eraser.hard_erase(
            tenant_id=caller.tenant_id,
            owner_user_id=owner,
            document_id=document_id,
            expected_source_digest=source_digest,
            relative_path=relative_path,
            independent_evidence_retained=retained,
        )
        if live is None:
            document_kind = document_kind or ""
        return _hard_erase_result(
            document_uri=document_uri,
            document_id=document_id,
            document_kind=document_kind,
            relative_path=relative_path,
            erased=erased,
        )

    def _prepare_memory_consolidation(
        self,
        target_document_uri: str,
        merged_edit: str,
        expected_target_digest: str,
        source_documents: Sequence[Mapping[str, str]],
        *,
        caller: TrustedRequestContext,
    ) -> _PreparedConsolidation:
        target = self._load_live(target_document_uri, caller)
        _require_sha256(expected_target_digest, "expected_target_digest")
        if target.state.raw_sha256 != expected_target_digest:
            raise DocumentConflictError("merge target expected digest does not match live Markdown")
        replacement_body = str(merged_edit or "")
        if not replacement_body.strip():
            raise ValueError("merged_edit must contain the complete target Markdown body")
        bounded_sources = tuple(source_documents)
        if not bounded_sources or len(bounded_sources) > 100:
            raise ValueError("merge requires between 1 and 100 source documents")
        sources: list[ConsolidationSource] = []
        seen_ids: set[str] = set()
        source_identity: list[tuple[str, str]] = []
        for item in bounded_sources:
            if set(item) != {"document_uri", "expected_digest"}:
                raise ValueError("merge source must contain only document_uri and expected_digest")
            source_uri = str(item["document_uri"])
            expected_digest = str(item["expected_digest"])
            _require_sha256(expected_digest, "source expected_digest")
            source = self._load_live(source_uri, caller)
            if source.document_id == target.document_id:
                raise ValueError("merge target cannot also be a redundant source")
            if source.document_id in seen_ids:
                raise ValueError("merge source documents must be unique")
            seen_ids.add(source.document_id)
            if source.state.raw_sha256 != expected_digest:
                raise DocumentConflictError("merge source expected digest does not match live Markdown")
            sources.append(
                ConsolidationSource(
                    document_id=source.document_id,
                    relative_path=source.relative_path,
                    raw_sha256=source.state.raw_sha256,
                    size=source.state.size,
                )
            )
            source_identity.append((source.document_uri, source.state.raw_sha256))
        sources.sort(
            key=lambda source: (
                source.document_id,
                source.relative_path,
                source.raw_sha256,
                source.size,
            )
        )
        parsed = parse_front_matter(
            target.raw_bytes,
            max_header_bytes=self.planner.max_front_matter_bytes,
            max_depth=self.planner.max_front_matter_depth,
        )
        body_bytes = replacement_body.encode("utf-8")
        if not body_bytes.startswith(b"\n"):
            body_bytes = b"\n" + body_bytes
        if not body_bytes.endswith(b"\n"):
            body_bytes += b"\n"
        after = parsed.header_bytes + body_bytes
        after_digest = hashlib.sha256(after).hexdigest()
        request_identity = [
            target.document_uri,
            target.state.raw_sha256,
            after_digest,
            sorted(source_identity),
        ]
        request_digest = hashlib.sha256(canonical_json(request_identity).encode()).hexdigest()
        plan = DocumentEditPlan(
            idempotency_key=f"merge-target:{request_digest}",
            tenant_id=target.tenant_id,
            owner_user_id=target.owner_user_id,
            edit_kind=DocumentEditKind.UPDATE,
            expected_state=target.state,
            evidence_digest=request_digest,
            edit_summary="merge memory documents into target",
            document_id=target.document_id,
            relative_path=target.relative_path,
            after_bytes=after,
            expected_registration_document_id=target.document_id,
        )
        return _PreparedConsolidation(
            target=target,
            plan=plan,
            sources=tuple(sources),
            request_digest=request_digest,
        )

    def _load_live(
        self,
        document_uri: str,
        caller: TrustedRequestContext,
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
        caller: TrustedRequestContext,
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
            return self.committer.recover_intent(
                plan.tenant_id,
                plan.owner_user_id,
                intent_id,
            )
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
    def _bind_document_uri(document_uri: str, caller: TrustedRequestContext) -> tuple[str, str]:
        owner, document_id = MemoryDocumentPathPolicy.parse_document_uri(document_uri)
        caller.assert_identity(user_id=owner)
        return owner, document_id

    @staticmethod
    def _require_trusted_user(caller: TrustedRequestContext, capability: str) -> None:
        caller.require(capability)
        if caller.actor_kind != "user":
            raise PermissionError("authoritative memory document commands require a trusted user actor")

    @staticmethod
    def _actor_binding(caller: TrustedRequestContext) -> str:
        return f"trusted:{caller.actor_kind}:{caller.actor_id}:{caller.user_id}"

    def _require_ready(self) -> None:
        if self.readiness is not None:
            self.readiness.require_ready()


def _explicit_proposal(
    content: str,
    *,
    occurred_at: str | None,
    target_hint: str | None,
    evidence_reference: str,
) -> MemoryEditProposal:
    raw_hint = str(target_hint or "").strip()
    normalized_hint = raw_hint.casefold().replace("-", "_")
    kind_aliases = {
        "profile": MemoryCandidateKind.PROFILE_FACT,
        "profile_fact": MemoryCandidateKind.PROFILE_FACT,
        "preference": MemoryCandidateKind.PREFERENCE,
        "preferences": MemoryCandidateKind.PREFERENCE,
        "entity": MemoryCandidateKind.ENTITY_NOTE,
        "entity_note": MemoryCandidateKind.ENTITY_NOTE,
        "topic": MemoryCandidateKind.TOPIC_NOTE,
        "topic_note": MemoryCandidateKind.TOPIC_NOTE,
        "episode": MemoryCandidateKind.EPISODE,
        "open_loop": MemoryCandidateKind.OPEN_LOOP,
        "experience": MemoryCandidateKind.EXPERIENCE,
    }
    subject_hint = ""
    prefix, separator, suffix = raw_hint.partition(":")
    if separator and prefix.casefold().replace("-", "_") in kind_aliases:
        kind = kind_aliases[prefix.casefold().replace("-", "_")]
        subject_hint = suffix.strip()
    else:
        kind = kind_aliases.get(normalized_hint, MemoryCandidateKind.TOPIC_NOTE)
        if raw_hint and normalized_hint not in kind_aliases:
            subject_hint = raw_hint
    title = subject_hint or _content_title(content)
    entity_hints = (title,) if kind == MemoryCandidateKind.ENTITY_NOTE else ()
    topic_hints = (title,) if kind == MemoryCandidateKind.TOPIC_NOTE else ()
    return MemoryEditProposal(
        candidate_kind=kind,
        title=title,
        body=content,
        evidence_refs=(evidence_reference,),
        subject=title,
        entity_hints=entity_hints,
        topic_hints=topic_hints,
        occurred_at=str(occurred_at or ""),
    )


def _content_title(content: str) -> str:
    first = next((line.strip() for line in content.splitlines() if line.strip()), "Memory")
    first = re.sub(r"^#{1,6}[ \t]+", "", first)
    collapsed = " ".join(first.split())
    return (collapsed[:120] or "Memory").rstrip()


def _remove_markdown_section(
    raw: bytes,
    anchor: str,
    *,
    max_header_bytes: int,
    max_depth: int,
) -> bytes:
    parsed = parse_front_matter(raw, max_header_bytes=max_header_bytes, max_depth=max_depth)
    target = _normalized_anchor(anchor)
    lines = parsed.body.splitlines(keepends=True)
    matches: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        match = _HEADING.fullmatch(line.rstrip("\r\n"))
        if match and " ".join(match.group(2).split()) == target:
            matches.append((index, len(match.group(1))))
    if len(matches) != 1:
        raise ValueError("section_anchor must match exactly one Markdown heading")
    start, level = matches[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = _HEADING.fullmatch(lines[index].rstrip("\r\n"))
        if match and len(match.group(1)) <= level:
            end = index
            break
    body = "".join(lines[:start] + lines[end:]).encode()
    return parsed.header_bytes + body


def _normalized_anchor(anchor: str) -> str:
    value = re.sub(r"^#{1,6}[ \t]+", "", str(anchor or "").strip())
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError("section_anchor is required")
    return normalized


def _revision_info(record: DocumentRevisionRecord) -> MemoryRevisionInfo:
    return MemoryRevisionInfo(
        document_revision=record.logical_revision,
        projection_generation=record.projection_generation,
        edit_kind=record.edit_kind.value,
        relative_path=record.relative_path,
        source_digest=record.raw_sha256,
        state=record.state,
        created_at=record.created_at,
        restorable=bool(record.content_blob_digest),
    )


def _hard_erase_result(
    *,
    document_uri: str,
    document_id: str,
    document_kind: str,
    relative_path: str,
    erased: DocumentEraseResult,
) -> ForgetResult:
    record = erased.record
    return ForgetResult(
        document_uri=document_uri,
        document_id=document_id,
        document_kind=document_kind,
        relative_path=relative_path,
        document_revision=record.document_revision_floor,
        source_digest="",
        changed=True,
        edit_summary="hard erase whole memory document",
        projection_status=record.status.value,
        mode="HARD_ERASE",
        recoverable=False,
        erasure_status=record.status.value,
        erasure_epoch=record.erasure_epoch,
        pending_backends=record.pending_backends,
        independent_evidence_retained=erased.independent_evidence_retained,
        media_disclaimer=erased.media_disclaimer,
    )


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


def _document_diff(before: bytes, after: bytes) -> bytes:
    try:
        before_lines = before.decode("utf-8", errors="strict").splitlines(keepends=True)
        after_lines = after.decode("utf-8", errors="strict").splitlines(keepends=True)
    except UnicodeDecodeError as exc:  # pragma: no cover - live documents are already strict UTF-8.
        raise ValueError("consolidation proposal Markdown must be UTF-8") from exc
    return "".join(
        unified_diff(
            before_lines,
            after_lines,
            fromfile="live-target-markdown",
            tofile="proposed-consolidated-markdown",
        )
    ).encode("utf-8")


def _replace_document_body(header_bytes: bytes, replacement_body: str) -> bytes:
    body_bytes = replacement_body.encode("utf-8")
    if body_bytes and not body_bytes.startswith(b"\n"):
        body_bytes = b"\n" + body_bytes
    if body_bytes and not body_bytes.endswith(b"\n"):
        body_bytes += b"\n"
    return header_bytes + body_bytes


def _review_consolidation_source_payload(
    owner_user_id: str,
    source: ReviewConsolidationSource,
) -> dict[str, object]:
    return {
        "document_uri": MemoryDocumentPathPolicy.document_uri(owner_user_id, source.document_id),
        "document_id": source.document_id,
        "relative_path": source.relative_path,
        "source_digest": source.raw_sha256,
        "size": source.size,
    }


__all__ = [
    "AdoptResult",
    "DocumentEditResult",
    "ForgetMode",
    "ForgetResult",
    "MemoryCommandService",
    "MemoryConsolidationProposalResult",
    "MemoryDocumentCommandResult",
    "MemoryHistoryResult",
    "MemoryRevisionInfo",
    "RememberResult",
]
