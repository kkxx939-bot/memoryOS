"""编排已封存记忆修改提案的预览、审核和提交。"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from difflib import unified_diff
from typing import Any, Literal, Protocol

from foundation.identity import LocalUserContext
from infrastructure.store.memory.review import (
    MemoryEditReviewRecord,
    MemoryEditReviewStatus,
    MemoryEditReviewStore,
    MemoryEditReviewWorkflow,
    ReviewConsolidationSource,
)
from memory.commit.consolidation import (
    ConsolidationResult,
    ConsolidationSource,
    MemoryDocumentConsolidator,
    consolidation_saga_id,
)
from memory.commit.document_commit import DocumentCommitResult, MemoryDocumentCommitter
from memory.core.model import (
    AbsentPath,
    DocumentEditKind,
    DocumentEditPlan,
    PresentPath,
)
from memory.core.structure.frontmatter import parse_front_matter
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.document_store import DocumentConflictError, DocumentNotFoundError
from memory.ports.erase import DocumentEraseStore

ReviewDecision = Literal["APPROVE", "REJECT", "CORRECT"]


class ReadinessGate(Protocol):
    def require_ready(self) -> None: ...


@dataclass(frozen=True)
class MemoryEditReviewResult:
    proposal_id: str
    status: str
    document_uri: str
    document_id: str
    document_kind: str
    relative_path: str
    document_revision: int
    source_digest: str
    proposed_source_digest: str
    proposed_diff_digest: str
    changed: bool
    edit_summary: str
    projection_status: str
    replacement_proposal_id: str = ""
    workflow_kind: str = MemoryEditReviewWorkflow.DOCUMENT_EDIT.value
    consolidation_sources: tuple[dict[str, object], ...] = ()
    consolidation_saga_id: str = ""
    consolidation_status: str = ""
    target_projection_generation: int = 0
    target_projection_confirmed: bool = False
    soft_forgotten_document_ids: tuple[str, ...] = ()
    pending_document_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryEditReviewPreview:
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
    workflow_kind: str = MemoryEditReviewWorkflow.DOCUMENT_EDIT.value
    consolidation_sources: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryEditReviewService:
    """审核、拒绝或修正一份不可变的记忆文档 CAS 修改提案。"""

    def __init__(
        self,
        review_store: MemoryEditReviewStore,
        committer: MemoryDocumentCommitter,
        *,
        erasure_store: DocumentEraseStore,
        readiness: ReadinessGate | None = None,
        consolidator: MemoryDocumentConsolidator | None = None,
    ) -> None:
        self.review_store = review_store
        self.committer = committer
        self.control_store = committer.control_store
        self.erasure_store = erasure_store
        self.readiness = readiness
        if consolidator is not None and consolidator.committer is not committer:
            raise ValueError("memory review consolidator must share the document committer")
        self.consolidator = consolidator

    def seal_edit_proposal(
        self,
        plan: DocumentEditPlan,
        *,
        proposed_diff: str | bytes,
    ) -> MemoryEditReviewResult:
        """Internal candidate path: persist exact before/after/diff without mutating live Markdown."""

        self._require_ready()
        self.erasure_store.assert_mutation_allowed(
            plan.tenant_id,
            plan.owner_user_id,
            plan.document_id,
        )
        record = self.review_store.seal(plan, proposed_diff=proposed_diff)
        return self._result(record, changed=False, projection_status="AWAITING_REVIEW")

    def read_proposed_diff(
        self,
        proposal_id: str,
        *,
        caller: LocalUserContext,
    ) -> bytes:
        self._require_ready()
        record = self._owned_record(proposal_id, caller)
        return self.review_store.load_proposed_diff(record)

    def preview_edit(
        self,
        proposal_id: str,
        *,
        caller: LocalUserContext,
    ) -> MemoryEditReviewPreview:
        """Return the caller-owned bounded diff needed for an informed decision."""

        self._require_ready()
        record = self._owned_record(proposal_id, caller)
        proposed_diff = self.review_store.load_proposed_diff(record).decode("utf-8", errors="strict")
        current = self._result(record, changed=False, projection_status="AWAITING_REVIEW")
        return MemoryEditReviewPreview(
            proposal_id=current.proposal_id,
            status=current.status,
            document_uri=current.document_uri,
            document_id=current.document_id,
            document_kind=current.document_kind,
            relative_path=current.relative_path,
            source_digest=current.source_digest,
            proposed_source_digest=current.proposed_source_digest,
            proposed_diff_digest=current.proposed_diff_digest,
            proposed_diff=proposed_diff,
            edit_summary=current.edit_summary,
            workflow_kind=current.workflow_kind,
            consolidation_sources=current.consolidation_sources,
        )

    def review_edit(
        self,
        proposal_id: str,
        decision: ReviewDecision,
        *,
        caller: LocalUserContext,
        corrected_edit: str | None = None,
    ) -> MemoryEditReviewResult:
        self._require_ready()
        record = self._owned_record(proposal_id, caller)
        normalized = str(decision or "").strip().upper()
        if normalized not in {"APPROVE", "REJECT", "CORRECT"}:
            raise ValueError("review decision must be APPROVE, REJECT or CORRECT")
        if normalized == "CORRECT":
            if corrected_edit is None:
                raise ValueError("CORRECT requires corrected_edit")
            return self._correct(record, corrected_edit)
        if corrected_edit is not None:
            raise ValueError("only CORRECT accepts corrected_edit")
        if normalized == "REJECT":
            rejected = self.review_store.transition(record, MemoryEditReviewStatus.REJECTED)
            return self._result(rejected, changed=False, projection_status="REJECTED")
        return self._approve(record, caller)

    def _approve(
        self,
        record: MemoryEditReviewRecord,
        caller: LocalUserContext,
    ) -> MemoryEditReviewResult:
        if record.workflow_kind is MemoryEditReviewWorkflow.CONSOLIDATION:
            return self._approve_consolidation(record, caller)
        if record.status == MemoryEditReviewStatus.APPROVED:
            return self._result(record, changed=True, projection_status="ENQUEUED")
        if record.status != MemoryEditReviewStatus.PENDING:
            raise ValueError("only a PENDING document edit proposal can be approved")
        self.erasure_store.assert_mutation_allowed(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        )
        plan = self.review_store.to_plan(record)
        result = self.committer.commit(
            plan,
            actor_binding=f"local:{caller.user_id}",
            evidence_reference=_review_evidence_reference(record),
        )
        approved = self.review_store.transition(
            record,
            MemoryEditReviewStatus.APPROVED,
            commit_intent_id=result.intent_id,
        )
        return self._result_from_commit(approved, result)

    def _approve_consolidation(
        self,
        record: MemoryEditReviewRecord,
        caller: LocalUserContext,
    ) -> MemoryEditReviewResult:
        if self.consolidator is None:
            raise RuntimeError("memory consolidation review is not configured")
        actor_binding = f"local:{caller.user_id}"
        if record.status == MemoryEditReviewStatus.APPROVED:
            result = self.consolidator.resume(
                tenant_id=record.tenant_id,
                owner_user_id=record.owner_user_id,
                saga_id=record.consolidation_saga_id,
                actor_binding=actor_binding,
            )
            return self._result(
                record,
                changed=True,
                projection_status=result.status.value,
                consolidation=result,
            )
        if record.status != MemoryEditReviewStatus.PENDING:
            raise ValueError("only a PENDING consolidation proposal can be approved")
        self.erasure_store.assert_mutation_allowed(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        )
        plan = self.review_store.to_plan(record)
        idempotency_key = f"review-consolidation:{record.proposal_id}"
        expected_saga_id = consolidation_saga_id(
            record.tenant_id,
            record.owner_user_id,
            hashlib.sha256(idempotency_key.encode()).hexdigest(),
        )
        existing_saga = self.consolidator.saga_store.load(
            record.tenant_id,
            record.owner_user_id,
            expected_saga_id,
        )
        sources = (
            _sealed_consolidation_sources(record)
            if existing_saga is not None
            else self._validated_consolidation_sources(record, plan)
        )
        result = self.consolidator.consolidate(
            plan,
            sources,
            idempotency_key=idempotency_key,
            actor_binding=actor_binding,
        )
        approved = self.review_store.transition(
            record,
            MemoryEditReviewStatus.APPROVED,
            consolidation_saga_id=result.saga_id,
        )
        return self._result(
            approved,
            changed=True,
            projection_status=result.status.value,
            consolidation=result,
        )

    def _validated_consolidation_sources(
        self,
        record: MemoryEditReviewRecord,
        plan: DocumentEditPlan,
    ) -> tuple[ConsolidationSource, ...]:
        target_state = self.committer.document_store.read_state(
            record.tenant_id,
            record.owner_user_id,
            record.relative_path,
        )
        if target_state != plan.expected_state:
            raise DocumentConflictError("consolidation review target changed after its copy-on-write proposal")
        if isinstance(target_state, PresentPath):
            target_raw = self.committer.document_store.read_raw(
                record.tenant_id,
                record.owner_user_id,
                relative_path=record.relative_path,
            )
            if hashlib.sha256(target_raw).hexdigest() != target_state.raw_sha256:
                raise DocumentConflictError("consolidation review target changed during validation")
            target_front_matter = parse_front_matter(
                target_raw,
                max_header_bytes=int(getattr(self.committer.document_store, "max_front_matter_bytes", 32 * 1024)),
                max_depth=int(getattr(self.committer.document_store, "max_front_matter_depth", 12)),
            )
            if target_front_matter.document_id != record.document_id:
                raise DocumentConflictError("consolidation review target identity changed")
        elif not isinstance(target_state, AbsentPath):
            raise DocumentConflictError("consolidation review target is unsafe")

        sources: list[ConsolidationSource] = []
        for source in record.consolidation_sources:
            self.erasure_store.assert_mutation_allowed(
                record.tenant_id,
                record.owner_user_id,
                source.document_id,
            )
            expected = PresentPath(source.relative_path, source.raw_sha256, source.size)
            state = self.committer.document_store.read_state(
                record.tenant_id,
                record.owner_user_id,
                source.relative_path,
            )
            if state != expected:
                raise DocumentConflictError("consolidation review source changed after its sealed proposal")
            raw = self.committer.document_store.read_raw(
                record.tenant_id,
                record.owner_user_id,
                relative_path=source.relative_path,
            )
            if hashlib.sha256(raw).hexdigest() != source.raw_sha256:
                raise DocumentConflictError("consolidation review source changed during validation")
            parsed = parse_front_matter(
                raw,
                max_header_bytes=int(getattr(self.committer.document_store, "max_front_matter_bytes", 32 * 1024)),
                max_depth=int(getattr(self.committer.document_store, "max_front_matter_depth", 12)),
            )
            if parsed.document_id != source.document_id:
                raise DocumentConflictError("consolidation review source identity changed")
            sources.append(
                ConsolidationSource(
                    document_id=source.document_id,
                    relative_path=source.relative_path,
                    raw_sha256=source.raw_sha256,
                    size=source.size,
                )
            )
        return tuple(sources)

    def _correct(
        self,
        record: MemoryEditReviewRecord,
        corrected_edit: str,
    ) -> MemoryEditReviewResult:
        corrected_plan, corrected_diff = self._corrected_plan(record, corrected_edit)
        if record.status == MemoryEditReviewStatus.CORRECTED:
            replacement = self.review_store.load(
                record.tenant_id,
                record.owner_user_id,
                record.replacement_proposal_id,
            )
            if replacement is None:
                raise DocumentNotFoundError("corrected replacement proposal is missing")
            if not _matches_correction(
                self.review_store,
                record,
                replacement,
                corrected_plan,
                corrected_diff,
            ):
                raise ValueError("review proposal is already bound to a different correction")
            return self._result(replacement, changed=False, projection_status="AWAITING_REVIEW")
        if record.status != MemoryEditReviewStatus.PENDING:
            raise ValueError("only a PENDING document edit proposal can be corrected")
        if (
            corrected_plan.tenant_id != record.tenant_id
            or corrected_plan.owner_user_id != record.owner_user_id
            or corrected_plan.document_id != record.document_id
            or corrected_plan.relative_path != record.relative_path
            or corrected_plan.expected_state != record.expected_state
        ):
            raise ValueError("corrected review must retain the exact tenant, owner, document, path and before CAS")
        self.erasure_store.assert_mutation_allowed(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        )
        replacement = self.review_store.seal(
            corrected_plan,
            proposed_diff=corrected_diff,
            independent_evidence_references=record.independent_evidence_references,
            workflow_kind=record.workflow_kind,
            consolidation_sources=record.consolidation_sources,
        )
        self.review_store.transition(
            record,
            MemoryEditReviewStatus.CORRECTED,
            replacement_proposal_id=replacement.proposal_id,
        )
        return self._result(replacement, changed=False, projection_status="AWAITING_REVIEW")

    def _corrected_plan(
        self,
        record: MemoryEditReviewRecord,
        corrected_edit: str,
    ) -> tuple[DocumentEditPlan, bytes]:
        if record.edit_kind not in {DocumentEditKind.CREATE, DocumentEditKind.UPDATE}:
            raise ValueError("CORRECT only supports create/update proposals; reject other proposal kinds")
        corrected_body = str(corrected_edit)
        if not corrected_body.strip():
            raise ValueError("corrected_edit is empty; reject or forget instead")
        original_after = self.review_store.load_after_blob(record)
        if original_after is None:
            raise ValueError("correctable review proposal has no exact after blob")
        parsed = parse_front_matter(
            original_after,
            max_header_bytes=int(getattr(self.committer.document_store, "max_front_matter_bytes", 32 * 1024)),
            max_depth=int(getattr(self.committer.document_store, "max_front_matter_depth", 12)),
        )
        corrected_bytes = corrected_body.encode()
        if not corrected_bytes.startswith(b"\n"):
            corrected_bytes = b"\n" + corrected_bytes
        if not corrected_bytes.endswith(b"\n"):
            corrected_bytes += b"\n"
        after = parsed.header_bytes + corrected_bytes
        if after == original_after:
            raise ValueError("corrected_edit does not change the sealed proposal")
        diff = "".join(
            unified_diff(
                original_after.decode().splitlines(keepends=True),
                after.decode().splitlines(keepends=True),
                fromfile="sealed-proposal",
                tofile="corrected-proposal",
            )
        ).encode()
        digest = hashlib.sha256(after).hexdigest()
        return (
            DocumentEditPlan(
                idempotency_key=f"correct:{record.proposal_id}:{digest}",
                tenant_id=record.tenant_id,
                owner_user_id=record.owner_user_id,
                edit_kind=record.edit_kind,
                expected_state=record.expected_state,
                evidence_digest=digest,
                edit_summary=f"corrected review: {record.edit_summary}"[:500],
                document_id=record.document_id,
                relative_path=record.relative_path,
                after_bytes=after,
                new_relative_path=record.new_relative_path,
                expected_new_state=record.expected_new_state,
                expected_registration_document_id=record.expected_registration_document_id,
            ),
            diff,
        )

    def _owned_record(
        self,
        proposal_id: str,
        caller: LocalUserContext,
    ) -> MemoryEditReviewRecord:
        record = self.review_store.load(caller.tenant_id, caller.user_id, proposal_id)
        if record is None:
            raise DocumentNotFoundError("memory edit review proposal does not exist")
        caller.assert_identity(user_id=record.owner_user_id, tenant_id=record.tenant_id)
        return record

    def _result_from_commit(
        self,
        record: MemoryEditReviewRecord,
        commit: DocumentCommitResult,
    ) -> MemoryEditReviewResult:
        control = commit.control or self.control_store.load_control(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        )
        return self._result(
            record,
            changed=commit.event is not None and not commit.no_op,
            projection_status="ENQUEUED" if commit.event is not None else "UNCHANGED",
            document_revision=control.logical_revision if control is not None else 0,
            source_digest=control.raw_sha256 if control is not None else "",
        )

    def _result(
        self,
        record: MemoryEditReviewRecord,
        *,
        changed: bool,
        projection_status: str,
        document_revision: int | None = None,
        source_digest: str | None = None,
        consolidation: ConsolidationResult | None = None,
    ) -> MemoryEditReviewResult:
        control = self.control_store.load_control(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        )
        if document_revision is None:
            document_revision = control.logical_revision if control is not None else 0
        if source_digest is None:
            if control is not None:
                source_digest = control.raw_sha256
            elif isinstance(record.expected_state, PresentPath):
                source_digest = record.expected_state.raw_sha256
            else:
                source_digest = ""
        assert document_revision is not None
        assert source_digest is not None
        return MemoryEditReviewResult(
            proposal_id=record.proposal_id,
            status=record.status.value,
            document_uri=MemoryDocumentPathPolicy.document_uri(
                record.owner_user_id,
                record.document_id,
            ),
            document_id=record.document_id,
            document_kind=MemoryDocumentPathPolicy.kind_for(record.relative_path).value,
            relative_path=record.relative_path,
            document_revision=document_revision,
            source_digest=source_digest,
            proposed_source_digest=record.after_blob_digest,
            proposed_diff_digest=record.proposed_diff_blob_digest,
            changed=changed,
            edit_summary=record.edit_summary,
            projection_status=projection_status,
            replacement_proposal_id=record.replacement_proposal_id,
            workflow_kind=record.workflow_kind.value,
            consolidation_sources=tuple(
                _consolidation_source_payload(record.owner_user_id, source) for source in record.consolidation_sources
            ),
            consolidation_saga_id=(
                consolidation.saga_id if consolidation is not None else record.consolidation_saga_id
            ),
            consolidation_status=(consolidation.status.value if consolidation is not None else ""),
            target_projection_generation=(
                consolidation.target_projection_generation if consolidation is not None else 0
            ),
            target_projection_confirmed=(
                consolidation.target_projection_confirmed if consolidation is not None else False
            ),
            soft_forgotten_document_ids=(
                consolidation.soft_forgotten_document_ids if consolidation is not None else ()
            ),
            pending_document_ids=(consolidation.pending_document_ids if consolidation is not None else ()),
        )

    def _require_ready(self) -> None:
        if self.readiness is not None:
            self.readiness.require_ready()

def _matches_correction(
    store: MemoryEditReviewStore,
    original: MemoryEditReviewRecord,
    replacement: MemoryEditReviewRecord,
    corrected_plan: DocumentEditPlan,
    corrected_diff: str | bytes,
) -> bool:
    diff = corrected_diff.encode() if isinstance(corrected_diff, str) else bytes(corrected_diff)
    replacement_plan = store.to_plan(replacement)
    return (
        replacement.proposed_diff_blob_digest == hashlib.sha256(diff).hexdigest()
        and replacement_plan.tenant_id == corrected_plan.tenant_id
        and replacement_plan.owner_user_id == corrected_plan.owner_user_id
        and replacement_plan.document_id == corrected_plan.document_id
        and replacement_plan.edit_kind == corrected_plan.edit_kind
        and replacement_plan.expected_state == corrected_plan.expected_state
        and replacement_plan.expected_new_state == corrected_plan.expected_new_state
        and replacement_plan.relative_path == corrected_plan.relative_path
        and replacement_plan.new_relative_path == corrected_plan.new_relative_path
        and replacement_plan.expected_registration_document_id == corrected_plan.expected_registration_document_id
        and replacement_plan.evidence_digest == corrected_plan.evidence_digest
        and replacement_plan.edit_summary == corrected_plan.edit_summary
        and replacement_plan.after_bytes == corrected_plan.after_bytes
        and replacement.workflow_kind == original.workflow_kind
        and replacement.consolidation_sources == original.consolidation_sources
    )


def _consolidation_source_payload(
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


def _sealed_consolidation_sources(
    record: MemoryEditReviewRecord,
) -> tuple[ConsolidationSource, ...]:
    return tuple(
        ConsolidationSource(
            document_id=source.document_id,
            relative_path=source.relative_path,
            raw_sha256=source.raw_sha256,
            size=source.size,
        )
        for source in record.consolidation_sources
    )


def _review_evidence_reference(record: MemoryEditReviewRecord) -> str:
    if record.independent_evidence_references:
        return record.independent_evidence_references[0]
    return f"review-proposal:{record.proposal_id}:sha256:{record.evidence_digest}"


__all__ = [
    "MemoryEditReviewResult",
    "MemoryEditReviewPreview",
    "MemoryEditReviewService",
    "ReviewDecision",
]
