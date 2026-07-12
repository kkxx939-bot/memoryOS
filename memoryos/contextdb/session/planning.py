"""Request-scoped state for canonical memory planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memoryos.memory.canonical.evidence import EvidenceRef
from memoryos.memory.canonical.proposal import MemorySemanticProposal
from memoryos.operations.model.context_operation import ContextOperation


@dataclass(frozen=True)
class ProposalPlanningInput:
    """One immutable proposal and the views selected for it."""

    proposal: MemorySemanticProposal
    retrieval_views: tuple[str, ...] = ()
    forced_pending_reason: str = ""


@dataclass(frozen=True)
class ProposalPlanningOutcome:
    proposal_id: str
    decision: str
    reason: str
    candidate_index: int | None = None
    security_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrefetchSnapshot:
    """Stable representation of canonical state observed before extraction."""

    uri: str
    revision: int
    payload_json: str


@dataclass(frozen=True)
class StagedObjectSnapshot:
    """A canonical object produced only inside this planning request."""

    uri: str
    payload_json: str


@dataclass(frozen=True)
class PlanningContext:
    """Everything needed to safely replan exactly one original request."""

    planning_id: str
    task_id: str
    archive_digest: str
    manifest_digest: str
    episode_id: str
    session_id: str
    tenant_id: str
    proposal_inputs: tuple[ProposalPlanningInput, ...]
    prefetch_snapshot: tuple[PrefetchSnapshot, ...]
    planned_against_revisions: tuple[tuple[str, int], ...]
    staged_objects: tuple[StagedObjectSnapshot, ...]
    scope_candidates: tuple[str, ...]
    evidence_references: tuple[EvidenceRef, ...]
    operation_group_identity: str
    admission_summary: tuple[tuple[str, int], ...] = ()
    proposal_outcomes: tuple[ProposalPlanningOutcome, ...] = ()
    extraction_security_flags: tuple[str, ...] = ()
    salience_fingerprint: str = ""
    salience_reasons: tuple[str, ...] = ()

    def assert_matches(
        self,
        *,
        task_id: str,
        session_id: str,
        tenant_id: str,
        archive_digest: str,
        manifest_digest: str,
    ) -> None:
        """Reject accidental reuse of a context for another request."""

        if self.task_id != task_id or self.session_id != session_id:
            raise PlanningContextMismatchError(
                f"planning context {self.planning_id} belongs to task={self.task_id} session={self.session_id}"
            )
        if self.archive_digest and archive_digest and self.archive_digest != archive_digest:
            raise PlanningContextMismatchError(
                f"planning context {self.planning_id} archive digest does not match the archived request"
            )
        if self.tenant_id != tenant_id:
            raise PlanningContextMismatchError(
                f"planning context {self.planning_id} tenant does not match the archived request"
            )
        if self.manifest_digest and manifest_digest and self.manifest_digest != manifest_digest:
            raise PlanningContextMismatchError(
                f"planning context {self.planning_id} manifest does not match the archived request"
            )


class PlanningContextMismatchError(RuntimeError):
    """Raised when a replan context is used for a different request."""


@dataclass(frozen=True)
class MemoryPlanningResult:
    """Typed operations plus the request-scoped context required for replan."""

    operations: tuple[ContextOperation, ...]
    context: PlanningContext

    def metadata(self) -> dict[str, Any]:
        return {
            "planning_id": self.context.planning_id,
            "archive_digest": self.context.archive_digest,
            "operation_group_identity": self.context.operation_group_identity,
            "operation_count": len(self.operations),
        }
