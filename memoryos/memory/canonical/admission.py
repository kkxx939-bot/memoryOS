"""记忆系统里的准入。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

from memoryos.adapters.agent_hooks.sanitizer import ENV_SECRET_RE, INLINE_SECRET_RE, PRIVATE_KEY_RE, SECRET_KEY_RE
from memoryos.memory.canonical.episode import EvidenceEpisode
from memoryos.memory.canonical.evidence import ProposalValidationResult
from memoryos.memory.canonical.proposal import EpistemicStatus, NormalizedSemanticAssessment
from memoryos.memory.canonical.scope import HIERARCHICAL_SCOPE_KINDS, MemoryScope
from memoryos.memory.canonical.semantic import MemorySemanticNormalizer
from memoryos.memory.schema import MemoryType, MemoryTypeRegistry

_PRIVATE_PROCESS_RE = re.compile(
    r"(?i)\b(chain of thought|scratchpad|internal reasoning|agent private|内部推理|草稿)\b"
)


class ProposalAdmissionDecision(str, Enum):
    """保存 ProposalAdmissionDecision 需要的这组数据。"""

    ACCEPT_FOR_RECONCILE = "ACCEPT_FOR_RECONCILE"
    PENDING = "PENDING"
    ARCHIVE_ONLY = "ARCHIVE_ONLY"
    PRIVATE_ONLY = "PRIVATE_ONLY"
    RESTRICTED = "RESTRICTED"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ProposalAdmissionResult:
    """保存 ProposalAdmissionResult 需要的这组数据。"""

    decision: ProposalAdmissionDecision
    reason: str


class ProposalAdmissionGate:
    """负责 ProposalAdmissionGate 这部分逻辑。"""

    def __init__(self, registry: MemoryTypeRegistry | None = None) -> None:
        self.registry = registry or MemoryTypeRegistry()

    def evaluate(
        self,
        validation: ProposalValidationResult,
        *,
        episode: EvidenceEpisode,
        memory_scope: MemoryScope,
        source_role: str,
    ) -> ProposalAdmissionResult:
        """处理 evaluate 这一步。"""

        proposal = validation.proposal
        try:
            memory_type = MemoryType(proposal.memory_type)
            schema = self.registry.get(memory_type)
        except ValueError:
            return ProposalAdmissionResult(ProposalAdmissionDecision.REJECT, "unsupported_memory_schema")
        try:
            memory_scope.validate_tenant(episode.tenant_id)
        except ValueError:
            return ProposalAdmissionResult(ProposalAdmissionDecision.REJECT, "cross_tenant_visibility")
        legal = {scope.key for scope in episode.legal_scope_candidates()}
        suggested = {scope.key for scope in validation.proposal.suggested_scope_refs}
        if not suggested.issubset(legal):
            return ProposalAdmissionResult(ProposalAdmissionDecision.REJECT, "illegal_scope_suggestion")
        if not validation.valid:
            prefix = (
                "PENDING_MISSING_EVIDENCE"
                if any(
                    error == "missing_evidence" or error.startswith(("missing_field_evidence:", "unknown_event:"))
                    for error in validation.errors
                )
                else "validation_failed"
            )
            return ProposalAdmissionResult(
                ProposalAdmissionDecision.PENDING,
                f"{prefix}:{','.join(validation.errors)}",
            )
        normalized_semantic = (
            proposal.semantic
            if isinstance(proposal.semantic, NormalizedSemanticAssessment)
            else MemorySemanticNormalizer().normalize(proposal).semantic
        )
        if not isinstance(normalized_semantic, NormalizedSemanticAssessment):
            return ProposalAdmissionResult(ProposalAdmissionDecision.PENDING, "semantic_not_normalized")
        if not normalized_semantic.schema_safe:
            return ProposalAdmissionResult(
                ProposalAdmissionDecision.PENDING,
                "semantic_schema_pending:" + ",".join(normalized_semantic.schema_errors),
            )
        subject = memory_scope.canonical_subject
        if subject is None:
            return ProposalAdmissionResult(ProposalAdmissionDecision.PENDING, "canonical_subject_missing")
        if subject.kind in HIERARCHICAL_SCOPE_KINDS and not subject.parent_path:
            return ProposalAdmissionResult(ProposalAdmissionDecision.PENDING, "scope_hierarchy_missing")
        if subject.inferred or memory_scope.authority.inferred:
            return ProposalAdmissionResult(ProposalAdmissionDecision.PENDING, "scope_authority_inferred")
        text = json.dumps(
            {"identity_fields": dict(proposal.identity_fields), "value_fields": dict(proposal.value_fields)},
            ensure_ascii=False,
            sort_keys=True,
        )
        if self._raw_tool_output(text):
            return ProposalAdmissionResult(ProposalAdmissionDecision.ARCHIVE_ONLY, "raw_tool_output")
        if self._secret_like(text):
            return ProposalAdmissionResult(ProposalAdmissionDecision.RESTRICTED, "secret_or_sensitive_content")
        if _PRIVATE_PROCESS_RE.search(text) or source_role in {"agent_private", "internal"}:
            return ProposalAdmissionResult(ProposalAdmissionDecision.PRIVATE_ONLY, "agent_private_process")
        if source_role == "user" and not schema.allow_user_source:
            return ProposalAdmissionResult(ProposalAdmissionDecision.REJECT, "user_source_not_allowed")
        if source_role in {"assistant", "agent"} and not schema.allow_assistant_source:
            return ProposalAdmissionResult(ProposalAdmissionDecision.PENDING, "assistant_source_not_authoritative")
        if source_role == "tool" and not schema.allow_tool_source:
            return ProposalAdmissionResult(ProposalAdmissionDecision.ARCHIVE_ONLY, "tool_source_not_allowed")
        if source_role == "tool" and proposal.epistemic_status != EpistemicStatus.OBSERVED:
            return ProposalAdmissionResult(ProposalAdmissionDecision.ARCHIVE_ONLY, "tool_claim_not_observed")
        if proposal.epistemic_status == EpistemicStatus.HYPOTHESIZED:
            return ProposalAdmissionResult(ProposalAdmissionDecision.PENDING, "hypothesis_requires_confirmation")
        threshold = 0.65 if proposal.epistemic_status == EpistemicStatus.EXPLICIT else 0.75
        if proposal.confidence < threshold:
            return ProposalAdmissionResult(ProposalAdmissionDecision.PENDING, "confidence_below_threshold")
        return ProposalAdmissionResult(ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE, "validated")

    def _secret_like(self, text: str) -> bool:
        return bool(
            PRIVATE_KEY_RE.search(text)
            or ENV_SECRET_RE.search(text)
            or INLINE_SECRET_RE.search(text)
            or ("<redacted" in text.casefold() and SECRET_KEY_RE.search(text))
            or re.search(r"(?i)\b(authorization\s*:|cookie\s*:)", text)
        )

    def _raw_tool_output(self, text: str) -> bool:
        normalized = text.casefold()
        return any(
            marker in normalized
            for marker in (
                "traceback (most recent call last)",
                "assertionerror",
                "exit code:",
                "stdout:",
                "stderr:",
            )
        )
