"""通过已配置的最小模型端口编排记忆语义候选提取。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from foundation.integrity import canonical_digest
from memory.core.formation.schema import MemoryCandidateSchema
from memory.core.model import MemoryEditProposal
from memory.formation.errors import (
    MemoryExtractionCandidateValidationError,
    MemoryExtractionConfigurationError,
    MemoryExtractionMalformedEnvelopeError,
    MemoryExtractionSecurityError,
    classify_memory_extraction_failure,
)
from memory.formation.llm.prompt import MemoryExtractionPromptBuilder
from memory.formation.llm.result import MemoryExtractionBatchResult, RejectedMemoryCandidate
from memory.formation.llm.validation import MemoryExtractionCandidateValidator
from memory.ports import MemoryExtractionModelProvider
from memory.formation.egress import EgressDecision, MemoryEgressPolicy
from pre.evidence import SessionArchiveEpisodeAdapter
from pre.session import SessionArchive


class LLMMemoryExtractorBackend:
    """调用通用大模型协议提取候选，不持有记忆路由、写入或删除权限。"""

    semantic_proposal_backend = True
    llm_semantic_backend = True

    def __init__(
        self,
        provider: MemoryExtractionModelProvider,
        *,
        prompt_builder: MemoryExtractionPromptBuilder | None = None,
        egress_policy: MemoryEgressPolicy | None = None,
        max_candidates: int = 32,
        max_body_bytes: int = 32 * 1024,
    ) -> None:
        if max_candidates < 1 or max_candidates > 128 or max_body_bytes < 1:
            raise ValueError("invalid extraction bounds")
        self.provider = provider
        self.prompt_builder = prompt_builder or MemoryExtractionPromptBuilder()
        self.egress_policy = egress_policy or MemoryEgressPolicy()
        self.max_candidates = max_candidates
        self.validator = MemoryExtractionCandidateValidator(max_body_bytes=max_body_bytes)

    @property
    def is_remote(self) -> bool:
        return bool(getattr(self.provider, "is_remote", True))

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryCandidateSchema],
    ) -> list[MemoryEditProposal]:
        """返回通过确定性校验、可交给下游规划器处理的候选。"""

        return list(self.extract_batch(archive, schemas).accepted)

    def extract_batch(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryCandidateSchema],
    ) -> MemoryExtractionBatchResult:
        """执行出站审查、模型调用和逐候选校验，并保留拒绝原因。"""

        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        assessment = self.egress_policy.evaluate(archive, episode, remote=self.is_remote)
        if self.is_remote and assessment.decision in {EgressDecision.DENY, EgressDecision.LOCAL_ONLY}:
            raise MemoryExtractionSecurityError("remote memory extraction is blocked by egress policy")
        prompt = self.prompt_builder.build(archive, schemas, episode)
        prompt = self.egress_policy.redact(prompt, assessment)
        try:
            # 模型、温度和供应商由外部装配配置；领域层只传递受审查后的文本 Prompt。
            response = self.provider.complete(prompt)
        except Exception as exc:
            raise classify_memory_extraction_failure(exc) from exc
        raw = response if isinstance(response, str) else getattr(response, "text", None)
        if not isinstance(raw, str):
            raise MemoryExtractionConfigurationError("memory model provider must return text")
        payload = self._parse(raw)
        rows = payload.get("candidates")
        if not isinstance(rows, list):
            raise MemoryExtractionMalformedEnvelopeError("memory extraction candidates must be an array")
        if len(rows) > self.max_candidates:
            raise MemoryExtractionMalformedEnvelopeError("memory extraction candidate count exceeds bound")
        accepted: list[MemoryEditProposal] = []
        rejected: list[RejectedMemoryCandidate] = []
        schemas_by_kind = {item.candidate_kind: item for item in schemas}
        for index, row in enumerate(rows):
            try:
                accepted.append(self.validator.proposal(row, episode, schemas_by_kind))
            except MemoryExtractionSecurityError:
                raise
            except (MemoryExtractionCandidateValidationError, ValueError, TypeError) as exc:
                rejected.append(RejectedMemoryCandidate(index=index, reason=str(exc)))
        return MemoryExtractionBatchResult(
            accepted=tuple(accepted),
            rejected=tuple(rejected),
            outbound_digest=canonical_digest(prompt),
            egress_decision=assessment.decision.value,
        )

    @staticmethod
    def _parse(raw: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MemoryExtractionMalformedEnvelopeError("memory extraction response is not valid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"candidates"}:
            raise MemoryExtractionMalformedEnvelopeError("memory extraction envelope must contain only candidates")
        return payload


__all__ = ["LLMMemoryExtractorBackend"]
