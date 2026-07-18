"""Strict LLM backend for storage-neutral memory edit proposals."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.integrity import canonical_digest
from memoryos.memory.documents.model import MemoryCandidateKind, MemoryEditProposal
from memoryos.memory.evidence import EvidenceEpisode, SessionArchiveEpisodeAdapter
from memoryos.memory.extraction.egress import EgressDecision, MemoryEgressPolicy
from memoryos.memory.extraction.errors import (
    MemoryExtractionCandidateValidationError,
    MemoryExtractionConfigurationError,
    MemoryExtractionMalformedEnvelopeError,
    MemoryExtractionSecurityError,
    classify_memory_extraction_failure,
)
from memoryos.memory.schema import MemoryCandidateSchema


class MemoryModelProvider(Protocol):
    is_remote: bool

    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class RejectedMemoryCandidate:
    index: int
    reason: str


@dataclass(frozen=True)
class MemoryExtractionBatchResult:
    accepted: tuple[MemoryEditProposal, ...]
    rejected: tuple[RejectedMemoryCandidate, ...]
    outbound_digest: str = ""
    egress_decision: str = EgressDecision.LOCAL_ONLY.value


class MemoryExtractionPromptBuilder:
    """Expose evidence and semantic fields while excluding all storage controls."""

    def build(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryCandidateSchema],
        episode: EvidenceEpisode,
    ) -> str:
        contract = {
            "task": "Extract durable semantic memory candidates from immutable session evidence.",
            "output": {
                "candidates": [
                    {
                        "candidate_kind": "one configured kind",
                        "title": "short heading",
                        "subject": "semantic subject",
                        "body": "grounded Markdown body",
                        "entity_hints": ["semantic entity labels"],
                        "topic_hints": ["semantic topic labels"],
                        "occurred_at": "ISO-8601 with timezone when known",
                        "temporal_status": "optional semantic status",
                        "relation_hints": ["semantic relations"],
                        "evidence_refs": ["event_id"],
                        "field_evidence_refs": {"body": ["event_id"]},
                        "confidence": 0.0,
                    }
                ]
            },
            "forbidden": [
                "path",
                "document_id",
                "tenant",
                "owner",
                "workspace authority",
                "ACL",
                "SQL",
                "delete",
                "hard erase",
                "projection generation",
                "final authority",
            ],
            "candidate_kinds": [
                {
                    "candidate_kind": item.candidate_kind.value,
                    "description": item.description,
                    "requires_occurred_at": item.requires_occurred_at,
                }
                for item in schemas
            ],
            "evidence": [
                {
                    "event_id": item.event_id,
                    "event_type": item.event_type,
                    "actor": item.actor.to_dict(),
                    "occurred_at": item.occurred_at.isoformat(),
                    "text": item.text(),
                }
                for item in episode.events
            ],
            "archive_binding": {
                "session_id": archive.session_id,
                "archive_uri": archive.archive_uri,
                "archive_digest": archive.archive_digest,
                "manifest_digest": archive.manifest_digest,
            },
        }
        return json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_ALLOWED_FIELDS = frozenset(
    {
        "candidate_kind",
        "title",
        "subject",
        "body",
        "entity_hints",
        "topic_hints",
        "occurred_at",
        "temporal_status",
        "relation_hints",
        "evidence_refs",
        "field_evidence_refs",
        "confidence",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "path",
        "relative_path",
        "absolute_path",
        "document_id",
        "document_uri",
        "tenant",
        "tenant_id",
        "owner",
        "owner_user_id",
        "workspace_id",
        "acl",
        "visibility",
        "authority",
        "sql",
        "delete",
        "hard_erase",
        "projection_generation",
        "final_authority",
    }
)


class LLMMemoryExtractorBackend:
    semantic_proposal_backend = True
    llm_semantic_backend = True

    def __init__(
        self,
        provider: MemoryModelProvider,
        *,
        model_id: str = "",
        prompt_builder: MemoryExtractionPromptBuilder | None = None,
        egress_policy: MemoryEgressPolicy | None = None,
        max_candidates: int = 32,
        max_body_bytes: int = 32 * 1024,
    ) -> None:
        if max_candidates < 1 or max_candidates > 128 or max_body_bytes < 1:
            raise ValueError("invalid extraction bounds")
        self.provider = provider
        self.model_id = str(model_id)
        self.prompt_builder = prompt_builder or MemoryExtractionPromptBuilder()
        self.egress_policy = egress_policy or MemoryEgressPolicy()
        self.max_candidates = max_candidates
        self.max_body_bytes = max_body_bytes

    @property
    def is_remote(self) -> bool:
        return bool(getattr(self.provider, "is_remote", True))

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryCandidateSchema],
    ) -> list[MemoryEditProposal]:
        return list(self.extract_batch_with_context(archive, schemas).accepted)

    def extract_with_context(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryCandidateSchema],
        **_: Any,
    ) -> list[MemoryEditProposal]:
        return self.extract(archive, schemas)

    def extract_batch_with_context(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryCandidateSchema],
        **_: Any,
    ) -> MemoryExtractionBatchResult:
        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        assessment = self.egress_policy.evaluate(archive, episode, remote=self.is_remote)
        if self.is_remote and assessment.decision in {EgressDecision.DENY, EgressDecision.LOCAL_ONLY}:
            raise MemoryExtractionSecurityError("remote memory extraction is blocked by egress policy")
        prompt = self.prompt_builder.build(archive, schemas, episode)
        prompt = self.egress_policy.redact(prompt, assessment)
        try:
            raw = self.provider.complete(prompt)
        except BaseException as exc:
            raise classify_memory_extraction_failure(exc) from exc
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
                accepted.append(self._proposal(row, episode, schemas_by_kind))
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

    def _proposal(
        self,
        raw: object,
        episode: EvidenceEpisode,
        schemas: Mapping[MemoryCandidateKind, MemoryCandidateSchema],
    ) -> MemoryEditProposal:
        if not isinstance(raw, Mapping):
            raise MemoryExtractionCandidateValidationError("candidate must be an object")
        keys = {str(key) for key in raw}
        if keys & _FORBIDDEN_KEYS:
            raise MemoryExtractionSecurityError("model attempted to author a trusted storage field")
        unknown = keys - _ALLOWED_FIELDS
        if unknown:
            raise MemoryExtractionCandidateValidationError(f"unsupported candidate fields: {sorted(unknown)}")
        try:
            kind = MemoryCandidateKind(str(raw.get("candidate_kind") or ""))
        except ValueError as exc:
            raise MemoryExtractionCandidateValidationError("candidate_kind is not configured") from exc
        schema = schemas.get(kind)
        if schema is None:
            raise MemoryExtractionCandidateValidationError("candidate_kind is not enabled")
        title = self._text(raw.get("title"), "title", maximum=240)
        body = self._text(raw.get("body"), "body", maximum=self.max_body_bytes)
        confidence = raw.get("confidence", 1.0)
        if isinstance(confidence, bool):
            raise MemoryExtractionCandidateValidationError("confidence must be numeric")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError) as exc:
            raise MemoryExtractionCandidateValidationError("confidence must be numeric") from exc
        if not math.isfinite(confidence_value) or not 0 <= confidence_value <= 1:
            raise MemoryExtractionCandidateValidationError("confidence must be between zero and one")
        evidence_refs = self._strings(raw.get("evidence_refs"), "evidence_refs", required=True)
        if not set(evidence_refs).issubset(episode.event_ids):
            raise MemoryExtractionCandidateValidationError("candidate references unknown evidence")
        field_refs = self._field_refs(raw.get("field_evidence_refs", {}), episode)
        occurred_at = str(raw.get("occurred_at") or "").strip()
        if schema.requires_occurred_at:
            self._timestamp(occurred_at)
        elif occurred_at:
            self._timestamp(occurred_at)
        actors: set[str] = set()
        for item in evidence_refs:
            evidence_event = episode.event(item)
            if evidence_event is not None:
                actors.add(evidence_event.actor.kind)
        if actors and actors.issubset({"assistant", "tool", "service"}) and not schema.allow_assistant_source:
            raise MemoryExtractionCandidateValidationError("candidate kind requires user or system evidence")
        return MemoryEditProposal(
            candidate_kind=kind,
            title=title,
            subject=str(raw.get("subject") or "").strip(),
            body=body,
            entity_hints=self._strings(raw.get("entity_hints", []), "entity_hints"),
            topic_hints=self._strings(raw.get("topic_hints", []), "topic_hints"),
            occurred_at=occurred_at,
            temporal_status=str(raw.get("temporal_status") or "").strip(),
            relation_hints=self._strings(raw.get("relation_hints", []), "relation_hints"),
            evidence_refs=evidence_refs,
            field_evidence_refs=field_refs,
            confidence=confidence_value,
        )

    @staticmethod
    def _text(value: object, name: str, *, maximum: int) -> str:
        if not isinstance(value, str) or not value.strip():
            raise MemoryExtractionCandidateValidationError(f"{name} is required")
        text = value.strip()
        if len(text.encode()) > maximum:
            raise MemoryExtractionCandidateValidationError(f"{name} exceeds bound")
        return text

    @staticmethod
    def _strings(value: object, name: str, *, required: bool = False) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            raise MemoryExtractionCandidateValidationError(f"{name} must be an array")
        rows = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
        if required and not rows:
            raise MemoryExtractionCandidateValidationError(f"{name} cannot be empty")
        if len(rows) > 32:
            raise MemoryExtractionCandidateValidationError(f"{name} exceeds bound")
        return rows

    def _field_refs(self, value: object, episode: EvidenceEpisode) -> dict[str, tuple[str, ...]]:
        if not isinstance(value, Mapping):
            raise MemoryExtractionCandidateValidationError("field_evidence_refs must be an object")
        result: dict[str, tuple[str, ...]] = {}
        for key, rows in value.items():
            field_name = str(key)
            if field_name not in {"title", "subject", "body", "occurred_at", "temporal_status"}:
                raise MemoryExtractionCandidateValidationError("field evidence key is not semantic")
            refs = self._strings(rows, f"field_evidence_refs.{field_name}", required=True)
            if not set(refs).issubset(episode.event_ids):
                raise MemoryExtractionCandidateValidationError("field references unknown evidence")
            result[field_name] = refs
        return result

    @staticmethod
    def _timestamp(value: str) -> None:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise MemoryExtractionCandidateValidationError("occurred_at must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise MemoryExtractionCandidateValidationError("occurred_at must include timezone")


class FakeMemoryModelProvider:
    is_remote = False

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self.response


__all__ = [
    "FakeMemoryModelProvider",
    "LLMMemoryExtractorBackend",
    "MemoryExtractionBatchResult",
    "MemoryExtractionPromptBuilder",
    "MemoryModelProvider",
    "RejectedMemoryCandidate",
]
