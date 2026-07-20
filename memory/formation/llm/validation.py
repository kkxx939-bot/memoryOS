"""把模型输出收敛为有证据、无存储权限的记忆语义候选。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime

from memory.core.formation.schema import MemoryCandidateSchema
from memory.core.model import MemoryCandidateKind, MemoryEditProposal
from memory.formation.errors import (
    MemoryExtractionCandidateValidationError,
    MemoryExtractionSecurityError,
)
from pre.evidence import EvidenceEpisode

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


class MemoryExtractionCandidateValidator:
    """执行字段、证据、时间和可信控制字段校验。"""

    def __init__(self, *, max_body_bytes: int) -> None:
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")
        self.max_body_bytes = int(max_body_bytes)

    def proposal(
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
        confidence_value = self._confidence(raw.get("confidence", 1.0))
        evidence_refs = self._strings(raw.get("evidence_refs"), "evidence_refs", required=True)
        if not set(evidence_refs).issubset(episode.event_ids):
            raise MemoryExtractionCandidateValidationError("candidate references unknown evidence")
        field_refs = self._field_refs(raw.get("field_evidence_refs", {}), episode, evidence_refs)
        occurred_at = str(raw.get("occurred_at") or "").strip()
        if schema.requires_occurred_at or occurred_at:
            self._timestamp(occurred_at)
        actors = {
            event.actor.kind
            for event_id in evidence_refs
            if (event := episode.event(event_id)) is not None
        }
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
    def _confidence(value: object) -> float:
        if isinstance(value, bool):
            raise MemoryExtractionCandidateValidationError("confidence must be numeric")
        try:
            result = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise MemoryExtractionCandidateValidationError("confidence must be numeric") from exc
        if not math.isfinite(result) or not 0 <= result <= 1:
            raise MemoryExtractionCandidateValidationError("confidence must be between zero and one")
        return result

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

    def _field_refs(
        self,
        value: object,
        episode: EvidenceEpisode,
        evidence_refs: tuple[str, ...],
    ) -> dict[str, tuple[str, ...]]:
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
            if not set(refs).issubset(evidence_refs):
                raise MemoryExtractionCandidateValidationError("field references must belong to candidate evidence")
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


__all__ = ["MemoryExtractionCandidateValidator"]
