from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from memoryos.memory.canonical.episode import EvidenceEpisode
from memoryos.memory.canonical.event import EventEnvelope
from memoryos.memory.canonical.proposal import EpistemicStatus, MemorySemanticProposal


def evidence_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceRef:
    event_id: str
    source_uri: str | None
    content_hash: str
    span_start: int | None = None
    span_end: int | None = None
    quoted_text_hash: str | None = None

    @classmethod
    def from_event(
        cls,
        event: EventEnvelope,
        *,
        source_uri: str | None = None,
        span_start: int | None = None,
        span_end: int | None = None,
    ) -> EvidenceRef:
        text = event.text()
        quoted_hash = None
        if span_start is not None and span_end is not None:
            quoted_hash = evidence_hash(text[span_start:span_end])
        return cls(
            event_id=event.event_id,
            source_uri=source_uri,
            content_hash=evidence_hash(text),
            span_start=span_start,
            span_end=span_end,
            quoted_text_hash=quoted_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_uri": self.source_uri,
            "content_hash": self.content_hash,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "quoted_text_hash": self.quoted_text_hash,
        }


@dataclass(frozen=True)
class ProposalValidationResult:
    valid: bool
    proposal: MemorySemanticProposal
    errors: tuple[str, ...] = ()
    unsupported_fields: tuple[str, ...] = ()


class ProposalEvidenceValidator:
    """Validates model references against immutable episode content."""

    def validate(self, proposal: MemorySemanticProposal, episode: EvidenceEpisode) -> ProposalValidationResult:
        errors: list[str] = []
        evidence_texts: list[str] = []
        evidence_actor_kinds: list[str] = []
        if not proposal.evidence_refs:
            errors.append("missing_evidence")
        for ref in proposal.evidence_refs:
            event = episode.event(ref.event_id)
            if event is None:
                errors.append(f"unknown_event:{ref.event_id}")
                continue
            text = event.text()
            evidence_actor_kinds.append(event.actor.kind)
            if evidence_hash(text) != ref.content_hash:
                errors.append(f"content_hash_mismatch:{ref.event_id}")
                continue
            if (ref.span_start is None) != (ref.span_end is None):
                errors.append(f"incomplete_span:{ref.event_id}")
                continue
            selected = text
            if ref.span_start is not None and ref.span_end is not None:
                if ref.span_start < 0 or ref.span_end <= ref.span_start or ref.span_end > len(text):
                    errors.append(f"invalid_span:{ref.event_id}")
                    continue
                selected = text[ref.span_start : ref.span_end]
                if ref.quoted_text_hash and evidence_hash(selected) != ref.quoted_text_hash:
                    errors.append(f"quoted_text_hash_mismatch:{ref.event_id}")
                    continue
            evidence_texts.append(selected)

        unsupported = self._unsupported_fields(proposal, evidence_texts)
        hardened = proposal
        if unsupported and proposal.epistemic_status in {EpistemicStatus.EXPLICIT, EpistemicStatus.OBSERVED}:
            hardened = replace(proposal, epistemic_status=EpistemicStatus.INFERRED)
            errors.append("unsupported_core_fields")
        semantic_errors = self._semantic_errors(proposal, evidence_texts, evidence_actor_kinds)
        if semantic_errors and hardened.epistemic_status == EpistemicStatus.EXPLICIT:
            hardened = replace(hardened, epistemic_status=EpistemicStatus.INFERRED)
        errors.extend(semantic_errors)
        return ProposalValidationResult(
            valid=not errors,
            proposal=hardened,
            errors=tuple(dict.fromkeys(errors)),
            unsupported_fields=tuple(unsupported),
        )

    def _semantic_errors(
        self,
        proposal: MemorySemanticProposal,
        evidence_texts: list[str],
        actor_kinds: list[str],
    ) -> list[str]:
        errors = []
        if proposal.epistemic_status == EpistemicStatus.EXPLICIT and not any(
            kind in {"user", "system"} for kind in actor_kinds
        ):
            errors.append("explicit_status_requires_authoritative_evidence")
        semantic = proposal.semantic
        speech = str(getattr(semantic.speech_act, "value", semantic.speech_act)).casefold()
        commitment = str(getattr(semantic.commitment, "value", semantic.commitment)).casefold()
        if speech in {"confirmation", "correction"} or commitment in {"confirmed", "committed"}:
            text = "\n".join(evidence_texts).casefold()
            signals = (
                "confirm",
                "confirmed",
                "decided",
                "adopted",
                "formally change",
                "remains active",
                "continue using",
                "must",
                "do not",
                "prefer",
                "确认",
                "决定",
                "采用",
                "正式改成",
                "继续使用",
                "必须",
                "禁止",
                "偏好",
                "喜欢",
            )
            negative_signals = (
                "no confirmation",
                "not confirmed",
                "only a future option",
                "future option; no",
                "尚未确认",
                "未确认",
                "只是候选",
                "仅供评估",
            )
            if any(signal in text for signal in negative_signals) or not any(signal in text for signal in signals):
                errors.append("semantic_confirmation_unsupported")
        return errors

    def _unsupported_fields(self, proposal: MemorySemanticProposal, evidence_texts: list[str]) -> list[str]:
        if not evidence_texts:
            return [
                *[f"identity.{key}" for key in proposal.identity_fields],
                *[f"value.{key}" for key in proposal.value_fields],
            ]
        haystack = "\n".join(evidence_texts).casefold()
        unsupported = []
        system_identity_fields = {str(item) for item in proposal.metadata.get("system_identity_fields", []) or []}
        for prefix, fields in (("identity", proposal.identity_fields), ("value", proposal.value_fields)):
            for key, value in fields.items():
                if prefix == "identity" and key in system_identity_fields:
                    continue
                if not self._supported(value, haystack):
                    unsupported.append(f"{prefix}.{key}")
        return unsupported

    def _supported(self, value: Any, haystack: str) -> bool:
        if value is None or value == "":
            return False
        if isinstance(value, bool | int | float):
            return str(value).casefold() in haystack
        if isinstance(value, str):
            candidates = {value.casefold(), value.replace("_", " ").casefold(), value.replace("-", " ").casefold()}
            return any(candidate in haystack for candidate in candidates)
        if isinstance(value, Mapping):
            return all(self._supported(item, haystack) for item in value.values())
        if isinstance(value, list | tuple | set):
            return all(self._supported(item, haystack) for item in value)
        return str(value).casefold() in haystack
