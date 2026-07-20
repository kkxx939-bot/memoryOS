"""只生成语义候选的保守本地规则提取器。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from memory.core.formation.schema import MemoryCandidateSchema
from memory.core.formation.signals import MemorySignal, detect_memory_signals, strip_remember_prefix
from memory.core.model import MemoryCandidateKind, MemoryEditProposal
from pre.evidence.session import SessionArchiveEpisodeAdapter, SessionArchiveView


class RuleFallbackExtractor:
    semantic_proposal_backend = True
    llm_semantic_backend = False

    @property
    def is_remote(self) -> bool:
        return False

    def extract(
        self,
        archive: SessionArchiveView,
        schemas: Sequence[MemoryCandidateSchema],
    ) -> list[MemoryEditProposal]:
        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        enabled = {item.candidate_kind: item for item in schemas}
        result: list[MemoryEditProposal] = []
        for event in episode.events:
            text = event.text().strip()
            if not text or event.actor.kind not in {"user", "system", "assistant"}:
                continue
            proposal = self._proposal(text, event.event_id, event.occurred_at, event.actor.kind, enabled)
            if proposal is not None:
                result.append(proposal)
        unique = {
            (item.candidate_kind, item.title.casefold(), item.body.casefold()): item
            for item in result
        }
        return list(unique.values())

    @staticmethod
    def _proposal(
        text: str,
        event_id: str,
        occurred_at: datetime,
        actor_kind: str,
        enabled: dict[MemoryCandidateKind, MemoryCandidateSchema],
    ) -> MemoryEditProposal | None:
        body, remembered = strip_remember_prefix(text)
        if len(body) < 4 or len(body.encode()) > 16 * 1024:
            return None
        signals = detect_memory_signals(body)
        kind: MemoryCandidateKind | None = None
        confidence = 0.72
        if MemorySignal.PREFERENCE in signals:
            kind = MemoryCandidateKind.PREFERENCE
            confidence = 0.86
        elif MemorySignal.PROFILE in signals:
            kind = MemoryCandidateKind.PROFILE_FACT
            confidence = 0.82
        elif MemorySignal.OPEN_LOOP in signals:
            kind = MemoryCandidateKind.OPEN_LOOP
            confidence = 0.78
        elif actor_kind == "assistant" and MemorySignal.EXPERIENCE in signals:
            kind = MemoryCandidateKind.EXPERIENCE
            confidence = 0.7
        elif remembered and MemorySignal.ENTITY in signals:
            kind = MemoryCandidateKind.ENTITY_NOTE
        elif remembered:
            kind = MemoryCandidateKind.TOPIC_NOTE
        if kind is None or kind not in enabled:
            return None
        schema = enabled[kind]
        if actor_kind == "assistant" and not schema.allow_assistant_source:
            return None
        title = RuleFallbackExtractor._title(body)
        timestamp = ""
        if schema.requires_occurred_at:
            resolved = occurred_at.astimezone(timezone.utc)
            timestamp = resolved.isoformat().replace("+00:00", "Z")
        entity_hints = (title,) if kind is MemoryCandidateKind.ENTITY_NOTE else ()
        topic_hints = (title,) if kind is MemoryCandidateKind.TOPIC_NOTE else ()
        return MemoryEditProposal(
            candidate_kind=kind,
            title=title,
            subject=title,
            body=body,
            entity_hints=entity_hints,
            topic_hints=topic_hints,
            occurred_at=timestamp,
            evidence_refs=(event_id,),
            field_evidence_refs={"body": (event_id,)},
            confidence=confidence,
        )

    @staticmethod
    def _title(text: str) -> str:
        compact = " ".join(text.split()).strip("。.!！?？")
        return compact[:80] or "记忆条目"


__all__ = ["RuleFallbackExtractor"]
