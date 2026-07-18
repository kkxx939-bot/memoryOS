"""Conservative local fallback that emits semantic proposals only."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timezone

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.documents.model import MemoryCandidateKind, MemoryEditProposal
from memoryos.memory.evidence import SessionArchiveEpisodeAdapter
from memoryos.memory.extraction.memory_extractor import MemoryExtractorBackend
from memoryos.memory.schema import MemoryCandidateSchema

_REMEMBER = re.compile(r"(?i)^(?:请)?(?:记住|记得|remember(?: that)?)\s*[:：,，]?\s*(.+)$")
_PREFERENCE = re.compile(r"(?i)(?:我(?:不)?喜欢|我更喜欢|我的偏好是|i (?:prefer|dislike|like))")
_PROFILE = re.compile(r"(?i)^(?:我是|我叫|我的.+是|i am|my name is)")
_OPEN_LOOP = re.compile(r"(?i)(?:待确认|尚未解决|以后再看|需要跟进|todo|open question|follow up)")
_EXPERIENCE = re.compile(r"(?i)(?:经验|教训|有效做法|复用|lesson|worked well|reusable)")
_ENTITY = re.compile(r"(?i)(?:项目|系统|产品|公司|组织|project|system|product|organization)")


class RuleFallbackExtractor(MemoryExtractorBackend):
    semantic_proposal_backend = True
    llm_semantic_backend = False

    @property
    def is_remote(self) -> bool:
        return False

    def extract(
        self,
        archive: SessionArchive,
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

    def extract_with_context(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryCandidateSchema],
        **_: object,
    ) -> list[MemoryEditProposal]:
        return self.extract(archive, schemas)

    @staticmethod
    def _proposal(
        text: str,
        event_id: str,
        occurred_at: datetime,
        actor_kind: str,
        enabled: dict[MemoryCandidateKind, MemoryCandidateSchema],
    ) -> MemoryEditProposal | None:
        remembered = _REMEMBER.match(text)
        body = (remembered.group(1) if remembered else text).strip()
        if len(body) < 4 or len(body.encode()) > 16 * 1024:
            return None
        kind: MemoryCandidateKind | None = None
        confidence = 0.72
        if _PREFERENCE.search(body):
            kind = MemoryCandidateKind.PREFERENCE
            confidence = 0.86
        elif _PROFILE.search(body):
            kind = MemoryCandidateKind.PROFILE_FACT
            confidence = 0.82
        elif _OPEN_LOOP.search(body):
            kind = MemoryCandidateKind.OPEN_LOOP
            confidence = 0.78
        elif actor_kind == "assistant" and _EXPERIENCE.search(body):
            kind = MemoryCandidateKind.EXPERIENCE
            confidence = 0.7
        elif remembered and _ENTITY.search(body):
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
        return compact[:80] or "Memory note"


__all__ = ["RuleFallbackExtractor"]
