"""把记忆候选确定性路由到有限的用户可见目录。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

from foundation.ids import stable_hash
from memory.core.model import MemoryCandidateKind, MemoryEditProposal

_SLUG_TOKEN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class MemoryDocumentRouter:
    def route(self, proposal: MemoryEditProposal) -> str:
        kind = proposal.candidate_kind
        if kind is MemoryCandidateKind.PROFILE_FACT:
            return "profile.md"
        if kind is MemoryCandidateKind.PREFERENCE:
            return "preferences.md"
        if kind is MemoryCandidateKind.OPEN_LOOP:
            return "knowledge/open-loops.md"
        subject = proposal.subject or proposal.title
        if kind is MemoryCandidateKind.ENTITY_NOTE:
            hint = proposal.entity_hints[0] if proposal.entity_hints else subject
            return f"knowledge/entities/{self.safe_slug(hint)}.md"
        if kind is MemoryCandidateKind.TOPIC_NOTE:
            hint = proposal.topic_hints[0] if proposal.topic_hints else subject
            return f"knowledge/topics/{self.safe_slug(hint)}.md"
        if kind is MemoryCandidateKind.EPISODE:
            return f"knowledge/episodes/{self._date(proposal.occurred_at)}-{self.safe_slug(subject)}.md"
        if kind is MemoryCandidateKind.EXPERIENCE:
            return f"experiences/{self._date(proposal.occurred_at)}-{self.safe_slug(subject)}.md"
        raise ValueError(f"unsupported memory candidate kind: {kind}")

    @staticmethod
    def safe_slug(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
        slug = _SLUG_TOKEN.sub("-", normalized).strip("-")[:120]
        return slug or f"note-{stable_hash(str(value), 16)}"

    @staticmethod
    def _date(value: str) -> str:
        if not value:
            raise ValueError("episode and experience candidates require occurred_at")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("occurred_at must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("occurred_at must include a timezone")
        return parsed.date().isoformat()


__all__ = ["MemoryDocumentRouter"]
