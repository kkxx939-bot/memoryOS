"""Storage-facing encoding of generic Session evidence events."""

from __future__ import annotations

from memoryos.contextdb.session.evidence_encoder import SessionEvidenceEvent
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.evidence.episode import SessionArchiveEpisodeAdapter


class SessionEvidenceArchiveEncoder:
    def encode(self, archive: SessionArchive) -> tuple[SessionEvidenceEvent, ...]:
        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        return tuple(
            SessionEvidenceEvent(
                payload=event.to_dict(),
                event_id=event.event_id,
                event_digest=event.digest,
                event_type=event.event_type,
                category=str(event.metadata.get("category") or "event"),
                occurred_at=event.occurred_at,
                ingested_at=event.ingested_at,
                sequence=event.sequence,
            )
            for event in episode.events
        )


__all__ = ["SessionEvidenceArchiveEncoder"]
