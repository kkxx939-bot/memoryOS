"""Memory-domain implementation of the session evidence encoder boundary."""

from __future__ import annotations

from memoryos.contextdb.session.evidence_encoder import (
    SessionEvidenceEvent,
    register_session_evidence_encoder,
)
from memoryos.contextdb.session.session_model import SessionArchive


class CanonicalSessionEvidenceEncoder:
    """Encode archives with Memory's canonical episode semantics."""

    def encode(self, archive: SessionArchive) -> tuple[SessionEvidenceEvent, ...]:
        # Keep the provider importable while ``memoryos.memory`` initializes;
        # the canonical package may itself be the caller that triggered that
        # initialization.
        from memoryos.memory.canonical.episode import SessionArchiveEpisodeAdapter

        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        return tuple(
            SessionEvidenceEvent(
                payload=event.to_dict(),
                event_id=event.event_id,
                event_digest=event.digest,
                event_type=event.event_type,
                category=str(event.metadata.get("category", "")),
                occurred_at=event.occurred_at,
                ingested_at=event.ingested_at,
                sequence=event.sequence,
            )
            for event in episode.events
        )


def register_default_session_evidence_encoder() -> None:
    """Install Memory's encoder for historical direct domain composition."""

    register_session_evidence_encoder(CanonicalSessionEvidenceEncoder())


__all__ = [
    "CanonicalSessionEvidenceEncoder",
    "register_default_session_evidence_encoder",
]
