"""把公共 Session 证据编码为存储层需要的窄事件结构。"""

from __future__ import annotations

from infrastructure.store.contracts.session_evidence import SessionEvidenceEvent
from pre.evidence.session import SessionArchiveEpisodeAdapter
from pre.session import SessionArchive


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
