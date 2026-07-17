"""ContextDB-owned boundary for encoding immutable session evidence.

The filesystem archive persists generic event snapshots, but it must not know
which domain turns a :class:`SessionArchive` into those snapshots.  A domain
encoder is registered at the composition boundary and returns this narrow,
storage-facing representation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from memoryos.contextdb.session.session_model import SessionArchive


@dataclass(frozen=True)
class SessionEvidenceEvent:
    """One immutable event payload plus its manifest reference fields."""

    payload: Mapping[str, Any]
    event_id: str
    event_digest: str
    event_type: str
    category: str
    occurred_at: Any
    ingested_at: Any
    sequence: int

    def manifest_reference(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_digest": self.event_digest,
            "event_type": self.event_type,
            "category": self.category,
            "occurred_at": self.occurred_at,
            "ingested_at": self.ingested_at,
            "sequence": self.sequence,
        }


class SessionEvidenceEncoder(Protocol):
    """Encode a session archive without exposing a domain event model."""

    def encode(self, archive: SessionArchive) -> tuple[SessionEvidenceEvent, ...]: ...


_encoder: SessionEvidenceEncoder | None = None


def register_session_evidence_encoder(encoder: SessionEvidenceEncoder) -> None:
    """Register the domain encoder at an explicit composition boundary."""

    global _encoder
    _encoder = encoder


def session_evidence_encoder() -> SessionEvidenceEncoder:
    """Return the configured encoder or fail closed when composition is absent."""

    encoder = _encoder
    if encoder is None:
        raise RuntimeError("Session evidence encoder is not registered")
    return encoder


__all__ = [
    "SessionEvidenceEncoder",
    "SessionEvidenceEvent",
    "register_session_evidence_encoder",
    "session_evidence_encoder",
]
