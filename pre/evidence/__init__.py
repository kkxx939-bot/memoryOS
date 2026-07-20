"""业务前置阶段形成、供各领域复用的不可变 Session 证据。"""

from pre.evidence.model import (
    ActorRef,
    EventEnvelope,
    EvidenceEpisode,
    OriginContext,
    ScopeRef,
    ScopeResolutionSource,
    SubjectRef,
    scope_from_external,
)
from pre.evidence.session import SessionArchiveEpisodeAdapter, SessionArchiveView

__all__ = [
    "ActorRef",
    "EvidenceEpisode",
    "EventEnvelope",
    "OriginContext",
    "ScopeRef",
    "ScopeResolutionSource",
    "SessionArchiveEpisodeAdapter",
    "SessionArchiveView",
    "SubjectRef",
    "scope_from_external",
]
