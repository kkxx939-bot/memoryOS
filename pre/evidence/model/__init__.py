"""证据领域模型的公开入口。"""

from pre.evidence.model.episode import EvidenceEpisode
from pre.evidence.model.event import ActorRef, EventEnvelope, OriginContext, SubjectRef
from pre.evidence.model.scope import ScopeRef, ScopeResolutionSource, scope_from_external

__all__ = [
    "ActorRef",
    "EvidenceEpisode",
    "EventEnvelope",
    "OriginContext",
    "ScopeRef",
    "ScopeResolutionSource",
    "SubjectRef",
    "scope_from_external",
]
