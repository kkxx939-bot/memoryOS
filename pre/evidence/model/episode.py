"""由一组不可变事件组成的 Session 证据单元。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pre.evidence.model.event import EventEnvelope, OriginContext, SubjectRef
from pre.evidence.model.scope import ScopeRef, ScopeResolutionSource, scope_from_external


@dataclass(frozen=True)
class EvidenceEpisode:
    """一次 Session 的规范化证据视图，不携带任何写入权限。"""

    episode_id: str
    tenant_id: str
    events: tuple[EventEnvelope, ...]
    started_at: datetime
    ended_at: datetime
    origin: OriginContext
    subjects: tuple[SubjectRef, ...]
    used_contexts: tuple[Mapping[str, Any], ...] = ()
    tool_results: tuple[Mapping[str, Any], ...] = ()
    source_uris: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("evidence episode must contain at least one event")
        if any(item.episode_id != self.episode_id or item.tenant_id != self.tenant_id for item in self.events):
            raise ValueError("evidence episode identity mismatch")
        if len({item.event_id for item in self.events}) != len(self.events):
            raise ValueError("evidence event IDs must be unique")
        object.__setattr__(self, "events", tuple(sorted(self.events, key=self.sort_key)))
        unique_subjects = {(item.kind, item.id, item.inferred): item for item in self.subjects}
        object.__setattr__(
            self,
            "subjects",
            tuple(unique_subjects[key] for key in sorted(unique_subjects)),
        )

    @staticmethod
    def sort_key(event: EventEnvelope) -> tuple[datetime, datetime, int, str]:
        return event.occurred_at, event.ingested_at, event.sequence, event.event_id

    @property
    def event_ids(self) -> frozenset[str]:
        return frozenset(item.event_id for item in self.events)

    def event(self, event_id: str) -> EventEnvelope | None:
        return next((item for item in self.events if item.event_id == event_id), None)

    def legal_scope_candidates(self) -> tuple[ScopeRef, ...]:
        """返回只能用于后续策略判断的候选作用域，不授予访问权限。"""

        rows = [scope_from_external("episode", self.episode_id), *self.origin.scope_refs]
        rows.extend(
            scope_from_external(
                item.kind,
                item.id,
                confidence=0.5 if item.inferred else 1.0,
                source=ScopeResolutionSource.INFERRED if item.inferred else ScopeResolutionSource.EVENT,
                inferred=item.inferred,
            )
            for item in self.subjects
            if item.kind in {"person", "user", "principal", "robot", "device", "asset"}
        )
        unique = {item.key: item for item in rows}
        return tuple(unique.values())


__all__ = ["EvidenceEpisode"]
