"""Build immutable evidence episodes from archived session rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.evidence.event import ActorRef, EventEnvelope, OriginContext, SubjectRef
from memoryos.memory.evidence.scope import ScopeRef, ScopeResolutionSource, scope_from_external

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _parse_time(value: Any, *, fallback: datetime = _EPOCH) -> tuple[datetime, bool, bool]:
    """Return a UTC timestamp plus bounded inferred and invalid markers."""

    if isinstance(value, datetime):
        parsed = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc), value.tzinfo is None, False
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return fallback, True, False
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return fallback, True, True
    inferred = parsed.tzinfo is None
    resolved = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return resolved.astimezone(timezone.utc), inferred, False


@dataclass(frozen=True)
class EvidenceEpisode:
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
        unique_subjects = {
            (item.kind, item.id, item.inferred): item
            for item in self.subjects
        }
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


class SessionArchiveEpisodeAdapter:
    """Normalize all archive rows at the single trusted evidence boundary."""

    def adapt(self, archive: SessionArchive) -> EvidenceEpisode:
        metadata = dict(archive.metadata or {})
        scope = dict(metadata.get("scope", {}) or {})
        tenant_id = str(metadata.get("tenant_id") or scope.get("tenant_id") or "default")
        origin = self._origin(metadata)
        subjects = self._subjects(archive, metadata)
        rows = list(self._rows(archive)) or [
            ("session", {"id": f"{archive.session_id}:empty", "content": f"Session {archive.session_id}"}, 0)
        ]
        events = tuple(
            self._event(archive, tenant_id, origin, subjects, category, row, index)
            for category, row, index in rows
        )
        times = [item.occurred_at for item in events]
        source_uris = tuple(
            dict.fromkeys(
                [archive.archive_uri]
                + [str(item.get("source_uri")) for item in archive.used_contexts if item.get("source_uri")]
            )
        )
        return EvidenceEpisode(
            episode_id=archive.session_id,
            tenant_id=tenant_id,
            events=events,
            started_at=min(times),
            ended_at=max(times),
            origin=origin,
            subjects=subjects,
            used_contexts=tuple(dict(item) for item in archive.used_contexts),
            tool_results=tuple(dict(item) for item in archive.tool_results),
            source_uris=source_uris,
        )

    @staticmethod
    def _rows(archive: SessionArchive) -> Iterable[tuple[str, dict[str, Any], int]]:
        collections = (
            ("message", archive.messages),
            ("observation", archive.observations),
            ("tool_result", archive.tool_results),
            ("action_result", archive.action_results),
            ("feedback", archive.feedback),
        )
        for category, rows in collections:
            for index, row in enumerate(rows):
                yield category, dict(row), index

    @staticmethod
    def _project_id(metadata: Mapping[str, Any], connect: Mapping[str, Any]) -> str:
        extra = dict(connect.get("extra", {}) or {})
        return str(
            metadata.get("project_id")
            or metadata.get("project")
            or connect.get("project_id")
            or extra.get("project_id")
            or extra.get("repo")
            or ""
        )

    def _origin(self, metadata: Mapping[str, Any]) -> OriginContext:
        connect = dict(metadata.get("connect", {}) or {})
        scope = dict(metadata.get("scope", {}) or {})
        raw = dict(metadata.get("origin", {}) or scope.get("origin", {}) or {})
        primary = self._scope(raw.get("primary_scope"))
        project_id = self._project_id(metadata, connect)
        if primary is None and project_id:
            primary = scope_from_external("project", project_id, source=ScopeResolutionSource.ORIGIN)
        qualifiers = tuple(item for item in (self._scope(row) for row in raw.get("qualifiers", []) or []) if item)
        return OriginContext(
            world_domain=str(raw.get("world_domain") or connect.get("world_domain") or "software"),
            connect_type=str(raw.get("connect_type") or connect.get("connect_type") or "agent"),
            adapter_id=str(raw.get("adapter_id") or connect.get("adapter_id") or metadata.get("adapter_id") or "generic_agent"),
            instance_id=str(raw.get("instance_id") or connect.get("agent_instance_id") or "") or None,
            primary_scope=primary,
            qualifiers=qualifiers,
        )

    @staticmethod
    def _subjects(archive: SessionArchive, metadata: Mapping[str, Any]) -> tuple[SubjectRef, ...]:
        scope = dict(metadata.get("scope", {}) or {})
        raw = metadata.get("subjects", []) or scope.get("subjects", []) or []
        result = tuple(
            SubjectRef(str(item["kind"]), str(item["id"]), inferred=bool(item.get("inferred")))
            for item in raw
            if isinstance(item, Mapping) and item.get("kind") and item.get("id")
        )
        return result or (SubjectRef("person", archive.user_id, inferred=True),)

    def _event(
        self,
        archive: SessionArchive,
        tenant_id: str,
        origin: OriginContext,
        subjects: tuple[SubjectRef, ...],
        category: str,
        row: dict[str, Any],
        index: int,
    ) -> EventEnvelope:
        explicit_role = bool(str(row.get("role") or "").strip())
        default_role = "tool" if category == "tool_result" else "user" if category == "message" else "system"
        role = str(row.get("role") or default_role).casefold()
        kind = role if role in {"user", "assistant", "tool", "system", "robot", "sensor", "service"} else "service"
        actor_id = str(row.get("actor_id") or (archive.user_id if kind == "user" else origin.adapter_id))
        archive_ingested, archive_ingested_inferred, archive_ingested_invalid = _parse_time(archive.created_at)
        ingested_raw = row.get("ingested_at")
        ingested_at, ingested_inferred, ingested_invalid = _parse_time(
            ingested_raw,
            fallback=archive_ingested,
        )
        if ingested_raw in (None, ""):
            ingested_inferred = True
        event_time = row.get("occurred_at") or row.get("event_time") or row.get("created_at")
        occurred_at, occurred_inferred, occurred_invalid = _parse_time(event_time, fallback=ingested_at)
        raw_sequence = row.get("sequence", row.get("source_sequence"))
        sequence_inferred = raw_sequence is None
        try:
            sequence = index if raw_sequence is None else int(raw_sequence)
            sequence_invalid = False
        except (TypeError, ValueError):
            sequence = index
            sequence_inferred = True
            sequence_invalid = True
        event_type = str(row.get("event_type") or category).upper()
        status = str(row.get("status") or row.get("result_status") or "").casefold()
        if category == "tool_result" and status in {"failed", "error", "timeout"}:
            event_type = "TOOL_FAILURE"
        elif category == "tool_result" and status in {"recovered", "retry_succeeded"}:
            event_type = "TOOL_RECOVERY"
        event_subjects = self._event_subjects(row, subjects)
        inferred_fields = []
        if not explicit_role:
            inferred_fields.append("actor.role")
        if not row.get("actor_id"):
            inferred_fields.append("actor.id")
        if any(item.inferred for item in event_subjects):
            inferred_fields.append("subjects")
        if occurred_inferred:
            inferred_fields.append("occurred_at")
        if ingested_inferred or archive_ingested_inferred:
            inferred_fields.append("ingested_at")
        if sequence_inferred:
            inferred_fields.append("sequence")
        invalid_fields = [
            field_name
            for field_name, invalid in (
                ("occurred_at", occurred_invalid),
                ("ingested_at", ingested_invalid or archive_ingested_invalid),
                ("sequence", sequence_invalid),
            )
            if invalid
        ]
        return EventEnvelope(
            event_id=str(row.get("event_id") or row.get("id") or row.get("message_id") or f"{category}:{index}"),
            event_type=event_type,
            tenant_id=tenant_id,
            actor=ActorRef(kind, actor_id, role=role, id_inferred=not row.get("actor_id"), role_inferred=not explicit_role),
            subjects=event_subjects,
            origin=origin,
            episode_id=archive.session_id,
            session_id=archive.session_id,
            occurred_at=occurred_at,
            ingested_at=ingested_at,
            sequence=sequence,
            occurred_at_inferred=occurred_inferred,
            ingested_at_inferred=ingested_inferred or archive_ingested_inferred,
            sequence_inferred=sequence_inferred,
            content=row,
            metadata={
                **dict(row.get("metadata", {}) or {}),
                "archive_uri": archive.archive_uri,
                "category": category,
                "inferred_fields": inferred_fields,
                "invalid_fields": invalid_fields,
            },
        )

    @staticmethod
    def _event_subjects(row: Mapping[str, Any], defaults: tuple[SubjectRef, ...]) -> tuple[SubjectRef, ...]:
        raw = row.get("subjects")
        if not isinstance(raw, list):
            return defaults
        result = tuple(
            SubjectRef(str(item["kind"]), str(item["id"]), inferred=bool(item.get("inferred")))
            for item in raw
            if isinstance(item, Mapping) and item.get("kind") and item.get("id")
        )
        return result or defaults

    @staticmethod
    def _scope(payload: Any) -> ScopeRef | None:
        if not isinstance(payload, Mapping) or not payload.get("kind") or not payload.get("id"):
            return None
        return scope_from_external(
            str(payload["kind"]),
            str(payload["id"]),
            namespace=str(payload.get("namespace") or "memoryos"),
            parent_id=str(payload["parent_id"]) if payload.get("parent_id") else None,
            parent_path=tuple(str(item) for item in payload.get("parent_path", []) or []),
            attributes=dict(payload.get("attributes", {}) or {}),
            confidence=float(payload.get("confidence", 1.0)),
            source=str(payload.get("source") or ScopeResolutionSource.EXPLICIT.value),
            inferred=bool(payload.get("inferred", False)),
        )


__all__ = ["EvidenceEpisode", "SessionArchiveEpisodeAdapter"]
