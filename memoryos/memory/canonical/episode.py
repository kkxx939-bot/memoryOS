"""Build immutable evidence episodes from archived session rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical.event import ActorRef, EventEnvelope, OriginContext, SubjectRef
from memoryos.memory.canonical.scope import ScopeRef, ScopeResolutionSource, scope_from_external

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _parse_time(value: Any, *, fallback: datetime = _EPOCH) -> tuple[datetime, bool, bool]:
    """Return UTC time plus inferred/invalid flags without wall-clock fallback."""

    if isinstance(value, datetime):
        inferred = value.tzinfo is None
        parsed = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc), inferred, False
    text = str(value or "").strip().replace("Z", "+00:00")
    if text:
        try:
            parsed = datetime.fromisoformat(text)
            inferred = parsed.tzinfo is None
            resolved = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            return resolved.astimezone(timezone.utc), inferred, False
        except ValueError:
            return fallback, True, True
    return fallback, True, False


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
        if any(event.episode_id != self.episode_id for event in self.events):
            raise ValueError("all events must belong to the evidence episode")
        if any(event.tenant_id != self.tenant_id for event in self.events):
            raise ValueError("all events must belong to the evidence episode tenant")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("evidence episode event IDs must be unique")
        ordered = tuple(sorted(self.events, key=self.sort_key))
        if ordered != self.events:
            object.__setattr__(self, "events", ordered)

    @staticmethod
    def sort_key(event: EventEnvelope) -> tuple[datetime, datetime, int, str]:
        return (event.occurred_at, event.ingested_at or event.occurred_at, event.sequence, event.event_id)

    @property
    def event_ids(self) -> frozenset[str]:
        return frozenset(event.event_id for event in self.events)

    def event(self, event_id: str) -> EventEnvelope | None:
        return next((event for event in self.events if event.event_id == event_id), None)

    def legal_scope_candidates(self) -> tuple[ScopeRef, ...]:
        scopes = [
            scope_from_external("episode", self.episode_id),
            *self.origin.scope_refs,
            *(
                scope_from_external(
                    subject.kind,
                    subject.id,
                    confidence=0.5 if subject.inferred else 1.0,
                    source=ScopeResolutionSource.INFERRED if subject.inferred else ScopeResolutionSource.EVENT,
                    inferred=subject.inferred,
                )
                for subject in self.subjects
                if subject.kind in {"person", "user", "principal", "robot", "device", "asset"}
            ),
        ]
        return tuple({scope.key: scope for scope in scopes}.values())


class SessionArchiveEpisodeAdapter:
    def adapt(self, archive: SessionArchive) -> EvidenceEpisode:
        metadata = dict(archive.metadata or {})
        boundary_scope = dict(metadata.get("scope", {}) or {})
        tenant_id = str(metadata.get("tenant_id") or boundary_scope.get("tenant_id") or "default")
        origin = self._origin(archive)
        subjects = self._subjects(archive, metadata)
        rows = list(self._rows(archive))
        if not rows:
            rows = [("session", {"id": f"{archive.session_id}:empty", "content": f"Session {archive.session_id}"}, 0)]
        events = tuple(
            sorted(
                (
                    self._event(archive, tenant_id, origin, subjects, category, row, source_index)
                    for category, row, source_index in rows
                ),
                key=EvidenceEpisode.sort_key,
            )
        )
        times = [event.occurred_at for event in events]
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

    def _origin(self, archive: SessionArchive) -> OriginContext:
        metadata = dict(archive.metadata or {})
        connect = dict(metadata.get("connect", {}) or {})
        boundary_scope = dict(metadata.get("scope", {}) or {})
        origin_payload = dict(metadata.get("origin", {}) or boundary_scope.get("origin", {}) or {})
        project_id = self._project_id(metadata, connect)
        primary_scope = self._scope(origin_payload.get("primary_scope"))
        if primary_scope is None and project_id:
            primary_scope = scope_from_external(
                "project",
                project_id,
                source=ScopeResolutionSource.ORIGIN,
                inferred=False,
            )
        qualifiers = tuple(
            scope
            for scope in (self._scope(item) for item in origin_payload.get("qualifiers", []) or [])
            if scope is not None
        )
        return OriginContext(
            world_domain=str(origin_payload.get("world_domain") or connect.get("world_domain") or "software"),
            connect_type=str(origin_payload.get("connect_type") or connect.get("connect_type") or "agent"),
            adapter_id=str(
                origin_payload.get("adapter_id")
                or connect.get("adapter_id")
                or metadata.get("adapter_id")
                or "generic_agent"
            ),
            instance_id=str(origin_payload.get("instance_id") or connect.get("agent_instance_id") or "") or None,
            primary_scope=primary_scope,
            qualifiers=qualifiers,
        )

    def _subjects(self, archive: SessionArchive, metadata: dict[str, Any]) -> tuple[SubjectRef, ...]:
        boundary_scope = dict(metadata.get("scope", {}) or {})
        raw = metadata.get("subjects", []) or boundary_scope.get("subjects", []) or []
        subjects = []
        for item in raw:
            if isinstance(item, Mapping) and item.get("kind") and item.get("id"):
                subjects.append(SubjectRef(str(item["kind"]), str(item["id"]), inferred=bool(item.get("inferred"))))
        if not subjects:
            subjects.append(SubjectRef("person", archive.user_id, inferred=True))
        return tuple(dict.fromkeys(subjects))

    def _rows(self, archive: SessionArchive) -> Iterable[tuple[str, dict[str, Any], int]]:
        collections = (
            ("message", archive.messages),
            ("observation", archive.observations),
            ("tool_result", archive.tool_results),
            ("action_result", archive.action_results),
            ("feedback", archive.feedback),
        )
        for category, rows in collections:
            for source_index, row in enumerate(rows):
                yield category, dict(row), source_index

    def _event(
        self,
        archive: SessionArchive,
        tenant_id: str,
        origin: OriginContext,
        subjects: tuple[SubjectRef, ...],
        category: str,
        row: dict[str, Any],
        source_index: int,
    ) -> EventEnvelope:
        explicit_role = bool(str(row.get("role") or "").strip())
        default_role = "tool" if category == "tool_result" else "user" if category == "message" else "system"
        role = str(row.get("role") or default_role).lower()
        actor_kind = (
            role if role in {"user", "assistant", "tool", "system", "robot", "sensor", "service"} else "service"
        )
        explicit_actor = bool(str(row.get("actor_id") or "").strip())
        actor_id = str(row.get("actor_id") or (archive.user_id if actor_kind == "user" else origin.adapter_id))
        event_id = str(row.get("event_id") or row.get("id") or row.get("message_id") or f"{category}:{source_index}")
        event_type = str(row.get("event_type") or category).upper()
        status = str(row.get("status") or row.get("result_status") or "").casefold()
        if category == "tool_result" and status in {"failed", "error", "timeout"}:
            event_type = "TOOL_FAILURE"
        elif category == "tool_result" and status in {"recovered", "retry_succeeded"}:
            event_type = "TOOL_RECOVERY"

        archive_ingested, archive_ingested_inferred, archive_ingested_invalid = _parse_time(archive.created_at)
        ingested_raw = row.get("ingested_at")
        ingested_at, ingested_inferred, ingested_invalid = _parse_time(
            ingested_raw,
            fallback=archive_ingested,
        )
        if ingested_raw in (None, ""):
            ingested_inferred = True
        # ``event_time`` is the Unified Context public name for when an
        # activity happened.  Session adapters historically used
        # ``occurred_at``; accept both at this single evidence boundary while
        # keeping the explicit legacy field authoritative when both exist.
        occurred_raw = row.get("occurred_at") or row.get("event_time") or row.get("created_at")
        occurred_at, occurred_inferred, occurred_invalid = _parse_time(occurred_raw, fallback=ingested_at)
        raw_sequence = row.get("sequence", row.get("source_sequence"))
        sequence_inferred = raw_sequence is None
        try:
            sequence = source_index if raw_sequence is None else int(raw_sequence)
            sequence_invalid = False
        except (TypeError, ValueError):
            sequence = source_index
            sequence_inferred = True
            sequence_invalid = True

        event_subjects = self._event_subjects(row, subjects)
        inferred_fields = []
        if not explicit_role:
            inferred_fields.append("actor.role")
        if not explicit_actor:
            inferred_fields.append("actor.id")
        if any(subject.inferred for subject in event_subjects):
            inferred_fields.append("subjects")
        if occurred_inferred:
            inferred_fields.append("occurred_at")
        if ingested_inferred or archive_ingested_inferred:
            inferred_fields.append("ingested_at")
        if sequence_inferred:
            inferred_fields.append("sequence")
        event_metadata = {
            **dict(row.get("metadata", {}) or {}),
            "archive_uri": archive.archive_uri,
            "category": category,
            "inferred_fields": inferred_fields,
            "invalid_fields": [
                field_name
                for field_name, invalid in (
                    ("occurred_at", occurred_invalid),
                    ("ingested_at", ingested_invalid or archive_ingested_invalid),
                    ("sequence", sequence_invalid),
                )
                if invalid
            ],
        }
        for key in ("salient", "memory_types", "canonical_memory_uris"):
            if key in row:
                event_metadata[key] = row[key]
        return EventEnvelope(
            event_id=event_id,
            event_type=event_type,
            tenant_id=tenant_id,
            actor=ActorRef(
                actor_kind,
                actor_id,
                role=role,
                id_inferred=not explicit_actor,
                role_inferred=not explicit_role,
            ),
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
            metadata=event_metadata,
        )

    def _event_subjects(self, row: Mapping[str, Any], defaults: tuple[SubjectRef, ...]) -> tuple[SubjectRef, ...]:
        raw = row.get("subjects")
        if not isinstance(raw, list):
            return defaults
        explicit = tuple(
            SubjectRef(str(item["kind"]), str(item["id"]), inferred=bool(item.get("inferred")))
            for item in raw
            if isinstance(item, Mapping) and item.get("kind") and item.get("id")
        )
        return explicit or defaults

    def _scope(self, payload: Any) -> ScopeRef | None:
        if not isinstance(payload, Mapping) or not payload.get("kind") or not payload.get("id"):
            return None
        return scope_from_external(
            str(payload["kind"]),
            str(payload["id"]),
            namespace=str(payload.get("namespace") or "memoryos"),
            parent_id=str(payload["parent_id"]) if payload.get("parent_id") else None,
            parent_path=tuple(str(item) for item in payload.get("parent_path", []) or []),
            attributes=dict(payload.get("attributes", {}) or {}),
            confidence=payload.get("confidence", 1.0),
            source=str(payload.get("source") or ScopeResolutionSource.EXPLICIT.value),
            inferred=bool(payload.get("inferred", False)),
        )

    def _project_id(self, metadata: dict[str, Any], connect: dict[str, Any]) -> str:
        extra = dict(connect.get("extra", {}) or {})
        return str(
            metadata.get("project_id")
            or metadata.get("project")
            or connect.get("project_id")
            or extra.get("project_id")
            or extra.get("repo")
            or ""
        )
