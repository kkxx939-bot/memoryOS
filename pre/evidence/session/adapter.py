"""把 Session 归档规范化为公共证据模型。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol

from pre.evidence.model.episode import EvidenceEpisode
from pre.evidence.model.event import ActorRef, EventEnvelope, OriginContext, SubjectRef
from pre.evidence.model.scope import ScopeRef, ScopeResolutionSource, scope_from_external

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _parse_time(value: Any, *, fallback: datetime = _EPOCH) -> tuple[datetime, bool, bool]:
    """返回 UTC 时间，并显式标记推断值和非法输入。"""

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


class SessionArchiveView(Protocol):
    """证据适配器所需的最小归档视图，避免公共 Evidence 反向依赖 Memory。"""

    @property
    def user_id(self) -> str: ...

    @property
    def session_id(self) -> str: ...

    @property
    def archive_uri(self) -> str: ...

    @property
    def created_at(self) -> str: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    @property
    def messages(self) -> Sequence[Mapping[str, Any]]: ...

    @property
    def observations(self) -> Sequence[Mapping[str, Any]]: ...

    @property
    def tool_results(self) -> Sequence[Mapping[str, Any]]: ...

    @property
    def action_results(self) -> Sequence[Mapping[str, Any]]: ...

    @property
    def feedback(self) -> Sequence[Mapping[str, Any]]: ...

    @property
    def used_contexts(self) -> Sequence[Mapping[str, Any]]: ...


class SessionArchiveEpisodeAdapter:
    """在唯一可信证据边界规范化全部 Session 事件。"""

    def adapt(self, archive: SessionArchiveView) -> EvidenceEpisode:
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
    def _rows(archive: SessionArchiveView) -> Iterable[tuple[str, dict[str, Any], int]]:
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
    def _subjects(archive: SessionArchiveView, metadata: Mapping[str, Any]) -> tuple[SubjectRef, ...]:
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
        archive: SessionArchiveView,
        tenant_id: str,
        origin: OriginContext,
        subjects: tuple[SubjectRef, ...],
        category: str,
        row: dict[str, Any],
        index: int,
    ) -> EventEnvelope:
        explicit_role = bool(str(row.get("role") or "").strip())
        # 缺省角色按事件来源确定。Observation 不能伪装成 system，
        # 否则传感器文本可能错误触发只允许用户或系统声明的长期记忆规则。
        default_role = {
            "message": "user",
            "observation": "sensor",
            "tool_result": "tool",
            "action_result": "service",
            "feedback": "user",
            "session": "system",
        }.get(category, "service")
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


__all__ = ["SessionArchiveEpisodeAdapter", "SessionArchiveView"]
