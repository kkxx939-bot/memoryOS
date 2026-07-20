"""构造 Session 的语义片段、资源引用、路径和时间字段。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from infrastructure.context.layers.generator import l0_abstract
from infrastructure.store.model.catalog import CatalogRecord, CatalogRecordKind
from pre.session import SessionArchive
from sanitization.context_projection import ContextProjectionSanitizer

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")


class SessionRecordBuilderMixin:
    """为 Session 投影器提供纯记录构造辅助方法。"""

    # 由 SessionContextProjector 绑定；在 Mixin 中显式声明以保持类型边界完整。
    sanitizer: ContextProjectionSanitizer
    semantic_segment_size: int
    vectorize_important_events: bool

    def _record(self, **kwargs: Any) -> CatalogRecord:
        return CatalogRecord(**kwargs).with_sanitized_projection(self.sanitizer)

    def _semantic_segments(
        self,
        archive: SessionArchive,
        event_records: Sequence[CatalogRecord],
        *,
        common: Mapping[str, Any],
        root_uri: str,
    ) -> list[CatalogRecord]:
        source_events = [
            record for record in event_records if record.record_kind != CatalogRecordKind.RESOURCE_REFERENCE.value
        ]
        # 一个语义片段只有一个结构化 ``event_time``，不能跨越多个本地时间线日期；
        # 否则记录会声明首事件时间，却又能从另一天的路径访问。切分过程保持确定性，
        # 并继续受现有最大片段长度约束。
        chunks: list[list[CatalogRecord]] = []
        chunk: list[CatalogRecord] = []
        chunk_timeline = ""
        for record in source_events:
            timeline = next((path for path in record.tree_paths if path.startswith("timeline/")), "")
            if not timeline:
                raise ValueError("session event projection has no Timeline path")
            if chunk and (len(chunk) >= self.semantic_segment_size or timeline != chunk_timeline):
                chunks.append(chunk)
                chunk = []
            if not chunk:
                chunk_timeline = timeline
            chunk.append(record)
        if chunk:
            chunks.append(chunk)

        segments: list[CatalogRecord] = []
        for segment_index, chunk in enumerate(chunks):
            digest = self.sanitizer.digest([record.source_digest for record in chunk])
            text = "\n".join(record.l1_text or record.l0_text for record in chunk)
            paths = tuple(dict.fromkeys(path for record in chunk for path in record.tree_paths))[:8]
            segments.append(
                self._record(
                    **common,
                    record_key=self._key(archive, "semantic_segment", digest),
                    uri=f"{archive.archive_uri.rstrip('/')}/context/segments/{segment_index}",
                    source_kind="semantic_segment",
                    record_kind=CatalogRecordKind.SEMANTIC_SEGMENT.value,
                    parent_uri=root_uri,
                    tree_paths=paths,
                    event_time=chunk[0].event_time,
                    ingested_at=chunk[-1].ingested_at,
                    title=f"Session {archive.session_id} segment {segment_index + 1}",
                    l0_text=l0_abstract(text),
                    l1_text=text,
                    source_uri=archive.archive_uri,
                    source_digest=digest,
                    metadata={
                        "archive_uri": archive.archive_uri,
                        "event_source_digests": [record.source_digest for record in chunk],
                        "vector_eligible": True,
                    },
                )
            )
        return segments

    def _resource_record(
        self,
        archive: SessionArchive,
        tool_record: CatalogRecord,
        raw: Mapping[str, Any],
        *,
        common: Mapping[str, Any],
        root_uri: str,
    ) -> CatalogRecord:
        name = str(tool_record.metadata.get("resource_name") or "resource")
        location = str(tool_record.metadata.get("resource_location") or "external")
        digest = self.sanitizer.digest(
            {"tool": tool_record.source_digest, "resource_name": name, "resource_location": location}
        )
        paths = tuple(dict.fromkeys((*tool_record.tree_paths, f"resources/{self._segment(location)}")))[:8]
        return self._record(
            **common,
            record_key=self._key(archive, "resource", digest),
            uri=f"{archive.archive_uri.rstrip('/')}/context/resources/{digest[:20]}",
            source_kind="resource_reference",
            record_kind=CatalogRecordKind.RESOURCE_REFERENCE.value,
            parent_uri=root_uri,
            tree_paths=paths,
            event_time=tool_record.event_time,
            ingested_at=tool_record.ingested_at,
            title=name,
            l0_text=f"{location} resource: {name}",
            l1_text=str(raw.get("description") or raw.get("summary") or name),
            source_uri=tool_record.source_uri,
            source_digest=tool_record.source_digest,
            metadata={
                "archive_uri": archive.archive_uri,
                "resource_name": name,
                "resource_location": location,
                "tool_result_record_key": tool_record.record_key,
                "vector_eligible": bool(
                    self.vectorize_important_events
                    and (raw.get("important") or raw.get("salient") or raw.get("pinned"))
                ),
            },
        )

    def _reference_records(
        self,
        archive: SessionArchive,
        values: Iterable[Mapping[str, Any]],
        *,
        kind: CatalogRecordKind,
        common: Mapping[str, Any],
        root_uri: str,
        base_paths: tuple[str, ...],
        fallback_event_time: object,
    ) -> list[CatalogRecord]:
        result: list[CatalogRecord] = []
        for index, value in enumerate(values):
            raw = dict(value)
            digest = self.sanitizer.digest(raw)
            text = self._event_text(raw)
            title = str(raw.get("title") or raw.get("name") or raw.get("skill_name") or f"{kind.value} {index + 1}")
            reference_event_time = raw.get("event_time") or raw.get("occurred_at") or fallback_event_time
            paths = [
                self._timeline_path(reference_event_time, archive.metadata),
                *(path for path in base_paths if not path.startswith("timeline/")),
            ]
            if kind is CatalogRecordKind.USED_SKILL:
                paths.append(f"skills/{self._segment(raw.get('skill_name') or raw.get('name') or title)}")
            result.append(
                self._record(
                    **common,
                    record_key=self._key(archive, kind.value, digest),
                    uri=f"{archive.archive_uri.rstrip('/')}/context/{kind.value}/{digest[:20]}",
                    source_kind=kind.value,
                    record_kind=kind.value,
                    parent_uri=root_uri,
                    tree_paths=tuple(dict.fromkeys(paths))[:8],
                    event_time=self._iso(reference_event_time),
                    ingested_at=self._iso(raw.get("ingested_at") or archive.created_at),
                    title=title,
                    l0_text=l0_abstract(text or title),
                    l1_text=text,
                    source_uri=archive.archive_uri,
                    source_digest=digest,
                    metadata={
                        "archive_uri": archive.archive_uri,
                        "source_reference_uri": str(raw.get("source_uri") or raw.get("uri") or ""),
                        "vector_eligible": False,
                    },
                )
            )
        return result

    def _event_paths(
        self,
        archive: SessionArchive,
        event_time: object,
        *,
        base_paths: tuple[str, ...],
        raw: Mapping[str, Any],
    ) -> tuple[str, ...]:
        # 事件节点属于事件真实发生的日期。Session 根节点可以带归档或 Episode
        # 时间线路径，但不能把写入时间路径复制给每个事件，否则同一个 Tool Result
        # 会错误地出现在多个日期下。
        paths = [
            self._timeline_path(event_time, archive.metadata),
            *(path for path in base_paths if not path.startswith("timeline/")),
        ]
        resource_name, location = self.sanitizer._resource_identity(raw)  # noqa: SLF001 - 统一策略边界
        if resource_name:
            paths.append(f"resources/{self._segment(location or 'external')}")
        return tuple(dict.fromkeys(paths))[:8]

    def _base_paths(
        self,
        archive: SessionArchive,
        *,
        event_time: object,
        workspace_id: str,
        adapter_id: str,
    ) -> tuple[str, ...]:
        paths = [
            f"sessions/{self._segment(archive.session_id)}",
            self._timeline_path(event_time, archive.metadata),
        ]
        if workspace_id:
            paths.append(f"projects/{self._segment(workspace_id)}")
        if adapter_id:
            paths.append(f"agents/{self._segment(adapter_id)}")
        return tuple(dict.fromkeys(paths))

    def _event_metadata(
        self,
        archive: SessionArchive,
        raw: Mapping[str, Any],
        *,
        category: str,
        event_id: str,
    ) -> dict[str, Any]:
        allowed = {
            key: raw[key]
            for key in (
                "tool_name",
                "name",
                "status",
                "result_status",
                "resource_uri",
                "file_uri",
                "path",
                "file_path",
                "absolute_path",
                "file_name",
                "filename",
                "resource_name",
                "resource_location",
                "important",
                "salient",
            )
            if key in raw
        }
        return {
            **allowed,
            "archive_uri": archive.archive_uri,
            "event_id": event_id,
            "event_category": category,
            "projection_source": "session_archive_event",
        }

    @staticmethod
    def _event_text(raw: Mapping[str, Any]) -> str:
        for key in ("content", "text", "raw_text", "scene", "output", "result", "summary", "description"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, Mapping | list) and value:
                return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return json.dumps(dict(raw), ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _event_title(category: str, raw: Mapping[str, Any], ordinal: int) -> str:
        return str(
            raw.get("title")
            or raw.get("file_name")
            or raw.get("filename")
            or raw.get("resource_name")
            or raw.get("tool_name")
            or f"{category.replace('_', ' ').title()} {ordinal + 1}"
        )

    @staticmethod
    def _key(archive: SessionArchive, kind: str, digest: str) -> str:
        return f"session:{archive.session_id}:{archive.manifest_digest}:{kind}:{digest[:32]}"

    @staticmethod
    def _segment(value: object) -> str:
        text = str(value or "unknown").strip()
        if _SAFE_SEGMENT.fullmatch(text):
            return text
        return "id-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]

    @classmethod
    def _timeline_path(cls, value: object, metadata: Mapping[str, Any]) -> str:
        parsed = cls._datetime(value)
        timezone_name = str(metadata.get("timezone") or metadata.get("time_zone") or "")
        if timezone_name:
            try:
                zone: tzinfo = ZoneInfo(timezone_name)
            except ZoneInfoNotFoundError:
                zone = timezone.utc
        else:
            # Session 未配置 IANA 时区时，保留证据携带的显式时区偏移。
            # 结构化 event_time 仍使用 UTC，只对受控的逻辑日期路径做本地化。
            zone = parsed.tzinfo or timezone.utc
        local = parsed.astimezone(zone)
        return f"timeline/{local.year:04d}/{local.month:02d}/{local.day:02d}"

    @staticmethod
    def _datetime(value: object) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("session projection timestamp must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("session projection timestamp must include timezone")
        return parsed

    @classmethod
    def _iso(cls, value: object) -> str:
        return cls._datetime(value).astimezone(timezone.utc).isoformat()


__all__ = ["SessionRecordBuilderMixin"]
