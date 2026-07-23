"""直接把 SessionArchive 原始集合编码为归档存储结构。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from foundation.integrity import canonical_digest
from infrastructure.store.contracts.session_archive_event import SessionArchiveEvent
from pre.session import SessionArchive

_COLLECTIONS = (
    "messages",
    "observations",
    "tool_results",
    "action_results",
    "feedback",
)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _normalized_time(value: Any, *, fallback: str) -> str:
    candidate = value if value not in (None, "") else fallback
    try:
        parsed = datetime.fromisoformat(str(candidate or "").replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.fromisoformat(str(fallback or "").replace("Z", "+00:00"))
        except ValueError:
            parsed = _EPOCH
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _rows(archive: SessionArchive) -> Iterable[tuple[str, Mapping[str, Any], int]]:
    ordinal = 0
    for category in _COLLECTIONS:
        values: Sequence[Mapping[str, Any]] = getattr(archive, category)
        for value in values:
            if not isinstance(value, Mapping):
                raise TypeError(f"SessionArchive {category} entries must be objects")
            yield category.removesuffix("s"), value, ordinal
            ordinal += 1


class CanonicalSessionArchiveEventEncoder:
    def encode(self, archive: SessionArchive) -> tuple[SessionArchiveEvent, ...]:
        encoded: list[SessionArchiveEvent] = []
        for category, source, ordinal in _rows(archive):
            raw = dict(source)
            event_id = str(raw.get("event_id") or raw.get("id") or raw.get("message_id") or f"{category}:{ordinal}")
            event_type = str(raw.get("event_type") or category).upper()
            ingested_at = _normalized_time(raw.get("ingested_at"), fallback=archive.created_at)
            occurred_at = _normalized_time(
                raw.get("occurred_at") or raw.get("event_time") or raw.get("created_at"),
                fallback=ingested_at,
            )
            body = {
                "event_id": event_id,
                "event_type": event_type,
                "session_id": archive.session_id,
                "occurred_at": occurred_at,
                "ingested_at": ingested_at,
                "sequence": ordinal,
                "content": raw,
                "metadata": {
                    "archive_uri": archive.archive_uri,
                    "category": category,
                },
            }
            event_digest = canonical_digest(body)
            encoded.append(
                SessionArchiveEvent(
                    payload={**body, "event_digest": event_digest},
                    event_id=event_id,
                    event_digest=event_digest,
                    event_type=event_type,
                    category=category,
                    occurred_at=occurred_at,
                    ingested_at=ingested_at,
                    sequence=ordinal,
                )
            )
        return tuple(encoded)


__all__ = ["CanonicalSessionArchiveEventEncoder"]
