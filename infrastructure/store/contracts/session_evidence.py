"""SessionArchive Store 依赖的证据编码窄协议。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pre.session import SessionArchive


@dataclass(frozen=True)
class SessionEvidenceEvent:
    """不可变事件正文及其 Manifest 引用字段。"""

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
    """把会话归档编码为存储层可持久化的事件。"""

    def encode(self, archive: SessionArchive) -> tuple[SessionEvidenceEvent, ...]: ...


__all__ = ["SessionEvidenceEncoder", "SessionEvidenceEvent"]
