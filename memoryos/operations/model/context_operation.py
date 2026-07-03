from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.contextdb.model.context_type import ContextType
from memoryos.core.ids import new_id
from memoryos.core.time import utc_now
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


@dataclass
class ContextOperation:
    context_type: ContextType
    action: OperationAction
    payload: dict
    user_id: str
    target_uri: str | None = None
    evidence: list[dict] = field(default_factory=list)
    confidence: float = 1.0
    source_uri: str | None = None
    source_episode_id: str | None = None
    source_session_id: str | None = None
    status: OperationStatus = OperationStatus.CANDIDATE
    operation_id: str = ""
    created_at: str = ""
    schema_version: str = "context_operation_v1"

    def __post_init__(self) -> None:
        if isinstance(self.context_type, str):
            self.context_type = ContextType(self.context_type)
        if isinstance(self.action, str):
            self.action = OperationAction(self.action)
        if isinstance(self.status, str):
            self.status = OperationStatus(self.status)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if not self.operation_id:
            self.operation_id = new_id("op")
        if not self.created_at:
            self.created_at = utc_now()

    def key(self) -> tuple[str, str | None]:
        return (self.context_type.value, self.target_uri)

    def to_dict(self) -> dict:
        return {
            "operation_id": self.operation_id,
            "user_id": self.user_id,
            "context_type": self.context_type.value,
            "action": self.action.value,
            "target_uri": self.target_uri,
            "payload": self.payload,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "source_uri": self.source_uri,
            "source_episode_id": self.source_episode_id,
            "source_session_id": self.source_session_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ContextOperation:
        return cls(
            operation_id=str(payload.get("operation_id", "")),
            user_id=str(payload["user_id"]),
            context_type=ContextType(str(payload["context_type"])),
            action=OperationAction(str(payload["action"])),
            target_uri=payload.get("target_uri"),
            payload=dict(payload.get("payload", {})),
            evidence=list(payload.get("evidence", [])),
            confidence=float(payload.get("confidence", 1.0)),
            source_uri=payload.get("source_uri"),
            source_episode_id=payload.get("source_episode_id"),
            source_session_id=payload.get("source_session_id"),
            status=OperationStatus(str(payload.get("status", OperationStatus.CANDIDATE.value))),
            created_at=str(payload.get("created_at", "")),
            schema_version=str(payload.get("schema_version", "context_operation_v1")),
        )
