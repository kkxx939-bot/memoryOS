"""操作提交里的上下文差异。"""

from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.core.ids import new_id, require_safe_path_segment
from memoryos.core.time import utc_now
from memoryos.operations.model.context_operation import ContextOperation


@dataclass
class ContextDiff:
    user_id: str
    operations: list[ContextOperation] = field(default_factory=list)
    pending_operations: list[ContextOperation] = field(default_factory=list)
    rejected_operations: list[ContextOperation] = field(default_factory=list)
    diff_id: str = ""
    created_at: str = ""
    schema_version: str = "context_diff_v1"

    def __post_init__(self) -> None:
        if not self.diff_id:
            self.diff_id = new_id("diff")
        require_safe_path_segment(self.diff_id, "diff_id")
        if not self.created_at:
            self.created_at = utc_now()

    def to_dict(self) -> dict:
        return {
            "diff_id": self.diff_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "operations": [operation.to_dict() for operation in self.operations],
            "pending_operations": [operation.to_dict() for operation in self.pending_operations],
            "rejected_operations": [operation.to_dict() for operation in self.rejected_operations],
        }
