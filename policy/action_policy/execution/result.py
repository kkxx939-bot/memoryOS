"""ActionPolicy 动作执行产生的客观结果。"""

from __future__ import annotations

from dataclasses import dataclass, field

from foundation.clock import utc_now
from foundation.ids import new_id


@dataclass(frozen=True)
class ActionResult:
    """记录工具调用事实，不在执行层计算策略奖励。"""

    action: str
    status: str
    executed: bool
    reason: str
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    resource_uris: list[str] = field(default_factory=list)
    skill_uris: list[str] = field(default_factory=list)
    output: dict = field(default_factory=dict)
    error: str = ""
    trace_id: str = field(default_factory=lambda: new_id("action_result"))
    started_at: str = field(default_factory=utc_now)
    finished_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        """转换为可归档的执行结果。"""

        return {
            "action": self.action,
            "status": self.status,
            "executed": self.executed,
            "reason": self.reason,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "resource_uris": self.resource_uris,
            "skill_uris": self.skill_uris,
            "output": self.output,
            "error": self.error,
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
