"""预测模块里的动作结果。"""

from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.core.ids import new_id
from memoryos.core.time import utc_now


@dataclass(frozen=True)
class ActionResult:
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

    def to_feedback(self, *, user_id: str, episode_id: str, policy_uri: str, scene_key: str) -> dict:
        if self.status == "success":
            reward = 1.0
            feedback_type = "execution_success"
        elif self.status in {"failed", "blocked"}:
            reward = -1.0
            feedback_type = "execution_failure"
        else:
            reward = 0.0
            feedback_type = "no_execution"
        return {
            "user_id": user_id,
            "episode_id": episode_id,
            "policy_uri": policy_uri,
            "scene_key": scene_key,
            "action": self.action,
            "feedback_type": feedback_type,
            "reward": reward,
            "action_result": self.to_dict(),
        }
