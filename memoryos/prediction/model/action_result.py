from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActionResult:
    action: str
    status: str
    executed: bool
    reason: str
    tool_name: str = ""
    resource_uris: list[str] = field(default_factory=list)
    skill_uris: list[str] = field(default_factory=list)
    output: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "status": self.status,
            "executed": self.executed,
            "reason": self.reason,
            "tool_name": self.tool_name,
            "resource_uris": self.resource_uris,
            "skill_uris": self.skill_uris,
            "output": self.output,
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
