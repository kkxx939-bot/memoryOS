"""把执行事实映射为策略学习使用的奖励反馈。"""

from __future__ import annotations

from typing import Any, Protocol


class ActionOutcome(Protocol):
    """反馈映射所需的最小执行结果协议。"""

    @property
    def action(self) -> str: ...

    @property
    def status(self) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...


def build_action_feedback(
    result: ActionOutcome,
    *,
    user_id: str,
    episode_id: str,
    policy_uri: str,
    scene_key: str,
) -> dict[str, Any]:
    """由策略层定义执行状态对应的奖励，而不是让执行层推断。"""

    if result.status == "success":
        reward = 1.0
        feedback_type = "execution_success"
    elif result.status in {"failed", "blocked"}:
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
        "action": result.action,
        "feedback_type": feedback_type,
        "reward": reward,
        "action_result": result.to_dict(),
    }
