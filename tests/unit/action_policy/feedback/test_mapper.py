"""ActionPolicy 执行反馈映射测试。"""

import pytest

from policy.action_policy.execution.result import ActionResult
from policy.action_policy.feedback import build_action_feedback


@pytest.mark.parametrize(
    ("status", "feedback_type", "reward"),
    [
        ("success", "execution_success", 1.0),
        ("failed", "execution_failure", -1.0),
        ("blocked", "execution_failure", -1.0),
        ("skipped", "no_execution", 0.0),
    ],
)
def test_action_feedback_is_owned_by_policy(
    status: str,
    feedback_type: str,
    reward: float,
) -> None:
    """执行层只报告状态，奖励语义由 ActionPolicy 统一形成。"""

    result = ActionResult(action="turn_on_ac", status=status, executed=status == "success", reason="test")

    feedback = build_action_feedback(
        result,
        user_id="u1",
        episode_id="episode-1",
        policy_uri="memoryos://user/u1/action_policies/hot/turn_on_ac",
        scene_key="hot",
    )

    assert feedback["feedback_type"] == feedback_type
    assert feedback["reward"] == reward
    assert feedback["action_result"] == result.to_dict()
