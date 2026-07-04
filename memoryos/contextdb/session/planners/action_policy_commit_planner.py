from __future__ import annotations

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.ids import stable_hash
from memoryos.memory.model.memory import Memory, MemoryKind
from memoryos.memory.update.memory_updater import MemoryUpdater
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class ActionPolicyCommitPlanner:
    def __init__(self) -> None:
        self.memory_updater = MemoryUpdater()

    def plan(self, archive: SessionArchive) -> list[ContextOperation]:
        operations: list[ContextOperation] = []
        for feedback in archive.feedback:
            policy_uri = feedback.get("policy_uri") or feedback.get("action_policy_uri") or self._policy_uri_from_feedback(archive.user_id, feedback)
            reward = float(feedback.get("reward", feedback.get("reward_value", 0.0)) or 0.0)
            explicit_rule = str(feedback.get("explicit_rule", ""))
            if reward >= 0:
                operations.append(
                    ContextOperation(
                        user_id=archive.user_id,
                        context_type=ContextType.ACTION_POLICY,
                        action=OperationAction.REWARD,
                        target_uri=policy_uri,
                        payload={"reward": reward or 0.1, "signal_type": feedback.get("feedback_type", "implicit_positive")},
                        evidence=[{"source": "session_feedback"}],
                        source_session_id=archive.session_id,
                    )
                )
            else:
                operations.append(
                    ContextOperation(
                        user_id=archive.user_id,
                        context_type=ContextType.ACTION_POLICY,
                        action=OperationAction.PENALIZE,
                        target_uri=policy_uri,
                        payload={
                            "penalty": abs(reward),
                            "signal_type": feedback.get("feedback_type", "implicit_negative"),
                            "explicit_rule": explicit_rule,
                        },
                        evidence=[{"source": "session_feedback"}],
                        source_session_id=archive.session_id,
                    )
                )
            if explicit_rule:
                operations.append(self.memory_updater.policy_rule(self._policy_memory(archive.user_id, explicit_rule, policy_uri), evidence=[{"source": "explicit_negative_feedback"}]))
                operations.append(
                    ContextOperation(
                        user_id=archive.user_id,
                        context_type=ContextType.ACTION_POLICY,
                        action=OperationAction.DISABLE,
                        target_uri=policy_uri,
                        payload={"explicit_rule": explicit_rule},
                        evidence=[{"source": "explicit_negative_feedback"}],
                        source_session_id=archive.session_id,
                    )
                )
        return operations

    def _policy_uri_from_feedback(self, user_id: str, feedback: dict) -> str:
        scene_key = str(feedback.get("scene_key", "default"))
        action = str(feedback.get("action", feedback.get("selected_action", "unknown")))
        return f"memoryos://user/{user_id}/action_policies/{scene_key}/{action}"

    def _policy_memory(self, user_id: str, rule: str, policy_uri: str) -> Memory:
        digest = stable_hash([user_id, rule, policy_uri], length=16)
        return Memory(
            uri=f"memoryos://user/{user_id}/memories/policies/{digest}",
            user_id=user_id,
            title=rule[:48] or "policy memory",
            content=rule,
            kind=MemoryKind.POLICY,
            confidence=1.0,
            constrains_policy_uris=[policy_uri],
        )
