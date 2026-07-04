from __future__ import annotations

import tempfile
import unittest

from memoryos.action_policy.model import ActionPolicy, ActionPolicyStatus, PenaltySignal, RewardSignal
from memoryos.action_policy.ranking import ActionPolicyRanker
from memoryos.action_policy.update import ActionPolicyUpdater, FeedbackCommitPlanner
from memoryos.behavior.model import BehaviorCase, BehaviorPattern, Observation, OpportunityStats
from memoryos.behavior.update import BehaviorLifecycleService, OpportunityAwareDecay
from memoryos.contextdb.model import ContextObject, ContextType
from memoryos.contextdb.session import SessionArchive, SessionArchiveStore, SessionCommitService
from memoryos.contextdb.store import FileSystemSourceStore, InMemoryIndexStore, InMemoryQueueStore
from memoryos.memory.model import MemoryAnchor
from memoryos.memory.update.memory_cooling import MemoryCoolingPolicy
from memoryos.operations.model import OperationAction
from memoryos.prediction.model import ActionContext, PredictionLedger, PredictionRequest
from memoryos.prediction.pipeline import PolicyGate, PredictionEngine


class PredictiveContextDatabaseTest(unittest.TestCase):
    def test_behavior_pattern_and_action_policy_require_memory_anchor(self) -> None:
        with self.assertRaises(ValueError):
            BehaviorPattern(
                user_id="gulf",
                scene_key="hot",
                trigger_conditions={},
                memory_anchor_uri="",
                case_refs=[],
                action_distribution=[],
            )
        with self.assertRaises(ValueError):
            ActionPolicy(user_id="gulf", scene_key="hot", action="turn_on_ac", memory_anchor_uri="")

    def test_behavior_lifecycle_creates_anchor_cluster_and_pattern(self) -> None:
        cases = [
            BehaviorCase(user_id="gulf", scene_key="hot", observation={}, user_actual_action="turn_on_ac"),
            BehaviorCase(user_id="gulf", scene_key="hot", observation={}, user_actual_action="turn_on_ac"),
            BehaviorCase(user_id="gulf", scene_key="hot", observation={}, user_actual_action="turn_on_fan"),
        ]
        result = BehaviorLifecycleService().evaluate("gulf", "hot", cases)
        self.assertIsInstance(result.memory_anchor, MemoryAnchor)
        self.assertIsNotNone(result.cluster)
        self.assertIsNotNone(result.pattern)
        assert result.memory_anchor is not None
        assert result.pattern is not None
        self.assertTrue(result.memory_candidate_required)
        self.assertEqual(result.pattern.memory_anchor_uri, result.memory_anchor.uri)

    def test_single_short_behavior_remains_temporary_case_only(self) -> None:
        case = BehaviorCase(user_id="gulf", scene_key="hot", observation={}, user_actual_action="turn_on_ac")
        result = BehaviorLifecycleService().evaluate("gulf", "hot", [case])
        self.assertEqual(result.temporary_cases, [case])
        self.assertIsNone(result.memory_anchor)
        self.assertIsNone(result.cluster)
        self.assertIsNone(result.pattern)

    def test_opportunity_decay_uses_opportunity_not_elapsed_time_only(self) -> None:
        pattern = BehaviorPattern(
            user_id="gulf",
            scene_key="hot",
            trigger_conditions={"context_tags": ["hot_environment"]},
            memory_anchor_uri="memoryos://user/gulf/memories/anchors/hot",
            case_refs=["c1"],
            action_distribution=[{"action": "turn_on_ac", "probability": 1.0}],
            opportunity=OpportunityStats(activation_count=1, missed_opportunity_count=0),
        )
        no_opportunity = OpportunityAwareDecay().evaluate(pattern, [])
        self.assertEqual(no_opportunity.opportunity_state, "no_opportunity")
        hot_obs = Observation(user_id="gulf", location="home", environment={"temperature": 30})
        activated = OpportunityAwareDecay().evaluate(pattern, [hot_obs])
        self.assertEqual(activated.opportunity_state, "opportunity_activated")

    def test_action_policy_reward_penalty_and_disable(self) -> None:
        policy = ActionPolicy(
            user_id="gulf",
            scene_key="hot",
            action="turn_on_ac",
            memory_anchor_uri="memoryos://user/gulf/memories/anchors/hot",
            auto_execute_allowed=True,
        )
        updater = ActionPolicyUpdater()
        updater.reward(policy, RewardSignal(reward=1.0, signal_type="explicit_positive"))
        self.assertGreater(policy.q_value, 0.5)
        updater.penalize(policy, PenaltySignal(penalty=1.0, explicit_rule="以后别自动开空调"))
        self.assertEqual(policy.status, ActionPolicyStatus.DISABLED_AUTO_EXECUTE)
        self.assertFalse(policy.auto_execute_allowed)

    def test_explicit_negative_rule_writes_policy_memory_and_disables_policy(self) -> None:
        policy = ActionPolicy(
            user_id="gulf",
            scene_key="hot",
            action="turn_on_ac",
            memory_anchor_uri="memoryos://user/gulf/memories/anchors/hot",
        )
        ops = FeedbackCommitPlanner().explicit_negative_rule_operations(
            policy,
            PenaltySignal(penalty=1.0, explicit_rule="以后别自动开空调"),
        )
        self.assertEqual([op.action for op in ops], [OperationAction.ADD, OperationAction.DISABLE])
        self.assertEqual(ops[0].context_type, ContextType.MEMORY)
        self.assertEqual(ops[1].context_type, ContextType.ACTION_POLICY)

    def test_memory_with_behavior_support_is_compressed_not_deleted(self) -> None:
        memory = ContextObject(
            uri="memoryos://user/gulf/memories/preferences/temperature",
            context_type=ContextType.MEMORY,
            title="Temperature preference",
            owner_user_id="gulf",
            behavior_support_hotness=0.8,
        )
        op = MemoryCoolingPolicy().cool(memory)
        self.assertEqual(op.action, OperationAction.COMPRESS)
        self.assertFalse(op.payload["delete"])

    def test_policy_gate_allows_auto_execute_only_after_gate(self) -> None:
        policy = ActionPolicy(
            user_id="gulf",
            scene_key="hot",
            action="turn_on_ac",
            memory_anchor_uri="memoryos://user/gulf/memories/anchors/hot",
            auto_execute_allowed=True,
            confidence=0.9,
            q_value=0.9,
        )
        candidate = ActionPolicyRanker().rank([policy])[0]
        action_context = ActionContext(user_id="gulf", candidate_actions=[candidate.action], packed_context={})
        decision = PolicyGate().evaluate(candidate, action_context, action_policy=policy, prediction_confidence=0.8)
        self.assertEqual(decision.mode, "execute")

    def test_prediction_engine_records_ledger_and_does_not_write_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = InMemoryIndexStore()
            anchor_obj = ContextObject(
                uri="memoryos://user/gulf/memories/anchors/hot",
                context_type=ContextType.MEMORY,
                title="hot weather anchor",
                owner_user_id="gulf",
            )
            index.upsert_index(anchor_obj, "hot weather home comfort")
            policy = ActionPolicy(
                user_id="gulf",
                scene_key="hot",
                action="turn_on_ac",
                memory_anchor_uri=anchor_obj.uri,
                auto_execute_allowed=True,
                q_value=0.9,
                confidence=0.9,
            )
            result = PredictionEngine(index, PredictionLedger(tmp)).process(
                PredictionRequest(
                    user_id="gulf",
                    episode_id="ep1",
                    observation={"raw_text": "room is hot", "location": "home", "environment": {"temperature": 30}},
                    available_actions=["turn_on_ac", "ask_user", "do_nothing"],
                    request_id="req1",
                ),
                policies=[policy],
            )
            self.assertEqual(result.memory_operations, [])
            self.assertTrue(result.decision.mode in {"execute", "ask_user"})

    def test_session_commit_two_phase_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = SessionArchive(
                user_id="gulf",
                session_id="s1",
                archive_uri="memoryos://user/gulf/sessions/history/archive_001",
                messages=[{"role": "user", "content": "room is hot"}],
                observations=[{"raw_text": "temperature 30"}],
            )
            queue = InMemoryQueueStore()
            service = SessionCommitService(SessionArchiveStore(tmp), queue, allow_plan_only=True)
            queued = service.sync_archive(archive)
            self.assertEqual(queued.status, "queued")
            done = service.async_commit(archive)
            self.assertTrue(done.done)
            source = FileSystemSourceStore(tmp)
            archive_dir = source._object_dir(archive.archive_uri)
            self.assertTrue((archive_dir / ".done").exists())
            self.assertTrue((archive_dir / "memory_diff.json").exists())
            self.assertTrue(queue.lease("semantic", 1))


if __name__ == "__main__":
    unittest.main()
