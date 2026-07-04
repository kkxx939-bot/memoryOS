from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.behavior.model.observation import Observation
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryQueueStore,
    InMemoryRelationStore,
)
from memoryos.prediction.model.prediction_ledger import PredictionLedger
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.pipeline.prediction_engine import PredictionEngine
from memoryos.prediction.pipeline.predictive_observation_processor import PredictiveObservationProcessor


class PredictiveHotRoomAcFlowTest(unittest.TestCase):
    def test_hot_room_ac_flow_uses_predictive_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            relations = InMemoryRelationStore()
            queue = InMemoryQueueStore()
            session_service = SessionCommitService(SessionArchiveStore(root), queue)
            observation = Observation(
                user_id="u1",
                raw_text="室温 30 度，用户在家",
                location="home",
                activity="resting",
                signals=["user_present"],
                environment={"temperature": 30},
            )
            scene_key = observation.scene_key
            anchor_uri = "memoryos://user/u1/memories/anchors/home_comfort"
            anchor = ContextObject(uri=anchor_uri, context_type=ContextType.MEMORY, title="home comfort anchor", owner_user_id="u1")
            source.write_object(anchor, content="User comfort anchor for hot home environment.")
            index.upsert_index(anchor, content="hot home comfort anchor")
            pattern = BehaviorPattern(
                user_id="u1",
                scene_key=scene_key,
                trigger_conditions={"location": "home", "temperature_gte": 29},
                memory_anchor_uri=anchor_uri,
                case_refs=["case-1", "case-2", "case-3"],
                action_distribution=[{"action": "turn_on_ac", "count": 3}],
                confidence=0.9,
                hotness=0.9,
            )
            pattern_obj = pattern.to_context_object()
            source.write_object(pattern_obj, content="hot room behavior pattern")
            index.upsert_index(pattern_obj, content="hot room user present turn_on_ac behavior pattern")
            policy = ActionPolicy(
                user_id="u1",
                scene_key=scene_key,
                action="turn_on_air_conditioner",
                memory_anchor_uri=anchor_uri,
                q_value=0.9,
                confidence=0.9,
                reward_score=5.0,
                auto_execute_allowed=True,
            )
            fan_policy = ActionPolicy(
                user_id="u1",
                scene_key=scene_key,
                action="turn_on_fan",
                memory_anchor_uri=anchor_uri,
                q_value=0.4,
                confidence=0.5,
            )
            for item in (policy, fan_policy):
                obj = item.to_context_object()
                source.write_object(obj, content=json.dumps(item.to_dict()))
                index.upsert_index(obj, content=f"{item.scene_key} {item.action}")
            resource = ContextObject(uri="memoryos://resources/devices/ac-living-room", context_type=ContextType.RESOURCE, title="living room AC", owner_user_id=None)
            skill = ContextObject(uri="memoryos://skills/smart_home/ac-control", context_type=ContextType.SKILL, title="AC control skill", owner_user_id=None)
            source.write_object(resource, content="device available")
            source.write_object(skill, content="skill available")
            for relation_type, target in (
                ("anchored_by", anchor_uri),
                ("supported_by", pattern_obj.uri),
                ("requires_resource", resource.uri),
                ("requires_skill", skill.uri),
            ):
                relations.add_relation(ContextRelation(source_uri=policy.uri, relation_type=relation_type, target_uri=target))
            engine = PredictionEngine(index, PredictionLedger(root), source_store=source, relation_store=relations)
            processor = PredictiveObservationProcessor(engine, session_commit_service=session_service)
            request = PredictionRequest(
                user_id="u1",
                episode_id="ep-hot-room",
                observation=observation,
                available_actions=["turn_on_ac", "turn_on_fan", "ask_user", "do_nothing"],
                token_budget=2000,
            )
            result = processor.process(request, [policy, fan_policy], archive_session=True)
            self.assertIn(result.candidates[0].action, {"turn_on_ac", "turn_on_air_conditioner"})
            slices = result.action_context.packed_context["slices"]
            self.assertTrue(slices["memory_anchor"]["items"])
            self.assertTrue(slices["action_policy"]["items"])
            self.assertTrue(slices["behavior_pattern"]["items"])
            self.assertIn(result.decision.mode, {"execute", "ask_user", "suggest", "do_nothing", "suppress", "blocked"})
            self.assertEqual(result.memory_operations, [])
            archive = SessionArchive(
                user_id="u1",
                session_id="ep-hot-room",
                archive_uri="memoryos://user/u1/sessions/history/ep-hot-room",
                observations=[result.observation.__dict__],
                predictions=[result.to_dict()],
                feedback=[{"policy_uri": policy.uri, "reward": 0.4, "feedback_type": "implicit_positive"}],
                used_contexts=[{"uri": uri} for uri in result.action_context.source_uris],
            )
            session_service.async_commit(archive)
            archive_dir = ContextURI.parse(archive.archive_uri).to_source_path(root)
            self.assertTrue((archive_dir / "behavior_diff.json").exists())
            self.assertTrue((archive_dir / "action_policy_diff.json").exists())


if __name__ == "__main__":
    unittest.main()
