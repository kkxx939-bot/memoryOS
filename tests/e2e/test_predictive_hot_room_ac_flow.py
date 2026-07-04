from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.behavior.model.observation import Observation
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.prediction.model.prediction_request import PredictionRequest


class PredictiveHotRoomAcFlowTest(unittest.TestCase):
    def test_hot_room_ac_flow_uses_predictive_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = MemoryOSClient(str(root))
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
            client.source_store.write_object(anchor, content="User comfort anchor for hot home environment.")
            client.index_store.upsert_index(anchor, content="hot home comfort anchor")
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
            client.source_store.write_object(pattern_obj, content="hot room behavior pattern")
            client.index_store.upsert_index(pattern_obj, content="hot room user present turn_on_ac behavior pattern")
            policy = ActionPolicy(
                user_id="u1",
                scene_key=scene_key,
                action="turn_on_air_conditioner",
                memory_anchor_uri=anchor_uri,
                q_value=0.9,
                confidence=0.9,
                reward_score=5.0,
                auto_execute_allowed=True,
                required_resource_uris=["memoryos://resources/devices/ac-living-room"],
                required_skill_uris=["memoryos://skills/smart_home/ac-control"],
                supported_behavior_pattern_uris=[pattern_obj.uri],
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
                client.source_store.write_object(obj, content=json.dumps(item.to_dict()))
                client.index_store.upsert_index(obj, content=f"{item.scene_key} {item.action}")
            resource = ContextObject(uri="memoryos://resources/devices/ac-living-room", context_type=ContextType.RESOURCE, title="living room AC", owner_user_id=None)
            skill = ContextObject(uri="memoryos://skills/smart_home/ac-control", context_type=ContextType.SKILL, title="AC control skill", owner_user_id=None)
            client.source_store.write_object(resource, content="device available")
            client.source_store.write_object(skill, content="skill available")
            client.index_store.upsert_index(resource, content="device available")
            client.index_store.upsert_index(skill, content="skill available")
            for relation_type, target in (
                ("anchored_by", anchor_uri),
                ("supported_by", pattern_obj.uri),
                ("requires_resource", resource.uri),
                ("requires_skill", skill.uri),
            ):
                client.relation_store.add_relation(ContextRelation(source_uri=policy.uri, relation_type=relation_type, target_uri=target, metadata={"owner_user_id": "u1", "tenant_id": "default"}))
            request = PredictionRequest(
                user_id="u1",
                episode_id="ep-hot-room",
                observation=observation,
                available_actions=["turn_on_ac", "turn_on_fan", "ask_user", "do_nothing"],
                token_budget=2000,
            )
            result = client.process_observation(request, [policy, fan_policy], archive_session=True, async_commit=True)
            self.assertIn(result.candidates[0].action, {"turn_on_ac", "turn_on_air_conditioner"})
            slices = result.action_context.packed_context["slices"]
            self.assertTrue(slices["memory_anchor"]["items"])
            self.assertTrue(slices["action_policy"]["items"])
            self.assertTrue(slices["behavior_pattern"]["items"])
            self.assertTrue(slices["resource"]["items"])
            self.assertTrue(slices["skill"]["items"])
            self.assertIn(result.decision.mode, {"execute", "ask_user", "suggest"})
            self.assertEqual(result.memory_operations, [])
            archive_dir = ContextURI.parse("memoryos://user/u1/sessions/history/ep-hot-room").to_source_path(root)
            for filename in ("memory_diff.json", "behavior_diff.json", "action_policy_diff.json", "context_diff.json"):
                payload = json.loads((archive_dir / filename).read_text(encoding="utf-8"))
                self.assertEqual(payload["status"], "committed")


if __name__ == "__main__":
    unittest.main()
