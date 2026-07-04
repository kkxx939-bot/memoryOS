from __future__ import annotations

from collections import Counter

from memoryos.behavior.model.behavior_case import BehaviorCase
from memoryos.behavior.model.behavior_pattern import BehaviorCluster, BehaviorPattern
from memoryos.behavior.update.behavior_case_writer import BehaviorCaseWriter
from memoryos.behavior.update.behavior_cluster_updater import BehaviorClusterUpdater
from memoryos.behavior.update.behavior_pattern_updater import BehaviorPatternUpdater
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.operations.model.context_operation import ContextOperation


class BehaviorCommitPlanner:
    def __init__(self) -> None:
        self.case_writer = BehaviorCaseWriter()
        self.cluster_updater = BehaviorClusterUpdater()
        self.pattern_updater = BehaviorPatternUpdater()

    def plan(self, archive: SessionArchive) -> list[ContextOperation]:
        operations: list[ContextOperation] = []
        cases_by_scene: dict[str, list[BehaviorCase]] = {}
        feedback_by_episode = {str(item.get("episode_id", item.get("request_id", ""))): item for item in archive.feedback}
        for observation in archive.observations:
            scene_key = str(observation.get("scene_key", observation.get("scene", "default")))
            prediction = self._prediction_for_scene(archive, scene_key)
            candidates = list(prediction.get("candidates", [])) if isinstance(prediction, dict) else []
            selected_action = self._selected_action(prediction)
            feedback = feedback_by_episode.get(str(observation.get("episode_id", "")), {})
            case = BehaviorCase(
                user_id=archive.user_id,
                scene_key=scene_key,
                observation=observation,
                predicted_candidates=candidates,
                selected_action=selected_action,
                executed_action=feedback.get("executed_action"),
                user_actual_action=feedback.get("actual_action"),
                feedback_type=str(feedback.get("feedback_type", "unknown")),
                reward=float(feedback.get("reward", feedback.get("reward_value", 0.0)) or 0.0),
                related_policy_uris=[str(item.get("policy_uri")) for item in candidates if isinstance(item, dict) and item.get("policy_uri")],
            )
            cases_by_scene.setdefault(scene_key, []).append(case)
            operations.append(self.case_writer.add_case(case))

        for scene_key, cases in cases_by_scene.items():
            anchor_uri = f"memoryos://user/{archive.user_id}/memories/anchors/{scene_key}_anchor"
            case_refs = [f"memoryos://user/{archive.user_id}/behavior/cases/{case.scene_key}/{case.case_id}" for case in cases]
            if len(cases) >= 2:
                cluster = BehaviorCluster(user_id=archive.user_id, scene_key=scene_key, memory_anchor_uri=anchor_uri, case_refs=case_refs)
                operations.append(self.cluster_updater.add_cluster(cluster))
            if len(cases) >= 3:
                actions = Counter(case.selected_action or case.executed_action or case.user_actual_action or "unknown" for case in cases)
                pattern = BehaviorPattern(
                    user_id=archive.user_id,
                    scene_key=scene_key,
                    trigger_conditions={"scene_key": scene_key},
                    memory_anchor_uri=anchor_uri,
                    case_refs=case_refs,
                    action_distribution=[{"action": action, "count": count} for action, count in actions.items()],
                    hotness=0.65,
                    confidence=0.72,
                )
                operations.append(self.pattern_updater.add_pattern(pattern))
        return operations

    def _prediction_for_scene(self, archive: SessionArchive, scene_key: str) -> dict:
        for prediction in archive.predictions:
            observation = prediction.get("observation", {}) if isinstance(prediction, dict) else {}
            if observation.get("scene_key") == scene_key:
                return prediction
        return archive.predictions[0] if archive.predictions else {}

    def _selected_action(self, prediction: dict) -> str | None:
        decision = prediction.get("decision", {}) if isinstance(prediction, dict) else {}
        action = decision.get("action")
        if action:
            return str(action)
        candidates = prediction.get("candidates", []) if isinstance(prediction, dict) else []
        if candidates and isinstance(candidates[0], dict):
            return str(candidates[0].get("action", ""))
        return None
