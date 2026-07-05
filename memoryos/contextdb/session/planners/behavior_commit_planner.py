from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from memoryos.behavior.model.behavior_case import BehaviorCase
from memoryos.behavior.model.behavior_pattern import BehaviorCluster, BehaviorPattern
from memoryos.behavior.update.behavior_case_writer import BehaviorCaseWriter
from memoryos.behavior.update.behavior_cluster_updater import BehaviorClusterUpdater
from memoryos.behavior.update.behavior_pattern_updater import BehaviorPatternUpdater
from memoryos.behavior.update.behavior_window import BehaviorWindowEvaluator
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.source_store import IndexStore, SourceStore
from memoryos.memory.model.memory import MemoryAnchor
from memoryos.memory.service.memory_updater import MemoryUpdater
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class BehaviorCommitPlanner:
    def __init__(self, index_store: IndexStore | None = None, source_store: SourceStore | None = None) -> None:
        self.index_store = index_store
        self.source_store = source_store
        self.case_writer = BehaviorCaseWriter()
        self.cluster_updater = BehaviorClusterUpdater()
        self.pattern_updater = BehaviorPatternUpdater()
        self.memory_updater = MemoryUpdater()
        self.window_evaluator = BehaviorWindowEvaluator()

    def plan(self, archive: SessionArchive) -> list[ContextOperation]:
        operations: list[ContextOperation] = []
        cases_by_scene: dict[str, list[BehaviorCase]] = {}
        for observation in archive.observations:
            scene_key = str(observation.get("scene_key", observation.get("scene", "default")))
            prediction = self._prediction_for_scene(archive, scene_key)
            candidates = list(prediction.get("candidates", [])) if isinstance(prediction, dict) else []
            selected_action = self._selected_action(prediction)
            feedback = self._feedback_for_observation(archive.feedback, observation, scene_key)
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
            current_case_refs = [f"memoryos://user/{archive.user_id}/behavior/cases/{case.scene_key}/{case.case_id}" for case in cases]
            history_records = self._history_records(archive.user_id, scene_key)
            decision = self.window_evaluator.evaluate(scene_key, cases, history_records)
            case_refs = decision.similar_refs_30d or [*current_case_refs]
            if decision.create_cluster:
                operations.append(self.memory_updater.add_memory(self._anchor(archive.user_id, scene_key, case_refs), evidence=[{"source": "behavior_cluster", "case_refs": case_refs}]))
                cluster = BehaviorCluster(user_id=archive.user_id, scene_key=scene_key, memory_anchor_uri=anchor_uri, case_refs=decision.similar_refs_3d)
                operations.append(self.cluster_updater.add_cluster(cluster))
            if decision.create_pattern:
                actions = Counter(case.selected_action or case.executed_action or case.user_actual_action or "unknown" for case in cases)
                pattern = BehaviorPattern(
                    user_id=archive.user_id,
                    scene_key=scene_key,
                    trigger_conditions={"scene_key": scene_key, "context_tags": list(decision.similarity_key)},
                    memory_anchor_uri=anchor_uri,
                    case_refs=case_refs,
                    action_distribution=[{"action": action, "count": count} for action, count in actions.items()],
                    hotness=0.65,
                    confidence=0.72,
                )
                operations.append(self.pattern_updater.add_pattern(pattern))
            if (decision.archive_stale_single or len(case_refs) == 1) and any(self._observation_age_days(item) > 3 for item in archive.observations):
                operations.append(
                    ContextOperation(
                        user_id=archive.user_id,
                        context_type=ContextType.BEHAVIOR_CASE,
                        action=OperationAction.ARCHIVE,
                        target_uri=case_refs[0],
                        payload={"reason": "single_behavior_without_repeat"},
                        evidence=[{"source": "behavior_lifecycle"}],
                        source_session_id=archive.session_id,
                    )
                )
        return operations

    def _feedback_for_observation(self, feedback_items: list[dict], observation: dict, scene_key: str) -> dict:
        if not feedback_items:
            return {}
        for key in ("episode_id", "request_id"):
            value = observation.get(key)
            if value:
                for item in feedback_items:
                    if str(item.get(key, "")) == str(value):
                        return item
        for item in feedback_items:
            if str(item.get("scene_key", "")) == scene_key:
                return item
        if len(feedback_items) == 1:
            return feedback_items[0]
        return {}

    def _history_records(self, user_id: str, scene_key: str) -> list[dict]:
        if self.index_store is None:
            return []
        records: list[dict] = []
        seen: set[str] = set()
        for context_type in (ContextType.BEHAVIOR_CASE, ContextType.BEHAVIOR_CLUSTER, ContextType.BEHAVIOR_PATTERN):
            hits = self.index_store.search(scene_key, filters={"owner_user_id": user_id, "context_type": context_type.value}, limit=20)
            for hit in hits:
                if hit.uri in seen:
                    continue
                seen.add(hit.uri)
                metadata = dict(hit.metadata)
                if self.source_store is not None:
                    try:
                        metadata = self.source_store.read_object(hit.uri).metadata
                    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                        pass
                records.append(self.window_evaluator.historical_record(hit.uri, metadata))
        return records

    def _anchor(self, user_id: str, scene_key: str, case_refs: list[str]) -> MemoryAnchor:
        anchor_key = f"{scene_key}_anchor"
        return MemoryAnchor(
            uri=f"memoryos://user/{user_id}/memories/anchors/{anchor_key}",
            user_id=user_id,
            title=f"{scene_key} behavior anchor",
            content=f"Recurring behavior theme for {scene_key}.",
            anchor_key=anchor_key,
            supporting_behavior_uris=case_refs,
        )

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

    def _observation_age_days(self, observation: dict) -> int:
        value = str(observation.get("observed_at") or observation.get("created_at") or "")
        if value:
            try:
                observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                observed = None
            if observed is not None:
                if observed.tzinfo is None:
                    observed = observed.replace(tzinfo=timezone.utc)
                return max(0, (datetime.now(timezone.utc) - observed).days)
        return int(observation.get("older_than_days", 0) or 0)
