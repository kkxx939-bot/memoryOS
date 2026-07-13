"""后台任务里的冷却任务。"""

from __future__ import annotations

from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.behavior.model.observation import Observation
from memoryos.behavior.model.opportunity import OpportunityStats
from memoryos.behavior.update.opportunity_decay import OpportunityAwareDecay
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.source_store import IndexStore, SourceStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.workers.readiness import require_source_store_ready


class CoolingWorker:
    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        committer: OperationCommitter,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.committer = committer
        self.decay = OpportunityAwareDecay()

    def process_behavior_patterns(
        self,
        user_id: str,
        recent_observations: list[Observation],
        limit: int = 100,
    ) -> dict:
        require_source_store_ready(self.source_store)
        query = " ".join(
            dict.fromkeys(
                tag
                for observation in recent_observations
                for tag in [
                    observation.raw_text,
                    observation.location,
                    observation.activity,
                    *observation.signals,
                    *observation.context_tags(),
                ]
                if tag
            )
        )
        hits = self.index_store.search(
            query or "behavior",
            filters={"owner_user_id": user_id, "context_type": ContextType.BEHAVIOR_PATTERN.value},
            limit=limit,
        )
        operations: list[ContextOperation] = []
        results = []
        for hit in hits:
            pattern = self._read_pattern(hit.uri)
            matching_observations = self._matching_observations(pattern, recent_observations)
            state_override = self._explicit_feedback_state(pattern, matching_observations)
            decay_result = self.decay.evaluate(pattern, recent_observations)
            state = state_override or decay_result.opportunity_state
            pattern_ops = self._operations_for_state(user_id, pattern, state, decay_result.q_value_delta)
            operations.extend(pattern_ops)
            results.append(
                {
                    "pattern_uri": pattern.uri,
                    "opportunity_state": state,
                    "reason": decay_result.reason,
                    "recent_opportunity_count": decay_result.recent_opportunity_count,
                    "recent_activation_count": decay_result.recent_activation_count,
                    "recent_missed_count": decay_result.recent_missed_count,
                    "recent_negative_count": decay_result.recent_negative_count,
                    "operation_ids": [operation.operation_id for operation in pattern_ops],
                }
            )
        diff = self.committer.commit(user_id, operations) if operations else None
        return {
            "processed": len(results),
            "operations": [operation.to_dict() for operation in operations],
            "committed_operation_ids": [operation.operation_id for operation in diff.operations] if diff else [],
            "results": results,
        }

    def _operations_for_state(
        self,
        user_id: str,
        pattern: BehaviorPattern,
        state: str,
        q_value_delta: float,
    ) -> list[ContextOperation]:
        if state == "no_opportunity":
            return []
        if state == "opportunity_activated":
            return [
                ContextOperation(
                    user_id=user_id,
                    context_type=ContextType.BEHAVIOR_PATTERN,
                    action=OperationAction.REFRESH_LAYERS,
                    target_uri=pattern.uri,
                    payload={"reason": "opportunity_activated"},
                    evidence=[{"source": "opportunity_aware_decay"}],
                )
            ]
        action = self._top_action(pattern)
        target_policy_uris = self._target_policy_uris(pattern)
        if state == "opportunity_missed":
            return [
                self._policy_operation(
                    user_id,
                    OperationAction.PENALIZE,
                    target_policy_uris,
                    pattern,
                    action,
                    {"penalty": min(0.3, abs(q_value_delta) or 0.08), "signal_type": "implicit_missed_opportunity"},
                )
            ]
        if state == "explicit_negative_rule":
            return [
                self._policy_operation(
                    user_id,
                    OperationAction.DISABLE,
                    target_policy_uris,
                    pattern,
                    action,
                    {"reason": "explicit_negative_rule"},
                )
            ]
        if state == "negative_feedback":
            return [
                self._policy_operation(
                    user_id,
                    OperationAction.PENALIZE,
                    target_policy_uris,
                    pattern,
                    action,
                    {"penalty": max(0.2, abs(q_value_delta) or 0.25), "signal_type": "negative_feedback"},
                )
            ]
        return []

    def _policy_operation(
        self,
        user_id: str,
        action_type: OperationAction,
        target_policy_uris: list[str],
        pattern: BehaviorPattern,
        action: str,
        payload: dict,
    ) -> ContextOperation:
        payload = {**payload, "scene_key": pattern.scene_key, "action": action}
        target_uri = target_policy_uris[0] if target_policy_uris else None
        return ContextOperation(
            user_id=user_id,
            context_type=ContextType.ACTION_POLICY,
            action=action_type,
            target_uri=target_uri,
            payload=payload,
            evidence=[{"source": "opportunity_aware_decay", "pattern_uri": pattern.uri}],
            confidence=0.8,
        )

    def _read_pattern(self, uri: str) -> BehaviorPattern:
        obj = self.source_store.read_object(uri)
        metadata = obj.metadata
        opportunity_payload = metadata.get("opportunity", {})
        opportunity = (
            OpportunityStats(**opportunity_payload) if isinstance(opportunity_payload, dict) else OpportunityStats()
        )
        return BehaviorPattern(
            user_id=str(obj.owner_user_id or metadata.get("user_id", "")),
            scene_key=str(metadata.get("scene_key", "")),
            trigger_conditions=dict(metadata.get("trigger_conditions", {})),
            memory_anchor_uri=str(metadata.get("memory_anchor_uri", "")),
            case_refs=list(metadata.get("case_refs", [])),
            action_distribution=list(metadata.get("action_distribution", [])),
            pattern_id=uri.rsplit("/", 1)[-1],
            opportunity=opportunity,
            hotness=float(obj.hotness),
            confidence=float(obj.behavior_support_hotness or metadata.get("confidence", 0.65)),
            status=str(metadata.get("status", "active")),
            updated_at=obj.updated_at,
        )

    def _explicit_feedback_state(self, pattern: BehaviorPattern, observations: list[Observation]) -> str | None:
        signals = {signal for observation in observations for signal in observation.signals}
        if "explicit_negative_rule" in signals:
            return "explicit_negative_rule"
        if "negative_feedback" in signals or "user_rejected" in signals:
            return "negative_feedback"
        return None

    def _matching_observations(self, pattern: BehaviorPattern, observations: list[Observation]) -> list[Observation]:
        expected = set(str(tag) for tag in pattern.trigger_conditions.get("context_tags", []) if tag)
        if not expected:
            return observations
        return [observation for observation in observations if expected.issubset(set(observation.context_tags()))]

    def _top_action(self, pattern: BehaviorPattern) -> str:
        if not pattern.action_distribution:
            return ""
        return str(max(pattern.action_distribution, key=lambda item: int(item.get("count", 0))).get("action", ""))

    def _target_policy_uris(self, pattern: BehaviorPattern) -> list[str]:
        values = pattern.trigger_conditions.get("related_policy_uris", [])
        return [str(value) for value in values if value]
