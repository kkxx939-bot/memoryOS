from __future__ import annotations

import json
from pathlib import Path

from memoryos.application.learning.intervention_policy_stats import PolicyStats
from memoryos.application.learning.rl_calibrator import ReinforcementPolicyLedger
from memoryos.application.learning.behavior_feedback import BehaviorStats
from memoryos.application.learning.behavior_patterns import BehaviorPatternStore
from memoryos.domain.memory.memory_item import MemoryItem, utc_now
from memoryos.infrastructure.repositories.memory_repository import MemoryStore


class LearningProcessor:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def apply_feedback_event(self, event: dict, episode_result: dict) -> dict:
        payload = dict(event.get("payload", {}))
        user_id = str(event.get("user_id") or payload.get("user_id", ""))
        episode_id = str(event.get("episode_id") or payload.get("episode_id", ""))
        result_path = self._learning_result_path(user_id, str(event.get("event_id", "")))
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["idempotent"] = True
            return result
        feedback = str(payload.get("feedback", ""))
        reward = float(payload.get("reward", 0.0))
        actual_action = payload.get("actual_action")
        action_params = payload.get("action_params") if isinstance(payload.get("action_params"), dict) else {}
        spontaneity = str(payload.get("spontaneity", "unknown"))
        intervention_result = str(payload.get("intervention_result", ""))
        correction = payload.get("correction")
        corrects_memory = bool(payload.get("corrects_memory", False))
        reward_breakdown = dict(payload.get("reward_breakdown", {}))
        behavior_reward = float(reward_breakdown.get("behavior_reward", reward))
        intervention_reward = float(reward_breakdown.get("intervention_reward", reward))

        prediction = episode_result.get("prediction", {})
        predicted_action = str(payload.get("predicted_action") or prediction.get("predicted_action", "unknown"))
        recommended_intervention = str(
            payload.get("recommended_intervention") or prediction.get("recommended_intervention", "unknown")
        )
        created_at = str(payload.get("created_at") or self._feedback_created_at(episode_result))

        behavior_update = BehaviorStats(self._behavior_stats_path(user_id)).record(
            retrieval_query=str(episode_result.get("retrieval_query") or episode_result.get("scene", "")),
            context_tags=[str(tag) for tag in episode_result.get("context_tags", [])],
            predicted_action=predicted_action,
            actual_action=actual_action,
            reward=behavior_reward,
        )
        pattern_update = None
        if actual_action:
            pattern_update = BehaviorPatternStore(self.store.root).record(
                user_id=user_id,
                episode_id=episode_id,
                retrieval_query=str(episode_result.get("retrieval_query") or episode_result.get("scene", "")),
                context_tags=[str(tag) for tag in episode_result.get("context_tags", [])],
                predicted_action=predicted_action,
                actual_action=str(actual_action),
                reward=reward,
                created_at=created_at,
                predicted_candidates=self._predicted_candidate_snapshot(episode_result),
                action_params=action_params,
                scene_features=self._scene_features(episode_result),
                spontaneity=spontaneity,
                intervention=recommended_intervention,
                intervention_result=intervention_result or feedback,
            )
        policy_update = PolicyStats(self._policy_stats_path(user_id)).record(
            predicted_action=predicted_action,
            recommended_intervention=recommended_intervention,
            reward=intervention_reward,
        )
        rl_update = ReinforcementPolicyLedger(self._rl_ledger_path(user_id)).record_feedback(
            episode_id=episode_id,
            predicted_action=predicted_action,
            actual_action=str(actual_action) if actual_action else None,
            reward=behavior_reward,
        )

        result = {
            "event_id": event.get("event_id"),
            "episode_id": episode_id,
            "created_at": created_at,
            "feedback": feedback,
            "reward": reward,
            "reward_breakdown": reward_breakdown,
            "predicted_action": predicted_action,
            "actual_action": actual_action,
            "action_params": action_params,
            "spontaneity": spontaneity,
            "intervention_result": intervention_result,
            "recommended_intervention": recommended_intervention,
            "correction": correction,
            "corrects_memory": corrects_memory,
            "behavior_update": behavior_update,
            "behavior_pattern_update": pattern_update,
            "policy_update": policy_update,
            "rl_update": rl_update,
        }
        if corrects_memory and correction:
            result["memory_event"] = self.store.record_event(
                user_id=user_id,
                event_type="memory_correction",
                text=f"Episode {episode_id} memory correction: {str(correction)}",
                tags=["memory_correction", "feedback"],
            )
            result["memory_corrections"] = self._apply_memory_correction(
                user_id=user_id,
                episode_id=episode_id,
                episode_result=episode_result,
                correction=str(correction),
            )
        if actual_action:
            result["case_memory"] = self._record_case_memory(
                user_id=user_id,
                episode_id=episode_id,
                episode_result=episode_result,
                actual_action=str(actual_action),
                action_params=action_params,
                spontaneity=spontaneity,
                intervention_result=intervention_result or feedback,
                reward=reward,
            )
            result["memory_consolidation"] = self._maybe_promote_behavior_pattern(user_id, pattern_update)
        result["idempotent"] = False
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def _feedback_created_at(self, episode_result: dict) -> str:
        observation = episode_result.get("observation") or {}
        observed_at = str(observation.get("observed_at") or "")
        return observed_at or utc_now()

    def _record_case_memory(
        self,
        user_id: str,
        episode_id: str,
        episode_result: dict,
        actual_action: str,
        action_params: dict,
        spontaneity: str,
        intervention_result: str,
        reward: float,
    ) -> dict:
        scene = str(episode_result.get("retrieval_query") or episode_result.get("scene") or "")
        prediction = episode_result.get("prediction", {})
        predicted_action = str(prediction.get("predicted_action", "unknown"))
        title = f"Case {actual_action} after {predicted_action}"
        text = "\n".join(
            [
                f"Scene: {scene}",
                f"Scene features: {json.dumps(self._scene_features(episode_result), ensure_ascii=False, sort_keys=True)}",
                f"Predicted candidates: {json.dumps(self._predicted_candidate_snapshot(episode_result), ensure_ascii=False, sort_keys=True)}",
                f"Predicted action: {predicted_action}",
                f"Actual action: {actual_action}",
                f"Action params: {json.dumps(action_params, ensure_ascii=False, sort_keys=True)}",
                f"Spontaneity: {spontaneity}",
                f"Intervention: {prediction.get('recommended_intervention', '')}",
                f"Intervention result: {intervention_result}",
                f"Reward: {reward}",
                f"Episode: {episode_id}",
            ]
        )
        item = MemoryItem(
            user_id=user_id,
            memory_type="case",
            title=title,
            text=text,
            tags=["case", f"actual_action:{actual_action}", f"predicted_action:{predicted_action}"],
            source=f"episode:{episode_id}:feedback",
            confidence=max(0.5, min(1.0, (reward + 1.0) / 2.0)),
        )
        path = self.store.add_memory(item)
        return {
            "uri": str(path.relative_to(self.store.root).as_posix()),
            "memory_type": "case",
            "actual_action": actual_action,
            "predicted_action": predicted_action,
        }

    def _scene_features(self, episode_result: dict) -> dict:
        observation = episode_result.get("observation") or {}
        if not isinstance(observation, dict):
            return {}
        environment = observation.get("environment") if isinstance(observation.get("environment"), dict) else {}
        return {
            "location": observation.get("location"),
            "activity": observation.get("activity"),
            "time_bucket": observation.get("time_of_day"),
            "duration_minutes": observation.get("computed_duration_minutes"),
            "thermal_level": observation.get("thermal_level"),
            "signals": observation.get("signals", []),
            "environment": environment,
        }

    def _predicted_candidate_snapshot(self, episode_result: dict) -> list[dict]:
        snapshot = []
        for candidate in episode_result.get("ranked_candidates", [])[:8]:
            snapshot.append(
                {
                    "action": candidate.get("action"),
                    "score": candidate.get("score"),
                    "prior": candidate.get("prior"),
                    "sources": candidate.get("sources", []),
                }
            )
        return snapshot

    def _maybe_promote_behavior_pattern(self, user_id: str, pattern_update: dict | None) -> dict | None:
        if not pattern_update:
            return None
        pattern_uri = str(pattern_update.get("pattern_uri", ""))
        if not pattern_uri:
            return None
        path = self.store.root / pattern_uri
        if not path.exists():
            return None
        pattern = json.loads(path.read_text(encoding="utf-8"))
        if not self._pattern_ready_for_long_term_memory(pattern):
            return {
                "promoted": False,
                "reason": "insufficient repeated evidence",
                "pattern_uri": pattern_uri,
                "sample_count": int(pattern.get("sample_count", 0)),
                "distinct_days": int(pattern.get("distinct_days", 0)),
                "evidence_confidence": float(pattern.get("evidence_confidence", 0.0)),
            }
        action = str(pattern.get("action", ""))
        domain = str(pattern.get("domain", "general"))
        group_id = str(pattern.get("group_id", ""))
        title = f"Behavior pattern: {action} in {domain}"
        text = "\n".join(
            [
                f"When context matches behavior group {group_id}, the user's observed actual action is usually {action}.",
                f"Domain: {domain}",
                f"Samples: {int(pattern.get('sample_count', 0))}",
                f"Distinct days: {int(pattern.get('distinct_days', 0))}",
                f"Average reward: {float(pattern.get('average_reward', 0.0)):.3f}",
                f"Evidence confidence: {float(pattern.get('evidence_confidence', 0.0)):.3f}",
                f"Pattern URI: {pattern_uri}",
            ]
        )
        tags = ["habit", "aggregated", f"action:{action}", f"domain:{domain}", f"promoted_behavior_pattern:{group_id}"]
        existing = self._existing_promoted_habit(user_id, group_id, action)
        if existing:
            update = self.store.update_memory(
                existing["path"],
                user_id=user_id,
                title=title,
                text=text,
                tags=sorted(set([*existing.get("tags", []), *tags])),
                metadata_patch={
                    "confidence": float(pattern.get("evidence_confidence", 0.75)),
                    "evidence_count": int(pattern.get("sample_count", 1)),
                    "positive_count": int(pattern.get("sample_count", 1)),
                    "negative_count": 0,
                    "source": f"behavior_pattern:{pattern_uri}",
                },
            )
            return {"promoted": True, "operation": "update", "uri": update["uri"], "pattern_uri": pattern_uri}
        item = MemoryItem(
            user_id=user_id,
            memory_type="habit",
            title=title,
            text=text,
            tags=tags,
            source=f"behavior_pattern:{pattern_uri}",
            confidence=float(pattern.get("evidence_confidence", 0.75)),
            evidence_count=int(pattern.get("sample_count", 1)),
            positive_count=int(pattern.get("sample_count", 1)),
            negative_count=0,
        )
        created_path = self.store.add_memory(item)
        return {
            "promoted": True,
            "operation": "create",
            "uri": str(created_path.relative_to(self.store.root).as_posix()),
            "pattern_uri": pattern_uri,
        }

    def _pattern_ready_for_long_term_memory(self, pattern: dict) -> bool:
        return (
            int(pattern.get("sample_count", 0)) >= 3
            and int(pattern.get("distinct_days", 0)) >= 2
            and float(pattern.get("average_reward", 0.0)) >= 0.5
            and float(pattern.get("evidence_confidence", 0.0)) >= 0.65
            and float(pattern.get("action_ratio", 1.0)) >= 0.6
        )

    def _existing_promoted_habit(self, user_id: str, group_id: str, action: str) -> dict | None:
        rows = self.store.hybrid_search(f"{group_id} {action}", user_id=user_id, memory_type="habit", limit=8)
        target_tag = f"promoted_behavior_pattern:{group_id}"
        action_tag = f"action:{action}"
        for row in rows:
            tags = {str(tag) for tag in row.get("tags", [])}
            if target_tag in tags and action_tag in tags:
                return row
        return None

    def _apply_memory_correction(
        self,
        user_id: str,
        episode_id: str,
        episode_result: dict,
        correction: str,
    ) -> list[dict]:
        prediction = episode_result.get("prediction", {})
        used_memories = [str(path) for path in prediction.get("used_memories", []) if path]
        updates = []
        for rel_path in used_memories:
            try:
                current = self.store.resolve_memory(rel_path, user_id)
            except FileNotFoundError:
                continue
            corrected_body = (
                str(current.get("content", "")).rstrip()
                + f"\n\n## Correction {utc_now()}\n\n"
                + f"Episode {episode_id}: {correction.strip()}\n"
            )
            negative_count = int(current.get("negative_count", 0)) + 1
            positive_count = int(current.get("positive_count", 1))
            confidence = max(0.1, float(current.get("confidence", 0.7)) - 0.2)
            update = self.store.update_memory(
                rel_path,
                user_id=user_id,
                text=corrected_body,
                metadata_patch={
                    "negative_count": negative_count,
                    "positive_count": positive_count,
                    "confidence": confidence,
                    "source": f"episode:{episode_id}:correction",
                },
            )
            updates.append({"uri": update["uri"], "confidence": confidence, "negative_count": negative_count})
        return updates

    def _policy_stats_path(self, user_id: str) -> Path:
        return self.store.root / "user" / user_id / "policy_stats.json"

    def _behavior_stats_path(self, user_id: str) -> Path:
        return self.store.root / "user" / user_id / "behavior_stats.json"

    def _rl_ledger_path(self, user_id: str) -> Path:
        return self.store.root / "user" / user_id / "rl" / "policy_ledger.json"

    def _learning_result_path(self, user_id: str, event_id: str) -> Path:
        return self.store.root / "user" / user_id / "events" / "learning_results" / f"{event_id or 'unknown'}.json"
