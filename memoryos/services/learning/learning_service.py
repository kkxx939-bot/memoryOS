from __future__ import annotations

import json
from pathlib import Path

from memoryos.domain.memory.memory_item import MemoryItem, utc_now
from memoryos.ports.repositories.memory_repository import MemoryRepository
from memoryos.services.learning.behavior_case_recorder import BehaviorCaseRecorder
from memoryos.services.learning.behavior_feedback import BehaviorStats
from memoryos.services.learning.intervention_policy_stats import PolicyStats
from memoryos.services.learning.rl_calibrator import ReinforcementPolicyLedger


class LearningProcessor:
    def __init__(self, store: MemoryRepository) -> None:
        self.store = store

    def apply_feedback_event(self, event: dict, episode_result: dict) -> dict:
        payload = dict(event.get("payload", {}))
        user_id = str(event.get("user_id") or payload.get("user_id", ""))
        episode_id = str(event.get("episode_id") or payload.get("episode_id", ""))
        event_id = str(event.get("event_id", ""))
        result_path = self._learning_result_path(user_id, event_id)
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["idempotent"] = True
            return result
        progress = self._begin_learning_event(user_id, event_id, episode_id)
        feedback = str(payload.get("feedback", ""))
        reward = float(payload.get("reward", 0.0))
        actual_action = payload.get("actual_action")
        raw_action_params = payload.get("action_params")
        action_params: dict = raw_action_params if isinstance(raw_action_params, dict) else {}
        spontaneity = str(payload.get("spontaneity", "unknown"))
        intervention_result = str(payload.get("intervention_result", ""))
        correction = payload.get("correction")
        corrects_memory = bool(payload.get("corrects_memory", False))
        reward_breakdown = dict(payload.get("reward_breakdown", {}))
        behavior_reward = float(reward_breakdown.get("behavior_reward", reward))
        intervention_reward = float(reward_breakdown.get("intervention_reward", reward))
        case_reward = max(float(reward), behavior_reward) if actual_action else behavior_reward

        prediction = episode_result.get("prediction", {})
        predicted_action = str(payload.get("predicted_action") or prediction.get("predicted_action", "unknown"))
        recommended_intervention = str(
            payload.get("recommended_intervention") or prediction.get("recommended_intervention", "unknown")
        )
        created_at = str(payload.get("created_at") or self._feedback_created_at(episode_result))

        retrieval_query = str(episode_result.get("retrieval_query") or episode_result.get("scene", ""))
        context_tags = [str(tag) for tag in episode_result.get("context_tags", [])]
        behavior_update = self._module_result(progress, "behavior_stats")
        if behavior_update is None:
            behavior_update = BehaviorStats(self._behavior_stats_path(user_id)).record(
                retrieval_query=retrieval_query,
                context_tags=context_tags,
                predicted_action=predicted_action,
                actual_action=actual_action,
                reward=behavior_reward,
                event_id=event_id,
            )
            progress = self._mark_learning_module(user_id, event_id, progress, "behavior_stats", behavior_update)
        pattern_update = None
        if actual_action:
            pattern_update = self._module_result(progress, "behavior_pattern")
            if pattern_update is None:
                pattern_update = BehaviorCaseRecorder(self.store.root).record_case(
                    user_id=user_id,
                    episode_id=episode_id,
                    retrieval_query=retrieval_query,
                    context_tags=context_tags,
                    predicted_action=predicted_action,
                    actual_action=str(actual_action),
                    reward=case_reward,
                    created_at=created_at,
                    predicted_candidates=self._predicted_candidate_snapshot(episode_result),
                    action_params=action_params,
                    scene_features=self._scene_features(episode_result),
                    spontaneity=spontaneity,
                    intervention=recommended_intervention,
                    intervention_result=intervention_result or feedback,
                    feedback_event_id=event_id,
                    reward_breakdown=reward_breakdown,
                )
                progress = self._mark_learning_module(user_id, event_id, progress, "behavior_pattern", pattern_update)
        policy_update = self._module_result(progress, "policy_stats")
        if policy_update is None:
            policy_update = PolicyStats(self._policy_stats_path(user_id)).record(
                predicted_action=predicted_action,
                recommended_intervention=recommended_intervention,
                reward=intervention_reward,
                event_id=event_id,
            )
            progress = self._mark_learning_module(user_id, event_id, progress, "policy_stats", policy_update)
        rl_update = self._module_result(progress, "rl")
        if rl_update is None:
            rl_update = ReinforcementPolicyLedger(self._rl_ledger_path(user_id)).record_feedback(
                episode_id=episode_id,
                predicted_action=predicted_action,
                actual_action=str(actual_action) if actual_action else None,
                reward=behavior_reward,
                event_id=event_id,
            )
            progress = self._mark_learning_module(user_id, event_id, progress, "rl", rl_update)

        result = {
            "event_id": event_id,
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
            memory_correction = self._module_result(progress, "memory_correction")
            if memory_correction is None:
                memory_correction = {
                    "memory_event": self._record_memory_correction_event(user_id, episode_id, event_id, str(correction)),
                    "memory_corrections": self._apply_memory_correction(
                        user_id=user_id,
                        episode_id=episode_id,
                        episode_result=episode_result,
                        correction=str(correction),
                    ),
                }
                progress = self._mark_learning_module(
                    user_id,
                    event_id,
                    progress,
                    "memory_correction",
                    memory_correction,
                )
            result.update(memory_correction)
        if actual_action:
            case_memory = self._module_result(progress, "case_memory")
            if case_memory is None:
                case_memory = self._record_case_memory(
                    user_id=user_id,
                    episode_id=episode_id,
                    event_id=event_id,
                    episode_result=episode_result,
                    actual_action=str(actual_action),
                    action_params=action_params,
                    spontaneity=spontaneity,
                    intervention_result=intervention_result or feedback,
                    user_reward=reward,
                    behavior_reward=behavior_reward,
                    case_reward=case_reward,
                    reward_breakdown=reward_breakdown,
                )
                progress = self._mark_learning_module(user_id, event_id, progress, "case_memory", case_memory)
            result["case_memory"] = case_memory
            consolidation = self._module_result(progress, "memory_consolidation")
            if consolidation is None:
                consolidation = self._maybe_promote_behavior_pattern(user_id, pattern_update)
                progress = self._mark_learning_module(
                    user_id,
                    event_id,
                    progress,
                    "memory_consolidation",
                    consolidation or {},
                )
            result["memory_consolidation"] = consolidation
        result["idempotent"] = False
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        self._mark_learning_applied(user_id, event_id, progress, result)
        return result

    def _feedback_created_at(self, episode_result: dict) -> str:
        observation = episode_result.get("observation") or {}
        observed_at = str(observation.get("observed_at") or "")
        return observed_at or utc_now()

    def _record_case_memory(
        self,
        user_id: str,
        episode_id: str,
        event_id: str,
        episode_result: dict,
        actual_action: str,
        action_params: dict,
        spontaneity: str,
        intervention_result: str,
        user_reward: float,
        behavior_reward: float,
        case_reward: float,
        reward_breakdown: dict,
    ) -> dict:
        existing = self._existing_case_memory(user_id, event_id)
        if existing:
            return existing
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
                f"User reward: {user_reward}",
                f"Case reward: {case_reward}",
                f"Reward breakdown: {json.dumps(reward_breakdown, ensure_ascii=False, sort_keys=True)}",
                f"Feedback event: {event_id}",
                f"Episode: {episode_id}",
            ]
        )
        confidence = max(0.35, min(1.0, (case_reward + 1.0) / 2.0))
        item = MemoryItem(
            user_id=user_id,
            memory_type="case",
            title=title,
            text=text,
            tags=[
                "case",
                f"actual_action:{actual_action}",
                f"predicted_action:{predicted_action}",
                f"feedback_event:{event_id}",
            ],
            source=f"episode:{episode_id}:feedback",
            confidence=confidence,
        )
        path = self.store.add_memory(item)
        return {
            "uri": str(path.relative_to(self.store.root).as_posix()),
            "memory_type": "case",
            "actual_action": actual_action,
            "predicted_action": predicted_action,
            "feedback_event_id": event_id,
            "confidence": confidence,
        }

    def _existing_case_memory(self, user_id: str, event_id: str) -> dict | None:
        if not event_id:
            return None
        rows = self.store.hybrid_search(event_id, user_id=user_id, memory_type="case", limit=8)
        target = f"feedback_event:{event_id}"
        for row in rows:
            if target in {str(tag) for tag in row.get("tags", [])}:
                return {
                    "uri": str(row.get("path", "")),
                    "memory_type": "case",
                    "actual_action": self._tag_value(row, "actual_action"),
                    "predicted_action": self._tag_value(row, "predicted_action"),
                    "feedback_event_id": event_id,
                    "idempotent": True,
                }
        return None

    def _tag_value(self, row: dict, prefix: str) -> str:
        marker = f"{prefix}:"
        for tag in row.get("tags", []):
            value = str(tag)
            if value.startswith(marker):
                return value.split(":", 1)[1]
        return ""

    def _scene_features(self, episode_result: dict) -> dict:
        if isinstance(episode_result.get("scene_features"), dict) and episode_result.get("scene_features"):
            return dict(episode_result["scene_features"])
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

    def _record_memory_correction_event(self, user_id: str, episode_id: str, event_id: str, correction: str) -> dict:
        existing = self.store.hybrid_search(event_id, user_id=user_id, memory_type="event", limit=8)
        target_tag = f"feedback_event:{event_id}"
        for row in existing:
            if target_tag in {str(tag) for tag in row.get("tags", [])}:
                return {"uri": str(row.get("path", "")), "idempotent": True}
        return self.store.record_event(
            user_id=user_id,
            event_type="memory_correction",
            text=f"Episode {episode_id} memory correction for feedback event {event_id}: {correction.strip()}",
            tags=["memory_correction", "feedback", target_tag],
        )

    def _policy_stats_path(self, user_id: str) -> Path:
        return self.store.root / "user" / user_id / "policy_stats.json"

    def _behavior_stats_path(self, user_id: str) -> Path:
        return self.store.root / "user" / user_id / "behavior_stats.json"

    def _rl_ledger_path(self, user_id: str) -> Path:
        return self.store.root / "user" / user_id / "rl" / "policy_ledger.json"

    def _learning_result_path(self, user_id: str, event_id: str) -> Path:
        return self.store.root / "user" / user_id / "events" / "learning_results" / f"{event_id or 'unknown'}.json"

    def _learning_progress_path(self, user_id: str, event_id: str) -> Path:
        return self.store.root / "user" / user_id / "events" / "learning_progress" / f"{event_id or 'unknown'}.json"

    def _begin_learning_event(self, user_id: str, event_id: str, episode_id: str) -> dict:
        path = self._learning_progress_path(user_id, event_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        progress = {
            "event_id": event_id,
            "episode_id": episode_id,
            "status": "processing",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "completed_modules": [],
            "module_results": {},
        }
        self._write_json(path, progress)
        return progress

    def _module_result(self, progress: dict, module_name: str) -> dict | None:
        if module_name not in progress.get("completed_modules", []):
            return None
        result = progress.get("module_results", {}).get(module_name)
        return dict(result) if isinstance(result, dict) else {}

    def _mark_learning_module(
        self,
        user_id: str,
        event_id: str,
        progress: dict,
        module_name: str,
        result: dict,
    ) -> dict:
        updated = dict(progress)
        completed = list(updated.get("completed_modules", []))
        if module_name not in completed:
            completed.append(module_name)
        updated["completed_modules"] = completed
        module_results = dict(updated.get("module_results", {}))
        module_results[module_name] = result
        updated["module_results"] = module_results
        updated["updated_at"] = utc_now()
        self._write_json(self._learning_progress_path(user_id, event_id), updated)
        return updated

    def _mark_learning_applied(self, user_id: str, event_id: str, progress: dict, result: dict) -> None:
        updated = dict(progress)
        updated["status"] = "applied"
        updated["applied_at"] = utc_now()
        updated["updated_at"] = updated["applied_at"]
        updated["result_path"] = str(self._learning_result_path(user_id, event_id).relative_to(self.store.root).as_posix())
        updated["result_summary"] = {
            "episode_id": result.get("episode_id"),
            "actual_action": result.get("actual_action"),
            "behavior_reward": result.get("reward_breakdown", {}).get("behavior_reward"),
            "intervention_reward": result.get("reward_breakdown", {}).get("intervention_reward"),
        }
        self._write_json(self._learning_progress_path(user_id, event_id), updated)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)
