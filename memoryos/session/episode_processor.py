from __future__ import annotations

import json
from pathlib import Path

from .memory.models import MemoryItem, utc_now
from .memory.extractor import MemoryOperation, RuleBasedExtractor
from ..predict.candidates import Candidate
from ..predict.interventions import InterventionDecision, InterventionSelector
from ..predict.policy_stats import PolicyStats
from ..predict.predictors import Prediction, RuleBasedPredictor
from ..predict.rl import PolicyState, ReinforcementPolicyLedger
from ..retrieve.behavior_feedback import BehaviorStats
from ..retrieve.behavior_patterns import BehaviorPatternStore
from ..retrieve.orchestrator import RetrievalOrchestrator
from ..observe.context import ObservationContext
from ..storage.memory_store import MemoryStore
from ..storage.paths import validate_identifier
from .memory.update_service import MemoryUpdateContext, MemoryUpdateService


class EpisodeProcessor:
    def __init__(
        self,
        store: MemoryStore,
        extractor=None,
        predictor=None,
    ) -> None:
        self.store = store
        self.extractor = extractor or RuleBasedExtractor()
        self.predictor = predictor or RuleBasedPredictor()
        self.intervention_selector = InterventionSelector()
        self.memory_updates = MemoryUpdateService(store)

    def process(
        self,
        user_id: str,
        episode_id: str,
        scene: str | None = None,
        observation: ObservationContext | dict | None = None,
        messages: list[dict[str, str]] | None = None,
        available_actions: list[str] | None = None,
        retrieval_limit: int = 8,
        memory_write_timing: str | None = None,
        episode_log_timing: str = "before_prediction",
        memory_commit_timing: str = "after_feedback",
    ) -> dict:
        validate_identifier(user_id, "user_id")
        validate_identifier(episode_id, "episode_id")
        self.store.init(user_id)
        if episode_log_timing != "before_prediction":
            raise ValueError("episode_log_timing must be before_prediction")
        if memory_commit_timing not in {"after_feedback", "explicit_or_after_feedback", "before_prediction", "deferred"}:
            raise ValueError(
                "memory_commit_timing must be after_feedback, explicit_or_after_feedback, before_prediction, or deferred"
            )
        if memory_write_timing is not None:
            memory_commit_timing = self._legacy_memory_commit_timing(memory_write_timing)
        observation_context = self._observation_context(observation)
        if observation_context is None and scene is None:
            raise ValueError("scene or observation is required")
        scene_text = observation_context.to_scene_text() if observation_context else str(scene)
        retrieval_query = observation_context.to_retrieval_query() if observation_context else scene_text
        context_tags = observation_context.context_tags() if observation_context else []
        available_actions = available_actions or ["ask_user", "do_nothing"]
        input_messages = messages or [{"role": "observation", "text": scene_text, "created_at": utc_now()}]
        memory_operations = self.extractor.extract(input_messages)
        explicit_memory_operations = self._explicit_memory_operations(memory_operations, input_messages)
        explicit_operation_ids = {id(operation) for operation in explicit_memory_operations}
        draft_memory_operations = [
            operation for operation in memory_operations if id(operation) not in explicit_operation_ids
        ]

        self._write_episode_file(
            user_id,
            episode_id,
            "episode_log.json",
            {
                "episode_id": episode_id,
                "status": "pending",
                "created_at": utc_now(),
                "scene": scene_text,
                "observation": observation_context.to_dict() if observation_context else None,
                "retrieval_query": retrieval_query,
                "context_tags": context_tags,
                "messages": input_messages,
            },
        )

        if memory_commit_timing == "before_prediction":
            memory_diff = self._apply_memory_operations(user_id, episode_id, memory_operations, observation_context)
        elif memory_commit_timing in {"after_feedback", "explicit_or_after_feedback"} and explicit_memory_operations:
            memory_diff = self._apply_memory_operations(user_id, episode_id, explicit_memory_operations, observation_context)
        else:
            memory_diff = self._empty_memory_diff(episode_id)

        retrieval = RetrievalOrchestrator(self.store, self._behavior_stats_path(user_id)).retrieve(
            user_id=user_id,
            query=retrieval_query,
            context_tags=context_tags,
            retrieval_limit=retrieval_limit,
        )
        memories = retrieval.memories_for_prediction()
        policy_stats = PolicyStats(self._policy_stats_path(user_id)).load()
        rl_ledger = ReinforcementPolicyLedger(self._rl_ledger_path(user_id))
        rl_state = rl_ledger.build_state(
            scene=scene_text,
            context_tags=context_tags,
            memories=memories,
            behavior_patterns=retrieval.behavior_patterns,
        )
        action_universe = self._action_universe(available_actions, memories, retrieval.behavior_patterns, retrieval.behavior_distribution)
        rl_action_scores = rl_ledger.action_scores(rl_state, action_universe)
        ranked_candidates = self._rank_candidates(
            scene_text,
            memories,
            available_actions,
            policy_stats,
            retrieval.behavior_patterns,
            retrieval.behavior_distribution,
            rl_action_scores,
        )
        intervention = self.intervention_selector.select(
            ranked_candidates[0] if ranked_candidates else None,
            available_actions,
            policy_stats,
        )
        prediction = self._prediction_from_candidates(ranked_candidates, intervention)
        rl_prediction = rl_ledger.record_prediction(
            user_id=user_id,
            episode_id=episode_id,
            state=rl_state,
            candidates=ranked_candidates,
            selected_action=prediction.predicted_action,
            intervention_action=intervention.action,
            action_scores=rl_action_scores,
        )

        pending_memory_operations = []
        if memory_commit_timing == "deferred":
            pending_source_operations = memory_operations
        else:
            pending_source_operations = draft_memory_operations
        if memory_commit_timing in {"after_feedback", "explicit_or_after_feedback", "deferred"}:
            pending_memory_operations = [
                self.memory_updates.operation_record(operation)
                for operation in pending_source_operations
            ]

        result = {
            "episode_id": episode_id,
            "episode_status": "predicted",
            "processed_at": utc_now(),
            "scene": scene_text,
            "observation": observation_context.to_dict() if observation_context else None,
            "retrieval_query": retrieval_query,
            "context_tags": context_tags,
            "episode_log_timing": episode_log_timing,
            "memory_commit_timing": memory_commit_timing,
            "memory_write_timing": memory_write_timing,
            "memory_diff": memory_diff,
            "pending_memory_operations": pending_memory_operations,
            "retrieval": retrieval.to_dict(),
            "memory_context": retrieval.memory_context.to_dict(),
            "architecture_layers": self._architecture_layers(
                memories=memories,
                retrieval=retrieval,
                rl_state=rl_state,
                intervention=intervention,
            ),
            "rl_state": rl_state.to_dict(),
            "rl_prediction": rl_prediction,
            "ranked_candidates": [candidate.to_dict() for candidate in ranked_candidates],
            "behavior_patterns": retrieval.behavior_patterns,
            "behavior_distribution": retrieval.behavior_distribution,
            "intervention": intervention.to_dict(),
            "prediction": prediction.to_dict(),
            "retrieved_memories": [
                {
                    "id": memory["id"],
                    "path": memory["path"],
                    "type": memory["type"],
                    "title": memory["title"],
                    "score": memory.get("score"),
                }
                for memory in memories
            ],
        }
        self._write_episode_file(user_id, episode_id, "episode_result.json", result)
        if pending_memory_operations:
            self._write_episode_file(
                user_id,
                episode_id,
                "pending_memory_operations.json",
                {"episode_id": episode_id, "operations": pending_memory_operations},
            )
        return result

    def _legacy_memory_commit_timing(self, memory_write_timing: str) -> str:
        mapping = {
            "before_prediction": "before_prediction",
            "after_prediction": "explicit_or_after_feedback",
            "deferred": "deferred",
        }
        if memory_write_timing not in mapping:
            raise ValueError("memory_write_timing must be before_prediction, after_prediction, or deferred")
        return mapping[memory_write_timing]

    def _explicit_memory_operations(
        self,
        operations: list[MemoryOperation],
        messages: list[dict[str, str]],
    ) -> list[MemoryOperation]:
        if self._messages_contain_explicit_memory_marker(messages):
            return operations
        explicit = []
        for operation in operations:
            tags = {str(tag) for tag in operation.tags}
            if "explicit_user_intent" in tags or "user_confirmed" in tags:
                explicit.append(operation)
        return explicit

    def _messages_contain_explicit_memory_marker(self, messages: list[dict[str, str]]) -> bool:
        markers = getattr(self.extractor, "markers", ("记住：", "记住:", "remember:", "Remember:"))
        for message in messages:
            text = str(message.get("text", ""))
            if any(marker in text for marker in markers):
                return True
        return False

    def _observation_context(self, observation: ObservationContext | dict | None) -> ObservationContext | None:
        if observation is None:
            return None
        if isinstance(observation, ObservationContext):
            return observation
        allowed = {
            "raw_text",
            "location",
            "activity",
            "started_at",
            "observed_at",
            "duration_minutes",
            "signals",
            "environment",
        }
        return ObservationContext(**{key: value for key, value in observation.items() if key in allowed})

    def _rank_candidates(
        self,
        scene: str,
        memories: list[dict],
        available_actions: list[str],
        policy_stats: dict,
        behavior_patterns: list[dict],
        behavior_distribution: list[dict],
        rl_action_scores: dict[str, float],
    ) -> list[Candidate]:
        if hasattr(self.predictor, "rank"):
            return self.predictor.rank(
                scene,
                memories,
                available_actions,
                policy_stats,
                behavior_patterns=behavior_patterns,
                behavior_distribution=behavior_distribution,
                rl_action_scores=rl_action_scores,
            )
        prediction = self.predictor.predict(scene, memories, available_actions)
        return [
            Candidate(
                action=prediction.predicted_action,
                need=prediction.predicted_need,
                prior=prediction.confidence,
                score=prediction.confidence,
                reason=prediction.reason,
                used_memories=prediction.used_memories,
            )
        ]

    def _prediction_from_candidates(self, candidates: list[Candidate], intervention: InterventionDecision) -> Prediction:
        if not candidates:
            return Prediction(
                predicted_action="unknown",
                predicted_need="unknown",
                confidence=0.0,
                recommended_intervention=intervention.action,
                reason="No candidates generated.",
            )
        top = candidates[0]
        return Prediction(
            predicted_action=top.action,
            predicted_need=top.need,
            confidence=max(0.0, min(1.0, top.score)),
            recommended_intervention=intervention.action,
            reason=top.reason,
            used_memories=top.used_memories,
        )

    def commit_pending_memory(self, user_id: str, episode_id: str) -> dict:
        validate_identifier(user_id, "user_id")
        validate_identifier(episode_id, "episode_id")
        path = self._episode_dir(user_id, episode_id) / "pending_memory_operations.json"
        if not path.exists():
            return self._empty_memory_diff(episode_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        operations = [
            MemoryOperation(
                action=raw["action"],
                memory_type=raw["memory_type"],
                title=raw["title"],
                text=raw["text"],
                tags=raw["tags"],
                confidence=float(raw["confidence"]),
                target=raw.get("target"),
                rationale=raw.get("rationale", ""),
                page_id=raw.get("page_id"),
                links=raw.get("links", []),
            )
            for raw in payload.get("operations", [])
        ]
        episode_result = self._read_episode_result(user_id, episode_id)
        observation_context = self._observation_context(episode_result.get("observation"))
        diff = self._apply_memory_operations(user_id, episode_id, operations, observation_context)
        path.unlink()
        return diff

    def record_feedback(
        self,
        user_id: str,
        episode_id: str,
        feedback: str,
        reward: float,
        actual_action: str | None = None,
        action_params: dict | None = None,
        spontaneity: str = "unknown",
        intervention_result: str = "",
        correction: str | None = None,
        corrects_memory: bool = False,
    ) -> dict:
        validate_identifier(user_id, "user_id")
        validate_identifier(episode_id, "episode_id")
        self.store.init(user_id)
        reward = max(-1.0, min(1.0, float(reward)))
        episode_result = self._read_episode_result(user_id, episode_id)
        prediction = episode_result.get("prediction", {})
        predicted_action = str(prediction.get("predicted_action", "unknown"))
        recommended_intervention = str(prediction.get("recommended_intervention", "unknown"))
        created_at = self._feedback_created_at(episode_result)
        behavior_update = BehaviorStats(self._behavior_stats_path(user_id)).record(
            retrieval_query=str(episode_result.get("retrieval_query") or episode_result.get("scene", "")),
            context_tags=[str(tag) for tag in episode_result.get("context_tags", [])],
            predicted_action=predicted_action,
            actual_action=actual_action,
            reward=reward,
        )
        pattern_update = None
        if actual_action:
            pattern_update = BehaviorPatternStore(self.store.root).record(
                user_id=user_id,
                episode_id=episode_id,
                retrieval_query=str(episode_result.get("retrieval_query") or episode_result.get("scene", "")),
                context_tags=[str(tag) for tag in episode_result.get("context_tags", [])],
                predicted_action=predicted_action,
                actual_action=actual_action,
                reward=reward,
                created_at=created_at,
                predicted_candidates=[
                    {
                        "action": candidate.get("action"),
                        "score": candidate.get("score"),
                        "prior": candidate.get("prior"),
                        "sources": candidate.get("sources", []),
                    }
                    for candidate in episode_result.get("ranked_candidates", [])
                ],
                action_params=action_params or {},
                scene_features=self._scene_features(episode_result),
                spontaneity=spontaneity,
                intervention=recommended_intervention,
                intervention_result=intervention_result or feedback,
            )
        policy_update = PolicyStats(self._policy_stats_path(user_id)).record(
            predicted_action=predicted_action,
            recommended_intervention=recommended_intervention,
            reward=reward,
        )
        rl_update = ReinforcementPolicyLedger(self._rl_ledger_path(user_id)).record_feedback(
            episode_id=episode_id,
            predicted_action=predicted_action,
            actual_action=actual_action,
            reward=reward,
        )
        feedback_record = {
            "episode_id": episode_id,
            "created_at": created_at,
            "feedback": feedback,
            "reward": reward,
            "predicted_action": predicted_action,
            "actual_action": actual_action,
            "action_params": action_params or {},
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
        consolidation = None
        if corrects_memory and correction:
            feedback_record["memory_event"] = self.store.record_event(
                user_id=user_id,
                event_type="memory_correction",
                text=f"Episode {episode_id} memory correction: {correction}",
                tags=["memory_correction", "feedback"],
            )
            feedback_record["memory_corrections"] = self._apply_memory_correction(
                user_id=user_id,
                episode_id=episode_id,
                episode_result=episode_result,
                correction=correction,
            )
        if actual_action:
            feedback_record["case_memory"] = self._record_case_memory(
                user_id=user_id,
                episode_id=episode_id,
                episode_result=episode_result,
                actual_action=actual_action,
                action_params=action_params or {},
                spontaneity=spontaneity,
                intervention_result=intervention_result or feedback,
                reward=reward,
            )
            consolidation = self._maybe_promote_behavior_pattern(user_id, pattern_update)
            feedback_record["memory_consolidation"] = consolidation
        self._append_episode_jsonl(user_id, episode_id, "feedback.jsonl", feedback_record)
        self._close_episode_result(user_id, episode_id, episode_result, feedback_record)
        return feedback_record

    def _apply_memory_operations(
        self,
        user_id: str,
        episode_id: str,
        extracted: list[MemoryOperation],
        observation_context: ObservationContext | None = None,
    ) -> dict:
        diff = self.memory_updates.apply(
            extracted,
            MemoryUpdateContext(
                user_id=user_id,
                source=f"episode:{episode_id}",
                diff_id=episode_id,
                day=self._memory_update_day(observation_context),
            ),
        )
        diff["episode_id"] = episode_id
        self._write_episode_file(user_id, episode_id, "memory_diff.json", diff)
        return diff

    def _empty_memory_diff(self, episode_id: str) -> dict:
        diff = self.memory_updates.empty_diff(episode_id)
        diff["episode_id"] = episode_id
        return diff

    def _operation_record(self, memory: MemoryOperation) -> dict:
        return self.memory_updates.operation_record(memory)

    def _memory_update_day(self, observation_context: ObservationContext | None) -> str | None:
        if not observation_context or not observation_context.observed_at:
            return None
        return observation_context.observed_at[:10]

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
        tags = [
            "habit",
            "aggregated",
            f"action:{action}",
            f"domain:{domain}",
            f"promoted_behavior_pattern:{group_id}",
        ]
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
        rows = self.store.hybrid_search(
            f"{group_id} {action}",
            user_id=user_id,
            memory_type="habit",
            limit=8,
        )
        target_tag = f"promoted_behavior_pattern:{group_id}"
        action_tag = f"action:{action}"
        for row in rows:
            tags = {str(tag) for tag in row.get("tags", [])}
            if target_tag in tags and action_tag in tags:
                return row
        return None

    def _close_episode_result(
        self,
        user_id: str,
        episode_id: str,
        episode_result: dict,
        feedback_record: dict,
    ) -> None:
        if not episode_result:
            return
        closed = dict(episode_result)
        closed["episode_status"] = "closed_with_feedback"
        closed["closed_at"] = feedback_record["created_at"]
        closed["actual_action"] = feedback_record.get("actual_action")
        closed["action_params"] = feedback_record.get("action_params", {})
        closed["spontaneity"] = feedback_record.get("spontaneity", "unknown")
        closed["feedback"] = feedback_record.get("feedback")
        closed["reward"] = feedback_record.get("reward")
        closed["feedback_record"] = feedback_record
        self._write_episode_file(user_id, episode_id, "episode_result.json", closed)

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
            updates.append(
                {
                    "uri": update["uri"],
                    "confidence": confidence,
                    "negative_count": negative_count,
                }
            )
        return updates

    def _action_universe(
        self,
        available_actions: list[str],
        memories: list[dict],
        behavior_patterns: list[dict],
        behavior_distribution: list[dict],
    ) -> list[str]:
        actions = {str(action) for action in available_actions if action}
        for item in behavior_patterns:
            if item.get("action"):
                actions.add(str(item["action"]))
        for item in behavior_distribution:
            if item.get("action"):
                actions.add(str(item["action"]))
        for memory in memories:
            for key in ("action", "actual_action", "predicted_action"):
                if memory.get(key):
                    actions.add(str(memory[key]))
            for tag in memory.get("tags", []):
                value = str(tag)
                if value.startswith("action:") or value.startswith("actual_action:"):
                    actions.add(value.split(":", 1)[1])
            for line in str(memory.get("content", "")).splitlines():
                if line.lower().strip().startswith("actual action:"):
                    actions.add(line.split(":", 1)[1].strip())
        return sorted(action for action in actions if action)

    def _architecture_layers(
        self,
        memories: list[dict],
        retrieval,
        rl_state: PolicyState,
        intervention: InterventionDecision,
    ) -> dict:
        return {
            "fact_layer": {
                "sources": ["episodes", "events", "feedback"],
                "retrieved_event_count": sum(1 for memory in memories if memory.get("type") in {"event", "case", "feedback"}),
            },
            "semantic_memory_layer": {
                "sources": ["profile", "preference", "habit", "trigger", "policy"],
                "retrieved_memory_count": len(memories),
                "route_trace": [route.to_dict() for route in retrieval.memory_context.route_trace],
            },
            "prediction_pattern_layer": {
                "behavior_pattern_count": len(retrieval.behavior_patterns),
                "behavior_distribution_count": len(retrieval.behavior_distribution),
                "rl_state_key": rl_state.key,
            },
            "decision_layer": {
                "intervention_action": intervention.action,
                "intervention_reason": intervention.reason,
            },
        }

    def _episode_dir(self, user_id: str, episode_id: str) -> Path:
        validate_identifier(user_id, "user_id")
        validate_identifier(episode_id, "episode_id")
        path = self.store.root / "user" / user_id / "episodes" / episode_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_episode_file(self, user_id: str, episode_id: str, filename: str, payload: dict) -> None:
        path = self._episode_dir(user_id, episode_id) / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_episode_jsonl(self, user_id: str, episode_id: str, filename: str, payload: dict) -> None:
        path = self._episode_dir(user_id, episode_id) / filename
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _read_episode_result(self, user_id: str, episode_id: str) -> dict:
        path = self._episode_dir(user_id, episode_id) / "episode_result.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _policy_stats_path(self, user_id: str) -> Path:
        return self.store.root / "user" / user_id / "policy_stats.json"

    def _behavior_stats_path(self, user_id: str) -> Path:
        return self.store.root / "user" / user_id / "behavior_stats.json"

    def _rl_ledger_path(self, user_id: str) -> Path:
        return self.store.root / "user" / user_id / "rl" / "policy_ledger.json"
