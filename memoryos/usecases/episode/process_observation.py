from __future__ import annotations

import json
from pathlib import Path

from memoryos.domain.actions.action_schema import ACTION_SCHEMA_VERSION
from memoryos.domain.feedback.reward_result import REWARD_MODEL_VERSION
from memoryos.domain.memory.memory_item import utc_now
from memoryos.domain.scene.observation import ObservationContext
from memoryos.domain.scene.scene_features import SceneFeatures
from memoryos.domain.scene.scene_signature import stable_scene_signature
from memoryos.observability.audit_log import AuditLogger
from memoryos.ports.repositories.memory_repository import MemoryRepository
from memoryos.security.path_safety import validate_identifier
from memoryos.services.learning.intervention_policy_stats import PolicyStats
from memoryos.services.learning.rl_calibrator import PolicyState, ReinforcementPolicyLedger
from memoryos.services.memory.extractor import MemoryOperation, RuleBasedExtractor
from memoryos.services.memory.update_service import MemoryUpdateContext, MemoryUpdateService
from memoryos.services.policy.policy_gate import POLICY_VERSION
from memoryos.services.prediction.candidate_generator import Candidate
from memoryos.services.prediction.prediction_service import Prediction, RuleBasedPredictor
from memoryos.services.retrieval.retrieval_service import RetrievalOrchestrator
from memoryos.usecases.episode.episode_files import EpisodeFileStore
from memoryos.usecases.episode.episode_state_machine import (
    CREATED,
    EPISODE_STATE_VERSION,
    FEEDBACK_PENDING,
    INTERVENTION_SELECTED,
    OBSERVED,
    PREDICTED,
    RETRIEVED,
)
from memoryos.usecases.feedback.record_feedback import FeedbackService
from memoryos.usecases.intervention.select_intervention import InterventionDecision, InterventionSelector


class EpisodeProcessor:
    def __init__(
        self,
        store: MemoryRepository,
        extractor=None,
        predictor=None,
    ) -> None:
        self.store = store
        self.extractor = extractor or RuleBasedExtractor()
        self.predictor = predictor or RuleBasedPredictor()
        self.intervention_selector = InterventionSelector()
        self.memory_updates = MemoryUpdateService(store)
        self.episode_files = EpisodeFileStore(store)

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
        scene_features = SceneFeatures.from_observation(observation_context) if observation_context else None
        scene_signature = stable_scene_signature(scene_features) if scene_features else ""
        available_actions = available_actions or ["ask_user", "do_nothing"]
        input_messages = messages or [{"role": "observation", "text": scene_text, "created_at": utc_now()}]
        memory_operations = self.extractor.extract(input_messages)
        explicit_memory_operations = self._explicit_memory_operations(memory_operations, input_messages)
        explicit_operation_ids = {id(operation) for operation in explicit_memory_operations}
        draft_memory_operations = [
            operation for operation in memory_operations if id(operation) not in explicit_operation_ids
        ]
        prediction_state_history = self._state_history(
            [
                (CREATED, "episode accepted"),
                (OBSERVED, "raw observation logged"),
            ]
        )

        self._write_episode_file(
            user_id,
            episode_id,
            "episode_log.json",
            {
                "episode_id": episode_id,
                "status": "pending",
                "episode_state": OBSERVED,
                "state_history": prediction_state_history,
                "versions": self._versions(),
                "created_at": utc_now(),
                "scene": scene_text,
                "observation": observation_context.to_dict() if observation_context else None,
                "scene_features": scene_features.to_dict() if scene_features else {},
                "scene_signature": scene_signature,
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
        prediction_state_history = self._state_history(
            [
                (CREATED, "episode accepted"),
                (OBSERVED, "raw observation logged"),
                (RETRIEVED, "historical memory and behavior context retrieved"),
                (PREDICTED, "behavior candidates ranked"),
                (INTERVENTION_SELECTED, "policy-gated intervention selected"),
                (FEEDBACK_PENDING, "waiting for actual action or user feedback"),
            ]
        )
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
            "episode_state": FEEDBACK_PENDING,
            "state_history": prediction_state_history,
            "versions": self._versions(),
            "processed_at": utc_now(),
            "scene": scene_text,
            "observation": observation_context.to_dict() if observation_context else None,
            "scene_features": scene_features.to_dict() if scene_features else {},
            "scene_signature": scene_signature,
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
        AuditLogger(self.store.root).record(
            user_id,
            "episode_predicted",
            {
                "episode_id": episode_id,
                "predicted_action": prediction.predicted_action,
                "recommended_intervention": prediction.recommended_intervention,
                "intervention_action": intervention.action,
                "top_candidate": ranked_candidates[0].to_dict() if ranked_candidates else None,
                "retrieved_memory_count": len(memories),
                "behavior_pattern_count": len(retrieval.behavior_patterns),
                "rl_state_key": rl_state.key,
            },
        )
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
        return FeedbackService(self.store).record_feedback(
            user_id=user_id,
            episode_id=episode_id,
            feedback=feedback,
            reward=reward,
            actual_action=actual_action,
            action_params=action_params,
            spontaneity=spontaneity,
            intervention_result=intervention_result,
            correction=correction,
            corrects_memory=corrects_memory,
        )

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
                "policy_version": POLICY_VERSION,
            },
        }

    def _versions(self) -> dict:
        return {
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "episode_state_version": EPISODE_STATE_VERSION,
            "reward_model_version": REWARD_MODEL_VERSION,
            "policy_version": POLICY_VERSION,
            "ranker_version": "ranker_v1",
        }

    def _state_history(self, states: list[tuple[str, str]]) -> list[dict]:
        created_at = utc_now()
        return [{"state": state, "reason": reason, "at": created_at} for state, reason in states]

    def _append_state_history(self, existing: list, states: list[tuple[str, str]]) -> list[dict]:
        history = [item for item in existing if isinstance(item, dict)]
        seen = {str(item.get("state", "")) for item in history}
        for item in self._state_history(states):
            if item["state"] not in seen:
                history.append(item)
                seen.add(item["state"])
        return history

    def _episode_dir(self, user_id: str, episode_id: str) -> Path:
        return self.episode_files.episode_dir(user_id, episode_id)

    def _write_episode_file(self, user_id: str, episode_id: str, filename: str, payload: dict) -> None:
        self.episode_files.write_json(user_id, episode_id, filename, payload)

    def _append_episode_jsonl(self, user_id: str, episode_id: str, filename: str, payload: dict) -> None:
        self.episode_files.append_jsonl(user_id, episode_id, filename, payload)

    def _read_episode_result(self, user_id: str, episode_id: str) -> dict:
        return self.episode_files.read_json(user_id, episode_id, "episode_result.json")

    def _policy_stats_path(self, user_id: str) -> Path:
        return self.store.root / "user" / user_id / "policy_stats.json"

    def _behavior_stats_path(self, user_id: str) -> Path:
        return self.store.root / "user" / user_id / "behavior_stats.json"

    def _rl_ledger_path(self, user_id: str) -> Path:
        return self.store.root / "user" / user_id / "rl" / "policy_ledger.json"
