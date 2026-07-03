from __future__ import annotations

import json
from pathlib import Path

from memoryos.application.episode.episode_state_machine import FEEDBACK_RECEIVED, LEARNING_QUEUED
from memoryos.application.feedback.feedback_event_store import FeedbackEventStore
from memoryos.domain.feedback.reward_result import compute_rewards
from memoryos.domain.memory.memory_item import utc_now
from memoryos.infrastructure.repositories.memory_repository import MemoryStore
from memoryos.infrastructure.safety.path_safety import validate_identifier
from memoryos.observability.audit_log import AuditLogger


class FeedbackService:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.events = FeedbackEventStore(store.root)

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
        reward_breakdown = compute_rewards(
            predicted_action=predicted_action,
            actual_action=actual_action,
            user_reward=reward,
            intervention_action=recommended_intervention,
            intervention_result=intervention_result or feedback,
            actual_params=action_params or {},
        )
        event_payload = {
            "user_id": user_id,
            "episode_id": episode_id,
            "created_at": created_at,
            "feedback": feedback,
            "reward": reward,
            "reward_breakdown": reward_breakdown.to_dict(),
            "predicted_action": predicted_action,
            "actual_action": actual_action,
            "action_params": action_params or {},
            "spontaneity": spontaneity,
            "intervention_result": intervention_result,
            "recommended_intervention": recommended_intervention,
            "correction": correction,
            "corrects_memory": corrects_memory,
        }
        feedback_event = self.events.append_feedback_event(user_id, episode_id, event_payload)
        outbox_event = self.events.append_outbox_event(user_id, feedback_event)
        record = {
            "episode_id": episode_id,
            "created_at": created_at,
            "feedback": feedback,
            "reward": reward,
            "reward_breakdown": reward_breakdown.to_dict(),
            "predicted_action": predicted_action,
            "actual_action": actual_action,
            "action_params": action_params or {},
            "spontaneity": spontaneity,
            "intervention_result": intervention_result,
            "recommended_intervention": recommended_intervention,
            "correction": correction,
            "corrects_memory": corrects_memory,
            "feedback_event": {
                "event_id": feedback_event["event_id"],
                "event_type": feedback_event["event_type"],
                "created_at": feedback_event["created_at"],
            },
            "outbox_event": outbox_event,
            "learning_status": "queued",
        }
        self._append_episode_jsonl(user_id, episode_id, "feedback.jsonl", record)
        self._mark_episode_feedback_queued(user_id, episode_id, episode_result, record)
        AuditLogger(self.store.root).record(
            user_id,
            "feedback_queued",
            {
                "episode_id": episode_id,
                "feedback_event_id": feedback_event["event_id"],
                "outbox_id": outbox_event["outbox_id"],
                "predicted_action": predicted_action,
                "actual_action": actual_action,
                "reward_breakdown": reward_breakdown.to_dict(),
            },
        )
        return record

    def _feedback_created_at(self, episode_result: dict) -> str:
        observation = episode_result.get("observation") or {}
        observed_at = str(observation.get("observed_at") or "")
        return observed_at or utc_now()

    def _mark_episode_feedback_queued(
        self,
        user_id: str,
        episode_id: str,
        episode_result: dict,
        feedback_record: dict,
    ) -> None:
        if not episode_result:
            return
        queued = dict(episode_result)
        queued["episode_status"] = "feedback_queued"
        queued["episode_state"] = LEARNING_QUEUED
        queued["state_history"] = self._append_state_history(
            episode_result.get("state_history", []),
            [
                (FEEDBACK_RECEIVED, "feedback event recorded"),
                (LEARNING_QUEUED, "learning event appended to local outbox"),
            ],
        )
        queued["actual_action"] = feedback_record.get("actual_action")
        queued["action_params"] = feedback_record.get("action_params", {})
        queued["spontaneity"] = feedback_record.get("spontaneity", "unknown")
        queued["feedback"] = feedback_record.get("feedback")
        queued["reward"] = feedback_record.get("reward")
        queued["feedback_record"] = feedback_record
        self._write_episode_file(user_id, episode_id, "episode_result.json", queued)

    def _append_state_history(self, existing: list, states: list[tuple[str, str]]) -> list[dict]:
        history = [item for item in existing if isinstance(item, dict)]
        seen = {str(item.get("state", "")) for item in history}
        at = utc_now()
        for state, reason in states:
            if state not in seen:
                history.append({"state": state, "reason": reason, "at": at})
                seen.add(state)
        return history

    def _episode_dir(self, user_id: str, episode_id: str) -> Path:
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
