from __future__ import annotations

import json
from pathlib import Path

from memoryos.adapters.events.jsonl_outbox import FeedbackEventStore
from memoryos.domain.memory.memory_item import utc_now
from memoryos.observability.audit_log import AuditLogger
from memoryos.ports.repositories.memory_repository import MemoryRepository
from memoryos.services.learning.learning_service import LearningProcessor
from memoryos.usecases.episode.episode_state_machine import CLOSED, LEARNING_APPLIED


class FeedbackWorker:
    def __init__(self, store: MemoryRepository) -> None:
        self.store = store
        self.events = FeedbackEventStore(store.root)
        self.learning = LearningProcessor(store)

    def process_pending(self, user_id: str | None = None, limit: int | None = None) -> dict:
        pending = self.events.pending_outbox_events(user_id=user_id, limit=limit)
        results = []
        for outbox_event in pending:
            event = dict(outbox_event.get("payload", {}))
            episode_user_id = str(event.get("user_id") or outbox_event.get("user_id", ""))
            episode_id = str(event.get("episode_id") or outbox_event.get("episode_id", ""))
            episode_result = self._read_episode_result(episode_user_id, episode_id)
            learning_result = self.learning.apply_feedback_event(event, episode_result)
            applied_outbox = self.events.mark_outbox_applied(outbox_event)
            self._append_episode_jsonl(
                episode_user_id,
                episode_id,
                "feedback.jsonl",
                {
                    **learning_result,
                    "outbox_event": applied_outbox,
                    "learning_status": "applied_by_worker",
                },
            )
            self._close_episode_result(episode_user_id, episode_id, episode_result, learning_result)
            AuditLogger(self.store.root).record(
                episode_user_id,
                "feedback_learning_applied",
                {
                    "episode_id": episode_id,
                    "outbox_id": outbox_event.get("outbox_id"),
                    "event_id": learning_result.get("event_id"),
                    "idempotent": learning_result.get("idempotent", False),
                    "actual_action": learning_result.get("actual_action"),
                    "behavior_reward": learning_result.get("reward_breakdown", {}).get("behavior_reward"),
                    "intervention_reward": learning_result.get("reward_breakdown", {}).get("intervention_reward"),
                },
            )
            results.append(
                {
                    "outbox_id": outbox_event.get("outbox_id"),
                    "episode_id": episode_id,
                    "learning_result": learning_result,
                    "outbox_event": applied_outbox,
                }
            )
        return {"processed": len(results), "results": results}

    def _close_episode_result(
        self,
        user_id: str,
        episode_id: str,
        episode_result: dict,
        learning_result: dict,
    ) -> None:
        if not episode_result:
            return
        closed = dict(episode_result)
        closed["episode_status"] = "closed_with_feedback"
        closed["episode_state"] = CLOSED
        closed["state_history"] = self._append_state_history(
            episode_result.get("state_history", []),
            [
                (LEARNING_APPLIED, "learning processor applied feedback event"),
                (CLOSED, "episode closed after feedback learning"),
            ],
        )
        closed["closed_at"] = learning_result.get("created_at", utc_now())
        closed["actual_action"] = learning_result.get("actual_action")
        closed["action_params"] = learning_result.get("action_params", {})
        closed["spontaneity"] = learning_result.get("spontaneity", "unknown")
        closed["feedback"] = learning_result.get("feedback")
        closed["reward"] = learning_result.get("reward")
        closed["learning_result"] = learning_result
        self._write_episode_file(user_id, episode_id, "episode_result.json", closed)

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

    def _read_episode_result(self, user_id: str, episode_id: str) -> dict:
        path = self._episode_dir(user_id, episode_id) / "episode_result.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_episode_file(self, user_id: str, episode_id: str, filename: str, payload: dict) -> None:
        path = self._episode_dir(user_id, episode_id) / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_episode_jsonl(self, user_id: str, episode_id: str, filename: str, payload: dict) -> None:
        path = self._episode_dir(user_id, episode_id) / filename
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
