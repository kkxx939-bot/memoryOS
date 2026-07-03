from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from memoryos.domain.actions.action_schema import action_spec, canonical_action
from memoryos.domain.memory.memory_item import utc_now
from memoryos.application.prediction.candidate_generator import Candidate


@dataclass
class PolicyState:
    key: str
    descriptor: dict

    def to_dict(self) -> dict:
        return {"key": self.key, "descriptor": self.descriptor}


@dataclass
class ActionValue:
    trials: int = 0
    total_reward: float = 0.0
    successes: int = 0
    failures: int = 0
    q_value: float = 0.0
    last_updated_at: str = ""
    average_reward: float = 0.0
    normalized_value: float = 0.5
    ucb_score: float = 0.5

    def update(self, reward: float, success: bool, state_visits: int, td_target: float, learning_rate: float) -> None:
        self.trials += 1
        self.total_reward += reward
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.q_value += learning_rate * (td_target - self.q_value)
        self.last_updated_at = utc_now()
        self.average_reward = self.total_reward / max(1, self.trials)
        blended_value = self.q_value * 0.70 + self.average_reward * 0.30
        self.normalized_value = max(0.0, min(1.0, (blended_value + 1.0) / 2.0))
        exploration = math.sqrt(math.log(max(state_visits, 2)) / max(1, self.trials))
        self.ucb_score = max(0.0, min(1.0, self.normalized_value + exploration * 0.12))

    @classmethod
    def from_dict(cls, payload: dict) -> "ActionValue":
        return cls(
            trials=int(payload.get("trials", 0)),
            total_reward=float(payload.get("total_reward", 0.0)),
            successes=int(payload.get("successes", 0)),
            failures=int(payload.get("failures", 0)),
            q_value=float(payload.get("q_value", 0.0)),
            last_updated_at=str(payload.get("last_updated_at", "")),
            average_reward=float(payload.get("average_reward", 0.0)),
            normalized_value=float(payload.get("normalized_value", 0.5)),
            ucb_score=float(payload.get("ucb_score", 0.5)),
        )

    def to_dict(self) -> dict:
        return {
            "trials": self.trials,
            "total_reward": round(self.total_reward, 6),
            "successes": self.successes,
            "failures": self.failures,
            "q_value": round(self.q_value, 6),
            "last_updated_at": self.last_updated_at,
            "average_reward": round(self.average_reward, 6),
            "normalized_value": round(self.normalized_value, 6),
            "ucb_score": round(self.ucb_score, 6),
        }


@dataclass
class StateValue:
    visits: int = 0
    actions: dict[str, ActionValue] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> "StateValue":
        return cls(
            visits=int(payload.get("visits", 0)),
            actions={
                str(action): ActionValue.from_dict(value)
                for action, value in dict(payload.get("actions", {})).items()
                if isinstance(value, dict)
            },
        )

    def to_dict(self) -> dict:
        return {
            "visits": self.visits,
            "actions": {action: value.to_dict() for action, value in sorted(self.actions.items())},
        }


class ReinforcementPolicyLedger:
    """Persistent Q-learning ledger for behavior prediction calibration."""

    def __init__(
        self,
        path: Path,
        exploration_bonus: float = 0.08,
        learning_rate: float = 0.35,
        discount_factor: float = 0.60,
    ) -> None:
        self.path = path
        self.exploration_bonus = exploration_bonus
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def build_state(
        self,
        scene: str,
        context_tags: list[str],
        memories: list[dict],
        behavior_patterns: list[dict],
    ) -> PolicyState:
        stable_tags = self._stable_context_tags(context_tags)
        descriptor = {
            "scene_signature": self._signature(" ".join(stable_tags) or scene),
            "context_tags": stable_tags,
        }
        key_material = json.dumps(descriptor, ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:24]
        return PolicyState(key=key, descriptor=descriptor)

    def action_scores(self, state: PolicyState, actions: list[str]) -> dict[str, float]:
        payload = self._load()
        state_value = StateValue.from_dict(payload.get("states", {}).get(state.key, {}))
        scores = {}
        for action in sorted({str(action) for action in actions if action}):
            value = state_value.actions.get(action)
            if value is None or value.trials == 0:
                scores[action] = 0.5 + self.exploration_bonus
            else:
                scores[action] = value.ucb_score
        return {action: round(max(0.0, min(1.0, score)), 6) for action, score in scores.items()}

    def record_prediction(
        self,
        user_id: str,
        episode_id: str,
        state: PolicyState,
        candidates: list[Candidate],
        selected_action: str,
        intervention_action: str,
        action_scores: dict[str, float],
    ) -> dict:
        payload = self._load()
        payload["states"].setdefault(state.key, StateValue().to_dict())
        prediction_record = {
            "user_id": user_id,
            "episode_id": episode_id,
            "created_at": utc_now(),
            "state": state.to_dict(),
            "selected_action": selected_action,
            "intervention_action": intervention_action,
            "action_scores": action_scores,
            "candidates": [
                {
                    "action": candidate.action,
                    "score": candidate.score,
                    "prior": candidate.prior,
                    "sources": candidate.sources,
                }
                for candidate in candidates
            ],
        }
        payload["predictions"][episode_id] = prediction_record
        self._save(payload)
        self._append_event("prediction_events.jsonl", prediction_record)
        return {
            "state_key": state.key,
            "selected_action": selected_action,
            "action_scores": action_scores,
            "ledger_uri": self._relative_uri(),
        }

    def record_feedback(
        self,
        episode_id: str,
        predicted_action: str,
        actual_action: str | None,
        reward: float,
        next_state: PolicyState | None = None,
    ) -> dict:
        payload = self._load()
        prediction = payload.get("predictions", {}).get(episode_id)
        if not prediction:
            return {"updated": False, "reason": "prediction not found"}
        state = prediction.get("state", {})
        state_key = str(state.get("key", ""))
        if not state_key:
            return {"updated": False, "reason": "state key missing"}
        states = payload.setdefault("states", {})
        state_value = StateValue.from_dict(states.get(state_key, {}))
        state_value.visits += 1
        actual = canonical_action(str(actual_action or "").strip())
        predicted = canonical_action(str(predicted_action or prediction.get("selected_action") or "unknown"))
        predicted_success = bool(actual and predicted == actual)
        predicted_reward = reward if predicted_success else min(float(reward), -0.5)
        next_best_q = self._next_best_q(payload, next_state)
        self._update_action(state_value, predicted, predicted_reward, predicted_success, next_best_q)
        actual_spec = action_spec(actual)
        if actual and actual != predicted and actual_spec.intervenable and actual_spec.risk_level not in {"private", "high"}:
            self._update_action(state_value, actual, max(float(reward), 0.5), True, next_best_q)
        states[state_key] = state_value.to_dict()
        transition_record = {
            "state_key": state_key,
            "action": predicted,
            "reward": predicted_reward,
            "next_state_key": next_state.key if next_state else "",
            "actual_action": actual,
            "terminal": next_state is None,
        }
        feedback_record = {
            "episode_id": episode_id,
            "created_at": utc_now(),
            "state_key": state_key,
            "predicted_action": predicted,
            "actual_action": actual,
            "reward": reward,
            "predicted_success": predicted_success,
            "state_value": states[state_key],
            "transition": transition_record,
        }
        payload.setdefault("feedback", []).append(feedback_record)
        payload.setdefault("transitions", []).append(transition_record)
        self._save(payload)
        self._append_event("feedback_events.jsonl", feedback_record)
        return {
            "updated": True,
            "state_key": state_key,
            "predicted_success": predicted_success,
            "state_value": states[state_key],
            "predicted_action_value": states[state_key]["actions"].get(predicted, {}),
            "actual_action_value": states[state_key]["actions"].get(actual, {}) if actual else {},
            "transition": transition_record,
        }

    def _update_action(self, state_value: StateValue, action: str, reward: float, success: bool, next_best_q: float) -> None:
        if not action:
            return
        value = state_value.actions.setdefault(action, ActionValue())
        bounded_reward = max(-1.0, min(1.0, float(reward)))
        td_target = bounded_reward + self.discount_factor * next_best_q
        value.update(bounded_reward, success, state_value.visits, td_target, self.learning_rate)

    def _next_best_q(self, payload: dict, next_state: PolicyState | None) -> float:
        if next_state is None:
            return 0.0
        state_value = StateValue.from_dict(payload.get("states", {}).get(next_state.key, {}))
        if not state_value.actions:
            return 0.0
        return max(value.q_value for value in state_value.actions.values())

    def _load(self) -> dict:
        if not self.path.exists():
            return {
                "version": 1,
                "algorithm": "contextual_q_learning_ucb",
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "states": {},
                "predictions": {},
                "feedback": [],
                "transitions": [],
            }
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_event(self, filename: str, payload: dict) -> None:
        path = self.path.parent / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _relative_uri(self) -> str:
        return str(self.path.as_posix())

    def _signature(self, text: str) -> str:
        normalized = " ".join(str(text).lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _stable_context_tags(self, context_tags: list[str]) -> list[str]:
        stable_prefixes = (
            "location_",
            "activity_",
            "duration_",
            "temperature_",
            "humidity_",
            "ac_status_",
            "fan_status_",
        )
        stable_values = {
            "morning",
            "noon",
            "afternoon",
            "evening",
            "night",
            "very_hot",
            "hot",
            "slightly_hot",
            "comfortable",
            "cold",
            "very_cold",
            "hot_environment",
            "cold_environment",
            "humid_environment",
            "sweating",
            "says_hot",
            "arrive_home",
            "computer_work",
            "computer_desk",
            "room",
        }
        values = set()
        for tag in context_tags:
            value = str(tag).strip().lower()
            if not value:
                continue
            if value in stable_values or any(value.startswith(prefix) for prefix in stable_prefixes):
                values.add(value)
        return sorted(values)
