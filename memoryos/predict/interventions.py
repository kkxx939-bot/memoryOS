from __future__ import annotations

from dataclasses import dataclass, field

from .candidates import Candidate


@dataclass
class InterventionDecision:
    action: str
    predicted_action: str
    predicted_need: str
    score: float
    reason: str
    features: dict[str, float] = field(default_factory=dict)
    alternatives: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "predicted_action": self.predicted_action,
            "predicted_need": self.predicted_need,
            "score": self.score,
            "reason": self.reason,
            "features": self.features,
            "alternatives": self.alternatives,
        }


class InterventionSelector:
    PHYSICAL_ACTIONS = {"turn_on_ac", "open_ac", "unlock_door", "send_message", "place_order"}

    def select(
        self,
        candidate: Candidate | None,
        available_actions: list[str],
        policy_stats: dict,
    ) -> InterventionDecision:
        if not candidate:
            action = self._pick_available(available_actions, ["do_nothing", "ask_user"])
            return InterventionDecision(
                action=action,
                predicted_action="unknown",
                predicted_need="unknown",
                score=0.0,
                reason="No behavior candidate was available.",
                features={"policy_reward": self._policy_reward("unknown", action, policy_stats)},
            )

        options = []
        preferred = self._preferred_interventions(candidate.action)
        available_preferred = [
            action
            for action in preferred
            if action in available_actions and self._permission_allows(action, available_actions, policy_stats)
        ]
        if not available_preferred:
            available_preferred = [
                action
                for action in (available_actions or ["do_nothing"])
                if self._permission_allows(action, available_actions, policy_stats)
            ] or ["do_nothing"]

        for rank, action in enumerate(available_preferred):
            preference_score = max(0.0, 1.0 - rank * 0.18)
            policy_reward = self._policy_reward(candidate.action, action, policy_stats)
            interruption_cost = self._interruption_cost(action)
            confidence = max(0.0, min(1.0, candidate.score))
            score = preference_score * 0.45 + policy_reward * 0.30 + confidence * 0.25 - interruption_cost
            options.append(
                {
                    "action": action,
                    "score": round(max(0.0, min(1.0, score)), 6),
                    "features": {
                        "preference_score": round(preference_score, 6),
                        "policy_reward": round(policy_reward, 6),
                        "behavior_confidence": round(confidence, 6),
                        "interruption_cost": interruption_cost,
                    },
                }
            )

        options.sort(key=lambda item: item["score"], reverse=True)
        top = options[0]
        return InterventionDecision(
            action=str(top["action"]),
            predicted_action=candidate.action,
            predicted_need=candidate.need,
            score=float(top["score"]),
            reason=f"Selected intervention for predicted user behavior: {candidate.action}.",
            features=top["features"],
            alternatives=options[1:],
        )

    def _preferred_interventions(self, action: str) -> list[str]:
        if action in {"open_ac", "turn_on_ac", "seek_cooling"}:
            return ["turn_on_ac", "ask_before_turning_on_ac", "ask_user", "do_nothing"]
        if action == "smoke":
            return ["remind_no_smoking", "ask_user", "do_nothing"]
        if action == "take_break":
            return ["suggest_break", "ask_user", "do_nothing"]
        if action in {"continue_working", "continue_current_activity"}:
            return ["do_nothing", "ask_user"]
        return ["ask_user", "do_nothing"]

    def _pick_available(self, available_actions: list[str], preferred: list[str]) -> str:
        for action in preferred:
            if action in available_actions:
                return action
        return available_actions[0] if available_actions else "do_nothing"

    def _interruption_cost(self, action: str) -> float:
        if action == "do_nothing":
            return 0.0
        if action in {"ask_user", "ask_before_turning_on_ac"}:
            return 0.08
        return 0.14

    def _policy_reward(self, predicted_action: str, intervention: str, policy_stats: dict) -> float:
        entry = None
        for alias in self._action_aliases(predicted_action):
            entry = policy_stats.get(f"{alias}::{intervention}")
            if entry:
                break
        if not entry:
            return 0.5
        average = float(entry.get("average_reward", 0.0))
        return max(0.0, min(1.0, (average + 1.0) / 2.0))

    def _permission_allows(self, action: str, available_actions: list[str], policy_stats: dict) -> bool:
        if action not in self.PHYSICAL_ACTIONS:
            return True
        permission = policy_stats.get(f"permission::{action}", {})
        if permission.get("allowed_without_confirmation") is True:
            return True
        return False

    def _action_aliases(self, action: str) -> list[str]:
        groups = [
            {"open_ac", "turn_on_ac", "seek_cooling"},
            {"continue_working", "continue_current_activity"},
        ]
        for group in groups:
            if action in group:
                return [action, *sorted(group - {action})]
        return [action]
