from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from memoryos.application.intervention.intervention_selector import InterventionSelector
from memoryos.application.memory.extractor import TextGenerationProvider
from memoryos.application.prediction.candidate_generator import CandidateGenerator
from memoryos.application.prediction.candidate_ranker import CandidateRanker


@dataclass
class Prediction:
    predicted_action: str
    predicted_need: str
    confidence: float
    recommended_intervention: str
    reason: str
    used_memories: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("Prediction confidence must be in [0, 1]")

    def to_dict(self) -> dict:
        return {
            "predicted_action": self.predicted_action,
            "predicted_need": self.predicted_need,
            "confidence": self.confidence,
            "recommended_intervention": self.recommended_intervention,
            "reason": self.reason,
            "used_memories": self.used_memories,
        }


class RuleBasedPredictor:
    def __init__(
        self,
        generator: CandidateGenerator | None = None,
        ranker: CandidateRanker | None = None,
        intervention_selector: InterventionSelector | None = None,
        policy_stats: dict | None = None,
    ) -> None:
        self.generator = generator or CandidateGenerator()
        self.ranker = ranker or CandidateRanker(generator=self.generator)
        self.intervention_selector = intervention_selector or InterventionSelector()
        self.policy_stats = policy_stats or {}

    def rank(
        self,
        scene: str,
        memories: list[dict],
        available_actions: list[str],
        policy_stats: dict | None = None,
        behavior_patterns: list[dict] | None = None,
        behavior_distribution: list[dict] | None = None,
        rl_action_scores: dict[str, float] | None = None,
    ) -> list:
        candidates = self.generator.generate(
            scene,
            memories,
            behavior_patterns=behavior_patterns,
            behavior_distribution=behavior_distribution,
        )
        return self.ranker.rank(
            scene,
            memories,
            behavior_distribution=behavior_distribution,
            candidates=candidates,
            rl_action_scores=rl_action_scores,
        )

    def predict(
        self,
        scene: str,
        memories: list[dict],
        available_actions: list[str],
    ) -> Prediction:
        candidates = self.rank(scene, memories, available_actions)
        if candidates:
            top = candidates[0]
            intervention = self.intervention_selector.select(top, available_actions, self.policy_stats)
            return Prediction(
                predicted_action=top.action,
                predicted_need=top.need,
                confidence=max(0.0, min(1.0, top.score)),
                recommended_intervention=intervention.action,
                reason=top.reason,
                used_memories=top.used_memories,
            )
        return Prediction(
            predicted_action="unknown",
            predicted_need="unknown",
            confidence=0.2,
            recommended_intervention=self._pick_action(available_actions, ["do_nothing", "ask_user"]),
            reason="No strong behavior signal found.",
            used_memories=[memory["path"] for memory in memories],
        )

    def _pick_action(self, available_actions: list[str], preferred: list[str]) -> str:
        for action in preferred:
            if action in available_actions:
                return action
        return available_actions[0] if available_actions else "do_nothing"


class JsonLLMPredictor:
    def __init__(self, provider: TextGenerationProvider) -> None:
        self.provider = provider

    def predict(
        self,
        scene: str,
        memories: list[dict],
        available_actions: list[str],
    ) -> Prediction:
        response = self.provider.complete(self.build_prompt(scene, memories))
        payload = self._load_json(response)
        used_memories = payload.get("used_memories", [])
        if not isinstance(used_memories, list):
            raise ValueError("Prediction used_memories must be a list")
        return Prediction(
            predicted_action=str(payload.get("predicted_action", "unknown")),
            predicted_need=str(payload.get("predicted_need", "unknown")),
            confidence=float(payload.get("confidence", 0.0)),
            recommended_intervention="do_nothing",
            reason=str(payload.get("reason", "")),
            used_memories=[str(value) for value in used_memories],
        )

    def build_prompt(self, scene: str, memories: list[dict]) -> str:
        memory_lines = []
        for memory in memories:
            abstract = memory.get("abstract") or memory.get("content", "")[:240]
            memory_lines.append(f"- {memory.get('path')}: [{memory.get('type')}] {memory.get('title')}: {abstract}")
        return f"""You are the behavior prediction layer for a personal memory system.

Use the current scene and retrieved memories to predict the user's likely next action or need.
Do not decide the final system intervention here; that is handled by a separate policy layer.
Return strict JSON. No markdown. No commentary.

Schema:
{{
  "predicted_action": "short action label",
  "predicted_need": "short need label",
  "confidence": 0.0,
  "reason": "brief auditable reason",
  "used_memories": ["memory/path.md"]
}}

Current scene:
{scene}

Retrieved memories:
{chr(10).join(memory_lines)}
"""

    def _load_json(self, response: str) -> dict:
        text = response.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Prediction response must be a JSON object")
        return payload
