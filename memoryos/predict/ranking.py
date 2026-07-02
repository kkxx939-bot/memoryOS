from __future__ import annotations

from .candidates import Candidate, CandidateGenerator


class CandidateRanker:
    def __init__(self, generator: CandidateGenerator | None = None) -> None:
        self.generator = generator or CandidateGenerator()

    def rank(
        self,
        scene: str,
        memories: list[dict],
        behavior_distribution: list[dict] | None = None,
        candidates: list[Candidate] | None = None,
        rl_action_scores: dict[str, float] | None = None,
    ) -> list[Candidate]:
        candidates = candidates or self.generator.generate(scene, memories)
        rl_action_scores = rl_action_scores or {}
        behavior_scores = {
            str(item["action"]): float(item.get("weighted_behavior_reward", item.get("behavior_reward_score", 0.5)))
            for item in (behavior_distribution or [])
        }
        ranked = []
        for candidate in candidates:
            memory_evidence = self._memory_evidence(candidate.action, memories)
            features = self._features(candidate, scene, memory_evidence, behavior_scores, rl_action_scores)
            score = (
                features["candidate_prior"] * 0.20
                + features["structured_action_match"] * 0.20
                + features["memory_support"] * 0.20
                + features["behavior_reward"] * 0.15
                + features["memory_hotness"] * 0.05
                + features["rl_policy_score"] * 0.20
            )
            candidate.features = features
            candidate.memory_evidence = memory_evidence
            candidate.used_memories = [item["path"] for item in memory_evidence if item["usage_weight"] > 0]
            candidate.score = round(score, 6)
            ranked.append(candidate)
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def _features(
        self,
        candidate: Candidate,
        scene: str,
        memory_evidence: list[dict],
        behavior_scores: dict[str, float],
        rl_action_scores: dict[str, float],
    ) -> dict[str, float]:
        memory_support = self._memory_support(memory_evidence)
        memory_hotness = max((float(item.get("hotness", 0.0)) for item in memory_evidence), default=0.0)
        return {
            "candidate_prior": candidate.prior,
            "structured_action_match": 1.0 if memory_evidence or "behavior_pattern" in candidate.sources else 0.0,
            "memory_support": memory_support,
            "behavior_reward": behavior_scores.get(candidate.action, 0.5),
            "memory_hotness": memory_hotness,
            "rl_policy_score": rl_action_scores.get(candidate.action, 0.5),
        }

    def _memory_evidence(self, action: str, memories: list[dict]) -> list[dict]:
        evidence = []
        for memory in memories:
            retrieval_weight = self._retrieval_weight(memory)
            usage_weight = self._usage_weight(action, memory)
            if usage_weight <= 0:
                continue
            evidence.append(
                {
                    "id": memory.get("id"),
                    "path": memory.get("path"),
                    "type": memory.get("type"),
                    "title": memory.get("title"),
                    "retrieval_weight": retrieval_weight,
                    "usage_weight": usage_weight,
                    "combined_weight": round(retrieval_weight * usage_weight, 6),
                    "hotness": float(memory.get("hotness", 0.0)),
                    "confidence": float(memory.get("confidence", 0.0)),
                    "support": "positive",
                }
            )
        evidence.sort(key=lambda item: item["combined_weight"], reverse=True)
        return evidence

    def _retrieval_weight(self, memory: dict) -> float:
        score = float(memory.get("final_score", memory.get("score", 0.0)) or 0.0)
        keyword = float(memory.get("keyword_score", 0.0) or 0.0)
        embedding = float(memory.get("embedding_score", 0.0) or 0.0)
        hotness = float(memory.get("hotness", 0.0) or 0.0)
        confidence = float(memory.get("confidence", 0.0) or 0.0)
        effective_weight = float(memory.get("effective_weight", 0.0) or 0.0)
        value = max(score, keyword * 0.25 + embedding * 0.35 + hotness * 0.10 + confidence * 0.10 + effective_weight * 0.20)
        return round(max(0.0, min(1.0, value)), 6)

    def _usage_weight(self, action: str, memory: dict) -> float:
        memory_type = str(memory.get("type", ""))
        effective_weight = float(memory.get("effective_weight", 1.0) or 0.0)
        type_weight = {
            "habit": 1.0,
            "trigger": 1.0,
            "preference": 0.85,
            "policy": 0.75,
            "feedback": 0.7,
            "intervention": 0.65,
            "case": 0.6,
            "profile": 0.55,
            "event": 0.45,
        }.get(memory_type, 0.4)
        if self._explicit_action_match(action, memory):
            return round(type_weight * effective_weight, 6)
        if action == "continue_current_activity" and memory_type in {"profile", "policy", "feedback"}:
            return round(type_weight * effective_weight * 0.35, 6)
        return 0.0

    def _explicit_action_match(self, action: str, memory: dict) -> bool:
        for tag in [str(tag) for tag in memory.get("tags", [])]:
            if tag == f"action:{action}" or tag == f"actual_action:{action}":
                return True
        for line in str(memory.get("content", "")).splitlines():
            lowered = line.lower().strip()
            if lowered.startswith("actual action:") and line.split(":", 1)[1].strip() == action:
                return True
        return False

    def _memory_support(self, memory_evidence: list[dict]) -> float:
        if not memory_evidence:
            return 0.0
        combined = sum(float(item["combined_weight"]) for item in memory_evidence[:5])
        return round(max(0.0, min(1.0, combined)), 6)
