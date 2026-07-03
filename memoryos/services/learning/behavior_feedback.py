from __future__ import annotations

import json
from pathlib import Path

from memoryos.domain.behavior.signatures import (
    MATCH_LEVEL_WEIGHTS,
    context_tokens,
    hash_tokens,
    is_coarse_token,
    is_exact_only_token,
    scene_signatures,
    text_tokens,
)


class BehaviorStats:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record(
        self,
        retrieval_query: str,
        context_tags: list[str],
        predicted_action: str,
        actual_action: str | None,
        reward: float,
        event_id: str = "",
    ) -> dict:
        data = self._load()
        if event_id:
            processed = data.setdefault("processed_events", {})
            if event_id in processed:
                result = dict(processed[event_id])
                result["idempotent"] = True
                return result
        signatures = self.scene_signatures(retrieval_query, context_tags)
        behavior_reward = self._behavior_reward(predicted_action, actual_action, reward)
        updated_entry = {}
        for level, signature in signatures.items():
            bucket = self._bucket(data, level, signature, retrieval_query, context_tags)
            updated_entry = self._update_bucket(bucket, predicted_action, actual_action, behavior_reward)

        result = {
            "signature": signatures["exact"],
            "signatures": signatures,
            "predicted_action": predicted_action,
            "actual_action": actual_action,
            "behavior_reward": behavior_reward,
            "entry": updated_entry,
            "event_id": event_id,
            "idempotent": False,
        }
        if event_id:
            data.setdefault("processed_events", {})[event_id] = result
        self._save(data)
        return result

    def distribution_for_scene(self, retrieval_query: str, context_tags: list[str]) -> list[dict]:
        data = self._load()
        signatures = self.scene_signatures(retrieval_query, context_tags)
        by_action: dict[str, dict] = {}
        for level in ("exact", "semantic", "coarse"):
            signature = signatures[level]
            bucket = data.get("signatures", {}).get(level, {}).get(signature)
            if not bucket:
                continue
            match_weight = MATCH_LEVEL_WEIGHTS[level]
            for action, entry in bucket.get("actions", {}).items():
                item = self._distribution_item(action, entry, level, match_weight, signature)
                existing = by_action.get(action)
                if existing is None or item["weighted_behavior_reward"] > existing["weighted_behavior_reward"]:
                    by_action[action] = item
        distribution = list(by_action.values())
        distribution.sort(
            key=lambda item: (
                item["weighted_prior"],
                item["weighted_behavior_reward"],
                item["actual_count"],
                item["predicted_count"],
            ),
            reverse=True,
        )
        return distribution

    def scene_signature(self, retrieval_query: str, context_tags: list[str]) -> str:
        return self.scene_signatures(retrieval_query, context_tags)["exact"]

    def scene_signatures(self, retrieval_query: str, context_tags: list[str]) -> dict[str, str]:
        return scene_signatures(retrieval_query, context_tags)

    def _bucket(self, data: dict, level: str, signature: str, retrieval_query: str, context_tags: list[str]) -> dict:
        bucket = data.setdefault("signatures", {}).setdefault(level, {}).setdefault(
            signature,
            {
                "signature": signature,
                "match_level": level,
                "retrieval_query": retrieval_query,
                "context_tags": context_tags,
                "actions": {},
            },
        )
        bucket["retrieval_query"] = retrieval_query
        bucket["context_tags"] = context_tags
        return bucket

    def _update_bucket(
        self,
        bucket: dict,
        predicted_action: str,
        actual_action: str | None,
        behavior_reward: float,
    ) -> dict:
        predicted_entry = self._action_entry(bucket, predicted_action)
        predicted_entry["predicted_count"] += 1
        predicted_entry["total_behavior_reward"] += behavior_reward
        predicted_entry["average_behavior_reward"] = predicted_entry["total_behavior_reward"] / predicted_entry["predicted_count"]
        predicted_entry["behavior_reward_score"] = self._normalize_reward(predicted_entry["average_behavior_reward"])

        if actual_action:
            if actual_action == predicted_action:
                predicted_entry["correct_count"] += 1
            predicted_entry["actual_actions"][actual_action] = int(predicted_entry["actual_actions"].get(actual_action, 0)) + 1
            actual_entry = self._action_entry(bucket, actual_action)
            actual_entry["actual_count"] += 1
            actual_entry["actual_support_score"] = min(1.0, actual_entry["actual_count"] / 5.0)
            actual_entry["behavior_reward_score"] = max(
                float(actual_entry.get("behavior_reward_score", 0.5)),
                min(1.0, 0.5 + actual_entry["actual_support_score"] * 0.5),
            )
        return predicted_entry

    def _distribution_item(
        self,
        action: str,
        entry: dict,
        match_level: str,
        match_weight: float,
        signature: str,
    ) -> dict:
        predicted_count = int(entry.get("predicted_count", 0))
        actual_count = int(entry.get("actual_count", 0))
        behavior_reward_score = float(entry.get("behavior_reward_score", 0.5))
        actual_support_score = float(entry.get("actual_support_score", 0.0))
        weighted_behavior_reward = behavior_reward_score * match_weight
        prior = max(0.0, min(0.85, 0.15 + actual_support_score * 0.45 + behavior_reward_score * 0.25))
        return {
            "action": action,
            "predicted_count": predicted_count,
            "actual_count": actual_count,
            "correct_count": int(entry.get("correct_count", 0)),
            "average_behavior_reward": float(entry.get("average_behavior_reward", 0.0)),
            "behavior_reward_score": behavior_reward_score,
            "weighted_behavior_reward": round(weighted_behavior_reward, 6),
            "actual_support_score": actual_support_score,
            "prior": round(prior, 6),
            "weighted_prior": round(prior * match_weight, 6),
            "actual_actions": entry.get("actual_actions", {}),
            "match_level": match_level,
            "match_weight": match_weight,
            "signature": signature,
        }

    def _context_tokens(self, retrieval_query: str, context_tags: list[str]) -> list[str]:
        return context_tokens(retrieval_query, context_tags)

    def _hash_tokens(self, tokens: list[str]) -> str:
        return hash_tokens(tokens)

    def _is_exact_only_token(self, token: str) -> bool:
        return is_exact_only_token(token)

    def _is_coarse_token(self, token: str) -> bool:
        return is_coarse_token(token)

    def _action_entry(self, bucket: dict, action: str) -> dict:
        return bucket.setdefault("actions", {}).setdefault(
            action,
            {
                "predicted_count": 0,
                "actual_count": 0,
                "correct_count": 0,
                "total_behavior_reward": 0.0,
                "average_behavior_reward": 0.0,
                "behavior_reward_score": 0.5,
                "actual_support_score": 0.0,
                "actual_actions": {},
            },
        )

    def _behavior_reward(self, predicted_action: str, actual_action: str | None, fallback_reward: float) -> float:
        return max(-1.0, min(1.0, float(fallback_reward)))

    def _normalize_reward(self, reward: float) -> float:
        return max(0.0, min(1.0, (float(reward) + 1.0) / 2.0))

    def _tokens(self, text: str) -> list[str]:
        return text_tokens(text)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"signatures": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
