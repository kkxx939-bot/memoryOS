from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from memoryos.infrastructure.providers.rerank_provider import RerankProvider, rerank_with_fallback
from memoryos.domain.memory.memory_item import slugify, utc_now
from .behavior_scoring import BehaviorEvidenceScorer
from memoryos.domain.behavior.signatures import MATCH_LEVEL_WEIGHTS, pattern_layered_token_sets, pattern_scene_signatures


class BehaviorPatternStore:
    def __init__(
        self,
        root: Path,
        *,
        merge_similarity_threshold: float = 0.56,
        active_evidence_limit: int = 120,
        rerank_provider: RerankProvider | None = None,
    ) -> None:
        self.root = root
        self.scorer = BehaviorEvidenceScorer()
        self.merge_similarity_threshold = merge_similarity_threshold
        self.active_evidence_limit = active_evidence_limit
        self.rerank_provider = rerank_provider

    def record(
        self,
        user_id: str,
        episode_id: str,
        retrieval_query: str,
        context_tags: list[str],
        predicted_action: str,
        actual_action: str,
        reward: float,
        created_at: str | None = None,
        predicted_candidates: list[dict] | None = None,
        action_params: dict | None = None,
        scene_features: dict | None = None,
        spontaneity: str = "unknown",
        intervention: str = "",
        intervention_result: str = "",
    ) -> dict:
        created_at = created_at or utc_now()
        domain = self._domain(context_tags, retrieval_query)
        signatures = pattern_scene_signatures(retrieval_query, context_tags)
        layers = pattern_layered_token_sets(retrieval_query, context_tags)
        pattern_path, group_id = self._resolve_pattern_path(
            user_id=user_id,
            domain=domain,
            layers=layers,
            fallback_group_id=signatures["semantic"],
            action=actual_action,
        )
        pattern = self._load_pattern(pattern_path) or self._new_pattern(
            domain=domain,
            group_id=group_id,
            semantic_signature=signatures["semantic"],
            action=actual_action,
            retrieval_query=retrieval_query,
            context_tags=context_tags,
        )
        evidence = {
            "episode_id": episode_id,
            "created_at": created_at,
            "retrieval_query": retrieval_query,
            "context_tags": context_tags,
            "scene_features": scene_features or {},
            "predicted_candidates": predicted_candidates or [],
            "predicted_action": predicted_action,
            "actual_action": actual_action,
            "action_params": action_params or {},
            "spontaneity": spontaneity,
            "intervention": intervention,
            "intervention_result": intervention_result,
            "reward": reward,
            "signatures": signatures,
        }
        self._append_evidence(pattern, evidence)
        self._recompute_pattern(pattern)
        pattern_path.parent.mkdir(parents=True, exist_ok=True)
        pattern_path.write_text(json.dumps(pattern, ensure_ascii=False, indent=2), encoding="utf-8")
        self._refresh_group(user_id, domain, pattern["group_id"])
        self._refresh_layers(user_id)
        return {
            "pattern_uri": str(pattern_path.relative_to(self.root).as_posix()),
            "group_uri": self._group_uri(user_id, domain, pattern["group_id"]),
            "domain": domain,
            "action": actual_action,
            "evidence_confidence": pattern["evidence_confidence"],
        }

    def distribution_for_scene(
        self,
        user_id: str,
        retrieval_query: str,
        context_tags: list[str],
        limit: int = 12,
    ) -> list[dict]:
        behavior_root = self.root / "user" / user_id / "behavior"
        if not behavior_root.exists():
            return []
        self._ensure_index(user_id)
        query_layers = pattern_layered_token_sets(retrieval_query, context_tags)
        domain = self._domain(context_tags, retrieval_query)
        items = []
        for row in self._index_rows(user_id, domain):
            match_level, similarity = self._best_layer_match(
                query_layers,
                {
                    "exact": set(row["exact_tokens"]),
                    "semantic": set(row["semantic_tokens"]),
                    "coarse": set(row["coarse_tokens"]),
                },
            )
            if similarity <= 0:
                continue
            match_weight = MATCH_LEVEL_WEIGHTS[match_level]
            confidence = float(row.get("evidence_confidence", 0.0))
            score = similarity * match_weight * confidence
            if score <= 0:
                continue
            pattern_path = self.root / str(row["pattern_uri"])
            pattern = self._load_pattern(pattern_path)
            if not pattern:
                continue
            items.append(
                {
                    "action": row["action"],
                    "source": "behavior_pattern",
                    "pattern_uri": str(row["pattern_uri"]),
                    "group_uri": str(row.get("group_uri", "")),
                    "domain": row["domain"],
                    "group_id": row["group_id"],
                    "sample_count": int(row.get("sample_count", 0)),
                    "distinct_days": int(row.get("distinct_days", 0)),
                    "average_reward": float(row.get("average_reward", 0.0)),
                    "prior": round(score, 6),
                    "evidence_confidence": confidence,
                    "prediction_coefficient": round(score, 6),
                    "match_level": match_level,
                    "match_weight": match_weight,
                    "similarity": round(similarity, 6),
                    "action_ratio": float(row.get("action_ratio", 0.0)),
                    "top_action_margin": float(row.get("top_action_margin", 0.0)),
                    "group_entropy": float(row.get("group_entropy", 0.0)),
                    "recent_7d_count": int(row.get("recent_7d_count", 0)),
                    "recent_30d_count": int(row.get("recent_30d_count", 0)),
                    "hotness": float(row.get("hotness", 0.0)),
                    "episodes": pattern.get("episodes", [])[-5:],
                    "action_distribution": self._load_group_distribution(str(row.get("group_uri", ""))),
                }
            )
        self._rerank_distribution(retrieval_query, items)
        items.sort(
            key=lambda item: (
                item["prediction_coefficient"],
                item["top_action_margin"],
                item["evidence_confidence"],
            ),
            reverse=True,
        )
        return items[:limit]

    def _rerank_distribution(self, query: str, items: list[dict]) -> None:
        if not items or self.rerank_provider is None:
            return
        documents = [self._rerank_document(item) for item in items]
        fallback_scores = [float(item.get("prediction_coefficient", item.get("prior", 0.0)) or 0.0) for item in items]
        rerank_scores = rerank_with_fallback(self.rerank_provider, query, documents, fallback_scores)
        for item, rerank_score, fallback in zip(items, rerank_scores, fallback_scores, strict=True):
            item["rerank_score"] = rerank_score
            blended = rerank_score * 0.60 + fallback * 0.40
            item["prediction_coefficient"] = round(max(0.0, min(1.0, blended)), 6)
            item["prior"] = item["prediction_coefficient"]

    def _rerank_document(self, item: dict) -> str:
        episodes = []
        for episode in item.get("episodes", [])[:5]:
            episodes.append(
                (
                    f"episode={episode.get('episode_id', '')}; "
                    f"query={episode.get('retrieval_query', '')}; "
                    f"predicted={episode.get('predicted_action', '')}; "
                    f"actual={episode.get('actual_action', '')}; "
                    f"reward={episode.get('reward', 0.0)}"
                )
            )
        return "\n".join(
            [
                f"behavior action: {item.get('action', '')}",
                f"domain: {item.get('domain', '')}",
                f"group: {item.get('group_id', '')}",
                f"sample_count: {item.get('sample_count', 0)}",
                f"distinct_days: {item.get('distinct_days', 0)}",
                f"confidence: {item.get('evidence_confidence', 0.0)}",
                f"episodes: {' | '.join(episodes)}",
            ]
        )

    def _new_pattern(
        self,
        domain: str,
        group_id: str,
        semantic_signature: str,
        action: str,
        retrieval_query: str,
        context_tags: list[str],
    ) -> dict:
        layers = pattern_layered_token_sets(retrieval_query, context_tags)
        return {
            "domain": domain,
            "group_id": group_id,
            "semantic_signature": semantic_signature,
            "action": action,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "context_tags": context_tags,
            "exact_tokens": sorted(layers["exact"]),
            "semantic_tokens": sorted(layers["semantic"]),
            "coarse_tokens": sorted(layers["coarse"]),
            "episodes": [],
            "old_evidence_summary": self._empty_old_summary(),
            "sample_count": 0,
            "distinct_days": 0,
            "average_reward": 0.0,
            "average_support": 0.0,
            "action_ratio": 1.0,
            "top_action_margin": 1.0,
            "group_entropy": 0.0,
            "recent_7d_count": 0,
            "recent_30d_count": 0,
            "hotness": 0.0,
            "evidence_confidence": 0.0,
        }

    def _append_evidence(self, pattern: dict, evidence: dict) -> None:
        episodes = pattern.setdefault("episodes", [])
        if any(item.get("episode_id") == evidence["episode_id"] for item in episodes):
            return
        episodes.append(evidence)
        self._compact_evidence(pattern)
        pattern["updated_at"] = evidence["created_at"]

    def _recompute_pattern(self, pattern: dict) -> None:
        episodes = pattern.get("episodes", [])
        old = pattern.get("old_evidence_summary", {})
        sample_count = len(episodes) + int(old.get("sample_count", 0))
        if sample_count == 0:
            return
        days = {self._day(item.get("created_at")) for item in episodes}
        days.update(str(day) for day in old.get("distinct_days", []))
        total_reward = sum(float(item.get("reward", 0.0)) for item in episodes) + float(old.get("total_reward", 0.0))
        average_reward = total_reward / sample_count
        total_support = sum(
            self.scorer.episode_support(
                weighted_similarity=1.0,
                reward=float(item.get("reward", 0.0)),
                created_at=item.get("created_at"),
            )
            for item in episodes
        ) + float(old.get("total_support", 0.0))
        average_support = total_support / sample_count
        recent_7d_count = self._recent_count(episodes, 7)
        recent_30d_count = self._recent_count(episodes, 30) + int(old.get("recent_30d_count", 0))
        consistency = float(pattern.get("action_ratio", 1.0))
        evidence_confidence = self.scorer.evidence_confidence(
            consistency=consistency,
            average_support=average_support,
            sample_count=sample_count,
            distinct_days=len(days),
            average_reward=average_reward,
        )
        pattern["sample_count"] = sample_count
        pattern["distinct_days"] = len(days)
        pattern["average_reward"] = average_reward
        pattern["average_support"] = average_support
        pattern["sample_strength"] = self.scorer.sample_strength(sample_count)
        pattern["diversity_strength"] = self.scorer.diversity_strength(len(days))
        pattern["recent_7d_count"] = recent_7d_count
        pattern["recent_30d_count"] = recent_30d_count
        pattern["hotness"] = self._pattern_hotness(evidence_confidence, recent_7d_count, recent_30d_count)
        pattern["evidence_confidence"] = evidence_confidence

    def _refresh_group(self, user_id: str, domain: str, group_id: str) -> None:
        group_path = self._group_path(user_id, domain, group_id)
        patterns = self._group_patterns(user_id, domain, group_id)
        total_samples = sum(int(pattern.get("sample_count", 0)) for pattern in patterns)
        action_distribution = []
        for pattern in patterns:
            sample_count = int(pattern.get("sample_count", 0))
            ratio = sample_count / total_samples if total_samples else 0.0
            action_distribution.append(
                {
                    "action": pattern.get("action"),
                    "sample_count": sample_count,
                    "probability": round(ratio, 6),
                    "ratio": round(ratio, 6),
                    "avg_reward": round(float(pattern.get("average_reward", 0.0)), 6),
                    "average_reward": round(float(pattern.get("average_reward", 0.0)), 6),
                    "evidence_confidence": float(pattern.get("evidence_confidence", 0.0)),
                    "confidence": float(pattern.get("evidence_confidence", 0.0)),
                    "negative_count": self._negative_count(pattern),
                    "last_seen_at": str(pattern.get("updated_at", "")),
                    "recency_weight": self._recency_weight(str(pattern.get("updated_at", ""))),
                    "param_distribution": self._param_distribution(pattern),
                    "spontaneity_distribution": self._field_distribution(pattern, "spontaneity"),
                }
            )
        action_distribution.sort(key=lambda item: item["sample_count"], reverse=True)
        entropy = self._entropy([item["ratio"] for item in action_distribution])
        top_ratio = float(action_distribution[0]["ratio"]) if action_distribution else 0.0
        second_ratio = float(action_distribution[1]["ratio"]) if len(action_distribution) > 1 else 0.0
        top_action_margin = round(top_ratio - second_ratio, 6)
        distinct_days = sorted(
            {
                self._day(evidence.get("created_at"))
                for pattern in patterns
                for evidence in pattern.get("episodes", [])
            }
        )
        group = {
            "group_id": group_id,
            "domain": domain,
            "updated_at": utc_now(),
            "total_samples": total_samples,
            "distinct_days": len(distinct_days),
            "group_entropy": entropy,
            "top_action": action_distribution[0]["action"] if action_distribution else "",
            "top_action_ratio": top_ratio,
            "top_action_margin": top_action_margin,
            "conflict_level": self._conflict_level(entropy, top_action_margin),
            "action_distribution": action_distribution,
            "negative_actions": self._negative_actions(action_distribution),
            "patterns": [
                {
                    "action": pattern.get("action"),
                    "pattern_uri": self._pattern_uri(user_id, domain, group_id, str(pattern.get("action", ""))),
                    "sample_count": int(pattern.get("sample_count", 0)),
                }
                for pattern in patterns
            ],
        }
        group_path.parent.mkdir(parents=True, exist_ok=True)
        group_path.write_text(json.dumps(group, ensure_ascii=False, indent=2), encoding="utf-8")
        for pattern in patterns:
            sample_count = int(pattern.get("sample_count", 0))
            pattern["action_ratio"] = round(sample_count / total_samples, 6) if total_samples else 0.0
            pattern["top_action_margin"] = top_action_margin
            pattern["group_entropy"] = entropy
            pattern["group_uri"] = str(group_path.relative_to(self.root).as_posix())
            self._recompute_pattern(pattern)
            path = self._pattern_path(user_id, domain, group_id, str(pattern.get("action", "")))
            path.write_text(json.dumps(pattern, ensure_ascii=False, indent=2), encoding="utf-8")
            self._index_pattern(user_id, path, pattern)

    def _load_group_distribution(self, group_uri: str) -> list[dict]:
        if not group_uri:
            return []
        path = self.root / group_uri
        group = self._load_pattern(path)
        if not group:
            return []
        return list(group.get("action_distribution", []))

    def _negative_count(self, pattern: dict) -> int:
        episodes = pattern.get("episodes", [])
        old = pattern.get("old_evidence_summary", {})
        return sum(1 for episode in episodes if float(episode.get("reward", 0.0)) < 0) + int(old.get("negative_count", 0))

    def _param_distribution(self, pattern: dict) -> dict:
        counts: dict[str, dict[str, int]] = {}
        for episode in pattern.get("episodes", []):
            params = episode.get("action_params", {})
            if not isinstance(params, dict):
                continue
            for key, value in params.items():
                bucket = counts.setdefault(str(key), {})
                bucket[str(value)] = bucket.get(str(value), 0) + 1
        distribution = {}
        for key, values in counts.items():
            total = sum(values.values())
            if total <= 0:
                continue
            distribution[key] = {
                value: round(count / total, 6)
                for value, count in sorted(values.items(), key=lambda item: item[1], reverse=True)
            }
        return distribution

    def _field_distribution(self, pattern: dict, field_name: str) -> dict:
        counts: dict[str, int] = {}
        for episode in pattern.get("episodes", []):
            value = str(episode.get(field_name, "") or "unknown")
            counts[value] = counts.get(value, 0) + 1
        total = sum(counts.values())
        if total <= 0:
            return {}
        return {
            value: round(count / total, 6)
            for value, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        }

    def _negative_actions(self, action_distribution: list[dict]) -> dict:
        negative = {}
        for item in action_distribution:
            negative_count = int(item.get("negative_count", 0))
            avg_reward = float(item.get("avg_reward", item.get("average_reward", 0.0)))
            if negative_count <= 0 and avg_reward >= 0:
                continue
            action = str(item.get("action", ""))
            negative[action] = {
                "negative_count": negative_count,
                "avg_reward": avg_reward,
                "confidence": float(item.get("confidence", 0.0)),
            }
        return negative

    def _recency_weight(self, value: str) -> float:
        if not value:
            return 0.0
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age_days = max((datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0, 0.0)
        return round(max(0.0, min(1.0, 1.0 / (1.0 + age_days / 14.0))), 6)

    def _refresh_layers(self, user_id: str) -> None:
        behavior_root = self.root / "user" / user_id / "behavior"
        domains = [path for path in behavior_root.iterdir() if path.is_dir()] if behavior_root.exists() else []
        root_lines = ["# Behavior Patterns", ""]
        for domain_path in sorted(domains):
            patterns = self._domain_patterns(domain_path)
            root_lines.append(f"- {domain_path.name}: {len(patterns)} patterns")
            self._write_domain_layers(domain_path, patterns)
        behavior_root.mkdir(parents=True, exist_ok=True)
        (behavior_root / ".abstract.md").write_text("\n".join(root_lines).strip() + "\n", encoding="utf-8")
        (behavior_root / ".overview.md").write_text("\n".join(root_lines).strip() + "\n", encoding="utf-8")

    def _write_domain_layers(self, domain_path: Path, patterns: list[dict]) -> None:
        patterns.sort(key=lambda item: float(item.get("evidence_confidence", 0.0)), reverse=True)
        lines = [f"# {domain_path.name}", "", f"- pattern_count: {len(patterns)}", "", "## Top patterns"]
        for pattern in patterns[:8]:
            lines.append(
                (
                    f"- action={pattern.get('action')} confidence={float(pattern.get('evidence_confidence', 0.0)):.3f} "
                    f"samples={pattern.get('sample_count', 0)} days={pattern.get('distinct_days', 0)} "
                    f"ratio={float(pattern.get('action_ratio', 0.0)):.3f}"
                )
            )
        text = "\n".join(lines).strip() + "\n"
        (domain_path / ".abstract.md").write_text(text, encoding="utf-8")
        (domain_path / ".overview.md").write_text(text, encoding="utf-8")

    def _domain_patterns(self, domain_path: Path) -> list[dict]:
        patterns = []
        for path in (domain_path / "patterns").glob("*.json"):
            pattern = self._load_pattern(path)
            if pattern:
                patterns.append(pattern)
        return patterns

    def _load_pattern(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _pattern_path(self, user_id: str, domain: str, group_id: str, action: str) -> Path:
        filename = f"{group_id}-{slugify(action)}.json"
        return self.root / "user" / user_id / "behavior" / domain / "patterns" / filename

    def _pattern_uri(self, user_id: str, domain: str, group_id: str, action: str) -> str:
        return str(self._pattern_path(user_id, domain, group_id, action).relative_to(self.root).as_posix())

    def _group_path(self, user_id: str, domain: str, group_id: str) -> Path:
        return self.root / "user" / user_id / "behavior" / domain / "groups" / f"{group_id}.json"

    def _group_uri(self, user_id: str, domain: str, group_id: str) -> str:
        return str(self._group_path(user_id, domain, group_id).relative_to(self.root).as_posix())

    def _resolve_pattern_path(
        self,
        *,
        user_id: str,
        domain: str,
        layers: dict[str, set[str]],
        fallback_group_id: str,
        action: str,
    ) -> tuple[Path, str]:
        self._ensure_index(user_id)
        best_group_id = ""
        best_score = 0.0
        for row in self._index_rows(user_id, domain):
            match_level, similarity = self._best_layer_match(
                layers,
                {
                    "exact": set(row["exact_tokens"]),
                    "semantic": set(row["semantic_tokens"]),
                    "coarse": set(row["coarse_tokens"]),
                },
            )
            score = similarity * MATCH_LEVEL_WEIGHTS[match_level]
            if score > best_score:
                best_group_id = str(row.get("group_id", ""))
                best_score = score
        group_id = best_group_id if best_score >= self.merge_similarity_threshold else fallback_group_id
        return self._pattern_path(user_id, domain, group_id, action), group_id

    def _group_patterns(self, user_id: str, domain: str, group_id: str) -> list[dict]:
        patterns = []
        pattern_dir = self.root / "user" / user_id / "behavior" / domain / "patterns"
        for path in pattern_dir.glob(f"{group_id}-*.json"):
            pattern = self._load_pattern(path)
            if pattern:
                patterns.append(pattern)
        return patterns

    def _empty_old_summary(self) -> dict:
        return {
            "sample_count": 0,
            "total_reward": 0.0,
            "total_support": 0.0,
            "negative_count": 0,
            "distinct_days": [],
            "first_seen": "",
            "last_seen": "",
            "recent_30d_count": 0,
        }

    def _compact_evidence(self, pattern: dict) -> None:
        episodes = pattern.setdefault("episodes", [])
        if len(episodes) <= self.active_evidence_limit:
            return
        episodes.sort(key=lambda item: str(item.get("created_at", "")))
        overflow = episodes[: len(episodes) - self.active_evidence_limit]
        pattern["episodes"] = episodes[len(overflow) :]
        summary = pattern.setdefault("old_evidence_summary", self._empty_old_summary())
        days = set(str(day) for day in summary.get("distinct_days", []))
        first_seen = str(summary.get("first_seen", ""))
        last_seen = str(summary.get("last_seen", ""))
        for item in overflow:
            created_at = str(item.get("created_at", ""))
            summary["sample_count"] = int(summary.get("sample_count", 0)) + 1
            summary["total_reward"] = float(summary.get("total_reward", 0.0)) + float(item.get("reward", 0.0))
            summary["total_support"] = float(summary.get("total_support", 0.0)) + self.scorer.episode_support(
                weighted_similarity=1.0,
                reward=float(item.get("reward", 0.0)),
                created_at=created_at,
            )
            if float(item.get("reward", 0.0)) < 0:
                summary["negative_count"] = int(summary.get("negative_count", 0)) + 1
            days.add(self._day(created_at))
            if not first_seen or created_at < first_seen:
                first_seen = created_at
            if not last_seen or created_at > last_seen:
                last_seen = created_at
        summary["distinct_days"] = sorted(days)
        summary["first_seen"] = first_seen
        summary["last_seen"] = last_seen

    def _recent_count(self, episodes: list[dict], window_days: int) -> int:
        now = datetime.now(timezone.utc)
        count = 0
        for item in episodes:
            created_at = item.get("created_at")
            if not created_at:
                continue
            try:
                parsed = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age_days = max((now - parsed).total_seconds() / 86400.0, 0.0)
            if age_days <= window_days:
                count += 1
        return count

    def _pattern_hotness(self, confidence: float, recent_7d_count: int, recent_30d_count: int) -> float:
        recent_7d = self.scorer.sample_strength(recent_7d_count)
        recent_30d = self.scorer.sample_strength(recent_30d_count)
        return round(max(0.0, min(1.0, confidence * 0.60 + recent_7d * 0.25 + recent_30d * 0.15)), 6)

    def _entropy(self, ratios: list[float]) -> float:
        value = -sum(ratio * math.log(ratio, 2) for ratio in ratios if ratio > 0)
        max_entropy = math.log(len(ratios), 2) if len(ratios) > 1 else 1.0
        return round(value / max_entropy if max_entropy else 0.0, 6)

    def _conflict_level(self, entropy: float, top_action_margin: float) -> str:
        if entropy >= 0.75 and top_action_margin < 0.25:
            return "high"
        if entropy >= 0.45 and top_action_margin < 0.45:
            return "medium"
        return "low"

    def _index_path(self, user_id: str) -> Path:
        return self.root / "user" / user_id / "behavior" / ".pattern_index.sqlite"

    def _connect_index(self, user_id: str) -> sqlite3.Connection:
        path = self._index_path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_index(self, user_id: str) -> None:
        with self._connect_index(user_id) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS behavior_patterns (
                    pattern_uri TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    group_uri TEXT NOT NULL,
                    action TEXT NOT NULL,
                    exact_tokens TEXT NOT NULL,
                    semantic_tokens TEXT NOT NULL,
                    coarse_tokens TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    distinct_days INTEGER NOT NULL,
                    average_reward REAL NOT NULL,
                    evidence_confidence REAL NOT NULL,
                    action_ratio REAL NOT NULL,
                    top_action_margin REAL NOT NULL,
                    group_entropy REAL NOT NULL,
                    recent_7d_count INTEGER NOT NULL,
                    recent_30d_count INTEGER NOT NULL,
                    hotness REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            count = conn.execute("SELECT COUNT(*) FROM behavior_patterns WHERE user_id = ?", (user_id,)).fetchone()[0]
        if count == 0:
            self._rebuild_index(user_id)

    def _rebuild_index(self, user_id: str) -> None:
        behavior_root = self.root / "user" / user_id / "behavior"
        if not behavior_root.exists():
            return
        with self._connect_index(user_id) as conn:
            conn.execute("DELETE FROM behavior_patterns WHERE user_id = ?", (user_id,))
        for pattern_path in behavior_root.glob("*/patterns/*.json"):
            pattern = self._load_pattern(pattern_path)
            if pattern:
                self._index_pattern(user_id, pattern_path, pattern)

    def _index_rows(self, user_id: str, domain: str) -> list[dict]:
        self._ensure_index(user_id)
        with self._connect_index(user_id) as conn:
            rows = conn.execute(
                """
                SELECT * FROM behavior_patterns
                WHERE user_id = ? AND domain = ?
                ORDER BY hotness DESC, evidence_confidence DESC, sample_count DESC
                """,
                (user_id, domain),
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    """
                    SELECT * FROM behavior_patterns
                    WHERE user_id = ?
                    ORDER BY hotness DESC, evidence_confidence DESC, sample_count DESC
                    """,
                    (user_id,),
                ).fetchall()
        return [self._decode_index_row(row) for row in rows]

    def _decode_index_row(self, row: sqlite3.Row) -> dict:
        item = dict(row)
        item["exact_tokens"] = json.loads(str(item.get("exact_tokens", "[]")))
        item["semantic_tokens"] = json.loads(str(item.get("semantic_tokens", "[]")))
        item["coarse_tokens"] = json.loads(str(item.get("coarse_tokens", "[]")))
        return item

    def _index_pattern(self, user_id: str, pattern_path: Path, pattern: dict) -> None:
        pattern_uri = str(pattern_path.relative_to(self.root).as_posix())
        group_id = str(pattern.get("group_id", pattern.get("semantic_signature", "")))
        group_uri = str(pattern.get("group_uri") or self._group_uri(user_id, str(pattern.get("domain", "general")), group_id))
        with self._connect_index(user_id) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO behavior_patterns (
                    pattern_uri, user_id, domain, group_id, group_uri, action,
                    exact_tokens, semantic_tokens, coarse_tokens,
                    sample_count, distinct_days, average_reward, evidence_confidence,
                    action_ratio, top_action_margin, group_entropy,
                    recent_7d_count, recent_30d_count, hotness, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pattern_uri,
                    user_id,
                    str(pattern.get("domain", "")),
                    group_id,
                    group_uri,
                    str(pattern.get("action", "")),
                    json.dumps(pattern.get("exact_tokens", []), ensure_ascii=False),
                    json.dumps(pattern.get("semantic_tokens", []), ensure_ascii=False),
                    json.dumps(pattern.get("coarse_tokens", []), ensure_ascii=False),
                    int(pattern.get("sample_count", 0)),
                    int(pattern.get("distinct_days", 0)),
                    float(pattern.get("average_reward", 0.0)),
                    float(pattern.get("evidence_confidence", 0.0)),
                    float(pattern.get("action_ratio", 0.0)),
                    float(pattern.get("top_action_margin", 0.0)),
                    float(pattern.get("group_entropy", 0.0)),
                    int(pattern.get("recent_7d_count", 0)),
                    int(pattern.get("recent_30d_count", 0)),
                    float(pattern.get("hotness", 0.0)),
                    str(pattern.get("updated_at", "")),
                ),
            )

    def _domain(self, context_tags: list[str], retrieval_query: str) -> str:
        for tag in context_tags:
            value = str(tag).strip().lower().replace(" ", "_")
            if value and not value.startswith(("temperature_", "humidity_", "duration_")):
                return slugify(value)
        tokens = pattern_layered_token_sets(retrieval_query, [])["coarse"]
        return slugify(sorted(tokens)[0]) if tokens else "general"

    def _best_layer_match(self, left: dict[str, set[str]], right: dict[str, set[str]]) -> tuple[str, float]:
        if left["exact"] and left["exact"] == right["exact"]:
            return "exact", 1.0
        best_level = "semantic"
        best_score = 0.0
        for level in ("semantic", "coarse"):
            score = self._jaccard(left[level], right[level])
            weighted_score = score * MATCH_LEVEL_WEIGHTS[level]
            if weighted_score > best_score:
                best_level = level
                best_score = weighted_score
        return best_level, self._jaccard(left[best_level], right[best_level])

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _day(self, value: str | None) -> str:
        if not value:
            return datetime.now(timezone.utc).date().isoformat()
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.date().isoformat()
