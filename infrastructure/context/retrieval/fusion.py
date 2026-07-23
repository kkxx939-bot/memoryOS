"""可解释的有界融合与语义去重。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from infrastructure.context.retrieval.query_plan import RetrievalQueryIntent, RetrievalQueryPlan


@dataclass(frozen=True)
class RetrievalScore:
    exact_score: float = 0.0
    lexical_score: float = 0.0
    vector_score: float = 0.0
    relation_score: float = 0.0
    recency_boost: float = 0.0
    hotness_boost: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            score = float(value)
            if not math.isfinite(score):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, max(0.0, min(1.0, score)))

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class RetrievalCandidate:
    record_key: str
    uri: str
    title: str
    context_type: str
    source_kind: str = ""
    record_kind: str = "context"
    text: str = ""
    l0_text: str = ""
    l1_text: str = ""
    l2_uri: str = ""
    source_uri: str = ""
    source_digest: str = ""
    tenant_id: str = ""
    owner_user_id: str = ""
    session_id: str = ""
    workspace_id: str = ""
    archive_digest: str = ""
    manifest_digest: str = ""
    event_time: str = ""
    hotness: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    branch_scores: Mapping[str, float] = field(default_factory=dict)
    branch_ranks: Mapping[str, int] = field(default_factory=dict)
    score: RetrievalScore = field(default_factory=RetrievalScore)

    def __post_init__(self) -> None:
        if not self.record_key or not self.uri:
            raise ValueError("retrieval candidate identity is required")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(
            self,
            "branch_scores",
            {str(key): _score(value) for key, value in self.branch_scores.items()},
        )
        ranks = {str(key): int(value) for key, value in self.branch_ranks.items()}
        if any(value < 1 for value in ranks.values()):
            raise ValueError("retrieval branch ranks must be positive")
        object.__setattr__(self, "branch_ranks", ranks)
        object.__setattr__(self, "hotness", _score(self.hotness))

    def with_branch(self, branch: str, score: float, rank: int) -> RetrievalCandidate:
        scores = dict(self.branch_scores)
        ranks = dict(self.branch_ranks)
        scores[branch] = max(scores.get(branch, 0.0), _score(score))
        ranks[branch] = min(ranks.get(branch, rank), rank)
        return replace(self, branch_scores=scores, branch_ranks=ranks)


class FusionRanker:
    """使用 RRF，并叠加依赖真实相关性的有界增益。

    词法、向量和关系原始分数只用于可观测性，不跨分支直接比较；核心融合分由各
    分支内的排名决定。
    """

    RRF_K = 60

    def fuse(
        self,
        branches: Mapping[str, Sequence[RetrievalCandidate]],
        *,
        plan: RetrievalQueryPlan,
        now: datetime | None = None,
    ) -> list[RetrievalCandidate]:
        if sum(len(values) for values in branches.values()) > plan.candidate_limit * max(1, len(branches)):
            raise ValueError("fusion input exceeds the bounded candidate plan")
        merged: dict[str, RetrievalCandidate] = {}
        for branch, candidates in branches.items():
            for rank, candidate in enumerate(candidates[: plan.candidate_limit], start=1):
                value = candidate.branch_scores.get(branch, candidate.score.final_score)
                existing = merged.get(candidate.record_key)
                merged[candidate.record_key] = (existing or candidate).with_branch(branch, value, rank)

        reference_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        ranked: list[RetrievalCandidate] = []
        for candidate in merged.values():
            rrf_terms = [1.0 / (self.RRF_K + rank) for rank in candidate.branch_ranks.values()]
            maximum_rrf = max(1, len(branches)) / (self.RRF_K + 1)
            rrf = sum(rrf_terms) / maximum_rrf if maximum_rrf else 0.0
            exact = candidate.branch_scores.get("exact", 0.0)
            lexical = candidate.branch_scores.get("lexical", 0.0)
            vector = candidate.branch_scores.get("vector", 0.0)
            relation = candidate.branch_scores.get("relation", 0.0)
            # RRF 只是跨分支排序信号，不属于语义证据；增益只能依赖真实分支相关分。
            base_relevance = max(exact, lexical, vector, relation)
            recency = self._recency(candidate.event_time, reference_now) * base_relevance * 0.08
            hotness = candidate.hotness * base_relevance * 0.05
            final = min(1.0, min(1.0, rrf) * 0.87 + exact * 0.05 + recency + hotness)
            ranked.append(
                replace(
                    candidate,
                    score=RetrievalScore(
                        exact_score=exact,
                        lexical_score=lexical,
                        vector_score=vector,
                        relation_score=relation,
                        recency_boost=recency,
                        hotness_boost=hotness,
                        final_score=final,
                    ),
                )
            )
        ranked.sort(key=lambda item: (-item.score.final_score, item.record_key))
        return self.dedupe(ranked, plan=plan)

    def apply_rerank(
        self,
        candidates: Sequence[RetrievalCandidate],
        scores: Mapping[str, float],
    ) -> list[RetrievalCandidate]:
        reranked = []
        for candidate in candidates:
            rerank_score = _score(scores.get(candidate.record_key, 0.0))
            final = min(1.0, candidate.score.final_score * 0.75 + rerank_score * 0.25)
            reranked.append(
                replace(
                    candidate,
                    score=replace(candidate.score, rerank_score=rerank_score, final_score=final),
                )
            )
        reranked.sort(key=lambda item: (-item.score.final_score, item.record_key))
        return reranked

    def dedupe(
        self,
        candidates: Iterable[RetrievalCandidate],
        *,
        plan: RetrievalQueryPlan,
        per_session_limit: int = 5,
    ) -> list[RetrievalCandidate]:
        seen: set[tuple[Any, ...]] = set()
        session_counts: dict[str, int] = {}
        result: list[RetrievalCandidate] = []
        for candidate in candidates:
            identity = self._identity(candidate, plan.query_intent)
            if identity in seen:
                continue
            if candidate.session_id:
                session_key = candidate.manifest_digest or candidate.archive_digest or candidate.session_id
                count = session_counts.get(session_key, 0)
                if count >= per_session_limit:
                    continue
                session_counts[session_key] = count + 1
            seen.add(identity)
            result.append(candidate)
            if len(result) >= plan.candidate_limit:
                break
        return result

    @staticmethod
    def _identity(candidate: RetrievalCandidate, intent: RetrievalQueryIntent) -> tuple[Any, ...]:
        del intent
        if candidate.source_kind in {"resource", "resource_reference"}:
            resource_uri = str(candidate.metadata.get("resource_uri") or candidate.source_uri)
            return ("resource", resource_uri, candidate.source_digest)
        if candidate.session_id:
            return (
                "session",
                candidate.manifest_digest
                or candidate.archive_digest
                or candidate.source_digest
                or candidate.record_key,
            )
        return ("context", candidate.source_digest or candidate.record_key)

    @staticmethod
    def _recency(value: str, now: datetime) -> float:
        if not value:
            return 0.0
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        if parsed.tzinfo is None:
            return 0.0
        age_days = max(0.0, (now - parsed.astimezone(timezone.utc)).total_seconds() / 86_400)
        return 1.0 / (1.0 + age_days / 30.0)


def _score(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


__all__ = ["FusionRanker", "RetrievalCandidate", "RetrievalScore"]
