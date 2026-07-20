from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from infrastructure.context.retrieval.fusion import FusionRanker, RetrievalCandidate
from infrastructure.context.retrieval.query_plan import RetrievalOptions, RetrievalQueryPlan


def _plan(**kwargs: Any) -> RetrievalQueryPlan:
    options = RetrievalOptions(**kwargs)
    return RetrievalQueryPlan.from_dict({"semantic_query": "ice cream", **options.to_dict()})


def _candidate(key: str, **kwargs: Any) -> RetrievalCandidate:
    return RetrievalCandidate(
        record_key=key,
        uri=f"memoryos://user/u1/context/{key}",
        title=key,
        context_type="memory",
        **kwargs,
    )


def test_rrf_does_not_compare_raw_branch_scores_and_records_components() -> None:
    lexical = _candidate("both", branch_scores={"lexical": 0.2})
    vector = _candidate("both", branch_scores={"vector": 0.99})
    relation = _candidate("relation", branch_scores={"relation": 1.0})

    results = FusionRanker().fuse(
        {"lexical": [lexical], "vector": [vector], "relation": [relation]},
        plan=_plan(),
        now=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )

    assert results[0].record_key == "both"
    assert results[0].score.lexical_score == 0.2
    assert results[0].score.vector_score == 0.99
    assert 0.0 <= results[0].score.final_score <= 1.0


def test_current_dedupes_live_document_generation_and_enforces_per_session_limit() -> None:
    candidates = [
        _candidate(
            f"document-projection-{index}",
            tenant_id="tenant-a",
            owner_user_id="u1",
            document_id="document-1",
            document_kind="preferences",
            projection_generation=3,
            branch_scores={"lexical": 1.0 - index / 10},
        )
        for index in range(2)
    ] + [
        _candidate(
            f"session-{index}",
            session_id="session-1",
            source_digest=f"digest-{index}",
            branch_scores={"lexical": 0.8},
        )
        for index in range(8)
    ]

    results = FusionRanker().fuse({"lexical": candidates}, plan=_plan(candidate_limit=20))

    assert sum(item.document_id == "document-1" for item in results) == 1
    assert sum(item.session_id == "session-1" for item in results) == 5


def test_recency_and_hotness_cannot_promote_irrelevant_candidate() -> None:
    relevant = _candidate("relevant", branch_scores={"lexical": 1.0})
    irrelevant = _candidate(
        "irrelevant",
        event_time="2026-07-14T00:00:00+00:00",
        hotness=1.0,
        branch_scores={"lexical": 0.0},
    )

    results = FusionRanker().fuse(
        {"lexical": [relevant, irrelevant]},
        plan=_plan(),
        now=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )

    assert results[0].record_key == "relevant"
    assert results[1].score.recency_boost == 0.0
    assert results[1].score.hotness_boost == 0.0
