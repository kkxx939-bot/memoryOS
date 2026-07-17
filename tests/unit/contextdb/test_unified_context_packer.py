from __future__ import annotations

from typing import Any

from memoryos.contextdb.retrieval.fusion import RetrievalCandidate, RetrievalScore
from memoryos.contextdb.retrieval.packing import ContextPacker
from memoryos.contextdb.retrieval.query_plan import RetrievalOptions, RetrievalQueryIntent, RetrievalQueryPlan


def _plan(**kwargs: Any) -> RetrievalQueryPlan:
    options = RetrievalOptions(**kwargs)
    return RetrievalQueryPlan.from_dict({"semantic_query": "query", **options.to_dict()})


def _candidate(key: str, **kwargs: Any) -> RetrievalCandidate:
    context_type = str(kwargs.pop("context_type", "session"))
    return RetrievalCandidate(
        record_key=key,
        uri=f"memoryos://user/u1/context/{key}",
        title=key,
        context_type=context_type,
        source_uri="memoryos://user/u1/sessions/history/s1",
        source_digest=f"digest-{key}",
        score=RetrievalScore(final_score=0.8),
        **kwargs,
    )


def test_layer_degrades_l2_l1_l0_uri_and_reports_required_fields() -> None:
    candidates = [
        _candidate("full", text="x" * 20, l1_text="overview", l0_text="abstract"),
        _candidate("overview", text="x" * 4_000, l1_text="short overview", l0_text="abstract"),
    ]

    packed = ContextPacker().pack(candidates, plan=_plan(token_budget=20, final_limit=2))

    assert packed["contexts"]
    assert packed["contexts"][0]["selected_layer"] in {"L2", "L1", "L0", "URI"}
    for item in [*packed["contexts"], *packed["dropped_contexts"]]:
        assert {
            "source_uri",
            "token_estimate",
            "canonical_validation_status",
            "projection_lag",
            "degraded_mode",
        }.issubset(item)


def test_packer_enforces_slot_session_and_resource_quotas() -> None:
    candidates = [
        _candidate(
            f"slot-{index}",
            context_type="memory",
            record_kind="current_slot",
            canonical_slot_id="same-slot",
            l0_text="memory",
        )
        for index in range(2)
    ] + [_candidate(f"session-{index}", session_id="s1", l0_text="session") for index in range(7)]

    packed = ContextPacker().pack(candidates, plan=_plan(token_budget=1_000, final_limit=20))

    selected = packed["contexts"]
    assert sum(item["metadata"].get("source_digest", "").startswith("digest-slot") for item in selected) == 1
    assert sum(item["record_key"].startswith("session-") for item in selected) <= 5
    assert {item["drop_reason"] for item in packed["dropped_contexts"]} >= {"slot_quota", "session_quota"}


def test_history_deduplicates_claim_revision_without_collapsing_slot_history() -> None:
    candidates = [
        _candidate(
            "claim-a-rev-1",
            context_type="memory",
            record_kind="claim_revision",
            canonical_slot_id="same-slot",
            canonical_claim_id="claim-a",
            canonical_revision=1,
            l0_text="first state",
        ),
        _candidate(
            "claim-b-rev-1",
            context_type="memory",
            record_kind="claim_revision",
            canonical_slot_id="same-slot",
            canonical_claim_id="claim-b",
            canonical_revision=1,
            l0_text="second state",
        ),
        _candidate(
            "claim-b-rev-1-duplicate",
            context_type="memory",
            record_kind="claim_revision",
            canonical_slot_id="same-slot",
            canonical_claim_id="claim-b",
            canonical_revision=1,
            l0_text="duplicate projection path",
        ),
    ]

    packed = ContextPacker().pack(
        candidates,
        plan=_plan(query_intent=RetrievalQueryIntent.HISTORY, token_budget=1_000, final_limit=10),
    )

    assert [item["record_key"] for item in packed["contexts"]] == [
        "claim-a-rev-1",
        "claim-b-rev-1",
    ]
    assert [item["drop_reason"] for item in packed["dropped_contexts"]] == ["claim_revision_quota"]


def test_coding_agent_priority_uses_record_and_source_kinds() -> None:
    candidates = [
        _candidate(
            "related-session",
            record_kind="session_root",
            session_id="older-session",
            l0_text="older related session",
        ),
        _candidate(
            "experience",
            context_type="memory",
            record_kind="current_slot",
            canonical_slot_id="experience-slot",
            metadata={"memory_type": "agent_experience"},
            l0_text="reusable experience",
        ),
        _candidate(
            "current-session",
            record_kind="session_root",
            session_id="current-session",
            l0_text="current session",
        ),
        _candidate(
            "resource",
            context_type="resource",
            source_kind="resource_reference",
            l0_text="repository source",
        ),
        _candidate(
            "decision",
            context_type="memory",
            record_kind="current_slot",
            canonical_slot_id="decision-slot",
            metadata={"memory_type": "project_decision"},
            l0_text="project decision",
        ),
        _candidate(
            "rule",
            context_type="memory",
            record_kind="current_slot",
            canonical_slot_id="rule-slot",
            metadata={"memory_type": "project_rule"},
            l0_text="project rule",
        ),
    ]

    packed = ContextPacker().pack(
        candidates,
        plan=_plan(
            source_kinds=("coding_agent",),
            session_ids=("current-session",),
            token_budget=1_000,
            final_limit=10,
        ),
    )

    assert [item["record_key"] for item in packed["contexts"]] == [
        "rule",
        "decision",
        "resource",
        "current-session",
        "experience",
        "related-session",
    ]
