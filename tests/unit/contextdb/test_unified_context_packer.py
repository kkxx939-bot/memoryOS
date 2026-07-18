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
            "source_validation_status",
            "projection_lag",
            "degraded_mode",
        }.issubset(item)


def test_packer_enforces_document_and_session_quotas() -> None:
    candidates = [
        _candidate(
            f"document-{index}",
            context_type="memory",
            record_kind="memory_document",
            tenant_id="tenant-a",
            owner_user_id="u1",
            document_id="same-document",
            document_kind="preferences",
            projection_generation=1,
            l0_text="memory",
        )
        for index in range(4)
    ] + [_candidate(f"session-{index}", session_id="s1", l0_text="session") for index in range(7)]

    packed = ContextPacker().pack(candidates, plan=_plan(token_budget=1_000, final_limit=20))

    selected = packed["contexts"]
    assert sum(item["record_key"].startswith("document-") for item in selected) == 3
    assert sum(item["record_key"].startswith("session-") for item in selected) <= 5
    assert {item["drop_reason"] for item in packed["dropped_contexts"]} >= {
        "document_quota",
        "session_quota",
    }


def test_history_applies_block_quota_per_document_without_collapsing_other_documents() -> None:
    candidates = [
        _candidate(
            f"first-document-block-{index}",
            context_type="memory",
            record_kind="memory_block",
            tenant_id="tenant-a",
            owner_user_id="u1",
            document_id="document-a",
            block_id=f"block-{index}",
            document_kind="topic",
            projection_generation=2,
            l0_text=f"first document block {index}",
        )
        for index in range(4)
    ] + [
        _candidate(
            "second-document-block",
            context_type="memory",
            record_kind="memory_block",
            tenant_id="tenant-a",
            owner_user_id="u1",
            document_id="document-b",
            block_id="block-0",
            document_kind="topic",
            projection_generation=1,
            l0_text="second document block",
        )
    ]

    packed = ContextPacker().pack(
        candidates,
        plan=_plan(query_intent=RetrievalQueryIntent.HISTORY, token_budget=1_000, final_limit=20),
    )

    assert [item["record_key"] for item in packed["contexts"]] == [
        "first-document-block-0",
        "first-document-block-1",
        "first-document-block-2",
        "second-document-block",
    ]
    assert [item["drop_reason"] for item in packed["dropped_contexts"]] == ["document_quota"]


def test_coding_agent_priority_uses_record_and_source_kinds() -> None:
    candidates = [
        _candidate(
            "related-session",
            record_kind="session_root",
            session_id="older-session",
            l0_text="older related session",
        ),
        _candidate(
            "preference",
            context_type="memory",
            record_kind="memory_document",
            document_id="preference-document",
            document_kind="preferences",
            l0_text="user preferences",
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
            "topic",
            context_type="memory",
            record_kind="memory_block",
            document_id="topic-document",
            block_id="topic-block",
            document_kind="topic",
            l0_text="project knowledge",
        ),
        _candidate(
            "experience",
            context_type="memory",
            record_kind="memory_document",
            document_id="experience-document",
            document_kind="experience",
            l0_text="reusable experience",
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
        "preference",
        "topic",
        "resource",
        "current-session",
        "experience",
        "related-session",
    ]
