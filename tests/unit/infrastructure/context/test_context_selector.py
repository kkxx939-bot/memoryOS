"""上下文候选选择策略测试。"""

from __future__ import annotations

from typing import Any

from infrastructure.context.retrieval.fusion import RetrievalCandidate, RetrievalScore
from infrastructure.context.retrieval.query_plan import RetrievalOptions, RetrievalQueryPlan
from infrastructure.context.selection import ContextSelector


def _plan(**kwargs: Any) -> RetrievalQueryPlan:
    options = RetrievalOptions(**kwargs)
    return RetrievalQueryPlan.from_dict({"semantic_query": "query", **options.to_dict()})


def _candidate(key: str, **kwargs: Any) -> RetrievalCandidate:
    return RetrievalCandidate(
        record_key=key,
        uri=f"memoryos://user/u1/context/{key}",
        title=key,
        context_type=str(kwargs.pop("context_type", "session")),
        source_uri="memoryos://user/u1/sessions/history/s1",
        source_digest=f"digest-{key}",
        score=RetrievalScore(final_score=0.8),
        **kwargs,
    )


def test_selector_uses_count_limit_without_token_fields() -> None:
    result = ContextSelector().select(
        [
            _candidate("first", text="full", l1_text="overview", l0_text="abstract"),
            _candidate("second", l1_text="overview", l0_text="abstract"),
        ],
        plan=_plan(final_limit=1),
    )

    assert len(result["contexts"]) == 1
    assert result["dropped_contexts"][0]["drop_reason"] == "final_limit"
    assert not any("token" in key for key in result["contexts"][0])
    assert not any("token" in key for key in result)


def test_selector_enforces_document_and_session_quotas() -> None:
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

    result = ContextSelector().select(candidates, plan=_plan(final_limit=20))

    assert sum(item["record_key"].startswith("document-") for item in result["contexts"]) == 3
    assert sum(item["record_key"].startswith("session-") for item in result["contexts"]) <= 5
    assert {item["drop_reason"] for item in result["dropped_contexts"]} >= {
        "document_quota",
        "session_quota",
    }


def test_selector_preserves_coding_agent_priority() -> None:
    candidates = [
        _candidate("related-session", record_kind="session_root", session_id="older-session", l0_text="old"),
        _candidate(
            "preference",
            context_type="memory",
            record_kind="memory_document",
            document_id="preference-document",
            document_kind="preferences",
            l0_text="preference",
        ),
        _candidate("current-session", record_kind="session_root", session_id="current-session", l0_text="current"),
        _candidate("resource", context_type="resource", source_kind="resource_reference", l0_text="resource"),
    ]

    result = ContextSelector().select(
        candidates,
        plan=_plan(source_kinds=("coding_agent",), session_ids=("current-session",), final_limit=10),
    )

    assert [item["record_key"] for item in result["contexts"]] == [
        "preference",
        "resource",
        "current-session",
        "related-session",
    ]
