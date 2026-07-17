from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval import (
    CanonicalResolutionMode,
    QueryPlan,
    QueryPlanner,
    RetrievalOptions,
    RetrievalQueryIntent,
    RetrievalQueryPlan,
    RetrievalScopeViolation,
    TrustedRetrievalScope,
    bind_trusted_scope,
    retrieval_options_from_legacy,
)
from memoryos.contextdb.retrieval.query_plan import (
    MAX_CANDIDATE_LIMIT,
    MAX_FINAL_LIMIT,
    MAX_TARGET_PATHS,
    MAX_TOKEN_BUDGET,
)


def test_legacy_query_plan_import_and_positional_contract_remain_compatible() -> None:
    plan = QueryPlan("tool output", "u1", [ContextType.SESSION], 512)

    assert plan.to_dict() == {
        "query": "tool output",
        "user_id": "u1",
        "context_types": ["session"],
        "token_budget": 512,
        "steps": [],
    }


def test_retrieval_options_are_normalized_serializable_and_round_trip() -> None:
    options = RetrievalOptions(
        target_uris=("memoryos://context/1", "memoryos://context/1", "memoryos://context/2"),
        target_paths=("/timeline//2026/07/14/", "resources/desktop"),
        context_types=(ContextType.SESSION, ContextType.RESOURCE),
        source_kinds=("TOOL_RESULT", "resource_reference"),
        tenant_id=" tenant-a ",
        owner_user_id=" u1 ",
        workspace_ids=("project-a", "project-a"),
        session_ids=("session-1",),
        adapter_id="codex",
        event_time_from="2026-07-14",
        event_time_to="2026-07-14",
        transaction_time_from="2026-07-14T01:00:00+08:00",
        transaction_time_to="2026-07-15T01:00:00+08:00",
        updated_at_from="2026-07-14",
        updated_at_to="2026-07-14",
        valid_at="2026-07-14T09:30:00+08:00",
        timezone="Asia/Singapore",
        query_intent=RetrievalQueryIntent.OPEN_RECALL,
        canonical_resolution_mode=CanonicalResolutionMode.VALIDATE,
        relation_expansion=True,
        candidate_limit=80,
        final_limit=12,
        token_budget=8_192,
        metadata_filters={"nested": {"states": ("ACTIVE", "SUPERSEDED")}, "score": 1.5},
        legacy_search_scope="default",
        legacy_retrieval_views=("project:project-a:rules",),
    )

    assert options.target_uris == ("memoryos://context/1", "memoryos://context/2")
    assert options.target_paths == ("timeline/2026/07/14", "resources/desktop")
    assert options.context_types == (ContextType.SESSION, ContextType.RESOURCE)
    assert options.source_kinds == ("tool_result", "resource_reference")
    assert options.event_time_from == "2026-07-13T16:00:00+00:00"
    assert options.event_time_to == "2026-07-14T16:00:00+00:00"
    assert options.transaction_time_from == "2026-07-13T17:00:00+00:00"
    assert options.transaction_time_to == "2026-07-14T17:00:00+00:00"
    assert options.updated_at_from == "2026-07-13T16:00:00+00:00"
    assert options.updated_at_to == "2026-07-14T16:00:00+00:00"
    assert options.valid_at == "2026-07-14T01:30:00+00:00"
    assert options.query_intent is RetrievalQueryIntent.OPEN_RECALL
    assert options.canonical_resolution_mode is CanonicalResolutionMode.VALIDATE

    payload = json.loads(options.to_json())
    assert RetrievalOptions.from_dict(payload) == options
    assert payload["metadata_filters"]["nested"]["states"] == ["ACTIVE", "SUPERSEDED"]


def test_query_plan_serializes_semantic_query_and_requires_valid_at_for_as_of() -> None:
    with pytest.raises(ValueError, match="requires valid_at"):
        QueryPlanner().build("database choice", options=RetrievalOptions(query_intent=RetrievalQueryIntent.AS_OF))

    plan = QueryPlanner().build(
        " database choice ",
        options=RetrievalOptions(
            query_intent=RetrievalQueryIntent.AS_OF,
            valid_at="2026-07-14",
            timezone="Asia/Singapore",
        ),
    )

    assert plan.semantic_query == "database choice"
    assert plan.valid_at == "2026-07-13T16:00:00+00:00"
    assert plan.query_intent is RetrievalQueryIntent.AS_OF
    assert RetrievalQueryPlan.from_dict(json.loads(plan.to_json())) == plan


def test_date_only_range_uses_local_day_and_handles_dst_boundary() -> None:
    options = RetrievalOptions(
        event_time_from="2026-03-08",
        event_time_to="2026-03-08",
        timezone="America/Los_Angeles",
    )

    assert options.event_time_from == "2026-03-08T08:00:00+00:00"
    assert options.event_time_to == "2026-03-09T07:00:00+00:00"


def test_planner_deterministically_maps_chinese_calendar_queries_to_time_semantics() -> None:
    planner = QueryPlanner(
        now_provider=lambda: datetime(2026, 7, 14, 1, 0, tzinfo=timezone.utc),
    )
    common = RetrievalOptions(timezone="Asia/Singapore")

    event_plan = planner.plan("7月14日发生了什么", options=common)
    valid_plan = planner.plan("7月14日时项目使用什么数据库", options=common)
    transaction_plan = planner.plan("7月14日系统新增了哪些记忆", options=common)

    assert event_plan.target_paths == ("timeline/2026/07/14",)
    assert event_plan.event_time_from == "2026-07-13T16:00:00+00:00"
    assert event_plan.event_time_to == "2026-07-14T16:00:00+00:00"
    assert event_plan.query_intent is RetrievalQueryIntent.OPEN_RECALL
    assert event_plan.valid_at is None
    assert event_plan.transaction_time_from is None
    assert valid_plan.valid_at == "2026-07-13T16:00:00+00:00"
    assert valid_plan.query_intent is RetrievalQueryIntent.AS_OF
    assert valid_plan.event_time_from is None
    assert transaction_plan.transaction_time_from == "2026-07-13T16:00:00+00:00"
    assert transaction_plan.transaction_time_to == "2026-07-14T16:00:00+00:00"
    assert transaction_plan.query_intent is RetrievalQueryIntent.HISTORY
    assert transaction_plan.event_time_from is None


def test_transaction_date_inference_preserves_non_default_caller_intent() -> None:
    planner = QueryPlanner(
        now_provider=lambda: datetime(2026, 7, 14, tzinfo=timezone.utc),
    )

    plan = planner.plan(
        "7月14日系统新增了哪些记忆",
        options=RetrievalOptions(
            query_intent=RetrievalQueryIntent.OPEN_RECALL,
            timezone="Asia/Singapore",
        ),
    )

    assert plan.query_intent is RetrievalQueryIntent.OPEN_RECALL
    assert plan.transaction_time_from == "2026-07-13T16:00:00+00:00"
    assert plan.transaction_time_to == "2026-07-14T16:00:00+00:00"


def test_transaction_date_inference_preserves_explicit_range_while_selecting_history() -> None:
    planner = QueryPlanner(
        now_provider=lambda: datetime(2026, 7, 14, tzinfo=timezone.utc),
    )

    plan = planner.plan(
        "7月14日系统新增了哪些记忆",
        options=RetrievalOptions(
            transaction_time_from="2025-01-02",
            transaction_time_to="2025-01-02",
            timezone="UTC",
        ),
    )

    assert plan.query_intent is RetrievalQueryIntent.HISTORY
    assert plan.transaction_time_from == "2025-01-02T00:00:00+00:00"
    assert plan.transaction_time_to == "2025-01-03T00:00:00+00:00"


def test_deterministic_date_inference_never_overrides_explicit_filters() -> None:
    planner = QueryPlanner(
        now_provider=lambda: datetime(2026, 7, 14, tzinfo=timezone.utc),
    )
    explicit_event = RetrievalOptions(
        target_paths=("resources/desktop",),
        event_time_from="2025-01-02",
        event_time_to="2025-01-02",
        timezone="UTC",
    )
    explicit_valid = RetrievalOptions(valid_at="2025-02-03", timezone="UTC")

    event_plan = planner.plan("7月14日发生了什么", options=explicit_event)
    valid_plan = planner.plan("7月14日时项目使用什么数据库", options=explicit_valid)

    assert event_plan.target_paths == ("resources/desktop",)
    assert event_plan.event_time_from == "2025-01-02T00:00:00+00:00"
    assert event_plan.event_time_to == "2025-01-03T00:00:00+00:00"
    assert valid_plan.valid_at == "2025-02-03T00:00:00+00:00"


def test_inferred_as_of_preserves_bound_scope_and_explicit_path_filters() -> None:
    planner = QueryPlanner(
        now_provider=lambda: datetime(2026, 7, 14, tzinfo=timezone.utc),
    )
    plan = planner.plan(
        "7月14日时项目使用什么数据库",
        options=RetrievalOptions(
            target_paths=("projects/project-a",),
            tenant_id="tenant-a",
            owner_user_id="u1",
            workspace_ids=("project-a",),
            timezone="Asia/Singapore",
        ),
        trusted_scope=TrustedRetrievalScope(
            tenant_id="tenant-a",
            owner_user_id="u1",
            workspace_ids=("project-a",),
        ),
    )

    assert plan.query_intent is RetrievalQueryIntent.AS_OF
    assert plan.valid_at == "2026-07-13T16:00:00+00:00"
    assert plan.tenant_id == "tenant-a"
    assert plan.owner_user_id == "u1"
    assert plan.workspace_ids == ("project-a",)
    assert plan.target_paths == ("projects/project-a",)


def test_deterministic_date_inference_uses_the_callers_local_current_year() -> None:
    planner = QueryPlanner(
        now_provider=lambda: datetime(2025, 12, 31, 23, 30, tzinfo=timezone.utc),
    )

    plan = planner.plan(
        "1月2日发生了什么",
        options=RetrievalOptions(timezone="Pacific/Kiritimati"),
    )

    assert plan.target_paths == ("timeline/2026/01/02",)
    assert plan.event_time_from == "2026-01-01T10:00:00+00:00"
    assert plan.event_time_to == "2026-01-02T10:00:00+00:00"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidate_limit": MAX_CANDIDATE_LIMIT + 1}, "candidate_limit"),
        ({"final_limit": MAX_FINAL_LIMIT + 1, "candidate_limit": MAX_FINAL_LIMIT + 1}, "final_limit"),
        ({"token_budget": MAX_TOKEN_BUDGET + 1}, "token_budget"),
        ({"candidate_limit": 5, "final_limit": 6}, "must not exceed"),
        ({"target_paths": tuple(f"timeline/{index}" for index in range(MAX_TARGET_PATHS + 1))}, "target_paths"),
        ({"relation_expansion": "true"}, "must be a bool"),
        ({"source_kinds": ("tool result",)}, "source_kind"),
        ({"target_paths": ("timeline/../private",)}, "target_path"),
        ({"metadata_filters": {"score": float("inf")}}, "finite"),
    ],
)
def test_options_reject_unbounded_or_unsafe_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        RetrievalOptions.from_dict(kwargs)


def test_options_reject_reversed_time_ranges_and_unknown_timezone() -> None:
    with pytest.raises(ValueError, match="event_time_from"):
        RetrievalOptions(
            event_time_from="2026-07-15T00:00:00Z",
            event_time_to="2026-07-14T00:00:00Z",
        )
    with pytest.raises(ValueError, match="updated_at_from"):
        RetrievalOptions(
            updated_at_from="2026-07-15T00:00:00Z",
            updated_at_to="2026-07-14T00:00:00Z",
        )
    with pytest.raises(ValueError, match="unknown timezone"):
        RetrievalOptions(timezone="Mars/Olympus_Mons")


def test_trusted_scope_is_inherited_and_explicit_filters_can_only_narrow_it() -> None:
    options = RetrievalOptions(
        target_paths=("resources/desktop",),
        workspace_ids=("project-a",),
        event_time_from="2026-07-14",
        event_time_to="2026-07-14",
        timezone="Asia/Singapore",
    )
    trusted = TrustedRetrievalScope(
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_ids=("project-a", "project-b"),
        session_ids=("session-1",),
        adapter_id="codex",
    )

    plan = QueryPlanner().plan("desktop file", options=options, trusted_scope=trusted)

    assert plan.tenant_id == "tenant-a"
    assert plan.owner_user_id == "u1"
    assert plan.workspace_ids == ("project-a",)
    assert plan.session_ids == ("session-1",)
    assert plan.adapter_id == "codex"
    assert plan.target_paths == ("resources/desktop",)
    assert plan.event_time_from == "2026-07-13T16:00:00+00:00"


@pytest.mark.parametrize(
    ("options", "trusted"),
    [
        (RetrievalOptions(tenant_id="attacker"), TrustedRetrievalScope(tenant_id="tenant-a")),
        (
            RetrievalOptions(workspace_ids=("project-b",)),
            TrustedRetrievalScope(workspace_ids=("project-a",)),
        ),
        (RetrievalOptions(), TrustedRetrievalScope(session_ids=())),
    ],
)
def test_trusted_scope_conflicts_fail_closed(
    options: RetrievalOptions,
    trusted: TrustedRetrievalScope,
) -> None:
    with pytest.raises(RetrievalScopeViolation):
        bind_trusted_scope(options, trusted)


def test_trusted_adapter_binds_agent_tree_and_rejects_cross_agent_paths() -> None:
    trusted = TrustedRetrievalScope(adapter_id="codex")

    bound = bind_trusted_scope(RetrievalOptions(target_paths=("agents",)), trusted)
    assert bound.target_paths == ("agents/codex",)

    with pytest.raises(RetrievalScopeViolation, match="adapter scope"):
        bind_trusted_scope(
            RetrievalOptions(target_paths=("agents/other",)),
            trusted,
        )


def test_trusted_scope_explicit_authorized_keys_narrow_the_available_scope_set() -> None:
    bound = bind_trusted_scope(
        RetrievalOptions(metadata_filters={"applicability_scope_keys": ["memoryos:team:platform"]}),
        TrustedRetrievalScope(
            owner_user_id="u1",
            workspace_ids=("project-a",),
            authorized_scope_keys=(
                "memoryos:principal:u1",
                "memoryos:workspace:project-a",
                "memoryos:team:platform",
            ),
        ),
    )

    assert bound.metadata_filters["applicability_scope_keys"] == ["memoryos:team:platform"]


def test_trusted_scope_without_explicit_filter_injects_every_authorized_scope() -> None:
    authorized = (
        "memoryos:principal:u1",
        "memoryos:workspace:project-a",
        "memoryos:team:platform",
    )

    bound = bind_trusted_scope(
        RetrievalOptions(),
        TrustedRetrievalScope(
            owner_user_id="u1",
            workspace_ids=("project-a",),
            authorized_scope_keys=authorized,
        ),
    )

    assert bound.metadata_filters["applicability_scope_keys"] == list(authorized)


def test_local_trusted_scope_without_authorization_envelope_keeps_legacy_explicit_scopes() -> None:
    bound = bind_trusted_scope(
        RetrievalOptions(
            metadata_filters={
                "applicability_scope_keys": [
                    "memoryos:environment:home-01",
                    "memoryos:asset:reachy-01",
                ]
            }
        ),
        TrustedRetrievalScope(owner_user_id="u1"),
    )

    assert bound.metadata_filters["applicability_scope_keys"] == [
        "memoryos:environment:home-01",
        "memoryos:asset:reachy-01",
        "memoryos:principal:u1",
    ]


def test_explicit_empty_authorized_scope_filter_is_unscoped_only() -> None:
    bound = bind_trusted_scope(
        RetrievalOptions(metadata_filters={"applicability_scope_keys": []}),
        TrustedRetrievalScope(
            owner_user_id="u1",
            authorized_scope_keys=("memoryos:principal:u1",),
        ),
    )

    assert bound.metadata_filters["applicability_scope_keys"] == []
    assert bound.metadata_filters["require_unscoped"] is True


@pytest.mark.parametrize(
    "forged_scope_key",
    [
        "memoryos:principal:victim",
        "memoryos:team:administrators",
        "memoryos:workspace:project-b",
    ],
)
def test_explicit_applicability_scope_keys_cannot_expand_trusted_grants(
    forged_scope_key: str,
) -> None:
    trusted = TrustedRetrievalScope(
        owner_user_id="u1",
        workspace_ids=("project-a",),
        authorized_scope_keys=(
            "memoryos:principal:u1",
            "memoryos:workspace:project-a",
        ),
    )

    with pytest.raises(RetrievalScopeViolation, match="exceed trusted caller scope"):
        bind_trusted_scope(
            RetrievalOptions(metadata_filters={"applicability_scope_keys": [forged_scope_key]}),
            trusted,
        )


def test_principal_only_workspace_is_not_exposed_as_an_applicability_key() -> None:
    bound = bind_trusted_scope(
        RetrievalOptions(),
        TrustedRetrievalScope(
            owner_user_id="u1",
            workspace_ids=("__memoryos_principal_only__",),
        ),
    )

    assert bound.metadata_filters["applicability_scope_keys"] == ["memoryos:principal:u1"]


def test_legacy_flat_kwargs_convert_to_structured_project_rule_plan() -> None:
    flat = {
        "user_id": "u1",
        "project_id": "project-a",
        "tenant_id": "tenant-a",
        "search_scope": "project_rules",
        "context_type": "memory",
        "source_kind": "canonical_projection",
        "claim_uris": ("memoryos://claim/1",),
        "slot_uris": ("memoryos://slot/1",),
        "limit": 12,
        "candidate_limit": 40,
        "connect_metadata": {"language": "python"},
        "memory_states": ("ACTIVE",),
        "query_intent": "CURRENT",
        "expand_relations": True,
    }

    options = retrieval_options_from_legacy(flat)
    plan = QueryPlanner().build_from_legacy(
        "sqlite",
        flat,
        trusted_scope=TrustedRetrievalScope(
            tenant_id="tenant-a",
            owner_user_id="u1",
            workspace_ids=("project-a",),
        ),
    )

    assert options.owner_user_id == "u1"
    assert options.workspace_ids == ("project-a",)
    assert options.target_paths == ("memories/rules",)
    assert options.target_uris == ("memoryos://claim/1", "memoryos://slot/1")
    assert options.context_types == (ContextType.MEMORY,)
    assert options.source_kinds == ("canonical_projection",)
    assert options.legacy_retrieval_views == ("project:project-a:rules",)
    assert options.metadata_filters == {
        "language": "python",
        "memory_states": ["ACTIVE"],
        "retrieval_views": ["project:project-a:rules"],
    }
    assert plan.final_limit == 12
    assert plan.candidate_limit == 40
    assert plan.relation_expansion is True


def test_legacy_candidate_scope_maps_to_options_intent_and_preserves_views() -> None:
    options = retrieval_options_from_legacy(
        {
            "user_id": "u1",
            "project_id": "project-a",
            "adapter_id": "codex",
            "search_scope": "candidates",
        }
    )

    assert options.query_intent is RetrievalQueryIntent.OPTIONS
    assert options.context_types == (ContextType.MEMORY,)
    assert options.metadata_filters["include_candidates"] is True
    assert options.legacy_retrieval_views == (
        "agent:codex:private",
        "user:u1:profile",
        "user:u1:preferences",
        "project:project-a:rules",
        "project:project-a:decisions",
        "project:project-a:knowledge",
        "project:project-a:agent_experience",
    )


def test_explicit_legacy_retrieval_views_override_scope_derivation() -> None:
    options = retrieval_options_from_legacy(
        {
            "user_id": "u1",
            "project_id": "project-a",
            "search_scope": "project_rules",
            "retrieval_views": ("custom:u1:memoryos",),
        }
    )

    assert options.legacy_retrieval_views == ("custom:u1:memoryos",)
    assert options.target_paths == ()
    assert options.workspace_ids == ("project-a",)


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (("SUPERSEDED",), RetrievalQueryIntent.HISTORY),
        (("RETRACTED",), RetrievalQueryIntent.HISTORY),
        (("CONFLICTED",), RetrievalQueryIntent.CONFLICTS),
        (("PROPOSED", "CONFLICTED"), RetrievalQueryIntent.OPTIONS),
        (("ACTIVE",), RetrievalQueryIntent.CURRENT),
    ],
)
def test_legacy_memory_states_infer_canonical_query_intent(
    states: tuple[str, ...],
    expected: RetrievalQueryIntent,
) -> None:
    options = retrieval_options_from_legacy({"memory_states": states})

    assert options.query_intent is expected


def test_explicit_legacy_query_intent_overrides_state_inference() -> None:
    options = retrieval_options_from_legacy({"memory_states": ("RETRACTED",), "query_intent": "CURRENT"})

    assert options.query_intent is RetrievalQueryIntent.CURRENT


@pytest.mark.parametrize(
    "flat",
    [
        {"surprise_filter": "value"},
        {"user_id": "u1", "owner_user_id": "u2"},
        {"project_id": "p1", "workspace_ids": ("p2",)},
        {"context_type": "session", "context_types": ("resource",)},
        {"limit": 5, "final_limit": 6},
        {"relation_expansion": True, "expand_relations": False},
        {"search_scope": "project_rules"},
    ],
)
def test_legacy_conversion_rejects_unknown_conflicting_or_under_scoped_inputs(flat: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        retrieval_options_from_legacy(flat)
