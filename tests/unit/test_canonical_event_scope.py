from __future__ import annotations

import pytest

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical import EpisodeSalienceGate, SessionArchiveEpisodeAdapter
from memoryos.memory.canonical.scope import (
    MemoryScope,
    ScopeRef,
    ScopeResolutionSource,
    ScopeSelector,
    VisibilityPolicy,
)


def test_coding_project_is_normalized_to_workspace_scope() -> None:
    episode = SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id="s1",
            archive_uri="memoryos://user/u1/sessions/history/s1",
            messages=[{"id": "m1", "role": "user", "content": "remember this"}],
            metadata={"project_id": "repo:memoryos", "connect": {"adapter_id": "codex"}},
        )
    )
    assert episode.origin.primary_scope is not None
    assert episode.origin.primary_scope.key == ScopeRef("memoryos", "workspace", "repo:memoryos").key
    assert episode.origin.primary_scope.source == ScopeResolutionSource.ORIGIN
    assert {scope.kind for scope in episode.legal_scope_candidates()} >= {"principal", "workspace", "episode"}


def test_reachy_origin_keeps_actor_subject_and_physical_qualifiers_distinct() -> None:
    episode = SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="user_1",
            session_id="interaction_1",
            archive_uri="memoryos://user/user_1/sessions/history/interaction_1",
            observations=[{"id": "obs1", "role": "robot", "actor_id": "reachy_01", "raw_text": "user lowered volume"}],
            metadata={
                "tenant_id": "home",
                "subjects": [{"kind": "person", "id": "user_1"}],
                "origin": {
                    "world_domain": "physical",
                    "connect_type": "robot",
                    "adapter_id": "reachy_mini",
                    "primary_scope": {"kind": "environment", "id": "home_01"},
                    "qualifiers": [
                        {"kind": "location", "id": "home_01:kitchen", "parent_id": "home_01"},
                        {"kind": "device", "id": "reachy_01", "parent_id": "home_01"},
                    ],
                },
            },
        )
    )
    event = episode.events[0]
    assert event.actor.kind == "robot" and event.actor.id == "reachy_01"
    assert event.subjects[0].kind == "person" and event.subjects[0].id == "user_1"
    assert episode.origin.primary_scope and episode.origin.primary_scope.kind == "environment"
    assert {scope.kind for scope in episode.origin.qualifiers} == {"location", "asset"}


def test_origin_applicability_and_visibility_are_independent_and_tenant_safe() -> None:
    kitchen = ScopeRef("memoryos", "location", "home_01:kitchen")
    principal = ScopeRef("memoryos", "principal", "user_1")
    home = ScopeRef("memoryos", "environment", "home_01")
    scope = MemoryScope(
        applicability=ScopeSelector((principal, home)),
        visibility=VisibilityPolicy("tenant_a", allowed_principal_ids=("user_1",)),
        origin_refs=(kitchen,),
    )
    scope.validate_tenant("tenant_a")
    assert scope.origin_refs != scope.applicability.all_of
    assert scope.visibility.permits(tenant_id="tenant_a", principal_id="user_1")
    assert not scope.visibility.permits(tenant_id="tenant_b", principal_id="user_1")


def test_core_rejects_unbounded_external_scope_kinds() -> None:
    try:
        ScopeRef("memoryos", "room", "kitchen")
    except ValueError as exc:
        assert "unsupported core scope kind" in str(exc)
    else:
        raise AssertionError("external room scope must be normalized at the adapter boundary")


def test_episode_rejects_duplicate_event_ids() -> None:
    with pytest.raises(ValueError, match="event IDs must be unique"):
        SessionArchiveEpisodeAdapter().adapt(
            SessionArchive(
                user_id="u1",
                session_id="s1",
                archive_uri="memoryos://user/u1/sessions/history/s1",
                messages=[
                    {"id": "same", "role": "user", "content": "remember one"},
                    {"id": "same", "role": "user", "content": "remember two"},
                ],
            )
        )


def test_adapter_salience_marker_and_tool_failure_are_structured_before_llm() -> None:
    marked = SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id="observation-window",
            archive_uri="memoryos://user/u1/sessions/history/observation-window",
            observations=[{"id": "o1", "role": "sensor", "raw_text": "volume changed", "salient": True}],
        )
    )
    assert EpisodeSalienceGate().evaluate(marked).reasons == ("adapter_marked",)
    failed = SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id="tool-window",
            archive_uri="memoryos://user/u1/sessions/history/tool-window",
            tool_results=[{"id": "t1", "status": "failed", "tool_output": "database unavailable"}],
        )
    )
    assert failed.events[0].event_type == "TOOL_FAILURE"
    assert EpisodeSalienceGate().evaluate(failed).salient
