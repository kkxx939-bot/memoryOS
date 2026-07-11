from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical import (
    AliasRegistry,
    Commitment,
    EpistemicStatus,
    EvidenceRef,
    MemoryScope,
    MemorySemanticNormalizer,
    MemorySemanticProposal,
    MemoryTransitionPolicy,
    ScopeRef,
    ScopeSelector,
    SemanticAssessment,
    SessionArchiveEpisodeAdapter,
    SpeechAct,
    StableMemoryIdentityResolver,
    VisibilityPolicy,
    bind_field_evidence,
)
from memoryos.memory.canonical.reconcile import MemorySemanticReconciler


def _explicit_bindings(
    identity_fields: Mapping[str, object],
    value_fields: Mapping[str, object],
    evidence_refs: tuple[EvidenceRef, ...],
) -> dict[str, tuple[EvidenceRef, ...]]:
    bindings = {
        **{f"identity.{key}": evidence_refs for key in identity_fields},
        **{f"value.{key}": evidence_refs for key in value_fields},
        "semantic.speech_act": evidence_refs,
        "semantic.commitment": evidence_refs,
        "semantic.temporal_scope": evidence_refs,
        "semantic.relation_to_existing": evidence_refs,
        "transition": evidence_refs,
    }
    return bind_field_evidence(identity_fields, value_fields, evidence_refs, bindings=bindings)


def _context():  # noqa: ANN202
    episode = SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id="s1",
            archive_uri="memoryos://user/u1/sessions/history/s1",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "content": "The primary storage backend is SQLite. PostgreSQL is a future option.",
                }
            ],
            metadata={
                "tenant_id": "t1",
                "project_id": "memoryos",
                "connect": {"adapter_id": "codex"},
            },
        )
    )
    assert episode.origin.primary_scope is not None
    scope = MemoryScope(
        ScopeSelector((episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        episode.origin.scope_refs,
    )
    return episode, scope


def _proposal(episode, value: str, speech: str, commitment: str, proposal_id: str):  # noqa: ANN001, ANN202
    assert episode.origin.primary_scope is not None
    identity_fields = {"decision_topic": "primary storage backend"}
    value_fields = {"canonical_value": value}
    evidence_refs = (EvidenceRef.from_event(episode.events[0], source_uri=episode.source_uris[0]),)
    return MemorySemanticNormalizer().normalize(
        MemorySemanticProposal(
            proposal_id=proposal_id,
            memory_type="project_decision",
            identity_fields=identity_fields,
            value_fields=value_fields,
            semantic=SemanticAssessment(speech, commitment, "current", "alternative"),
            epistemic_status=EpistemicStatus.EXPLICIT,
            suggested_scope_refs=(episode.origin.primary_scope,),
            related_memory_ids=(),
            evidence_refs=evidence_refs,
            field_evidence_refs=_explicit_bindings(identity_fields, value_fields, evidence_refs),
            confidence=0.95,
            extractor_version="fake",
            metadata={"transition_evidence_validated": True},
        )
    )


def _apply(proposal, scope, slot=None, claims=()):  # noqa: ANN001, ANN202
    identity = StableMemoryIdentityResolver().resolve(proposal, scope, tenant_id="t1", owner_user_id="u1")
    reconciliation = MemorySemanticReconciler().reconcile(proposal, identity, slot=slot, claims=claims)
    return identity, MemoryTransitionPolicy().apply(proposal, identity, reconciliation)


def test_slot_identity_ignores_body_and_claim_values_share_slot() -> None:
    episode, scope = _context()
    sqlite = _proposal(episode, "SQLite", "confirmation", "confirmed", "p-sqlite")
    postgres = _proposal(episode, "PostgreSQL", "future_option", "exploratory", "p-postgres")
    resolver = StableMemoryIdentityResolver()
    sqlite_identity = resolver.resolve(sqlite, scope, tenant_id="t1", owner_user_id="u1")
    postgres_identity = resolver.resolve(postgres, scope, tenant_id="t1", owner_user_id="u1")
    body_changed = replace(sqlite, metadata={"body": "a completely different summary"})
    assert resolver.resolve(body_changed, scope, tenant_id="t1", owner_user_id="u1").slot_id == sqlite_identity.slot_id
    assert sqlite_identity.slot_id == postgres_identity.slot_id
    assert sqlite_identity.claim_id != postgres_identity.claim_id


def test_different_workspaces_do_not_share_slot_and_agents_in_one_workspace_do() -> None:
    episode, scope = _context()
    proposal = _proposal(episode, "SQLite", "confirmation", "confirmed", "p1")
    resolver = StableMemoryIdentityResolver()
    first = resolver.resolve(proposal, scope, tenant_id="t1", owner_user_id="u1")
    other_scope = replace(
        scope,
        applicability=ScopeSelector((ScopeRef("memoryos", "workspace", "other"),)),
    )
    other = resolver.resolve(proposal, other_scope, tenant_id="t1", owner_user_id="u1")
    agent_changed = replace(proposal, model_id="another-agent")
    assert first.slot_id != other.slot_id
    assert first.slot_id == resolver.resolve(agent_changed, scope, tenant_id="t1", owner_user_id="u1").slot_id


def test_tenant_and_canonical_subject_are_stable_slot_boundary_not_author() -> None:
    episode, scope = _context()
    proposal = _proposal(episode, "SQLite", "confirmation", "confirmed", "p-owner")
    resolver = StableMemoryIdentityResolver()
    first = resolver.resolve(proposal, scope, tenant_id="t1", owner_user_id="u1")
    other_owner = resolver.resolve(proposal, scope, tenant_id="t1", owner_user_id="u2")
    other_tenant = resolver.resolve(proposal, scope, tenant_id="t2", owner_user_id="u1")
    assert first.slot_id == other_owner.slot_id
    assert first.slot_uri == other_owner.slot_uri
    assert first.slot_id != other_tenant.slot_id


def test_alias_registry_maps_reachy_names_to_one_stable_asset() -> None:
    aliases = AliasRegistry(
        {
            "scope:asset": {
                "Reachy Mini": "reachy_01",
                "reachy-mini": "reachy_01",
                "客厅机器人": "reachy_01",
            }
        }
    )
    assert {
        aliases.canonical_scope(ScopeRef("memoryos", "asset", name)).id
        for name in ("Reachy Mini", "reachy-mini", "客厅机器人")
    } == {"reachy_01"}


def test_sqlite_active_postgres_proposed_then_confirmation_atomically_supersedes_sqlite() -> None:
    episode, scope = _context()
    sqlite = _proposal(episode, "SQLite", "confirmation", "confirmed", "p-sqlite")
    _, first = _apply(sqlite, scope)
    assert first.claims[0].current.state == "ACTIVE"

    postgres = _proposal(episode, "PostgreSQL", "future_option", "exploratory", "p-postgres")
    _, second = _apply(postgres, scope, first.slot, first.claims)
    states = {claim.canonical_value: claim.current.state for claim in second.claims}
    assert states == {"sqlite": "ACTIVE", "postgresql": "PROPOSED"}

    confirmed = replace(
        postgres,
        proposal_id="p-confirm",
        semantic=replace(
            postgres.semantic,
            speech_act=SpeechAct.CONFIRMATION,
            commitment=Commitment.CONFIRMED,
        ),
    )
    _, third = _apply(confirmed, scope, second.slot, second.claims)
    states = {claim.canonical_value: claim.current.state for claim in third.claims}
    assert states == {"sqlite": "SUPERSEDED", "postgresql": "ACTIVE"}
    assert third.slot.active_claim_id == next(
        claim.claim_id for claim in third.claims if claim.canonical_value == "postgresql"
    )
    assert len([claim for claim in third.claims if claim.current.state == "ACTIVE"]) == 1


def test_authoritative_explicit_observation_cannot_bypass_confirmation_transition() -> None:
    episode, scope = _context()
    observation = _proposal(episode, "SQLite", "observation", "weak", "p-observation")
    _, transition = _apply(observation, scope)
    assert transition.claims[0].current.state == "PROPOSED"


def test_claim_revision_history_links_previous_and_validity_interval() -> None:
    episode, scope = _context()
    first = _proposal(episode, "SQLite", "confirmation", "confirmed", "p-first")
    _, initial = _apply(first, scope)
    corrected = replace(
        first,
        proposal_id="p-corrected",
        value_fields={"canonical_value": "SQLite", "reason": "verified"},
        semantic=replace(first.semantic, speech_act=SpeechAct.CORRECTION),
    )
    _, updated = _apply(corrected, scope, initial.slot, initial.claims)
    revisions = updated.claims[0].revisions
    assert len(revisions) == 2
    assert revisions[1].previous_revision == 1
    assert revisions[0].valid_to == revisions[1].valid_from
