from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
)
from memoryos.memory.canonical import (
    ActiveClaimInvariantError,
    AliasRegistry,
    CanonicalMemoryInvariantError,
    CanonicalMemoryRepository,
    EpistemicStatus,
    EvidenceRef,
    MemoryClaim,
    MemoryScope,
    MemorySemanticNormalizer,
    MemorySemanticProposal,
    MemorySlot,
    MemoryTransactionPlanner,
    MemoryTransitionPolicy,
    NormalizedSemanticAssessment,
    PendingSemanticReconciliation,
    ProposalAdmissionDecision,
    ProposalAdmissionGate,
    ProposalValidationResult,
    RevisionSequenceError,
    ScopeRef,
    ScopeResolutionSource,
    ScopeSelector,
    SemanticAssessment,
    SemanticRelation,
    SessionArchiveEpisodeAdapter,
    StableMemoryIdentityResolver,
    TransitionProfile,
    VisibilityPolicy,
    bind_field_evidence,
    scope_key_candidates_from_payload,
    scope_key_from_payload,
)
from memoryos.memory.canonical.reconcile import MemorySemanticReconciler
from memoryos.memory.canonical.scope import scope_keys_from_payloads
from memoryos.memory.canonical.state import MemoryRevision
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


def _write_committed_canonical_fixture(
    source: FileSystemSourceStore,
    entries: list[tuple[ContextObject, str]],
    *,
    key: str,
) -> None:
    """Persist canonical fixtures behind an integrity-valid transaction marker."""

    transaction_id = f"tx-{key}"
    idempotency_key = f"idem-{key}"
    operations: list[ContextOperation] = []
    for index, (raw_obj, content) in enumerate(entries):
        obj = raw_obj
        owner_user_id = obj.owner_user_id
        assert isinstance(owner_user_id, str)
        obj.metadata = {
            **dict(obj.metadata or {}),
            "canonical_transaction_id": transaction_id,
            "canonical_idempotency_key": idempotency_key,
        }
        source.write_object(obj, content=content)
        operations.append(
            ContextOperation(
                operation_id=f"op-{key}-{index}",
                user_id=owner_user_id,
                context_type=obj.context_type,
                action=OperationAction.ADD,
                target_uri=obj.uri,
                status=OperationStatus.COMMITTED,
                payload={
                    "canonical_memory": True,
                    "transaction_id": transaction_id,
                    "idempotency_key": idempotency_key,
                    "tenant_id": obj.tenant_id,
                    "expected_revision": 0,
                    "context_object": obj.to_dict(),
                    "content": content,
                },
            )
        )
    assert operations
    assert len({operation.user_id for operation in operations}) == 1
    committer = OperationCommitter(
        source,
        InMemoryIndexStore(),
        str(source.root),
        tenant_id=source.tenant_id,
    )
    diff = ContextDiff(
        user_id=operations[0].user_id,
        operations=operations,
        diff_id=f"diff-{transaction_id}",
    )
    marker = committer._transaction_marker(idempotency_key)
    committer._write_transaction_marker(marker, diff, operations)
    committer._validate_transaction_marker(marker, operations)


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
        "semantic.utterance_mode": evidence_refs,
        "semantic.attribution": evidence_refs,
        "semantic.durability": evidence_refs,
        "semantic.modal_force": evidence_refs,
        "semantic.atomicity": evidence_refs,
        "transition": evidence_refs,
    }
    return bind_field_evidence(
        identity_fields,
        value_fields,
        evidence_refs,
        bindings=bindings,
        semantic_contract_version="v3",
    )


def _episode():  # noqa: ANN202
    return SessionArchiveEpisodeAdapter().adapt(
        SessionArchive(
            user_id="u1",
            session_id="s1",
            archive_uri="memoryos://user/u1/sessions/history/s1",
            messages=[{"id": "m1", "role": "user", "content": "I confirm storage is SQLite."}],
            metadata={"tenant_id": "t1", "project_id": "workspace-a"},
        )
    )


def _scope(*scopes: ScopeRef) -> MemoryScope:
    return MemoryScope(ScopeSelector(tuple(scopes)), VisibilityPolicy("t1"), tuple(scopes))


def _proposal(
    value: object = "SQLite",
    *,
    identity_fields: dict | None = None,
    semantic: SemanticAssessment | None = None,
    metadata: dict | None = None,
    proposal_id: str = "p1",
) -> MemorySemanticProposal:
    episode = _episode()
    text = episode.events[0].text()
    evidence = (
        EvidenceRef.from_event(
            episode.events[0],
            source_uri=episode.source_uris[0],
            span_start=0,
            span_end=len(text),
        ),
    )
    identity = identity_fields or {"decision_topic": "storage"}
    values = {"canonical_value": value}
    raw_semantic = semantic or SemanticAssessment("confirmation", "confirmed", "current", "unrelated")
    v3_semantic = replace(
        raw_semantic,
        utterance_mode="assertion",
        attribution="source_actor",
        durability="durable",
        modal_force="none",
        atomicity="atomic",
    )
    return MemorySemanticNormalizer().normalize(
        MemorySemanticProposal(
            proposal_id=proposal_id,
            memory_type="project_decision",
            identity_fields=identity,
            value_fields=values,
            semantic=v3_semantic,
            epistemic_status=EpistemicStatus.EXPLICIT,
            suggested_scope_refs=(ScopeRef("memoryos", "workspace", "workspace-a"),),
            related_memory_ids=(),
            evidence_refs=evidence,
            field_evidence_refs=_explicit_bindings(identity, values, evidence),
            confidence=0.99,
            extractor_version="test_v3",
            prompt_version="test_v3",
            semantic_contract_version="v3",
            atomic_evidence_ref=evidence[0],
            metadata={
                "source_role": "user",
                "transition_evidence_validated": True,
                "semantic_contract_validated": True,
                "atomic_evidence_validated": True,
                **dict(metadata or {}),
            },
        )
    )


def _apply(proposal: MemorySemanticProposal, scope: MemoryScope, slot=None, claims=()):  # noqa: ANN001, ANN202
    identity = StableMemoryIdentityResolver().resolve(proposal, scope, tenant_id="t1", owner_user_id="u1")
    reconciled = MemorySemanticReconciler().reconcile(proposal, identity, slot=slot, claims=claims)
    return identity, reconciled, MemoryTransitionPolicy().apply(proposal, identity, reconciled)


def test_scope_key_contains_parent_path_and_has_one_v2_candidate() -> None:
    first = ScopeRef("memoryos", "asset", "camera", parent_id="workspace-a")
    second = ScopeRef("memoryos", "asset", "camera", parent_id="workspace-b")
    unparented = ScopeRef.from_dict({"namespace": "memoryos", "kind": "asset", "id": "camera"})

    assert first.key != second.key
    assert scope_key_from_payload(first.to_dict()) == first.key
    assert scope_key_candidates_from_payload(first.to_dict()) == (first.key,)
    assert unparented.key == "memoryos:asset:camera"
    assert unparented.source == ScopeResolutionSource.EXPLICIT


def test_scope_parent_fields_must_be_consistent_and_hierarchical() -> None:
    with pytest.raises(ValueError, match="parent_id"):
        scope_keys_from_payloads(
            [
                {
                    "namespace": "memoryos",
                    "kind": "location",
                    "id": "desk",
                    "parent_id": "floor-b",
                    "parent_path": ["building", "floor-a"],
                }
            ]
        )
    with pytest.raises(ValueError, match="does not support parent"):
        ScopeRef("memoryos", "workspace", "w1", parent_id="organization-a")


@pytest.mark.parametrize(
    "payload",
    [
        {"namespace": "memoryos", "kind": "workspace", "id": 123},
        {"namespace": "memoryos", "kind": "asset", "id": "camera", "parent_path": "workspace-a"},
        {"namespace": "memoryos", "kind": "asset", "id": "camera", "attributes": [["x", "y"]]},
        {"namespace": "memoryos", "kind": "workspace", "id": "w1", "inferred": "false"},
    ],
)
def test_scope_payload_types_fail_closed_without_string_coercion(payload: dict) -> None:
    with pytest.raises(ValueError):
        scope_keys_from_payloads([payload])


def test_parent_scope_key_and_logical_path_have_one_canonical_identity() -> None:
    short = ScopeRef("memoryos", "asset", "camera", parent_id="workspace-a")
    full = ScopeRef(
        "memoryos",
        "asset",
        "camera",
        parent_id="workspace-a",
        parent_path=("memoryos:workspace:workspace-a",),
    )
    expanded = ScopeRef(
        "memoryos",
        "location",
        "desk",
        parent_path=("memoryos:location:path:building-a/floor-1",),
    )
    logical = ScopeRef(
        "memoryos",
        "location",
        "desk",
        parent_path=("building-a", "floor-1"),
    )
    other_building = ScopeRef(
        "memoryos",
        "location",
        "desk",
        parent_path=("memoryos:location:path:building-b/floor-1",),
    )

    assert short.key == full.key
    assert expanded.parent_path == ("building-a", "floor-1")
    assert expanded.key == logical.key
    assert expanded.key != other_building.key


def test_identity_v2_uses_subject_not_author_and_preserves_tenant_boundary() -> None:
    proposal = _proposal()
    scope = _scope(ScopeRef("memoryos", "workspace", "workspace-a"))
    resolver = StableMemoryIdentityResolver()

    first = resolver.resolve(proposal, scope, tenant_id="t1", owner_user_id="u1")
    other_author = resolver.resolve(proposal, scope, tenant_id="t1", owner_user_id="u2")
    other_tenant = resolver.resolve(proposal, scope, tenant_id="t2", owner_user_id="u1")

    assert first.identity_algorithm_version == "identity_v2"
    assert first.slot_id == other_author.slot_id
    assert first.slot_uri == other_author.slot_uri
    assert first.slot_id != other_tenant.slot_id


def test_same_asset_id_under_different_workspace_does_not_collide() -> None:
    identity = {"entity_type": "camera", "canonical_entity_id": "front"}
    proposal = replace(_proposal(identity_fields=identity), memory_type="entity")
    resolver = StableMemoryIdentityResolver()
    first_scope = _scope(
        ScopeRef("memoryos", "workspace", "workspace-a"),
        ScopeRef("memoryos", "asset", "front"),
    )
    second_scope = _scope(
        ScopeRef("memoryos", "workspace", "workspace-b"),
        ScopeRef("memoryos", "asset", "front"),
    )

    first = resolver.resolve(proposal, first_scope, tenant_id="t1", owner_user_id="u1")
    second = resolver.resolve(proposal, second_scope, tenant_id="t1", owner_user_id="u2")

    assert first.canonical_subject is not None and first.canonical_subject.inferred
    assert second.canonical_subject is not None
    assert first.canonical_subject.key != second.canonical_subject.key
    assert first.slot_id != second.slot_id


def test_claim_identity_uses_core_value_and_applicability_not_arbitrary_metadata() -> None:
    scope = _scope(ScopeRef("memoryos", "workspace", "workspace-a"))
    resolver = StableMemoryIdentityResolver()
    first = _proposal(
        {"engine": "SQLite", "mode": "WAL"},
        identity_fields={"decision_topic": {"area": "storage", "level": "primary"}},
    )
    reordered = _proposal(
        {"mode": "WAL", "engine": "SQLite"},
        identity_fields={"decision_topic": {"level": "primary", "area": "storage"}},
        proposal_id="p2",
    )
    metadata_only = replace(first, value_fields={**dict(first.value_fields), "semantic_qualifier": "local-only"})
    qualified = replace(first, value_fields={**dict(first.value_fields), "environment": "local-only"})

    first_id = resolver.resolve(first, scope, tenant_id="t1", owner_user_id="u1")
    reordered_id = resolver.resolve(reordered, scope, tenant_id="t1", owner_user_id="u1")
    metadata_only_id = resolver.resolve(metadata_only, scope, tenant_id="t1", owner_user_id="u1")
    qualified_id = resolver.resolve(qualified, scope, tenant_id="t1", owner_user_id="u1")

    assert first_id.slot_id == reordered_id.slot_id
    assert first_id.claim_id == reordered_id.claim_id
    assert first_id.claim_id == metadata_only_id.claim_id
    assert first_id.claim_id != qualified_id.claim_id


def test_alias_registry_only_normalizes_values_before_v2_hashing() -> None:
    aliases = AliasRegistry({"project_decision:value": {"postgres": "postgresql"}})
    resolver = StableMemoryIdentityResolver(aliases)
    scope = _scope(ScopeRef("memoryos", "workspace", "workspace-a"))
    first = resolver.resolve(_proposal("Postgres"), scope, tenant_id="t1", owner_user_id="u1")
    second = resolver.resolve(_proposal("PostgreSQL"), scope, tenant_id="t1", owner_user_id="u2")

    assert first.slot_uri == second.slot_uri
    assert first.claim_uri == second.claim_uri


def test_repository_rejects_non_v2_slot_instead_of_migrating(tmp_path) -> None:  # noqa: ANN001
    proposal = _proposal()
    scope = replace(
        _scope(ScopeRef("memoryos", "workspace", "workspace-a")),
        canonical_subject=ScopeRef("memoryos", "workspace", "workspace-a"),
    )
    identity = StableMemoryIdentityResolver().resolve(proposal, scope, tenant_id="t1", owner_user_id="u1")
    slot = MemorySlot(
        identity.slot_id,
        identity.slot_uri,
        proposal.memory_type,
        identity.slot_identity,
        identity.scope_keys,
        canonical_subject_key=identity.canonical_subject_key,
        canonical_subject=identity.canonical_subject,
    )
    obj = slot.to_context_object(tenant_id="t1", owner_user_id="u1", scope=scope.to_dict())
    obj.metadata["identity_algorithm_version"] = "identity_v1"
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    _write_committed_canonical_fixture(source, [(obj, "")], key="identity-v1-slot")

    with pytest.raises(CanonicalMemoryInvariantError, match="not Identity V2"):
        CanonicalMemoryRepository(source).load_uri(identity.slot_uri)


def test_transaction_plan_is_v2_only_and_has_no_redirect_payload() -> None:
    proposal = _proposal()
    scope = _scope(ScopeRef("memoryos", "workspace", "workspace-a"))
    identity, _reconciled, transition = _apply(proposal, scope)
    first = MemoryTransactionPlanner().build(
        proposal,
        scope,
        transition,
        tenant_id="t1",
        owner_user_id="u1",
        episode_id="s1",
    )
    second = MemoryTransactionPlanner().build(
        proposal,
        scope,
        transition,
        tenant_id="t1",
        owner_user_id="u1",
        episode_id="s1",
    )

    assert first.idempotency_key == second.idempotency_key
    snapshot = first.operations[0].context_object.to_dict()
    payload = first.to_context_operations(user_id="u1", tenant_id="t1", episode_id="s1")[0].payload
    assert payload["identity_algorithm_version"] == "identity_v2"
    assert payload["schema_version"] == "canonical_memory_v2"
    assert "identity_alias_operations" not in payload
    assert first.operations[0].context_object.to_dict() == snapshot


def test_unknown_semantics_fail_closed_at_normalization_and_admission() -> None:
    proposal = _proposal(semantic=SemanticAssessment("invented", "confirmed", "current", "unrelated"))
    assert isinstance(proposal.semantic, NormalizedSemanticAssessment)
    assert proposal.semantic.speech_act.value == "SCHEMA_MISMATCH"
    validation = ProposalValidationResult(True, proposal)
    result = ProposalAdmissionGate().evaluate(
        validation,
        episode=_episode(),
        memory_scope=replace(
            _scope(ScopeRef("memoryos", "workspace", "workspace-a")),
            canonical_subject=ScopeRef("memoryos", "workspace", "workspace-a"),
        ),
        source_role="user",
    )
    assert result.decision == ProposalAdmissionDecision.PENDING
    assert "semantic_schema_pending" in result.reason


def test_ambiguous_relation_never_becomes_alternative_or_changes_state() -> None:
    scope = _scope(ScopeRef("memoryos", "workspace", "workspace-a"))
    initial = _proposal()
    _identity, _reconciled, first = _apply(initial, scope)
    ambiguous = _proposal(
        "PostgreSQL",
        proposal_id="p2",
        semantic=SemanticAssessment("observation", "weak", "current", "unrelated"),
    )
    identity = StableMemoryIdentityResolver().resolve(ambiguous, scope, tenant_id="t1", owner_user_id="u1")
    reconciled = MemorySemanticReconciler().reconcile(
        ambiguous,
        identity,
        slot=first.slot,
        claims=first.claims,
    )
    assert reconciled.relation == SemanticRelation.AMBIGUOUS
    with pytest.raises(PendingSemanticReconciliation):
        MemoryTransitionPolicy().apply(ambiguous, identity, reconciled)


def test_duplicate_is_noop_and_contradiction_requires_review_without_changing_current() -> None:
    scope = _scope(ScopeRef("memoryos", "workspace", "workspace-a"))
    initial = _proposal()
    _identity, _reconciled, first = _apply(initial, scope)
    duplicate = replace(initial, proposal_id="duplicate")
    _duplicate_identity, duplicate_reconciled, duplicate_transition = _apply(
        duplicate,
        scope,
        first.slot,
        first.claims,
    )
    assert duplicate_reconciled.relation == SemanticRelation.DUPLICATE
    assert duplicate_transition.changed_claim_ids == ()
    assert duplicate_transition.slot.revision == first.slot.revision

    contradiction = _proposal(
        "PostgreSQL",
        proposal_id="contradiction",
        semantic=SemanticAssessment("observation", "confirmed", "current", "contradicts"),
    )
    contradiction = replace(
        contradiction,
        related_claim_ids=(first.slot.active_claim_id,),
            metadata={
                **dict(contradiction.metadata),
                "relation_target_binding_validated": True,
                "semantic_relation_evidence_validated": True,
            },
    )
    contradiction_identity = StableMemoryIdentityResolver().resolve(
        contradiction,
        scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    contradiction_reconciled = MemorySemanticReconciler().reconcile(
        contradiction,
        contradiction_identity,
        slot=first.slot,
        claims=first.claims,
    )
    assert contradiction_reconciled.relation == SemanticRelation.CONTRADICTS
    with pytest.raises(PendingSemanticReconciliation, match="nonfinal_relation_requires_review"):
        MemoryTransitionPolicy().apply(contradiction, contradiction_identity, contradiction_reconciled)
    assert first.slot.active_claim_id is not None
    assert {claim.canonical_value: claim.current.state for claim in first.claims} == {"sqlite": "ACTIVE"}


def test_multiple_active_claims_raise_structured_invariant_before_reconciliation() -> None:
    evidence = _proposal().evidence_refs
    claims = tuple(
        MemoryClaim(
            f"c{index}",
            f"memoryos://user/u1/memories/canonical/slots/s/claims/c{index}",
            "s",
            value,
            TransitionProfile.AUTHORITATIVE_STATE,
            (
                MemoryRevision(
                    1,
                    "ACTIVE",
                    {"canonical_value": value},
                    evidence,
                    f"p{index}",
                    "UNRELATED",
                    "EXPLICIT",
                ),
            ),
        )
        for index, value in enumerate(("sqlite", "postgresql"), start=1)
    )
    slot = MemorySlot(
        "s", "memoryos://user/u1/memories/canonical/slots/s", "project_decision", {}, (), ("c1", "c2"), "c1", 2
    )
    with pytest.raises(ActiveClaimInvariantError) as caught:
        slot.validate_claims(claims)
    assert caught.value.active_claim_ids == ("c1", "c2")


def test_repository_rejects_persisted_multiple_active_claims(tmp_path) -> None:  # noqa: ANN001
    proposal = _proposal()
    subject = ScopeRef("memoryos", "workspace", "workspace-a")
    scope = replace(_scope(subject), canonical_subject=subject)
    identity = StableMemoryIdentityResolver().resolve(
        proposal,
        scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    claims = tuple(
        MemoryClaim(
            f"c{index}",
            f"{identity.slot_uri}/claims/c{index}",
            identity.slot_id,
            value,
            TransitionProfile.AUTHORITATIVE_STATE,
            (
                MemoryRevision(
                    1,
                    "ACTIVE",
                    {"canonical_value": value},
                    proposal.evidence_refs,
                    f"p{index}",
                    "UNRELATED",
                    "EXPLICIT",
                ),
            ),
            identity.identity_algorithm_version,
            identity.canonical_subject_key,
        )
        for index, value in enumerate(("sqlite", "postgresql"), start=1)
    )
    slot = MemorySlot(
        identity.slot_id,
        identity.slot_uri,
        proposal.memory_type,
        identity.slot_identity,
        identity.scope_keys,
        tuple(claim.claim_id for claim in claims),
        claims[0].claim_id,
        1,
        identity.identity_algorithm_version,
        identity.canonical_subject_key,
        canonical_subject=identity.canonical_subject,
    )
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    entries = []
    for claim in claims:
        entries.append(
            (
                claim.to_context_object(
                    tenant_id="t1",
                    owner_user_id="u1",
                    memory_type=proposal.memory_type,
                    scope=scope.to_dict(),
                ),
                "",
            )
        )
    entries.append(
        (
            slot.to_context_object(
                tenant_id="t1",
                owner_user_id="u1",
                scope=scope.to_dict(),
            ),
            "",
        )
    )
    _write_committed_canonical_fixture(source, entries, key="multiple-active-claims")

    with pytest.raises(ActiveClaimInvariantError):
        CanonicalMemoryRepository(source).load(identity)


def test_non_contiguous_revision_history_is_rejected() -> None:
    evidence = _proposal().evidence_refs
    with pytest.raises(RevisionSequenceError):
        MemoryClaim(
            "c1",
            "memoryos://user/u1/memories/canonical/slots/s/claims/c1",
            "s",
            "sqlite",
            TransitionProfile.AUTHORITATIVE_STATE,
            (
                MemoryRevision(1, "ACTIVE", {"canonical_value": "sqlite"}, evidence, "p1", "UNRELATED", "EXPLICIT"),
                MemoryRevision(3, "ACTIVE", {"canonical_value": "sqlite"}, evidence, "p3", "DUPLICATE", "EXPLICIT"),
            ),
        )


def test_late_correction_is_pending_and_does_not_replace_current() -> None:
    scope = _scope(ScopeRef("memoryos", "workspace", "workspace-a"))
    initial = _proposal(metadata={"effective_at": "2026-01-01T00:00:00+00:00"})
    _identity, _reconciled, first = _apply(initial, scope)
    late = _proposal(
        "PostgreSQL",
        proposal_id="late",
        semantic=SemanticAssessment("correction", "confirmed", "past", "corrects"),
        metadata={"effective_at": "2025-01-01T00:00:00+00:00"},
    )
    late_identity = StableMemoryIdentityResolver().resolve(
        late,
        scope,
        tenant_id="t1",
        owner_user_id="u1",
    )
    reconciled = MemorySemanticReconciler().reconcile(
        late,
        late_identity,
        slot=first.slot,
        claims=first.claims,
    )

    assert reconciled.historical_only
    assert reconciled.relation == SemanticRelation.AMBIGUOUS
    with pytest.raises(PendingSemanticReconciliation):
        MemoryTransitionPolicy().apply(late, late_identity, reconciled)
    assert first.slot.active_claim_id is not None
    assert {claim.canonical_value: claim.current.state for claim in first.claims} == {"sqlite": "ACTIVE"}
