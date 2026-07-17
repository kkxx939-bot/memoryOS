from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import InMemoryRelationStore
from memoryos.memory.canonical import (
    CanonicalMemoryRepository,
    Commitment,
    SemanticRelation,
    SpeechAct,
)
from memoryos.memory.canonical.final_state import (
    IdentityValidationError,
    OperationCompletenessError,
    RevisionEvidenceError,
)
from memoryos.operations.model.operation_action import OperationAction
from tests.support.canonical_transactions import (
    _explicit_bindings,
    _persisted_episode,
    _plan,
    _proposal,
    _replacement_proposal,
    _reviewed_resolution_plan,
    _setup,
    _supplement_proposal,
)


def _replacement_case(tmp_path):  # noqa: ANN001, ANN202
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    initial = _proposal(episode, "initial-sqlite", "SQLite", "confirmation", "confirmed")
    identity, _transition, plan = _plan(source, episode, scope, initial)
    committer.commit(
        "u1",
        plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id),
    )
    current_claim = next(
        claim for claim in CanonicalMemoryRepository(source).load(identity)[1] if claim.current.state == "ACTIVE"
    )
    replacement_episode = _persisted_episode(
        tmp_path,
        SessionArchive(
            user_id="u1",
            session_id="replacement-final-boundary",
            archive_uri="memoryos://user/u1/sessions/history/replacement-final-boundary",
            messages=[
                {
                    "id": "replace",
                    "role": "user",
                    "content": "Formally change the primary storage backend from SQLite to PostgreSQL now.",
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
        ),
    )
    replacement = _replacement_proposal(
        replacement_episode,
        "replace-postgresql",
        "PostgreSQL",
        current_claim,
    )
    replacement_plan = _reviewed_resolution_plan(
        source,
        committer,
        replacement_episode,
        replacement,
        command_suffix="final-boundary",
    )
    operations = list(replacement_plan.operations)
    # The production committer binds the immutable planning identity before
    # invoking the final-state safety boundary.
    committer._ensure_canonical_planning_digest(operations)
    return source, committer, identity, current_claim, episode, scope, operations


def _claim_operation(operations, *, canonical_value: str):  # noqa: ANN001, ANN202
    return next(
        operation
        for operation in operations
        if dict(operation.payload["context_object"]["metadata"]).get("canonical_value") == canonical_value
    )


def _slot_operation(operations):  # noqa: ANN001, ANN202
    return next(
        operation
        for operation in operations
        if dict(operation.payload["context_object"]["metadata"]).get("canonical_kind") == "slot"
    )


def _refresh_claim_content(operation) -> None:  # noqa: ANN001
    metadata = operation.payload["context_object"]["metadata"]
    revisions = metadata["revisions"]
    current_revision = int(metadata.get("current_revision", metadata["revision"]))
    current = next(item for item in revisions if int(item["revision"]) == current_revision)
    operation.payload["content"] = json.dumps(
        {
            "slot_id": metadata["slot_id"],
            "claim_id": metadata["claim_id"],
            "canonical_value": metadata["canonical_value"],
            "current": current,
            "latest_revision": metadata["revision"],
            "revisions": revisions,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def test_legal_replacement_overlay_validates_before_any_write(tmp_path) -> None:  # noqa: ANN001
    _source, committer, _identity, _claim, _episode, _scope, operations = _replacement_case(tmp_path)

    validated = committer.final_state_validator.validate(
        operations,
        tenant_id="t1",
        owner_user_id="u1",
    )

    assert validated is not None
    assert {claim.current.state for claim in validated.claims} == {"ACTIVE", "SUPERSEDED"}
    assert validated.slot.active_claim_id == next(
        claim.claim_id for claim in validated.claims if claim.current.state == "ACTIVE"
    )


@pytest.mark.parametrize("mutation", ["modify", "delete", "reorder", "skip"])
def test_final_validator_rejects_any_historical_revision_rewrite(tmp_path, mutation: str) -> None:  # noqa: ANN001
    _source, committer, _identity, _claim, _episode, _scope, legal = _replacement_case(tmp_path)
    operations = deepcopy(legal)
    old_claim = _claim_operation(operations, canonical_value="sqlite")
    metadata = old_claim.payload["context_object"]["metadata"]
    revisions = metadata["revisions"]
    if mutation == "modify":
        revisions[0]["value_fields"]["canonical_value"] = "tampered"
    elif mutation == "delete":
        revisions.pop(0)
    elif mutation == "reorder":
        revisions.reverse()
    else:
        revisions[-1]["revision"] += 1
        metadata["revision"] += 1

    with pytest.raises(ValueError):
        committer.final_state_validator.validate(
            operations,
            tenant_id="t1",
            owner_user_id="u1",
        )


def test_final_validator_recomputes_slot_and_claim_identity_v2(tmp_path) -> None:  # noqa: ANN001
    _source, committer, _identity, _claim, _episode, _scope, legal = _replacement_case(tmp_path)
    forged_slot = deepcopy(legal)
    _slot_operation(forged_slot).payload["context_object"]["metadata"]["identity_fields"] = {
        "decision_topic": "forged slot identity"
    }
    with pytest.raises(IdentityValidationError):
        committer.final_state_validator.validate(
            forged_slot,
            tenant_id="t1",
            owner_user_id="u1",
        )


@pytest.mark.parametrize("field_kind", ["identity", "value"])
def test_final_validator_rejects_proposal_fields_outside_memory_type_schema(
    tmp_path,
    field_kind: str,
) -> None:  # noqa: ANN001
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "schema-boundary", "SQLite", "confirmation", "confirmed")
    _identity, _transition, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer._ensure_canonical_planning_digest(operations)
    proof_field = "identity_fields" if field_kind == "identity" else "value_fields"
    forged_field = "forged_identity_selector" if field_kind == "identity" else "forged_semantic_field"
    for operation in operations:
        operation.payload["proposal_proofs"][0][proof_field][forged_field] = "not schema declared"

    with pytest.raises(RevisionEvidenceError, match=rf"{field_kind} fields violate.*schema"):
        committer.final_state_validator.validate(
            operations,
            tenant_id="t1",
            owner_user_id="u1",
        )


@pytest.mark.parametrize(
    ("target_kind", "field", "value"),
    [
        ("claim", "memory_type", "preference"),
        ("claim", "canonical_transaction_id", "memory_tx_forged"),
        ("slot", "canonical_idempotency_key", "forged-idempotency"),
        ("claim", "commit_group_id", "forged-group"),
    ],
)
def test_final_validator_rejects_forged_materialized_domain_mirrors(
    tmp_path,
    target_kind: str,
    field: str,
    value: str,
) -> None:  # noqa: ANN001
    _source, committer, _identity, _claim, _episode, _scope, legal = _replacement_case(tmp_path)
    operations = deepcopy(legal)
    operation = (
        _slot_operation(operations)
        if target_kind == "slot"
        else _claim_operation(operations, canonical_value="postgresql")
    )
    operation.payload["context_object"]["metadata"][field] = value

    with pytest.raises(ValueError):
        committer.final_state_validator.validate(
            operations,
            tenant_id="t1",
            owner_user_id="u1",
        )


def test_final_validator_rejects_forged_slot_action(tmp_path) -> None:  # noqa: ANN001
    _source, committer, _identity, _claim, _episode, _scope, legal = _replacement_case(tmp_path)
    operations = deepcopy(legal)
    _slot_operation(operations).action = OperationAction.ADD

    with pytest.raises(ValueError):
        committer.final_state_validator.validate(
            operations,
            tenant_id="t1",
            owner_user_id="u1",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("claim_id", "forged-payload-claim"),
        ("memory_type", "preference"),
        ("policy_version", "forged-policy"),
        ("schema_version", "forged-schema"),
    ],
)
def test_final_validator_rejects_forged_transaction_domain_mirrors(
    tmp_path,
    field: str,
    value: str,
) -> None:  # noqa: ANN001
    _source, committer, _identity, _claim, _episode, _scope, legal = _replacement_case(tmp_path)
    operations = deepcopy(legal)
    operation = _claim_operation(operations, canonical_value="postgresql")
    operation.payload[field] = value

    with pytest.raises(ValueError):
        committer.final_state_validator.validate(
            operations,
            tenant_id="t1",
            owner_user_id="u1",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transition_profile", "OBSERVATIONAL"),
        ("policy_version", "forged-policy"),
        ("schema_version", "forged-schema"),
        ("epistemic_status", "INFERRED"),
        ("relation", "SUPPLEMENTS"),
    ],
)
def test_final_validator_rejects_forged_transition_semantics(
    tmp_path,
    field: str,
    value: str,
) -> None:  # noqa: ANN001
    _source, committer, _identity, _claim, _episode, _scope, legal = _replacement_case(tmp_path)
    operations = deepcopy(legal)
    operation = _claim_operation(operations, canonical_value="postgresql")
    metadata = operation.payload["context_object"]["metadata"]
    if field == "transition_profile":
        metadata[field] = value
    else:
        metadata["revisions"][-1][field] = value
        if field in {"epistemic_status", "relation"}:
            mirror = "semantic_relation" if field == "relation" else field
            metadata[mirror] = value
        _refresh_claim_content(operation)

    with pytest.raises(ValueError):
        committer.final_state_validator.validate(
            operations,
            tenant_id="t1",
            owner_user_id="u1",
        )


def test_final_validator_recomputes_proposal_fingerprint_from_bound_proof(tmp_path) -> None:  # noqa: ANN001
    _source, committer, _identity, _claim, _episode, _scope, legal = _replacement_case(tmp_path)
    operations = deepcopy(legal)
    for operation in operations:
        operation.payload["proposal_fingerprints"] = ["forged-fingerprint"]
        metadata = operation.payload["context_object"]["metadata"]
        if metadata.get("canonical_kind") != "claim":
            continue
        metadata["revisions"][-1]["proposal_fingerprint"] = "forged-fingerprint"
        _refresh_claim_content(operation)

    with pytest.raises(RevisionEvidenceError, match="proposal proof"):
        committer.final_state_validator.validate(
            operations,
            tenant_id="t1",
            owner_user_id="u1",
        )

    forged_claim = deepcopy(legal)
    claim = _claim_operation(forged_claim, canonical_value="postgresql")
    claim.payload["context_object"]["metadata"]["claim_id"] = "forged-claim"
    with pytest.raises(IdentityValidationError):
        committer.final_state_validator.validate(
            forged_claim,
            tenant_id="t1",
            owner_user_id="u1",
        )


def test_final_validator_rejects_omitted_supersede_extra_operation_and_bad_active_pointer(
    tmp_path,
) -> None:  # noqa: ANN001
    _source, committer, _identity, _claim, _episode, _scope, legal = _replacement_case(tmp_path)

    omitted = [
        operation
        for operation in deepcopy(legal)
        if not (
            dict(operation.payload["context_object"]["metadata"]).get("canonical_kind") == "claim"
            and dict(operation.payload["context_object"]["metadata"]).get("canonical_value") == "sqlite"
        )
    ]
    with pytest.raises(ValueError):
        committer.final_state_validator.validate(omitted, tenant_id="t1", owner_user_id="u1")

    extra = deepcopy(legal)
    unrelated = deepcopy(_claim_operation(extra, canonical_value="postgresql"))
    unrelated.operation_id += "_extra"
    unrelated.target_uri = f"{unrelated.target_uri}-extra"
    unrelated.payload["context_object"]["uri"] = unrelated.target_uri
    unrelated.payload["context_object"]["metadata"]["claim_id"] = "extra"
    extra.append(unrelated)
    with pytest.raises(ValueError):
        committer.final_state_validator.validate(extra, tenant_id="t1", owner_user_id="u1")

    mismatched = deepcopy(legal)
    _slot_operation(mismatched).payload["context_object"]["metadata"]["active_claim_id"] = "missing"
    with pytest.raises(ValueError):
        committer.final_state_validator.validate(mismatched, tenant_id="t1", owner_user_id="u1")

    no_pointer = deepcopy(legal)
    _slot_operation(no_pointer).payload["context_object"]["metadata"]["active_claim_id"] = None
    with pytest.raises(ValueError):
        committer.final_state_validator.validate(no_pointer, tenant_id="t1", owner_user_id="u1")


def test_final_validator_never_allows_historical_claim_membership_to_be_pruned(
    tmp_path,
) -> None:  # noqa: ANN001
    _source, committer, _identity, _claim, _episode, _scope, legal = _replacement_case(tmp_path)
    operations = deepcopy(legal)
    old_claim = _claim_operation(operations, canonical_value="sqlite")
    old_claim_id = str(old_claim.payload["context_object"]["metadata"]["claim_id"])
    operations.remove(old_claim)
    slot_operation = _slot_operation(operations)
    slot_metadata = slot_operation.payload["context_object"]["metadata"]
    slot_metadata["claim_ids"] = [claim_id for claim_id in slot_metadata["claim_ids"] if claim_id != old_claim_id]
    slot_operation.payload["content"] = json.dumps(
        {
            "slot_id": slot_metadata["slot_id"],
            "identity_algorithm_version": slot_metadata["identity_algorithm_version"],
            "canonical_subject": slot_metadata["canonical_subject"],
            "identity_fields": slot_metadata["identity_fields"],
            "claim_ids": slot_metadata["claim_ids"],
            "active_claim_id": slot_metadata["active_claim_id"],
            "revision": slot_metadata["revision"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    # RelationStore is derived data.  Its accidental absence must never turn
    # an otherwise invalid history-pruning transaction into a legal one.
    assert isinstance(committer.relation_store, InMemoryRelationStore)
    committer.relation_store.relations.clear()

    with pytest.raises(OperationCompletenessError, match="removes historical"):
        committer.final_state_validator.validate(
            operations,
            tenant_id="t1",
            owner_user_id="u1",
        )


def test_final_validator_rejects_multiple_active_claims(tmp_path) -> None:  # noqa: ANN001
    _source, committer, _identity, _claim, _episode, _scope, legal = _replacement_case(tmp_path)
    operations = deepcopy(legal)
    old_claim = _claim_operation(operations, canonical_value="sqlite")
    old_claim.payload["context_object"]["metadata"]["revisions"][-1]["state"] = "ACTIVE"
    old_claim.payload["context_object"]["metadata"]["state"] = "ACTIVE"

    with pytest.raises(ValueError):
        committer.final_state_validator.validate(operations, tenant_id="t1", owner_user_id="u1")


def test_final_validator_rejects_empty_noop_revision(tmp_path) -> None:  # noqa: ANN001
    source, committer, identity, target, episode, scope, _replacement = _replacement_case(tmp_path)
    supplement = _supplement_proposal(
        episode,
        "noop-supplement",
        target,
        speech_act=SpeechAct.CONFIRMATION,
        commitment=Commitment.CONFIRMED,
    )
    supplement = replace(
        supplement,
        semantic=replace(
            supplement.semantic,
            relation_to_existing=SemanticRelation.SUPPLEMENTS,
        ),
        related_slot_ids=(target.slot_id,),
        related_claim_ids=(target.claim_id,),
        field_evidence_refs=_explicit_bindings(
            dict(supplement.identity_fields),
            dict(supplement.value_fields),
            supplement.evidence_refs,
        ),
    )
    _resolved, _transition, plan = _plan(source, episode, scope, supplement)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer._ensure_canonical_planning_digest(operations)
    claim_operation = next(
        operation
        for operation in operations
        if dict(operation.payload["context_object"]["metadata"]).get("claim_id") == target.claim_id
    )
    revisions = claim_operation.payload["context_object"]["metadata"]["revisions"]
    assert len(revisions) == 2
    revisions[-1]["state"] = revisions[-2]["state"]
    revisions[-1]["value_fields"] = deepcopy(revisions[-2]["value_fields"])
    revisions[-1]["qualifiers"] = deepcopy(revisions[-2]["qualifiers"])

    with pytest.raises(OperationCompletenessError, match="no-op"):
        committer.final_state_validator.validate(
            operations,
            tenant_id="t1",
            owner_user_id="u1",
        )
    assert CanonicalMemoryRepository(source).load(identity)[0] is not None
