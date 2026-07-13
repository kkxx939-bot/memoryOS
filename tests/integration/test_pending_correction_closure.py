from __future__ import annotations

from pathlib import Path

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical import (
    Atomicity,
    Attribution,
    CanonicalMemoryRepository,
    Durability,
    EpistemicStatus,
    EvidenceRef,
    MemorySemanticProposal,
    ModalForce,
    PendingReason,
    SemanticAssessment,
    SessionArchiveEpisodeAdapter,
    UtteranceMode,
    bind_field_evidence,
)
from memoryos.memory.canonical.current_head import artifact_root_for, load_current_head
from memoryos.memory.canonical.review_command import PendingReviewCommandStore


def _archive(session: str, text: str, *, task_id: str) -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id=session,
        archive_uri=f"memoryos://user/u1/sessions/history/{session}",
        messages=[{"id": f"{session}-m1", "role": "user", "content": text}],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
        task_id=task_id,
        created_at="2026-07-12T01:00:00Z",
    )


def _proposal(archive: SessionArchive, value: str, proposal_id: str) -> MemorySemanticProposal:
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    event = episode.events[0]
    text = event.text()
    evidence = (
        EvidenceRef.from_event(
            event,
            source_uri=archive.archive_uri,
            span_start=0,
            span_end=len(text),
        ),
    )
    identity = {"decision_topic": "primary storage backend"}
    values = {"canonical_value": value}
    bindings = {
        **{f"identity.{key}": evidence for key in identity},
        **{f"value.{key}": evidence for key in values},
        **{
            f"semantic.{field}": evidence
            for field in (
                "speech_act",
                "commitment",
                "temporal_scope",
                "relation_to_existing",
                "utterance_mode",
                "attribution",
                "durability",
                "modal_force",
                "atomicity",
            )
        },
        "transition": evidence,
    }
    assert episode.origin.primary_scope is not None
    return MemorySemanticProposal(
        proposal_id=proposal_id,
        memory_type="project_decision",
        identity_fields=identity,
        value_fields=values,
        semantic=SemanticAssessment(
            "confirmation",
            "confirmed",
            "current",
            "unrelated",
            UtteranceMode.ASSERTION.value,
            Attribution.SOURCE_ACTOR.value,
            Durability.DURABLE.value,
            ModalForce.NONE.value,
            Atomicity.ATOMIC.value,
        ),
        epistemic_status=EpistemicStatus.EXPLICIT,
        suggested_scope_refs=(episode.origin.primary_scope,),
        related_memory_ids=(),
        evidence_refs=evidence,
        field_evidence_refs=bind_field_evidence(
            identity,
            values,
            evidence,
            bindings=bindings,
            semantic_contract_version="v3",
        ),
        confidence=0.99,
        extractor_version="test_extractor_v3",
        prompt_version="test_prompt_v3",
        semantic_contract_version="v3",
        atomic_evidence_ref=evidence[0],
        metadata={
            "source_role": "user",
            "semantic_contract_validated": True,
            "atomic_evidence_validated": True,
            "transition_evidence_validated": True,
        },
    )


def _create_nonreviewable_pending(
    client: MemoryOSClient,
    archive: SessionArchive,
    proposal: MemorySemanticProposal,
    reason: PendingReason,
) -> dict:
    client.session_archive_store.write_sync_archive(archive)
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    formed = client.session_commit_service.memory_planner.formation.plan_pending(
        proposal,
        archive=archive,
        episode=episode,
        reason=reason.value,
        retrieval_views=["project:memoryos:decisions"],
        commit_group_id=f"create:{archive.task_id}",
    )
    client.committer.commit("u1", list(formed.operations))
    return client.list_pending(user_id="u1", tenant_id="t1", lifecycle_states=["PENDING"])[0]


def test_nonreviewable_pending_requires_new_linked_proposal_and_one_receipt(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="t1")
    old_archive = _archive(
        "old",
        "The primary storage backend is SQLite.",
        task_id="old-extraction-task",
    )
    old_proposal = _proposal(old_archive, "SQLite", "old-invalid-proposal")
    pending = _create_nonreviewable_pending(
        client,
        old_archive,
        old_proposal,
        PendingReason.NEEDS_SCHEMA_REPAIR,
    )

    for decision in ("CONFIRM", "CONFIRM_AND_APPLY"):
        with pytest.raises(ValueError, match="does not allow"):
            client.review_pending(
                user_id="u1",
                pending_uri=pending["uri"],
                decision=decision,
                expected_lifecycle_revision=pending["lifecycle_revision"],
                expected_proposal_fingerprint=pending["proposal_fingerprint"],
                command_id=f"forbidden-{decision}",
                tenant_id="t1",
            )

    corrected_archive = _archive(
        "corrected",
        "The primary storage backend is PostgreSQL.",
        task_id="new-extraction-task",
    )
    client.session_archive_store.write_sync_archive(corrected_archive)
    corrected_proposal = _proposal(corrected_archive, "PostgreSQL", "corrected-proposal")
    request = {
        "user_id": "u1",
        "pending_uri": pending["uri"],
        "decision": "CORRECT",
        "expected_lifecycle_revision": pending["lifecycle_revision"],
        "expected_proposal_fingerprint": pending["proposal_fingerprint"],
        "command_id": "correct-schema-pending",
        "tenant_id": "t1",
        "corrected_proposal": corrected_proposal,
    }
    result = client.review_pending(**request)
    assert client.review_pending(**request) == result
    assert result["status"] == LifecycleState.REJECTED.value
    assert len(result["corrected_claim_uris"]) == 1

    repository = CanonicalMemoryRepository(client.source_store, client.relation_store)
    predecessor = repository.load_pending(
        pending["uri"],
        tenant_id="t1",
        owner_user_id="u1",
    )
    assert predecessor.lifecycle_state == LifecycleState.REJECTED
    assert predecessor.lifecycle_history[-1]["reason"].startswith("structured_correction:")
    claim_uri = result["corrected_claim_uris"][0]
    claims = repository.load_uri(claim_uri.rsplit("/claims/", 1)[0])[1]
    claim = next(item for item in claims if item.uri == claim_uri)
    assert claim.current.qualifiers["corrects_pending_uri"] == pending["uri"]
    assert claim.current.qualifiers["corrects_pending_fingerprint"] == pending["proposal_fingerprint"]

    root = artifact_root_for(client.source_store)
    assert root is not None
    pending_head, pending_receipt, _pending_snapshot = load_current_head(
        root,
        pending["uri"],
        canonical_kind="pending_proposal",
    )
    claim_head, claim_receipt, _claim_snapshot = load_current_head(
        root,
        result["corrected_claim_uris"][0],
        canonical_kind="claim",
    )
    assert pending_head["receipt_digest"] == claim_head["receipt_digest"]
    assert pending_receipt["receipt_digest"] == claim_receipt["receipt_digest"]
    correction = next(
        item
        for item in pending_receipt["operations"]
        if dict(item.get("payload", {}) or {}).get("canonical_pending_correction") is True
    )
    assert correction["payload"]["corrected_claim_uris"] == result["corrected_claim_uris"]


def test_fallback_correction_requires_a_new_extraction_task(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="t1")
    archive = _archive(
        "fallback",
        "The primary storage backend is SQLite and PostgreSQL.",
        task_id="fallback-task",
    )
    old = _proposal(archive, "SQLite", "fallback-old")
    pending = _create_nonreviewable_pending(
        client,
        archive,
        old,
        PendingReason.FALLBACK_REQUIRES_REEXTRACTION,
    )
    corrected = _proposal(archive, "PostgreSQL", "fallback-corrected")
    with pytest.raises(ValueError, match="re-extraction in a new task"):
        client.review_pending(
            user_id="u1",
            pending_uri=pending["uri"],
            decision="CORRECT",
            expected_lifecycle_revision=pending["lifecycle_revision"],
            expected_proposal_fingerprint=pending["proposal_fingerprint"],
            command_id="same-task-correction",
            tenant_id="t1",
            corrected_proposal=corrected,
        )
    current = CanonicalMemoryRepository(client.source_store, client.relation_store).load_pending(
        pending["uri"],
        tenant_id="t1",
        owner_user_id="u1",
    )
    assert current.lifecycle_state == LifecycleState.PENDING


def test_historical_only_correction_cannot_resolve_pending_as_current_effect(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="t1")
    pending_archive = _archive(
        "historical-pending",
        "The primary storage backend needs schema repair.",
        task_id="historical-pending-task",
    )
    pending = _create_nonreviewable_pending(
        client,
        pending_archive,
        _proposal(pending_archive, "SQLite", "historical-pending-proposal"),
        PendingReason.NEEDS_SCHEMA_REPAIR,
    )
    corrected_archive = _archive(
        "historical-correction",
        "The primary storage backend was PostgreSQL.",
        task_id="historical-correction-task",
    )
    client.session_archive_store.write_sync_archive(corrected_archive)
    corrected_proposal = _proposal(
        corrected_archive,
        "PostgreSQL",
        "historical-corrected-proposal",
    )
    command_id = "historical-correction-command"
    command = PendingReviewCommandStore(tmp_path, tenant_id="t1").begin(
        command_id,
        owner_user_id="u1",
        pending_uri=pending["uri"],
        decision="CORRECT",
        expected_lifecycle_revision=pending["lifecycle_revision"],
        expected_proposal_fingerprint=pending["proposal_fingerprint"],
        reason="historical correction boundary test",
        correction_proposal_digest=stable_hash(
            [corrected_proposal.to_dict()],
            length=64,
        ),
    )
    formation = client.session_commit_service.memory_planner.formation
    formed = formation.plan_pending_correction(
        pending["uri"],
        corrected_proposal,
        archive=corrected_archive,
        episode=SessionArchiveEpisodeAdapter().adapt(corrected_archive),
        tenant_id="t1",
        owner_user_id="u1",
        commit_group_id="historical-correction-group",
        reason=f"structured_correction:{command_id}",
        review_command_id=command_id,
        review_decision="CORRECT",
        review_request_digest=str(command["request_digest"]),
    )
    claim_operation = next(
        operation
        for operation in formed.operations
        if isinstance((payload := operation.payload.get("context_object")), dict)
        and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "claim"
        and dict(payload.get("metadata", {}) or {}).get("state") == "ACTIVE"
    )
    claim_metadata = claim_operation.payload["context_object"]["metadata"]
    assert claim_metadata["current_revision"] == 1
    claim_metadata["revisions"][0].setdefault("qualifiers", {})["non_current_historical"] = True

    with pytest.raises(
        ValueError,
        match="current revision pointer selects a historical-only revision",
    ):
        client.committer.commit("u1", list(formed.operations))

    current = CanonicalMemoryRepository(client.source_store, client.relation_store).load_pending(
        pending["uri"],
        tenant_id="t1",
        owner_user_id="u1",
    )
    assert current.lifecycle_state == LifecycleState.PENDING
