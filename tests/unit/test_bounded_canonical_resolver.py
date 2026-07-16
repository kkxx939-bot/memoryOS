from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

import memoryos.contextdb.retrieval.canonical_resolver as resolver_module
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.canonical_resolver import BoundedCanonicalResolver
from memoryos.contextdb.retrieval.fusion import RetrievalCandidate
from memoryos.contextdb.retrieval.packing import ContextPacker, ContextPackingPolicy
from memoryos.contextdb.retrieval.query_plan import RetrievalQueryIntent, RetrievalQueryPlan
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.security.context_projection import ContextProjectionSanitizationError


def _scope() -> dict[str, object]:
    principal: dict[str, object] = {
        "namespace": "memoryos",
        "kind": "principal",
        "id": "u1",
        "parent_id": None,
        "parent_path": [],
        "attributes": {},
        "confidence": 1.0,
        "source": "explicit",
        "inferred": False,
    }
    return {
        "canonical_subject": principal,
        "applicability": {"all_of": [principal]},
        "visibility": {
            "tenant_id": "tenant-a",
            "allowed_principal_ids": ["u1"],
            "allowed_service_ids": [],
            "private": True,
        },
        "authority": {"principal_ids": ["u1"], "service_ids": [], "inferred": False},
        "origin_refs": [],
    }


def _candidate(slot_id: str, claim_id: str) -> RetrievalCandidate:
    slot_uri = f"memoryos://user/u1/memories/canonical/slots/{slot_id}"
    return RetrievalCandidate(
        record_key=f"slot:{slot_id}:current",
        uri=f"{slot_uri}/serving/current",
        title="current state",
        context_type="memory",
        source_kind="canonical_current_slot",
        record_kind="current_slot",
        canonical_slot_id=slot_id,
        canonical_claim_id=claim_id,
        canonical_revision=1,
        metadata={"canonical_slot_uri": slot_uri},
    )


def test_failed_claim_read_is_counted_and_never_exceeds_bounded_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def read(_source: object, uri: str, _relations: object) -> object:
        calls.append(uri)
        if "/claims/" in uri:
            raise FileNotFoundError(uri)
        slot_id = uri.rsplit("/", 1)[-1]
        slot = ContextObject(
            uri=uri,
            context_type=ContextType.MEMORY,
            title="slot",
            owner_user_id="u1",
            tenant_id="tenant-a",
            metadata={
                "canonical_kind": "slot",
                "slot_id": slot_id,
                "active_claim_id": f"claim-{slot_id}",
                "scope": _scope(),
                "asserted_by": "u1",
            },
        )
        return SimpleNamespace(object=slot)

    monkeypatch.setattr(resolver_module, "read_committed_canonical", read)
    plan = RetrievalQueryPlan(
        semantic_query="state",
        tenant_id="tenant-a",
        owner_user_id="u1",
        query_intent=RetrievalQueryIntent.CURRENT,
        candidate_limit=3,
        final_limit=2,
        metadata_filters={"applicability_scope_keys": ["memoryos:principal:u1"]},
    )

    result = BoundedCanonicalResolver(cast(SourceStore, object())).resolve(
        (_candidate("slot-a", "claim-slot-a"), _candidate("slot-b", "claim-slot-b")),
        plan=plan,
    )

    assert result.candidates == ()
    assert result.source_reads == len(calls) == 4
    assert result.source_reads <= plan.candidate_limit + 2
    assert [item["drop_reason"] for item in result.dropped] == [
        "canonical_unavailable",
        "canonical_unavailable",
    ]


def test_source_read_bound_keeps_reranked_prefix_and_marks_lower_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = BoundedCanonicalResolver(cast(SourceStore, object()))
    validation_order: list[str] = []

    def validate_current(
        candidate: RetrievalCandidate,
        _plan: RetrievalQueryPlan,
    ) -> tuple[RetrievalCandidate, int]:
        validation_order.append(candidate.record_key)
        return candidate, 2

    monkeypatch.setattr(resolver, "_validate_current", validate_current)
    candidates = tuple(_candidate(f"slot-{index}", f"claim-{index}") for index in range(8))
    plan = RetrievalQueryPlan(
        semantic_query="ranked state",
        tenant_id="tenant-a",
        owner_user_id="u1",
        query_intent=RetrievalQueryIntent.CURRENT,
        candidate_limit=10,
        final_limit=5,
    )

    result = resolver.resolve(candidates, plan=plan)

    expected_prefix = [candidate.record_key for candidate in candidates[:6]]
    assert validation_order == expected_prefix
    assert [candidate.record_key for candidate in result.candidates] == expected_prefix
    assert [item["record_key"] for item in result.dropped] == [
        candidates[6].record_key,
        candidates[7].record_key,
    ]
    assert {item["canonical_validation_status"] for item in result.dropped} == {
        "not_validated_bound"
    }
    assert result.canonical_candidates == 8
    assert result.canonical_validated == 6
    assert result.source_reads == plan.candidate_limit + 2


def _canonical_objects(
    *,
    slot_id: str,
    claim_id: str,
    value: str,
    revisions: list[dict[str, object]] | None = None,
) -> tuple[ContextObject, ContextObject]:
    slot_uri = f"memoryos://user/u1/memories/canonical/slots/{slot_id}"
    revision_rows = revisions or [
        {
            "revision": 1,
            "state": "ACTIVE",
            "value_fields": {"canonical_value": value},
            "valid_from": "2026-07-14T00:00:00+00:00",
            "valid_to": None,
            "qualifiers": {},
        }
    ]
    slot = ContextObject(
        uri=slot_uri,
        context_type=ContextType.MEMORY,
        title="slot",
        owner_user_id="u1",
        tenant_id="tenant-a",
        metadata={
            "canonical_kind": "slot",
            "slot_id": slot_id,
            "active_claim_id": claim_id,
            "scope": _scope(),
            "asserted_by": "u1",
        },
    )
    claim = ContextObject(
        uri=f"{slot_uri}/claims/{claim_id}",
        context_type=ContextType.MEMORY,
        title=f"canonical value {value}",
        owner_user_id="u1",
        tenant_id="tenant-a",
        metadata={
            "canonical_kind": "claim",
            "memory_type": "preference",
            "slot_id": slot_id,
            "claim_id": claim_id,
            "canonical_value": value,
            "revision": len(revision_rows),
            "current_revision": len(revision_rows),
            "state": str(revision_rows[-1]["state"]),
            "revisions": revision_rows,
            "scope": _scope(),
            "asserted_by": "u1",
        },
    )
    return slot, claim


def _install_canonical_reads(
    monkeypatch: pytest.MonkeyPatch,
    *,
    slot: ContextObject,
    claim: ContextObject,
) -> None:
    def read(_source: object, uri: str, _relations: object) -> object:
        return SimpleNamespace(object=claim if "/claims/" in uri else slot)

    monkeypatch.setattr(resolver_module, "read_committed_canonical", read)


def test_current_source_egress_sanitizes_text_title_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "api_key=sk-canonical-secret-123456 /Users/u1/Desktop/private-plan.md"
    slot, claim = _canonical_objects(slot_id="slot-secret", claim_id="claim-secret", value=secret)
    _install_canonical_reads(monkeypatch, slot=slot, claim=claim)
    resolver = BoundedCanonicalResolver(cast(SourceStore, object()))
    monkeypatch.setattr(resolver, "_current_slot_proof_matches", lambda *_args, **_kwargs: True)
    plan = RetrievalQueryPlan(
        semantic_query="secret state",
        tenant_id="tenant-a",
        owner_user_id="u1",
        query_intent=RetrievalQueryIntent.CURRENT,
        candidate_limit=3,
        final_limit=2,
        metadata_filters={"applicability_scope_keys": ["memoryos:principal:u1"]},
    )

    result = resolver.resolve((_candidate("slot-secret", "claim-secret"),), plan=plan)

    assert len(result.candidates) == 1
    resolved = result.candidates[0]
    public_payload = repr((resolved.title, resolved.text, dict(resolved.metadata)))
    assert "canonical-secret-123456" not in public_payload
    assert "/Users/u1" not in public_payload
    assert resolved.metadata["canonical_value"] == "api_key=<redacted> desktop/private-plan.md"
    assert resolved.metadata["projection_sanitized"] is True
    assert resolved.metadata["projection_redacted"] is True


def test_current_result_rebuilds_business_metadata_from_proved_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authoritative_value = "authoritative current B"
    tampered_value = "unproved current A"
    slot, claim = _canonical_objects(
        slot_id="slot-current-metadata",
        claim_id="claim-current-metadata",
        value=authoritative_value,
    )
    slot.metadata.update(
        {
            "memory_type": "preference",
            "identity_algorithm_version": "identity-v2",
            "identity_fields": {"subject": "dessert", "dimension": "flavor"},
            "scope_keys": ["memoryos:principal:u1"],
            "revision": 7,
        }
    )
    claim.metadata.update(
        {
            "identity_algorithm_version": "identity-v2",
            "transition_profile": "AUTHORITATIVE_STATE",
            "connect": {"run_mode": "context_reduction", "adapter_id": "codex"},
            "retrieval_views": ["project:memoryos:rules"],
        }
    )
    _install_canonical_reads(monkeypatch, slot=slot, claim=claim)
    resolver = BoundedCanonicalResolver(cast(SourceStore, object()))
    monkeypatch.setattr(resolver, "_current_slot_proof_matches", lambda *_args, **_kwargs: True)
    candidate = replace(
        _candidate("slot-current-metadata", "claim-current-metadata"),
        metadata={
            "canonical_slot_uri": slot.uri,
            "catalog_record_key": "slot:slot-current-metadata:current",
            "retrieval_scores": {"lexical": 0.75},
            "current_leak": tampered_value,
            "canonical_value": tampered_value,
            "value_fields": {"canonical_value": tampered_value},
            "revisions": [{"revision": 99, "value_fields": {"canonical_value": tampered_value}}],
            "evidence_refs": [{"excerpt": tampered_value}],
            "field_evidence_refs": {"value.canonical_value": [{"excerpt": tampered_value}]},
            "identity_fields": {"tampered": tampered_value},
            "memory_type": tampered_value,
            "qualifiers": {"display_fields": {"summary": tampered_value}},
            "proposal_id": tampered_value,
            "relation": tampered_value,
            "previous_revision": tampered_value,
            "connect": {"run_mode": "action_capable", "leak": tampered_value},
            "retrieval_views": [tampered_value],
        },
    )
    plan = RetrievalQueryPlan(
        semantic_query="current metadata",
        tenant_id="tenant-a",
        owner_user_id="u1",
        query_intent=RetrievalQueryIntent.CURRENT,
        candidate_limit=3,
        final_limit=2,
        metadata_filters={"applicability_scope_keys": ["memoryos:principal:u1"]},
    )

    result = resolver.resolve((candidate,), plan=plan)

    assert len(result.candidates) == 1
    metadata = dict(result.candidates[0].metadata)
    assert tampered_value not in repr(metadata)
    assert "current_leak" not in metadata
    assert metadata["canonical_value"] == authoritative_value
    assert metadata["value_fields"] == {"canonical_value": authoritative_value}
    assert metadata["revisions"][0]["value_fields"] == {"canonical_value": authoritative_value}
    assert metadata["identity_fields"] == {"subject": "dessert", "dimension": "flavor"}
    assert metadata["memory_type"] == "preference"
    assert metadata["connect"] == {"run_mode": "context_reduction", "adapter_id": "codex"}
    assert metadata["retrieval_views"] == ["project:memoryos:rules"]
    assert metadata["retrieval_scores"] == {"lexical": 0.75}


def test_history_source_egress_sanitizes_revision_value_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "Authorization: Bearer canonical-history-secret /Users/u1/.ssh/id_rsa"
    revisions: list[dict[str, object]] = [
        {
            "revision": 1,
            "state": "ACTIVE",
            "value_fields": {"canonical_value": secret},
            "valid_from": "2026-07-14T00:00:00+00:00",
            "valid_to": None,
            "qualifiers": {},
        }
    ]
    slot, claim = _canonical_objects(
        slot_id="slot-history-secret",
        claim_id="claim-history-secret",
        value=secret,
        revisions=revisions,
    )
    _install_canonical_reads(monkeypatch, slot=slot, claim=claim)
    resolver = BoundedCanonicalResolver(cast(SourceStore, object()))
    monkeypatch.setattr(
        resolver,
        "_validate_historical_publication",
        lambda *_args, **_kwargs: dict(revisions[0]),
    )
    claim_uri = claim.uri
    candidate = RetrievalCandidate(
        record_key="claim:claim-history-secret:revision:1",
        uri=claim_uri,
        title="stale wrong title canonical-history-secret",
        text="stale wrong L2 canonical-history-secret /Users/u1/private.txt",
        l0_text="stale wrong L0 canonical-history-secret",
        l1_text="stale wrong L1 canonical-history-secret",
        context_type="memory",
        source_kind="canonical_claim",
        record_kind="claim_revision",
        canonical_slot_id="slot-history-secret",
        canonical_claim_id="claim-history-secret",
        canonical_revision=1,
        metadata={
            "canonical_claim_uri": claim_uri,
            "source_revision": 1,
            "canonical_value": secret,
        },
    )
    plan = RetrievalQueryPlan(
        semantic_query="history",
        tenant_id="tenant-a",
        owner_user_id="u1",
        query_intent=RetrievalQueryIntent.HISTORY,
        candidate_limit=3,
        final_limit=2,
        token_budget=8,
        metadata_filters={"applicability_scope_keys": ["memoryos:principal:u1"]},
    )

    result = resolver.resolve((candidate,), plan=plan)

    assert len(result.candidates) == 1
    resolved = result.candidates[0]
    public_payload = repr((resolved.title, resolved.text, dict(resolved.metadata)))
    assert "canonical-history-secret" not in public_payload
    assert "/Users/u1" not in public_payload
    assert "stale wrong" not in repr(
        (resolved.title, resolved.text, resolved.l0_text, resolved.l1_text)
    )
    assert resolved.text == resolved.l0_text == resolved.l1_text
    assert resolved.metadata["projection_sanitized"] is True
    assert resolved.metadata["projection_redacted"] is True
    packed = ContextPacker(
        policy=ContextPackingPolicy(max_l2_items=0),
    ).pack(result.candidates, plan=plan)
    packed_payload = repr(packed)
    assert packed["selected_count"] == 1
    assert "canonical-history-secret" not in packed_payload
    assert "/Users/u1" not in packed_payload
    assert "stale wrong" not in packed_payload


@pytest.mark.parametrize(
    "intent",
    (
        RetrievalQueryIntent.HISTORY,
        RetrievalQueryIntent.AS_OF,
        RetrievalQueryIntent.CONFLICTS,
        RetrievalQueryIntent.OPTIONS,
    ),
)
def test_revision_result_metadata_is_bound_to_requested_revision_value(
    monkeypatch: pytest.MonkeyPatch,
    intent: RetrievalQueryIntent,
) -> None:
    current_value = "current aggregate A"
    historical_value = "late non-current revision B"
    revisions: list[dict[str, object]] = [
        {
            "revision": 1,
            "state": "ACTIVE",
            "value_fields": {"canonical_value": current_value},
            "valid_from": "2026-07-01T00:00:00+00:00",
            "valid_to": None,
            "qualifiers": {},
        },
        {
            "revision": 2,
            "state": "ACTIVE",
            "value_fields": {"canonical_value": historical_value},
            "valid_from": "2026-06-01T00:00:00+00:00",
            "valid_to": None,
            "qualifiers": {"non_current_historical": True},
        },
    ]
    slot, claim = _canonical_objects(
        slot_id="slot-late-history",
        claim_id="claim-late-history",
        value=current_value,
        revisions=revisions,
    )
    _install_canonical_reads(monkeypatch, slot=slot, claim=claim)
    resolver = BoundedCanonicalResolver(cast(SourceStore, object()))
    monkeypatch.setattr(
        resolver,
        "_validate_historical_publication",
        lambda *_args, **_kwargs: dict(revisions[1]),
    )
    candidate = RetrievalCandidate(
        record_key="claim:claim-late-history:revision:2",
        uri=claim.uri,
        title=current_value,
        text=current_value,
        context_type="memory",
        source_kind="canonical_claim",
        record_kind="claim_revision",
        canonical_slot_id="slot-late-history",
        canonical_claim_id="claim-late-history",
        canonical_revision=2,
        metadata={
            "canonical_claim_uri": claim.uri,
            "source_revision": 2,
            "canonical_value": current_value,
            "current_leak": current_value,
            "revision": current_value,
            "current_revision": current_value,
            "value_fields": {"canonical_value": current_value},
            "revisions": [{"revision": 99, "value_fields": {"canonical_value": current_value}}],
            "evidence_refs": [{"excerpt": current_value}],
            "field_evidence": {"value.canonical_value": current_value},
            "field_evidence_refs": {"value.canonical_value": [{"excerpt": current_value}]},
            "qualifiers": {"display_fields": {"summary": current_value}},
            "display_fields": {"summary": current_value},
            "display_field_evidence_refs": {"summary": [{"excerpt": current_value}]},
            "proposal_id": current_value,
            "relation": current_value,
            "previous_revision": current_value,
        },
    )
    plan = RetrievalQueryPlan(
        semantic_query="historical value",
        tenant_id="tenant-a",
        owner_user_id="u1",
        query_intent=intent,
        valid_at=("2026-06-15T00:00:00+00:00" if intent == RetrievalQueryIntent.AS_OF else None),
        candidate_limit=3,
        final_limit=2,
        metadata_filters={"applicability_scope_keys": ["memoryos:principal:u1"]},
    )

    result = resolver.resolve((candidate,), plan=plan)

    assert len(result.candidates) == 1
    resolved = result.candidates[0]
    assert resolved.title == historical_value
    assert resolved.text == historical_value
    assert resolved.metadata["canonical_value"] == historical_value
    assert resolved.metadata["value_fields"] == {"canonical_value": historical_value}
    assert resolved.metadata["revisions"][0]["value_fields"] == {"canonical_value": historical_value}
    assert "current_leak" not in resolved.metadata
    assert current_value not in repr((resolved.title, resolved.text, dict(resolved.metadata)))


def test_canonical_source_sanitization_failure_drops_candidate_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slot, claim = _canonical_objects(
        slot_id="slot-fail-closed",
        claim_id="claim-fail-closed",
        value="safe value",
    )
    _install_canonical_reads(monkeypatch, slot=slot, claim=claim)
    resolver = BoundedCanonicalResolver(cast(SourceStore, object()))
    monkeypatch.setattr(resolver, "_current_slot_proof_matches", lambda *_args, **_kwargs: True)

    def fail(**_kwargs: object) -> object:
        raise ContextProjectionSanitizationError("forced sanitizer failure")

    monkeypatch.setattr(resolver.sanitizer, "sanitize", fail)
    plan = RetrievalQueryPlan(
        semantic_query="state",
        tenant_id="tenant-a",
        owner_user_id="u1",
        query_intent=RetrievalQueryIntent.CURRENT,
        candidate_limit=3,
        final_limit=2,
        metadata_filters={"applicability_scope_keys": ["memoryos:principal:u1"]},
    )

    result = resolver.resolve((_candidate("slot-fail-closed", "claim-fail-closed"),), plan=plan)

    assert result.candidates == ()
    assert result.canonical_validated == 0
    assert result.dropped[0]["drop_reason"] == "canonical_unavailable"
    assert result.dropped[0]["error_type"] == "ContextProjectionSanitizationError"
