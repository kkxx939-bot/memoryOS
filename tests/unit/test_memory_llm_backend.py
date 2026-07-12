from __future__ import annotations

import json
from dataclasses import replace

import pytest

from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical import MemorySemanticProposal, SessionArchiveEpisodeAdapter
from memoryos.memory.canonical.prefetch import PrefetchedMemory
from memoryos.memory.canonical.proposal import (
    Atomicity,
    Attribution,
    Durability,
    ModalForce,
    NormalizedSemanticAssessment,
    UtteranceMode,
)
from memoryos.memory.extraction import FakeMemoryModelProvider, LLMMemoryExtractorBackend
from memoryos.memory.extraction.llm_backend import MemoryExtractionPromptBuilder
from memoryos.memory.schema import AdmissionDecision, MemoryTypeRegistry


def _archive(content: str = "Please remember this memory extraction test.") -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"id": "m1", "role": "user", "content": content}],
        metadata={"connect": {"adapter_id": "codex"}},
    )


def _atomic_ref(event_id: str, text: str = "I") -> dict:
    return {"event_id": event_id, "span_start": 0, "span_end": len(text)}


def _semantic_fields(**overrides: str) -> dict[str, str]:
    semantic = {
        "speech_act": "confirmation",
        "commitment": "confirmed",
        "temporal_scope": "current",
        "relation_to_existing": "unrelated",
        "utterance_mode": "assertion",
        "attribution": "source_actor",
        "durability": "durable",
        "modal_force": "none",
        "atomicity": "atomic",
    }
    semantic.update(overrides)
    return semantic


def _field_evidence(
    identity_fields: dict,
    value_fields: dict,
    event_id: str,
    *,
    atomic_ref: dict | None = None,
) -> dict:
    atomic = [atomic_ref or _atomic_ref(event_id)]
    return {
        **{f"identity.{key}": atomic for key in identity_fields},
        **{f"value.{key}": atomic for key in value_fields},
        **{f"semantic.{key}": atomic for key in _semantic_fields()},
        "transition": atomic,
    }


def test_fake_llm_extractor_backend_outputs_semantic_proposals() -> None:
    text = "I prefer findings first during code reviews."
    atomic_ref = _atomic_ref("m1", text)
    response = json.dumps(
        {
            "candidates": [
                {
                    "memory_type": "preference",
                    "proposal_id": "p-review-style",
                    "identity_fields": {"subject": "code reviews", "dimension": "findings first"},
                    "value_fields": {"canonical_value": "findings first during code reviews"},
                    "semantic": _semantic_fields(modal_force="prefer"),
                    "epistemic_status": "EXPLICIT",
                    "suggested_scope_refs": [],
                    "evidence_refs": [{"event_id": "m1"}],
                    "atomic_evidence_ref": atomic_ref,
                    "field_evidence_refs": _field_evidence(
                        {"subject": "code reviews", "dimension": "findings first"},
                        {"canonical_value": "findings first during code reviews"},
                        "m1",
                        atomic_ref=atomic_ref,
                    ),
                    "confidence": 0.9,
                    "source_role": "user",
                }
            ]
        }
    )
    backend = LLMMemoryExtractorBackend(FakeMemoryModelProvider(response))
    archive = _archive(text)

    operations = MemoryCommitPlanner(extractor=backend).plan(archive)

    assert operations.operations[0].payload["admission"]["decision"] == AdmissionDecision.ACCEPT.value


def test_public_llm_extract_returns_only_semantic_proposals() -> None:
    candidate = _semantic_candidate()
    backend = LLMMemoryExtractorBackend(FakeMemoryModelProvider(json.dumps({"candidates": [candidate]})))

    extracted = backend.extract(
        _archive("I confirm my answers preference: concise answers."),
        MemoryTypeRegistry().list(),
    )

    assert extracted and all(isinstance(item, MemorySemanticProposal) for item in extracted)
    assert extracted[0].semantic_contract_version == "v3"
    assert extracted[0].atomic_evidence_ref is not None


@pytest.mark.parametrize(
    "text",
    [
        "我的职业是软件测试工程师。",
        "Je travaille comme ingénieur logiciel.",
        "本项目数据库选型定为 CockroachDB。",
    ],
)
def test_semantic_extractor_runs_before_rule_based_salience_telemetry(text: str) -> None:
    prompts: list[str] = []
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": []}), prompts=prompts)
    )

    planned = MemoryCommitPlanner(extractor=backend).plan(_archive(text))

    assert len(prompts) == 1
    assert planned.operations == ()


def test_remote_provider_never_receives_secret_bearing_prompt() -> None:
    prompts: list[str] = []

    class CapturingRemoteProvider:
        is_remote = True

        def complete(self, prompt: str) -> str:
            prompts.append(prompt)
            return json.dumps({"candidates": []})

    archive = _archive("Remember this: OPENAI_API_KEY=sk-live-secret")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    result = LLMMemoryExtractorBackend(CapturingRemoteProvider()).extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert prompts == []
    assert result.accepted == ()
    assert result.security_flags == ("privacy_egress_blocked",)


def test_opaque_existing_candidate_ref_maps_to_internal_ids_without_exposing_uri() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    candidate = _semantic_candidate(
        semantic=_semantic_fields(relation_to_existing="duplicate"),
        related_candidate_refs=["existing_0"],
    )
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [candidate]}))
    )
    existing = PrefetchedMemory(
        uri="memoryos://user/u1/memories/canonical/slots/s1/claims/c1",
        memory_type="preference",
        state="ACTIVE",
        revision=1,
        slot_id="s1",
        claim_id="c1",
        canonical_value="concise",
        identity_fields={"subject": "answers", "dimension": "preference"},
        scope={},
        l0="concise",
        l1="concise answers",
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(existing,),
        episode=episode,
    )

    assert len(result.accepted) == 1
    assert result.accepted[0].related_memory_ids == ()
    assert result.accepted[0].related_slot_ids == ("s1",)
    assert result.accepted[0].related_claim_ids == ("c1",)


def test_replacement_and_supplement_require_one_compatible_active_target() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    active = PrefetchedMemory(
        uri="memoryos://user/u1/memories/canonical/slots/s1/claims/c1",
        memory_type="preference",
        state="ACTIVE",
        revision=1,
        slot_id="s1",
        claim_id="c1",
        canonical_value="concise",
        identity_fields={"subject": "answers", "dimension": "length"},
        scope={},
        l0="concise",
        l1="concise answers",
    )
    cases = [
        (_semantic_candidate(semantic=_semantic_fields(relation_to_existing="supersedes")), (active,)),
        (
            _semantic_candidate(
                semantic=_semantic_fields(relation_to_existing="supplements"),
                related_candidate_refs=["existing_0", "existing_1"],
            ),
            (active, active),
        ),
        (
            _semantic_candidate(
                semantic=_semantic_fields(relation_to_existing="supersedes"),
                related_candidate_refs=["existing_0"],
            ),
            (replace(active, state="PROPOSED"),),
        ),
    ]

    for candidate, existing in cases:
        backend = LLMMemoryExtractorBackend(
            FakeMemoryModelProvider(json.dumps({"candidates": [candidate]}))
        )
        result = backend.extract_batch_with_context(
            archive,
            MemoryTypeRegistry().list(),
            existing_memories=existing,
            episode=episode,
        )
        assert result.accepted == ()
        assert "related active claim" in result.rejected[0].reason or "compatible active claim" in result.rejected[0].reason


def test_v2_proposal_payload_remains_readable_but_new_semantics_fail_closed() -> None:
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [_semantic_candidate()]}))
    )
    proposal = backend.extract(
        _archive("I confirm my answers preference: concise answers."),
        MemoryTypeRegistry().list(),
    )[0]
    payload = proposal.to_dict()
    payload.pop("semantic_contract_version")
    payload.pop("atomic_evidence_ref")
    for field_name in ("utterance_mode", "attribution", "durability", "modal_force", "atomicity"):
        payload["semantic"].pop(field_name)

    restored = MemorySemanticProposal.from_dict(payload)

    assert restored.semantic_contract_version == "v2"
    assert restored.atomic_evidence_ref is None
    assert restored.semantic.utterance_mode == UtteranceMode.UNKNOWN
    assert restored.semantic.attribution == Attribution.UNKNOWN
    assert restored.semantic.durability == Durability.UNKNOWN
    assert restored.semantic.modal_force == ModalForce.UNKNOWN
    assert restored.semantic.atomicity == Atomicity.UNKNOWN
    assert isinstance(restored.semantic, NormalizedSemanticAssessment)
    assert restored.semantic.schema_safe is False


def test_v3_candidate_requires_all_semantic_fields_and_exact_atomic_span() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    missing_semantic = dict(_semantic_candidate()["semantic"])
    missing_semantic.pop("durability")
    candidates = (
        _semantic_candidate(proposal_id="missing-semantic", semantic=missing_semantic),
        _semantic_candidate(
            proposal_id="missing-atomic-span",
            atomic_evidence_ref={"event_id": "m1"},
        ),
    )
    result = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": candidates}))
    ).extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert result.accepted == ()
    assert [item.proposal_id for item in result.rejected] == ["missing-semantic", "missing-atomic-span"]
    assert "missing:durability" in result.rejected[0].reason
    assert "requires an exact span" in result.rejected[1].reason


def test_v3_semantic_fields_must_bind_only_to_atomic_evidence() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    candidate = _semantic_candidate()
    candidate["field_evidence_refs"]["semantic.durability"] = [{"event_id": "m1"}]

    result = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [candidate]}))
    ).extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert result.accepted == ()
    assert "semantic.durability must bind only to atomic_evidence_ref" in result.rejected[0].reason


def test_direct_operation_parser_is_not_publicly_exported() -> None:
    import memoryos.memory.extraction as extraction

    assert not hasattr(extraction, "MemoryExtractionJsonParser")


def test_llm_backend_cannot_bypass_admission() -> None:
    raw_text = "pytest failure: pytest failed\nTraceback (most recent call last):\nAssertionError"
    secret_text = "My preference is api_key=sk-test secret"
    raw_ref = _atomic_ref("m1", raw_text)
    secret_ref = _atomic_ref("m2", secret_text)
    response = json.dumps(
        {
            "candidates": [
                {
                    "memory_type": "preference",
                    "proposal_id": "p-raw",
                    "identity_fields": {"subject": "pytest", "dimension": "failure"},
                    "value_fields": {
                        "canonical_value": "pytest failed\nTraceback (most recent call last):\nAssertionError"
                    },
                        "semantic": _semantic_fields(
                            speech_act="observation",
                            commitment="weak",
                            temporal_scope="past",
                        ),
                    "epistemic_status": "EXPLICIT",
                    "suggested_scope_refs": [],
                        "evidence_refs": [{"event_id": "m1"}],
                        "atomic_evidence_ref": raw_ref,
                        "field_evidence_refs": _field_evidence(
                        {"subject": "pytest", "dimension": "failure"},
                        {"canonical_value": "pytest failed\nTraceback (most recent call last):\nAssertionError"},
                            "m1",
                            atomic_ref=raw_ref,
                    ),
                    "confidence": 0.99,
                    "source_role": "user",
                },
                {
                    "memory_type": "preference",
                    "proposal_id": "p-secret",
                    "identity_fields": {"subject": "api_key", "dimension": "secret"},
                    "value_fields": {"canonical_value": "api_key=sk-test"},
                        "semantic": _semantic_fields(),
                    "epistemic_status": "EXPLICIT",
                    "suggested_scope_refs": [],
                        "evidence_refs": [{"event_id": "m2"}],
                        "atomic_evidence_ref": secret_ref,
                    "field_evidence_refs": _field_evidence(
                        {"subject": "api_key", "dimension": "secret"},
                        {"canonical_value": "api_key=sk-test"},
                            "m2",
                            atomic_ref=secret_ref,
                    ),
                    "confidence": 0.99,
                    "source_role": "user",
                },
            ]
        }
    )
    planner = MemoryCommitPlanner(extractor=LLMMemoryExtractorBackend(FakeMemoryModelProvider(response)))

    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[
            {
                "id": "m1",
                "role": "user",
                "content": raw_text,
            },
            {"id": "m2", "role": "user", "content": secret_text},
        ],
        metadata={"connect": {"adapter_id": "codex"}},
    )
    operations = planner.plan(archive)

    assert operations.operations == ()
    assert "privacy_or_sensitivity_risk" in operations.context.salience_reasons
    assert operations.context.proposal_outcomes
    assert all(item.decision != "ACCEPT_FOR_RECONCILE" for item in operations.context.proposal_outcomes)


def test_llm_backend_rejects_illegal_memory_type() -> None:
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [_semantic_candidate(memory_type="tool_log")]}))
    )

    archive = _archive()
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )
    assert result.accepted == ()
    assert len(result.rejected) == 1
    assert "memory_type is not allowed" in result.rejected[0].reason


def _semantic_candidate(**overrides) -> dict:  # noqa: ANN003
    identity_fields = {"subject": "answers", "dimension": "length"}
    value_fields = {"canonical_value": "concise"}
    atomic_ref = _atomic_ref("m1")
    candidate = {
        "proposal_id": "p1",
        "memory_type": "preference",
        "identity_fields": identity_fields,
        "value_fields": value_fields,
        "semantic": _semantic_fields(),
        "epistemic_status": "EXPLICIT",
        "suggested_scope_refs": [],
        "evidence_refs": [{"event_id": "m1"}],
        "atomic_evidence_ref": atomic_ref,
        "field_evidence_refs": _field_evidence(
            identity_fields,
            value_fields,
            "m1",
            atomic_ref=atomic_ref,
        ),
        "confidence": 0.9,
        "source_role": "user",
    }
    candidate.update(overrides)
    return candidate


@pytest.mark.parametrize("forbidden", ["target_uri", "tenant_id", "user_id", "revision", "visibility_policy", "action"])
def test_semantic_llm_backend_rejects_operation_and_authority_fields(forbidden: str) -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    candidate = _semantic_candidate(**{forbidden: "forged"})
    backend = LLMMemoryExtractorBackend(FakeMemoryModelProvider(json.dumps({"candidates": [candidate]})))
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )
    assert result.accepted == ()
    assert len(result.rejected) == 1
    assert "unknown fields" in result.rejected[0].reason
    assert "forbidden_output_field_rejected" in result.security_flags


def test_semantic_llm_backend_rejects_unknown_evidence_scope_duplicate_ids_and_bad_hash() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    cases = [
        [_semantic_candidate(evidence_refs=[{"event_id": "missing"}])],
        [_semantic_candidate(evidence_refs=[{"event_id": "m1", "event_digest": "bad"}])],
        [_semantic_candidate(evidence_refs=[{"event_id": "m1", "content_hash": "bad"}])],
        [
            _semantic_candidate(),
            _semantic_candidate(),
        ],
        [_semantic_candidate(suggested_scope_refs=[{"namespace": "memoryos", "kind": "workspace", "id": "forged"}])],
        [_semantic_candidate(suggested_scope_refs=["forged"])],
        [_semantic_candidate(related_candidate_refs="not-a-list")],
        [_semantic_candidate(field_evidence_refs={})],
    ]
    for case_index, candidates in enumerate(cases):
        backend = LLMMemoryExtractorBackend(FakeMemoryModelProvider(json.dumps({"candidates": candidates})))
        result = backend.extract_batch_with_context(
            archive,
            MemoryTypeRegistry().list(),
            existing_memories=(),
            episode=episode,
        )
        if case_index == 3:
            assert len(result.accepted) == 1
            assert len(result.rejected) == 1
            assert "duplicate proposal_id" in result.rejected[0].reason
        else:
            assert result.accepted == ()
            assert len(result.rejected) == len(candidates)


def test_semantic_llm_backend_rejects_unknown_response_envelope_fields() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [_semantic_candidate()], "tenant_id": "forged"}))
    )
    with pytest.raises(ValueError, match="response contains unknown fields"):
        backend.extract_with_context(
            archive,
            MemoryTypeRegistry().list(),
            existing_memories=(),
            episode=episode,
        )


def test_llm_source_role_spoof_is_rejected() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"id": "m1", "role": "assistant", "content": "I recommend concise answers."}],
        metadata={"connect": {"adapter_id": "codex"}},
    )
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [_semantic_candidate(source_role="user")]}))
    )
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )
    assert result.accepted == ()
    assert "source_role does not match" in result.rejected[0].reason
    assert "source_authority_rejected" in result.security_flags


def test_llm_pure_system_evidence_preserves_system_source_role() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"id": "m1", "role": "system", "content": "Answers must remain concise."}],
        metadata={"connect": {"adapter_id": "codex"}},
    )
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [_semantic_candidate(source_role="system")]}))
    )
    episode = SessionArchiveEpisodeAdapter().adapt(archive)

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert len(result.accepted) == 1
    assert result.accepted[0].metadata["source_role"] == "system"


def test_llm_system_evidence_uses_system_authority_for_admission() -> None:
    text = "I confirm the answers length preference is concise."
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[
            {
                "id": "m1",
                "role": "system",
                "content": text,
            }
        ],
        metadata={"connect": {"adapter_id": "codex"}},
    )
    atomic_ref = _atomic_ref("m1", text)
    candidate = _semantic_candidate(
        source_role="system",
        semantic=_semantic_fields(modal_force="prefer"),
        atomic_evidence_ref=atomic_ref,
        field_evidence_refs=_field_evidence(
            {"subject": "answers", "dimension": "length"},
            {"canonical_value": "concise"},
            "m1",
            atomic_ref=atomic_ref,
        ),
    )
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [candidate]}))
    )

    operations = MemoryCommitPlanner(extractor=backend).plan(archive)

    assert operations.operations
    assert operations.operations[0].payload["admission"]["decision"] == AdmissionDecision.ACCEPT.value
    object_metadata = operations.operations[0].payload["context_object"]["metadata"]
    assert object_metadata["admission_score_components"]["source_authority"] == 1.0


def test_llm_mixed_context_uses_atomic_assertion_actor_for_source_authority() -> None:
    user_text = "Yes, keep it concise."
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[
            {"id": "m1", "role": "system", "content": "Answers should remain concise."},
            {"id": "m2", "role": "user", "content": user_text},
        ],
        metadata={"connect": {"adapter_id": "codex"}},
    )
    atomic_ref = _atomic_ref("m2", user_text)
    candidate = _semantic_candidate(
        source_role="user",
        evidence_refs=[{"event_id": "m1"}, {"event_id": "m2"}],
        atomic_evidence_ref=atomic_ref,
        field_evidence_refs=_field_evidence(
            {"subject": "answers", "dimension": "length"},
            {"canonical_value": "concise"},
            "m1",
            atomic_ref=atomic_ref,
        ),
    )
    backend = LLMMemoryExtractorBackend(FakeMemoryModelProvider(json.dumps({"candidates": [candidate]})))
    episode = SessionArchiveEpisodeAdapter().adapt(archive)

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert len(result.accepted) == 1
    assert result.accepted[0].metadata["source_role"] == "user"
    assert result.accepted[0].atomic_evidence_ref is not None
    assert result.accepted[0].atomic_evidence_ref.event_id == "m2"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("speech_act", "invented_act"),
        ("commitment", "certainly_maybe"),
        ("temporal_scope", "whenever"),
        ("relation_to_existing", "replaces_everything"),
        ("utterance_mode", "rhetorical_magic"),
        ("attribution", "everybody"),
        ("durability", "forever_maybe"),
        ("modal_force", "force_push"),
        ("atomicity", "several_but_one"),
    ],
)
def test_semantic_llm_backend_rejects_unknown_enums(field_name: str, value: str) -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    semantic = dict(_semantic_candidate()["semantic"])
    semantic[field_name] = value
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [_semantic_candidate(semantic=semantic)]}))
    )
    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )
    assert result.accepted == ()
    assert f"{field_name} is not allowed" in result.rejected[0].reason


def test_semantic_llm_backend_rejects_related_memory_outside_prefetch() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(
            json.dumps({"candidates": [_semantic_candidate(related_candidate_refs=["memoryos://user/u2/forged"])]})
        )
    )
    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )
    assert result.accepted == ()
    assert "illegal reference" in result.rejected[0].reason
    assert "target_reference_rejected" in result.security_flags


def test_llm_candidate_failures_are_isolated_with_audit_reasons() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    candidates = [_semantic_candidate(proposal_id=f"p{index}") for index in range(4)]
    invalid = _semantic_candidate(proposal_id="bad")
    invalid["semantic"] = {**invalid["semantic"], "commitment": "invented"}
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [*candidates, invalid]}))
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert len(result.accepted) == 4
    assert len(result.rejected) == 1
    assert result.rejected[0].proposal_id == "bad"
    assert "commitment is not allowed" in result.rejected[0].reason


def test_v3_parser_rejects_missing_canonical_value_without_dropping_valid_candidate() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    values = {"preference": "concise answers"}
    invalid = _semantic_candidate(
        proposal_id="missing-canonical-value",
        value_fields=values,
        field_evidence_refs=_field_evidence(
            {"subject": "answers", "dimension": "length"},
            values,
            "m1",
        ),
    )
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(
            json.dumps({"candidates": [_semantic_candidate(proposal_id="valid"), invalid]})
        )
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert [item.proposal_id for item in result.accepted] == ["valid"]
    assert [item.proposal_id for item in result.rejected] == ["missing-canonical-value"]
    assert "requires value_fields.canonical_value" in result.rejected[0].reason


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("speech_act", "future_option"),
        ("speech_act", "evaluation request"),
        ("attribution", "source actor"),
        ("modal_force", "conditional-forbid"),
    ],
)
def test_v3_parser_rejects_undeclared_enum_alias(field_name: str, invalid_value: str) -> None:
    archive = _archive("Maybe PostgreSQL can be a future option.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    invalid = _semantic_candidate(proposal_id="alias-enum")
    invalid["semantic"] = {**invalid["semantic"], field_name: invalid_value}
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(
            json.dumps({"candidates": [_semantic_candidate(proposal_id="valid"), invalid]})
        )
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert [item.proposal_id for item in result.accepted] == ["valid"]
    assert f"{field_name} is not allowed" in result.rejected[0].reason


def test_nested_target_and_delete_controls_reject_only_that_candidate() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    valid = [_semantic_candidate(proposal_id=f"valid-{index}") for index in range(4)]
    forbidden_values = {
        "canonical_value": "concise",
        "target_uri": "memoryos://user/u2/memories/canonical/claims/victim",
        "action": "DELETE",
    }
    forbidden = _semantic_candidate(
        proposal_id="nested-operation",
        value_fields=forbidden_values,
        field_evidence_refs=_field_evidence(
            {"subject": "answers", "dimension": "length"},
            forbidden_values,
            "m1",
        ),
    )
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [*valid, forbidden]}))
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert [item.proposal_id for item in result.accepted] == [f"valid-{index}" for index in range(4)]
    assert [item.proposal_id for item in result.rejected] == ["nested-operation"]
    assert "value_fields.action" in result.rejected[0].reason
    assert "value_fields.target_uri" in result.rejected[0].reason
    assert "forbidden_output_field_rejected" in result.rejected[0].security_flags
    assert "operation_control_rejected" in result.rejected[0].security_flags
    assert "target_reference_rejected" in result.rejected[0].security_flags
    assert "forbidden_output_field_rejected" in result.security_flags


@pytest.mark.parametrize("location", ["identity", "scope_attributes", "condition"])
def test_recursive_control_field_rejection_covers_all_nested_candidate_structures(location: str) -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    candidate = _semantic_candidate()
    if location == "identity":
        identity = {
            "subject": {"label": "answers", "target_uri": "memoryos://user/u2/forged"},
            "dimension": "length",
        }
        candidate["identity_fields"] = identity
        candidate["field_evidence_refs"] = _field_evidence(identity, candidate["value_fields"], "m1")
    elif location == "scope_attributes":
        scope = episode.legal_scope_candidates()[0].to_dict()
        scope["attributes"] = {"operation": {"action": "UPDATE"}}
        candidate["suggested_scope_refs"] = [scope]
    else:
        values = {
            "canonical_value": "concise",
            "condition": {"operation": {"action": "DELETE"}},
        }
        candidate["value_fields"] = values
        candidate["field_evidence_refs"] = _field_evidence(candidate["identity_fields"], values, "m1")
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [candidate]}))
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert result.accepted == ()
    assert len(result.rejected) == 1
    assert "forbidden control fields" in result.rejected[0].reason
    assert "forbidden_output_field_rejected" in result.security_flags


def test_same_key_forged_scope_attributes_are_rejected_and_valid_selection_uses_episode_scope() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    archive.metadata["origin"] = {
        "primary_scope": {
            "namespace": "memoryos",
            "kind": "workspace",
            "id": "project-1",
            "attributes": {"branch": "main"},
        }
    }
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    legal_scope = next(scope for scope in episode.legal_scope_candidates() if scope.kind == "workspace")
    minimal_selection = {
        "namespace": legal_scope.namespace,
        "kind": legal_scope.kind,
        "id": legal_scope.id,
    }
    forged_selection = legal_scope.to_dict()
    forged_selection["attributes"] = {"branch": "forged"}
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(
            json.dumps(
                {
                    "candidates": [
                        _semantic_candidate(
                            proposal_id="valid-scope",
                            suggested_scope_refs=[minimal_selection],
                        ),
                        _semantic_candidate(
                            proposal_id="forged-scope",
                            suggested_scope_refs=[forged_selection],
                        ),
                    ]
                }
            )
        )
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert [item.proposal_id for item in result.accepted] == ["valid-scope"]
    assert result.accepted[0].suggested_scope_refs == (legal_scope,)
    assert dict(result.accepted[0].suggested_scope_refs[0].attributes) == {"branch": "main"}
    assert [item.proposal_id for item in result.rejected] == ["forged-scope"]
    assert "differs from legal scope: attributes" in result.rejected[0].reason
    assert "scope_authority_rejected" in result.rejected[0].security_flags


@pytest.mark.parametrize(
    "field_name",
    [
        "targetUri",
        "objectURI",
        "databaseOperation",
        "requestedAction",
        "deleteRequest",
        "updateCommand",
        "targeturi",
        "TARGETURI",
        "objecturi",
        "dboperation",
        "deletecommand",
        "claimRevision",
        "claimrevision",
        "revisionhistory",
        "transaction_action",
    ],
)
def test_recursive_control_field_rejection_normalizes_control_field_spellings(field_name: str) -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    values = {"canonical_value": "concise", "condition": {field_name: "forged"}}
    candidate = _semantic_candidate(
        value_fields=values,
        field_evidence_refs=_field_evidence(
            {"subject": "answers", "dimension": "length"},
            values,
            "m1",
        ),
    )
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [candidate]}))
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert result.accepted == ()
    assert field_name in result.rejected[0].reason
    assert "forbidden_output_field_rejected" in result.security_flags


def test_recursive_control_rejection_preserves_ordinary_semantic_condition_keys() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    values = {
        "canonical_value": "concise",
        "condition": {
            "cooperation": "peer review",
            "interaction": "code review",
            "update_frequency": "weekly",
        },
    }
    candidate = _semantic_candidate(
        value_fields=values,
        field_evidence_refs=_field_evidence(
            {"subject": "answers", "dimension": "length"},
            values,
            "m1",
        ),
    )
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [candidate]}))
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert [item.proposal_id for item in result.accepted] == ["p1"]
    assert dict(result.accepted[0].value_fields["condition"]) == values["condition"]
    assert result.rejected == ()


@pytest.mark.parametrize(
    "confidence",
    [float("nan"), float("inf"), float("-inf"), -0.1, 1.1, True, "not-a-number"],
)
def test_invalid_model_confidence_isolated_before_proposal_construction(confidence: object) -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(
            json.dumps(
                {
                    "candidates": [
                        _semantic_candidate(proposal_id="invalid-confidence", confidence=confidence),
                        _semantic_candidate(proposal_id="valid-confidence", confidence=0.1),
                    ]
                }
            )
        )
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert [item.proposal_id for item in result.accepted] == ["valid-confidence"]
    assert [item.proposal_id for item in result.rejected] == ["invalid-confidence"]
    assert "confidence must be a finite number between 0 and 1" in result.rejected[0].reason
    assert "candidate_confidence_rejected" in result.rejected[0].security_flags
    assert "candidate_schema_rejected" in result.rejected[0].security_flags
    assert "evidence_integrity_rejected" not in result.rejected[0].security_flags


def test_unknown_value_field_is_rejected_by_memory_type_schema_without_dropping_valid_candidate() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    unknown_values = {"canonical_value": "concise", "invented_semantic_field": "forged"}
    invalid = _semantic_candidate(
        proposal_id="unknown-value-field",
        value_fields=unknown_values,
        field_evidence_refs=_field_evidence(
            {"subject": "answers", "dimension": "length"},
            unknown_values,
            "m1",
        ),
    )
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(
            json.dumps({"candidates": [_semantic_candidate(proposal_id="valid"), invalid]})
        )
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert [item.proposal_id for item in result.accepted] == ["valid"]
    assert [item.proposal_id for item in result.rejected] == ["unknown-value-field"]
    assert "value_fields contains fields outside the preference schema" in result.rejected[0].reason
    assert "candidate_value_schema_rejected" in result.rejected[0].security_flags


def test_system_or_legacy_schema_field_is_not_added_to_model_value_whitelist() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    values = {"canonical_value": "concise", "content": "model-authored legacy projection"}
    candidate = _semantic_candidate(
        value_fields=values,
        field_evidence_refs=_field_evidence(
            {"subject": "answers", "dimension": "length"},
            values,
            "m1",
        ),
    )
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [candidate]}))
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert result.accepted == ()
    assert "value_fields contains fields outside the preference schema: content" in result.rejected[0].reason
    assert "candidate_value_schema_rejected" in result.rejected[0].security_flags


def test_llm_identity_failure_is_isolated_from_four_valid_candidates() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    valid = [_semantic_candidate(proposal_id=f"valid-{index}") for index in range(4)]
    invalid_identity = {"subject": "answers"}
    invalid = _semantic_candidate(
        proposal_id="invalid-identity",
        identity_fields=invalid_identity,
        field_evidence_refs=_field_evidence(
            invalid_identity,
            {"canonical_value": "concise"},
            "m1",
        ),
    )
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [*valid, invalid]}))
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert len(result.accepted) == 4
    assert [item.proposal_id for item in result.rejected] == ["invalid-identity"]
    assert "slot identity mismatch: missing:dimension" in result.rejected[0].reason


def test_llm_unknown_slot_identity_field_is_rejected_per_candidate() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    identity = {"subject": "answers", "dimension": "length", "target_uri": "forged"}
    candidate = _semantic_candidate(
        identity_fields=identity,
        field_evidence_refs=_field_evidence(identity, {"canonical_value": "concise"}, "m1"),
    )
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [candidate]}))
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert result.accepted == ()
    assert "identity_fields.target_uri" in result.rejected[0].reason
    assert "forbidden_output_field_rejected" in result.rejected[0].security_flags


def test_forged_evidence_rejects_only_that_candidate() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    forged = _semantic_candidate(
        proposal_id="forged",
        evidence_refs=[{"event_id": "m1", "content_hash": "forged"}],
    )
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(
            json.dumps({"candidates": [_semantic_candidate(proposal_id="valid"), forged]})
        )
    )

    result = backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert [item.proposal_id for item in result.accepted] == ["valid"]
    assert [item.proposal_id for item in result.rejected] == ["forged"]
    assert "evidence_integrity_rejected" in result.security_flags


def test_invalid_json_is_an_envelope_failure() -> None:
    archive = _archive()
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    backend = LLMMemoryExtractorBackend(FakeMemoryModelProvider("not-json"))
    with pytest.raises(ValueError, match="not valid JSON"):
        backend.extract_batch_with_context(
            archive,
            MemoryTypeRegistry().list(),
            existing_memories=(),
            episode=episode,
        )


def test_prompt_contains_schema_enums_and_prefetched_slot_claim_identity() -> None:
    prompt = MemoryExtractionPromptBuilder().build(
        _archive(),
        MemoryTypeRegistry().list(),
        existing_memories=(
            PrefetchedMemory(
                uri="memoryos://user/u1/memories/canonical/slots/s1/claims/c1",
                memory_type="project_decision",
                state="ACTIVE",
                revision=3,
                slot_id="s1",
                claim_id="c1",
                canonical_value="sqlite",
                identity_fields={"decision_topic": "primary_storage_backend"},
                scope={"applicability": {"all_of": []}},
                l0="SQLite",
                l1="SQLite is active",
            ),
        ),
    )
    assert "MEMORY_SCHEMAS=" in prompt
    assert "Allowed speech_act values:" in prompt
    assert "Allowed utterance_mode values:" in prompt
    assert "Allowed attribution values:" in prompt
    assert "Allowed durability values:" in prompt
    assert "Allowed modal_force values:" in prompt
    assert "Allowed atomicity values:" in prompt
    assert "atomic_evidence_ref must select one exact non-empty span" in prompt
    assert "requires a separate structured review before changing or retracting an ACTIVE claim" in prompt
    assert '"candidate_ref": "existing_0"' in prompt
    assert '"slot_id": "s1"' not in prompt
    assert '"claim_id": "c1"' not in prompt
    assert "memoryos://user/u1/memories/canonical" not in prompt
    assert '"identity_fields": {"decision_topic": "primary_storage_backend"}' in prompt
    assert "CONFIRMATION only confirms the candidate; it never means SUPERSEDES" in prompt
    assert (
        "'Do not use Redis unless it is only a short-term cache' => CONDITIONAL_FORBID Redis "
        "with the cache exception retained."
    ) in prompt
    assert '"claim_identity_fields"' in prompt


def test_prompt_contains_concrete_semantics_for_every_memory_type_field() -> None:
    registry = MemoryTypeRegistry()
    prompt = MemoryExtractionPromptBuilder().build(_archive(), registry.list())
    serialized = prompt.split("MEMORY_SCHEMAS=", 1)[1].split("\nEXISTING_MEMORIES=", 1)[0]
    payload = {item["memory_type"]: item for item in json.loads(serialized)}

    for schema in registry.list():
        contract = payload[schema.memory_type.value]["field_semantics"]
        all_declared_fields = {
            *schema.required_fields,
            *schema.optional_fields,
            *schema.slot_identity_fields,
            *schema.claim_identity_fields,
            *schema.applicability_fields,
            *schema.display_fields,
            *schema.provenance_fields,
            *schema.field_merge_rules,
        }
        all_declared_fields.discard("*")
        assert all_declared_fields.issubset(contract)
        assert all(contract[field_name]["meaning"] for field_name in all_declared_fields)
        for field_name in (*schema.required_fields, *schema.optional_fields):
            assert contract[field_name]["meaning"]
            expected_placement = (
                "identity_fields" if field_name in schema.slot_identity_fields else "value_fields"
            )
            assert contract[field_name]["placement"] == expected_placement
            assert contract[field_name]["identity_role"]
        for field_name in schema.required_fields:
            assert contract[field_name]["legacy_projection_required"] is True
            if field_name not in schema.slot_identity_fields and field_name != "canonical_value":
                assert contract[field_name]["required"] is False
        for field_name in schema.slot_identity_fields:
            assert contract[field_name]["meaning"]
            assert contract[field_name]["placement"] == "identity_fields"
            assert contract[field_name]["identity_role"] == "slot_identity"
            assert contract[field_name]["model_authored"] is True
        for field_name in schema.provenance_fields:
            assert contract[field_name]["placement"] == "system_metadata"
            assert contract[field_name]["identity_role"] == "provenance_only"
            assert contract[field_name]["model_authored"] is False
        model_fields = {
            *schema.required_fields,
            *schema.optional_fields,
            *schema.claim_identity_fields,
            *schema.display_fields,
            *schema.applicability_fields,
            "canonical_value",
        }
        legacy_only = set(schema.field_merge_rules) - model_fields - set(schema.slot_identity_fields)
        for field_name in legacy_only:
            assert contract[field_name]["placement"] == "legacy_projection"
            assert contract[field_name]["identity_role"] == "legacy_projection_only"
            assert contract[field_name]["model_authored"] is False

    project_decision = payload["project_decision"]["field_semantics"]
    assert project_decision["canonical_value"]["identity_role"] == "claim_identity"
    assert "storage target" in project_decision["project_id"]["meaning"]
    assert project_decision["rationale"]["identity_role"] == "display_only"
    assert "model_authored value is false" in prompt
