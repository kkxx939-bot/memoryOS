from __future__ import annotations

import json
from typing import Any, cast

import pytest

from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical import MemorySemanticProposal, SessionArchiveEpisodeAdapter
from memoryos.memory.canonical.prefetch import PrefetchedMemory
from memoryos.memory.extraction import FakeMemoryModelProvider, LLMMemoryExtractorBackend
from memoryos.memory.extraction.llm_backend import MemoryExtractionPromptBuilder
from memoryos.memory.extraction.llm_memory_extractor import LLMMemoryExtractor
from memoryos.memory.schema import AdmissionDecision, MemoryTypeRegistry


def _archive(content: str = "Please remember this memory extraction test.") -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        messages=[{"id": "m1", "role": "user", "content": content}],
        metadata={"connect": {"adapter_id": "codex"}},
    )


def _field_evidence(identity_fields: dict, value_fields: dict, event_id: str) -> dict:
    ref = [{"event_id": event_id}]
    return {
        **{f"identity.{key}": ref for key in identity_fields},
        **{f"value.{key}": ref for key in value_fields},
        "semantic.speech_act": ref,
        "semantic.temporal_scope": ref,
        "transition": ref,
    }


def test_fake_llm_extractor_backend_outputs_semantic_proposals() -> None:
    response = json.dumps(
        {
            "candidates": [
                {
                    "memory_type": "preference",
                    "proposal_id": "p-review-style",
                    "identity_fields": {"subject": "code reviews", "dimension": "findings first"},
                    "value_fields": {"canonical_value": "findings first during code reviews"},
                    "semantic": {
                        "speech_act": "confirmation",
                        "commitment": "confirmed",
                        "temporal_scope": "current",
                        "relation_to_existing": "unrelated",
                    },
                    "epistemic_status": "EXPLICIT",
                    "suggested_scope_refs": [],
                    "evidence_refs": [{"event_id": "m1"}],
                    "field_evidence_refs": _field_evidence(
                        {"subject": "code reviews", "dimension": "findings first"},
                        {"canonical_value": "findings first during code reviews"},
                        "m1",
                    ),
                    "confidence": 0.9,
                    "source_role": "user",
                }
            ]
        }
    )
    backend = LLMMemoryExtractorBackend(FakeMemoryModelProvider(response))
    archive = _archive("I prefer findings first during code reviews.")

    operations = MemoryCommitPlanner(extractor=backend).plan(archive)

    assert operations[0].payload["admission"]["decision"] == AdmissionDecision.ACCEPT.value


def test_public_llm_extract_returns_only_semantic_proposals() -> None:
    candidate = _semantic_candidate()
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(json.dumps({"candidates": [candidate]}))
    )

    extracted = backend.extract(
        _archive("I confirm my answers preference: concise answers."),
        MemoryTypeRegistry().list(),
    )

    assert extracted and all(isinstance(item, MemorySemanticProposal) for item in extracted)


def test_direct_operation_parser_is_not_publicly_exported() -> None:
    import memoryos.memory.extraction as extraction

    assert not hasattr(extraction, "MemoryExtractionJsonParser")


def test_llm_backend_cannot_bypass_admission() -> None:
    response = json.dumps(
        {
            "candidates": [
                {
                    "memory_type": "preference",
                    "proposal_id": "p-raw",
                    "identity_fields": {"subject": "pytest", "dimension": "failure"},
                    "value_fields": {"canonical_value": "pytest failed\nTraceback (most recent call last):\nAssertionError"},
                    "semantic": {"speech_act": "observation", "commitment": "weak", "temporal_scope": "past"},
                    "epistemic_status": "EXPLICIT",
                    "suggested_scope_refs": [],
                    "evidence_refs": [{"event_id": "m1"}],
                    "field_evidence_refs": _field_evidence(
                        {"subject": "pytest", "dimension": "failure"},
                        {"canonical_value": "pytest failed\nTraceback (most recent call last):\nAssertionError"},
                        "m1",
                    ),
                    "confidence": 0.99,
                    "source_role": "user",
                },
                {
                    "memory_type": "preference",
                    "proposal_id": "p-secret",
                    "identity_fields": {"subject": "api_key", "dimension": "secret"},
                    "value_fields": {"canonical_value": "api_key=sk-test"},
                    "semantic": {"speech_act": "confirmation", "commitment": "confirmed", "temporal_scope": "current"},
                    "epistemic_status": "EXPLICIT",
                    "suggested_scope_refs": [],
                    "evidence_refs": [{"event_id": "m2"}],
                    "field_evidence_refs": _field_evidence(
                        {"subject": "api_key", "dimension": "secret"},
                        {"canonical_value": "api_key=sk-test"},
                        "m2",
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
            {"id": "m1", "role": "user", "content": "pytest failure: pytest failed\nTraceback (most recent call last):\nAssertionError"},
            {"id": "m2", "role": "user", "content": "My preference is api_key=sk-test secret"},
        ],
        metadata={"connect": {"adapter_id": "codex"}},
    )
    operations = planner.plan(archive)

    assert operations == []
    assert {result.decision.value for result in planner.last_canonical_results} == {"ARCHIVE_ONLY", "RESTRICTED"}


def test_llm_backend_rejects_illegal_memory_type() -> None:
    backend = LLMMemoryExtractorBackend(FakeMemoryModelProvider('{"candidates":[{"memory_type":"tool_log","content":"x","fields":{},"source_role":"user"}]}'))

    with pytest.raises(ValueError):
        backend.extract(_archive(), MemoryTypeRegistry().list())


def _semantic_candidate(**overrides) -> dict:  # noqa: ANN003
    identity_fields = {"subject": "answers", "dimension": "length"}
    value_fields = {"canonical_value": "concise"}
    candidate = {
        "proposal_id": "p1",
        "memory_type": "preference",
        "identity_fields": identity_fields,
        "value_fields": value_fields,
        "semantic": {
            "speech_act": "confirmation",
            "commitment": "confirmed",
            "temporal_scope": "current",
            "relation_to_existing": "unrelated",
        },
        "epistemic_status": "EXPLICIT",
        "suggested_scope_refs": [],
        "evidence_refs": [{"event_id": "m1"}],
        "field_evidence_refs": _field_evidence(identity_fields, value_fields, "m1"),
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
    with pytest.raises(ValueError, match="unknown fields"):
        backend.extract_with_context(
            archive,
            MemoryTypeRegistry().list(),
            existing_memories=(),
            episode=episode,
        )


def test_semantic_llm_backend_rejects_unknown_evidence_scope_duplicate_ids_and_bad_hash() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    cases = [
        [_semantic_candidate(evidence_refs=[{"event_id": "missing"}])],
        [_semantic_candidate(evidence_refs=[{"event_id": "m1", "content_hash": "bad"}])],
        [
            _semantic_candidate(),
            _semantic_candidate(),
        ],
        [
            _semantic_candidate(
                suggested_scope_refs=[{"namespace": "memoryos", "kind": "workspace", "id": "forged"}]
            )
        ],
        [_semantic_candidate(suggested_scope_refs=["forged"])],
        [_semantic_candidate(related_claim_ids="not-a-list")],
        [_semantic_candidate(evidence_refs=[])],
        [_semantic_candidate(field_evidence_refs={})],
    ]
    for candidates in cases:
        backend = LLMMemoryExtractorBackend(FakeMemoryModelProvider(json.dumps({"candidates": candidates})))
        with pytest.raises(ValueError):
            backend.extract_with_context(
                archive,
                MemoryTypeRegistry().list(),
                existing_memories=(),
                episode=episode,
            )


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


def test_llm_source_role_spoof_is_rejected_and_legacy_direct_operation_backend_is_rejected(tmp_path) -> None:  # noqa: ANN001
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
    with pytest.raises(ValueError, match="source_role does not match"):
        backend.extract_with_context(
            archive,
            MemoryTypeRegistry().list(),
            existing_memories=(),
            episode=episode,
        )

    from memoryos.api.sdk.client import MemoryOSClient

    with pytest.raises(TypeError, match="semantic candidate/proposal backend"):
        MemoryOSClient(
            str(tmp_path),
            memory_extractor=cast(Any, LLMMemoryExtractor(cast(Any, FakeMemoryModelProvider("{}")))),
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("speech_act", "invented_act"),
        ("commitment", "certainly_maybe"),
        ("temporal_scope", "whenever"),
        ("relation_to_existing", "replaces_everything"),
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
    with pytest.raises(ValueError, match=f"{field_name} is not allowed"):
        backend.extract_with_context(
            archive,
            MemoryTypeRegistry().list(),
            existing_memories=(),
            episode=episode,
        )


def test_semantic_llm_backend_rejects_related_memory_outside_prefetch() -> None:
    archive = _archive("I confirm my answers preference: concise answers.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    backend = LLMMemoryExtractorBackend(
        FakeMemoryModelProvider(
            json.dumps({"candidates": [_semantic_candidate(related_memory_ids=["memoryos://user/u2/forged"])]})
        )
    )
    with pytest.raises(ValueError, match="illegal reference"):
        backend.extract_with_context(
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
    assert '"slot_id": "s1"' in prompt
    assert '"claim_id": "c1"' in prompt
    assert '"identity_fields": {"decision_topic": "primary_storage_backend"}' in prompt
