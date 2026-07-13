from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.memory.canonical import (
    EpisodeSalienceGate,
    MemorySemanticProposal,
    SessionArchiveEpisodeAdapter,
)
from memoryos.memory.canonical.prefetch import PrefetchedMemory
from memoryos.memory.canonical.salience_ledger import (
    DurableSalienceLedger,
    SalienceLedgerIntegrityError,
)
from memoryos.memory.extraction import (
    EgressDecision,
    LLMMemoryExtractorBackend,
    MemoryEgressPolicy,
    MemoryExtractionBatchResult,
    MemoryExtractionConfigurationError,
    MemoryExtractionMalformedEnvelopeError,
    MemoryExtractionRateLimitError,
    MemoryExtractionTimeoutError,
    MemoryExtractionTransportError,
    SensitivityCategory,
)
from memoryos.memory.schema import MemoryTypeRegistry, MemoryTypeSchema


class _SequenceProvider:
    def __init__(self, outcomes: list[object], *, remote: bool = False) -> None:
        self.outcomes = list(outcomes)
        self.is_remote = remote
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return str(outcome)


class _RateLimitError(RuntimeError):
    status_code = 429


class _ServiceUnavailableError(RuntimeError):
    status_code = 503


class _SDKTimeoutError(RuntimeError):
    pass


def _archive(text: str, *, metadata: dict | None = None) -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id="extraction-policy",
        archive_uri="memoryos://user/u1/sessions/history/extraction-policy",
        messages=[{"id": "m1", "role": "user", "content": text}],
        metadata={"tenant_id": "t1", "project_id": "memoryos", **dict(metadata or {})},
        task_id="extraction-policy-task",
        created_at="2026-07-12T00:00:00+00:00",
    )


def _persisted_archive(root: Path, archive: SessionArchive) -> SessionArchive:
    SessionArchiveStore(root, tenant_id="t1").write_sync_archive(archive)
    return archive


def _extract(backend: LLMMemoryExtractorBackend, archive: SessionArchive):  # noqa: ANN202
    return backend.extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=SessionArchiveEpisodeAdapter().adapt(archive),
    )


@pytest.mark.parametrize(
    ("failure", "expected_type"),
    [
        (TimeoutError("timeout"), MemoryExtractionTimeoutError),
        (ConnectionError("network"), MemoryExtractionTransportError),
        (_RateLimitError("limited"), MemoryExtractionRateLimitError),
        (_ServiceUnavailableError("provider failed"), MemoryExtractionTransportError),
        (_SDKTimeoutError("provider failed"), MemoryExtractionTimeoutError),
    ],
)
def test_retryable_provider_failures_have_finite_typed_retry(
    failure: BaseException,
    expected_type: type[BaseException],
) -> None:
    provider = _SequenceProvider([failure, type(failure)(str(failure))])
    backend = LLMMemoryExtractorBackend(
        provider,
        max_attempts=2,
        retry_backoff_seconds=(),
    )

    with pytest.raises(expected_type):
        _extract(backend, _archive("Remember this durable project decision."))

    assert provider.calls == 2


def test_malformed_top_level_envelope_retries_but_configuration_error_does_not() -> None:
    recovered_provider = _SequenceProvider([json.dumps({"wrong": []}), json.dumps({"candidates": []})])
    recovered = _extract(
        LLMMemoryExtractorBackend(
            recovered_provider,
            max_attempts=2,
            retry_backoff_seconds=(),
        ),
        _archive("Remember this durable project decision."),
    )
    assert recovered.accepted == ()
    assert recovered_provider.calls == 2

    malformed_provider = _SequenceProvider(["[]", json.dumps({"candidates": "invalid"})])
    with pytest.raises(MemoryExtractionMalformedEnvelopeError):
        _extract(
            LLMMemoryExtractorBackend(
                malformed_provider,
                max_attempts=2,
                retry_backoff_seconds=(),
            ),
            _archive("Remember this durable project decision."),
        )
    assert malformed_provider.calls == 2

    configuration_provider = _SequenceProvider([RuntimeError("unsupported provider bug")])
    with pytest.raises(MemoryExtractionConfigurationError):
        _extract(
            LLMMemoryExtractorBackend(configuration_provider, max_attempts=3),
            _archive("Remember this durable project decision."),
        )
    assert configuration_provider.calls == 1


@pytest.mark.parametrize(
    ("text", "decision"),
    [
        ("Remember this: PASSWORD=hunter2", EgressDecision.DENY.value),
        ("Remember this: my password is hunter2", EgressDecision.DENY.value),
        ("Remember this: my HIV diagnosis is private.", EgressDecision.LOCAL_ONLY.value),
        ("My passport number is P1234567.", EgressDecision.LOCAL_ONLY.value),
        ("My home address is 1 Main Street.", EgressDecision.LOCAL_ONLY.value),
        ("My bank account number is 12345678.", EgressDecision.LOCAL_ONLY.value),
        ("This private relationship must remain confidential.", EgressDecision.LOCAL_ONLY.value),
        ("This is an off the record conversation.", EgressDecision.LOCAL_ONLY.value),
        ("This is our confidential source code configuration.", EgressDecision.LOCAL_ONLY.value),
    ],
)
def test_sensitive_remote_egress_never_calls_provider_or_retries(text: str, decision: str) -> None:
    provider = _SequenceProvider([json.dumps({"candidates": []})], remote=True)
    result = _extract(LLMMemoryExtractorBackend(provider, max_attempts=3), _archive(text))

    assert provider.calls == 0
    assert result.egress_decision == decision
    assert result.rejected
    assert result.egress_audit == {
        "outbound_digest": "",
        "decision": decision,
        "provider": "_SequenceProvider",
        "model": "",
    }


def test_remote_redaction_precedes_outbound_digest_and_local_sensitive_extraction_is_supported() -> None:
    remote = _SequenceProvider([json.dumps({"candidates": []})], remote=True)
    policy = MemoryEgressPolicy(redact_categories=(SensitivityCategory.CONTACT,))
    redacted = _extract(
        LLMMemoryExtractorBackend(remote, egress_policy=policy),
        _archive("Remember this contact: alice@example.com"),
    )
    assert remote.calls == 1
    assert "alice@example.com" not in remote.prompts[0]
    assert "REDACTED_CONTACT" in remote.prompts[0]
    assert redacted.egress_decision == EgressDecision.ALLOW_REDACTED.value
    assert redacted.egress_audit and redacted.egress_audit["outbound_digest"]

    local = _SequenceProvider([json.dumps({"candidates": []})], remote=False)
    local_result = _extract(
        LLMMemoryExtractorBackend(local),
        _archive("Remember this: my HIV diagnosis is private."),
    )
    assert local.calls == 1
    assert local_result.egress_decision == EgressDecision.LOCAL_ONLY.value


def test_remote_prompt_never_duplicates_unclassified_raw_message_fields() -> None:
    provider = _SequenceProvider([json.dumps({"candidates": []})], remote=True)
    archive = _archive("Remember this durable project decision.")
    archive.messages[0]["api_key"] = "sk-hidden-outside-content"

    result = _extract(LLMMemoryExtractorBackend(provider), archive)

    assert result.accepted == ()
    assert result.egress_decision == EgressDecision.DENY.value
    assert provider.calls == 0
    assert provider.prompts == []


def test_remote_egress_policy_covers_prefetched_memory_and_unstructured_labels() -> None:
    existing = PrefetchedMemory(
        uri="memoryos://user/u1/memories/canonical/slots/s/claims/c",
        memory_type="user_profile",
        state="ACTIVE",
        revision=1,
        slot_id="s",
        claim_id="c",
        canonical_value="alice@example.com",
        identity_fields={"attribute_key": "contact"},
        scope={},
        l0="contact",
        l1="alice@example.com",
    )
    archive = _archive("Remember this durable project decision.")
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    default_provider = _SequenceProvider([json.dumps({"candidates": []})], remote=True)
    default_result = LLMMemoryExtractorBackend(default_provider).extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(existing,),
        episode=episode,
    )
    assert default_provider.calls == 0
    assert default_result.egress_decision == EgressDecision.LOCAL_ONLY.value

    redacted_provider = _SequenceProvider([json.dumps({"candidates": []})], remote=True)
    redacted_result = LLMMemoryExtractorBackend(
        redacted_provider,
        egress_policy=MemoryEgressPolicy(redact_categories=(SensitivityCategory.CONTACT,)),
    ).extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(existing,),
        episode=episode,
    )
    assert redacted_provider.calls == 1
    assert "alice@example.com" not in redacted_provider.prompts[0]
    assert redacted_result.egress_decision == EgressDecision.ALLOW_REDACTED.value

    labelled_provider = _SequenceProvider([json.dumps({"candidates": []})], remote=True)
    labelled_archive = _archive(
        "Remember this project alias.",
        metadata={"sensitivity": ["contact"]},
    )
    labelled_result = LLMMemoryExtractorBackend(
        labelled_provider,
        egress_policy=MemoryEgressPolicy(redact_categories=(SensitivityCategory.CONTACT,)),
    ).extract_batch_with_context(
        labelled_archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=SessionArchiveEpisodeAdapter().adapt(labelled_archive),
    )
    assert labelled_provider.calls == 0
    assert labelled_result.egress_decision == EgressDecision.LOCAL_ONLY.value


class _CountingExtractor:
    semantic_proposal_backend = True
    llm_semantic_backend = True

    def __init__(self) -> None:
        self.calls = 0

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> Sequence[MemorySemanticProposal]:
        del archive, schemas
        self.calls += 1
        return []


class _LocalCountingExtractor(_CountingExtractor):
    is_remote = False


class _UntrustedRedactionExtractor(_CountingExtractor):
    egress_policy_enforced = True


class _PoisonedAuditExtractor(_CountingExtractor):
    def extract_batch_with_context(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
        *,
        existing_memories: Sequence[object],
        episode: object,
    ) -> MemoryExtractionBatchResult:
        del archive, schemas, existing_memories, episode
        self.calls += 1
        return MemoryExtractionBatchResult(
            (),
            egress_decision=EgressDecision.DENY.value,
            egress_audit={
                "outbound_digest": "attacker-controlled",
                "decision": EgressDecision.DENY.value,
                "provider": "forged",
                "model": "forged",
                "raw_prompt": "must never be persisted",
            },
        )


@pytest.mark.parametrize("boundary", ["duplicate", "budget", "privacy"])
def test_salience_boundary_controls_main_planner_model_calls(tmp_path: Path, boundary: str) -> None:
    extractor = _CountingExtractor()
    archive = _archive("Remember this durable project rule.")
    if boundary == "privacy":
        archive.messages[0]["content"] = "Remember this: OPENAI_API_KEY=sk-sensitive"
    elif boundary == "budget":
        archive.metadata["memory_planning"] = {"consumed_budget": 1, "max_episode_budget": 1}
    else:
        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        fingerprint = EpisodeSalienceGate().evaluate(episode).episode_fingerprint
        archive.metadata["memory_planning"] = {"seen_episode_fingerprints": [fingerprint]}
    planner = MemoryCommitPlanner(
        extractor=extractor,
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    )
    _persisted_archive(tmp_path, archive)

    result = planner.plan(archive)

    assert extractor.calls == 0
    audit = dict(result.context.egress_audit)
    assert set(audit) == {"outbound_digest", "decision", "provider", "model"}
    assert audit["decision"] == result.context.egress_decision
    assert audit["outbound_digest"] == ""
    assert result.operations == ()
    assert result.context.proposal_outcomes[0].decision in {"ARCHIVE_ONLY", "RESTRICTED"}


def test_durable_salience_ledger_skips_same_episode_across_different_sessions(tmp_path: Path) -> None:
    extractor = _LocalCountingExtractor()
    planner = MemoryCommitPlanner(
        extractor=extractor,
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    )

    def archive(session_id: str) -> SessionArchive:
        return SessionArchive(
            user_id="u1",
            session_id=session_id,
            archive_uri=f"memoryos://user/u1/sessions/history/{session_id}",
            messages=[
                {
                    "id": f"{session_id}-message",
                    "role": "user",
                    "content": "Remember this durable project rule.",
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
            task_id=f"task-{session_id}",
            created_at="2026-07-12T00:00:00+00:00",
        )

    first_archive = _persisted_archive(tmp_path, archive("first-session"))
    second_archive = _persisted_archive(tmp_path, archive("second-session"))
    first = planner.plan(first_archive)
    second = planner.plan(second_archive)

    assert extractor.calls == 1
    assert first.context.salience_duplicate is False
    assert second.context.salience_duplicate is True
    assert second.operations == ()
    assert second.context.proposal_outcomes[0].reason == "duplicate_episode"


def test_salience_reservation_rejects_same_task_rebound_to_different_episode(tmp_path: Path) -> None:
    """A crash between reservation and envelope cannot reuse stale admission."""

    gate = EpisodeSalienceGate()
    ledger = DurableSalienceLedger(tmp_path, tenant_id="t1")
    quiet = _archive("hello")
    changed = _archive("Remember this durable project rule.")
    first_episode = SessionArchiveEpisodeAdapter().adapt(quiet)
    changed_episode = SessionArchiveEpisodeAdapter().adapt(changed)

    reserved = ledger.reserve(
        gate,
        first_episode,
        task_id=quiet.task_id,
        user_id=quiet.user_id,
        project_id="memoryos",
    )
    assert reserved.decision.salient is False

    with pytest.raises(
        SalienceLedgerIntegrityError,
        match="different semantic episode",
    ):
        ledger.reserve(
            gate,
            changed_episode,
            task_id=changed.task_id,
            user_id=changed.user_id,
            project_id="memoryos",
        )


def test_salience_reservation_rejects_cross_tenant_episode_before_artifacts(tmp_path: Path) -> None:
    episode = SessionArchiveEpisodeAdapter().adapt(_archive("Remember this durable decision."))
    ledger = DurableSalienceLedger(tmp_path, tenant_id="tenant-b")

    with pytest.raises(SalienceLedgerIntegrityError, match="ledger tenant boundary"):
        ledger.reserve(
            EpisodeSalienceGate(),
            episode,
            task_id="cross-tenant-task",
            user_id="u1",
            project_id="memoryos",
        )

    assert not list(tmp_path.rglob("*"))


def test_structured_explicit_remember_bypasses_model_without_generic_identity(tmp_path: Path) -> None:
    extractor = _CountingExtractor()
    client = MemoryOSClient(str(tmp_path), memory_extractor=extractor)

    result = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )

    assert result["status"] == "COMMITTED"
    assert extractor.calls == 0
    replay = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )
    assert replay["uri"] == result["uri"]
    assert replay["diff_id"] == result["diff_id"]
    assert replay["transaction_id"] == result["transaction_id"]
    assert replay["receipt_digest"] == result["receipt_digest"]
    assert replay["idempotent_replay"] is True
    assert extractor.calls == 0


def test_planner_boundary_defaults_unknown_extractors_to_remote_and_blocks_sensitive_payload(
    tmp_path: Path,
) -> None:
    remote = _CountingExtractor()
    remote_root = tmp_path / "remote"
    remote_archive = _persisted_archive(
        remote_root,
        _archive("Remember this durable fact: my HIV diagnosis is private."),
    )
    blocked = MemoryCommitPlanner(
        extractor=remote,
        source_store=FileSystemSourceStore(remote_root, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    ).plan(remote_archive)

    assert remote.calls == 0
    assert blocked.context.egress_decision == EgressDecision.LOCAL_ONLY.value
    assert blocked.context.proposal_outcomes[0].decision == "RESTRICTED"
    assert dict(blocked.context.egress_audit)["outbound_digest"] == ""

    local = _LocalCountingExtractor()
    local_root = tmp_path / "local"
    local_archive = _persisted_archive(
        local_root,
        _archive("Remember this durable fact: my HIV diagnosis is private."),
    )
    allowed = MemoryCommitPlanner(
        extractor=local,
        source_store=FileSystemSourceStore(local_root, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    ).plan(local_archive)
    assert local.calls == 1
    assert allowed.context.egress_decision == EgressDecision.LOCAL_ONLY.value


def test_planner_does_not_trust_custom_remote_redaction_claim(tmp_path: Path) -> None:
    extractor = _UntrustedRedactionExtractor()
    archive = _persisted_archive(tmp_path, _archive("Remember this contact: alice@example.com"))
    result = MemoryCommitPlanner(
        extractor=extractor,
        egress_policy=MemoryEgressPolicy(redact_categories=(SensitivityCategory.CONTACT,)),
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    ).plan(archive)

    assert extractor.calls == 0
    assert result.operations == ()
    assert result.context.egress_decision == EgressDecision.LOCAL_ONLY.value
    assert dict(result.context.egress_audit)["outbound_digest"] == ""


def test_planner_configured_policy_is_the_backend_prompt_policy(tmp_path: Path) -> None:
    provider = _SequenceProvider([json.dumps({"candidates": []})], remote=True)
    backend = LLMMemoryExtractorBackend(provider)
    archive = _persisted_archive(
        tmp_path,
        _archive("Remember this contact: alice@example.com"),
    )

    result = MemoryCommitPlanner(
        extractor=backend,
        egress_policy=MemoryEgressPolicy(redact_categories=(SensitivityCategory.CONTACT,)),
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    ).plan(archive)

    assert provider.calls == 1
    assert "alice@example.com" not in provider.prompts[0]
    assert result.context.egress_decision == EgressDecision.ALLOW_REDACTED.value
    assert dict(result.context.egress_audit)["decision"] == EgressDecision.ALLOW_REDACTED.value


def test_planner_does_not_trust_model_result_to_rewrite_egress_audit(tmp_path: Path) -> None:
    extractor = _PoisonedAuditExtractor()
    archive = _persisted_archive(
        tmp_path,
        _archive("Remember this durable project decision."),
    )

    result = MemoryCommitPlanner(
        extractor=extractor,
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    ).plan(archive)

    audit = dict(result.context.egress_audit)
    assert extractor.calls == 1
    assert result.context.egress_decision == EgressDecision.ALLOW.value
    assert set(audit) == {"outbound_digest", "decision", "provider", "model"}
    assert audit["decision"] == EgressDecision.ALLOW.value
    assert len(audit["outbound_digest"]) == 64
    envelope = next((tmp_path / "tenants" / "t1" / "system" / "planning-envelopes").glob("*.json")).read_text(
        encoding="utf-8"
    )
    assert "raw_prompt" not in envelope
    assert "must never be persisted" not in envelope


def test_explicit_identity_fields_are_part_of_command_and_slot_identity(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    first = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "primary storage backend"},
    )
    second = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "analytics storage backend"},
    )

    assert first["uri"] != second["uri"]
    slot_objects = [client.read(result["uri"].rsplit("/claims/", 1)[0])["object"] for result in (first, second)]
    assert {item["metadata"]["identity_fields"]["decision_topic"] for item in slot_objects} == {
        "primary-storage-backend",
        "analytics-storage-backend",
    }
    assert all(
        item["metadata"]["identity_fields"]["decision_topic"] not in {"profile", "preference", "project rule"}
        for item in slot_objects
    )


@pytest.mark.parametrize("unstable", [float("nan"), float("inf"), float("-inf")])
def test_explicit_remember_rejects_non_finite_identity_before_persisting_command(
    tmp_path: Path,
    unstable: float,
) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="t1")
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    with pytest.raises(ValueError, match="identity field decision_topic must be finite"):
        client.remember(
            user_id="u1",
            content="PostgreSQL",
            memory_type="project_decision",
            project_id="memoryos",
            identity_fields={"decision_topic": unstable},
        )

    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before


@pytest.mark.parametrize(
    ("title", "identity_fields", "error"),
    [
        ("", None, "requires identity_fields"),
        ("project decision", None, "too generic"),
        ("", {"subject": "storage"}, "identity_fields mismatch"),
    ],
)
def test_explicit_remember_rejects_missing_generic_or_wrong_schema_identity_before_archive(
    tmp_path: Path,
    title: str,
    identity_fields: dict[str, str] | None,
    error: str,
) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="t1")
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    with pytest.raises(ValueError, match=error):
        client.remember(
            user_id="u1",
            title=title,
            content="PostgreSQL",
            memory_type="project_decision",
            project_id="memoryos",
            identity_fields=identity_fields,
        )

    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before
