from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.context_assembler import ContextAssembler
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryRelationStore,
)
from memoryos.memory.canonical import CanonicalMemoryRepository, MemorySemanticProposal
from memoryos.memory.canonical.current_head import load_current_head
from memoryos.memory.canonical.episode import SessionArchiveEpisodeAdapter
from memoryos.memory.canonical.formation import CandidateProposalAdapter, CanonicalMemoryFormationService
from memoryos.memory.canonical.history import (
    CanonicalHistoryIntegrityError,
    validate_canonical_receipt_history,
)
from memoryos.memory.extraction import FakeMemoryModelProvider, LLMMemoryExtractorBackend
from memoryos.memory.schema import (
    MemoryCandidateDraft,
    MemoryType,
    MemoryTypeRegistry,
    MemoryTypeSchema,
)
from memoryos.operations.commit.effect_marker import validate_marker
from memoryos.operations.commit.outbox_envelope import prepared_intent_digest, validate_outbox
from memoryos.operations.commit.planning_proof import (
    CANONICAL_PREPARED_INTENT_SCHEMA_VERSION,
)
from memoryos.operations.commit.receipt import load_transaction_receipt
from memoryos.runtime.readiness import RuntimeReadinessState
from tests.unit.test_canonical_transaction_commit import _plan, _proposal, _setup


def test_historical_canonical_receipt_does_not_validate_against_current_source(tmp_path: Path) -> None:
    source, _index, _queue, relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "immutable-receipt", "SQLite", "confirmation", "confirmed")
    identity, _transition, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id)
    committer.commit("u1", operations)

    receipt_path = committer._transaction_marker(str(operations[0].payload["idempotency_key"]))
    old_receipt = receipt_path.read_bytes()
    claim = source.read_object(identity.claim_uri)
    claim.title = "a later legal revision changed the current Source snapshot"
    source.write_object(claim, content=source.read_content(claim.layers.l2_uri or claim.uri))

    validate_marker(
        receipt_path,
        source,
        relations,
        transaction_id=str(operations[0].payload["transaction_id"]),
        idempotency_key=str(operations[0].payload["idempotency_key"]),
        tenant_id="t1",
        user_id="u1",
        operation_ids=[item.operation_id for item in operations],
    )
    assert receipt_path.read_bytes() == old_receipt


def test_memory_artifact_dependency_graph_is_acyclic(tmp_path: Path) -> None:
    source, _index, queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(episode, "artifact-dag", "SQLite", "confirmation", "confirmed")
    identity, _transition, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(user_id="u1", tenant_id="t1", episode_id=episode.episode_id)
    committer.commit("u1", operations)

    receipt_path = committer._transaction_marker(str(operations[0].payload["idempotency_key"]))
    receipt = load_transaction_receipt(receipt_path)
    outbox_path = committer._outbox_path(str(operations[0].payload["transaction_id"]))
    outbox = validate_outbox(json.loads(outbox_path.read_text(encoding="utf-8")))
    head, _head_receipt, _snapshot = load_current_head(committer.artifact_root, identity.claim_uri)
    intent_path = committer.planning_proofs.canonical_intent_path(str(receipt["transaction_id"]))
    immutable_intent = committer.planning_proofs.load_canonical_intent(
        str(receipt["transaction_id"]),
        operations=operations,
        prepared_intent_digest=str(receipt["prepared_intent_digest"]),
    )

    assert receipt["prepared_intent_digest"] == prepared_intent_digest(outbox)
    assert receipt["prepared_intent_schema_version"] == CANONICAL_PREPARED_INTENT_SCHEMA_VERSION
    assert intent_path.exists()
    assert immutable_intent["prepared_intent_digest"] == receipt["prepared_intent_digest"]
    assert not ({"outbox_digest", "outbox_path", "receipt_path"} & set(receipt))
    assert outbox["receipt_digest"] == receipt["receipt_digest"]
    assert head["receipt_digest"] == receipt["receipt_digest"]
    assert queue.get(f"outbox_{receipt['transaction_id']}") is not None

    # Edges point from a published artifact to artifacts it references.  They
    # are derived from the asserted persisted fields above, so a future
    # receipt -> committed-outbox reference makes this traversal fail.
    dependencies = {
        "planning": set(),
        "prepared_intent": {"planning"},
        "diff": {"planning"},
        "receipt": {"planning", "prepared_intent", "diff"},
        "current_head": {"receipt"},
        "committed_outbox": {"prepared_intent", "receipt"},
        "projection_queue": {"committed_outbox"},
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise AssertionError(f"artifact dependency cycle at {node}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in dependencies[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for artifact in dependencies:
        visit(artifact)


def test_receipt_history_rejects_tampered_immutable_prepared_intent(
    tmp_path: Path,
) -> None:
    source, _index, _queue, _relations, committer, episode, scope = _setup(tmp_path)
    proposal = _proposal(
        episode,
        "prepared-intent-tamper",
        "SQLite",
        "confirmation",
        "confirmed",
    )
    _identity, _transition, plan = _plan(source, episode, scope, proposal)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=episode.episode_id,
    )
    committer.commit("u1", operations)

    receipt_path = committer._transaction_marker(str(operations[0].payload["idempotency_key"]))
    receipt_bytes = receipt_path.read_bytes()
    transaction_id = str(operations[0].payload["transaction_id"])
    intent_path = committer.planning_proofs.canonical_intent_path(transaction_id)
    payload = json.loads(intent_path.read_text(encoding="utf-8"))
    payload["artifact_digest"] = "0" * 64
    intent_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        CanonicalHistoryIntegrityError,
        match="durable matching prepared intent",
    ):
        validate_canonical_receipt_history(
            committer.artifact_root,
            tenant_id="t1",
        )

    assert receipt_path.read_bytes() == receipt_bytes
    assert receipt_path.exists()


def test_startup_fails_closed_on_tampered_immutable_prepared_intent(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    result = client.remember(
        user_id="u1",
        content="SQLite",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "prepared intent startup"},
    )
    head, receipt, _snapshot = load_current_head(tmp_path, result["uri"])
    receipt_path = tmp_path / str(head["receipt_path"])
    receipt_bytes = receipt_path.read_bytes()
    intent_path = client.committer.planning_proofs.canonical_intent_path(str(receipt["transaction_id"]))
    payload = json.loads(intent_path.read_text(encoding="utf-8"))
    payload["artifact_digest"] = "0" * 64
    intent_path.write_text(json.dumps(payload), encoding="utf-8")

    restarted = MemoryOSClient(str(tmp_path))

    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert any(
        "prepared-intent" in reason or "PlanningProofIntegrityError" in reason
        for reason in restarted.readiness.snapshot()["reasons"]
    )
    assert receipt_path.read_bytes() == receipt_bytes


def test_startup_does_not_regenerate_missing_current_schema_prepared_intent(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    result = client.remember(
        user_id="u1",
        content="SQLite",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields={"decision_topic": "missing prepared intent startup"},
    )
    head, receipt, _snapshot = load_current_head(tmp_path, result["uri"])
    receipt_path = tmp_path / str(head["receipt_path"])
    receipt_bytes = receipt_path.read_bytes()
    intent_path = client.committer.planning_proofs.canonical_intent_path(str(receipt["transaction_id"]))
    intent_path.unlink()

    restarted = MemoryOSClient(str(tmp_path))

    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert any("lost its immutable prepared intent" in reason for reason in restarted.readiness.snapshot()["reasons"])
    assert not intent_path.exists()
    assert receipt_path.read_bytes() == receipt_bytes


def _pending_archive() -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id="pending-uncommitted",
        archive_uri="memoryos://user/u1/sessions/history/pending-uncommitted",
        messages=[{"id": "m1", "role": "user", "content": "Maybe use Redis later."}],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
    )


def test_pending_source_written_without_receipt_and_head_is_not_visible(tmp_path: Path) -> None:
    source = FileSystemSourceStore(tmp_path, tenant_id="t1")
    archive = _pending_archive()
    episode = SessionArchiveEpisodeAdapter().adapt(archive)
    proposal = CandidateProposalAdapter().adapt(
        MemoryCandidateDraft(
            memory_type=MemoryType.PROJECT_DECISION,
            title="primary database",
            content="Redis",
            fields={"decision_topic": "primary database", "project_id": "memoryos"},
            confidence=0.4,
            source_role="user",
            source_adapter_id="codex",
            source_session_id=archive.session_id,
            source_message_ids=["m1"],
        ),
        episode,
        archive,
    )
    formed = CanonicalMemoryFormationService(source).plan_pending(
        proposal,
        archive=archive,
        episode=episode,
        reason="review_required",
        retrieval_views=["project:memoryos:decisions"],
        commit_group_id="pending-uncommitted",
    )
    operation = formed.operations[0]
    desired = ContextObject.from_dict(operation.payload["context_object"])
    source.write_object(desired, content=str(operation.payload["content"]))

    repository = CanonicalMemoryRepository(source)
    with pytest.raises(FileNotFoundError):
        repository.load_pending(str(operation.target_uri), tenant_id="t1", owner_user_id="u1")
    assert repository.list_pending(tenant_id="t1", owner_user_id="u1") == ()
    index = InMemoryIndexStore()
    index.upsert_index(desired, content="Redis")
    assembler = ContextAssembler(ContextDB(source, index, InMemoryRelationStore()))
    assert (
        assembler.search(
            "Redis",
            user_id="u1",
            tenant_id="t1",
            project_id="memoryos",
            context_type="memory",
            search_scope="candidates",
        )
        == []
    )


class _CrashAfterMetadataStore(FileSystemSourceStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root, tenant_id="t1")
        self.armed = False

    def _write_atomic(self, path: Path, content: str) -> None:
        super()._write_atomic(path, content)
        if self.armed and path.name == ".meta.json":
            self.armed = False
            raise SystemExit("crash after canonical metadata")


def test_canonical_bundle_crash_exposes_only_complete_old_or_new_generation(tmp_path: Path) -> None:
    store = _CrashAfterMetadataStore(tmp_path)
    uri = "memoryos://user/u1/memories/canonical/slots/s1/claims/c1"
    old = ContextObject(
        uri=uri,
        context_type=ContextType.MEMORY,
        title="old",
        owner_user_id="u1",
        tenant_id="t1",
        metadata={"canonical_kind": "claim", "revision": 1},
    )
    new = ContextObject.from_dict(old.to_dict())
    new.title = "new"
    new.metadata = {**new.metadata, "revision": 2}
    store.write_object(old, content="old")
    store.armed = True
    with pytest.raises(SystemExit):
        store.write_object(new, content="new")

    observed = (store.read_object(uri).title, store.read_content(uri))
    assert observed in {("old", "old"), ("new", "new")}


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


def test_durable_planning_envelope_prevents_model_recall_for_same_task(tmp_path: Path) -> None:
    extractor = _CountingExtractor()
    planner = MemoryCommitPlanner(
        extractor=extractor,
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
        relation_store=InMemoryRelationStore(),
    )
    archive = SessionArchive(
        user_id="u1",
        session_id="durable-planning",
        archive_uri="memoryos://user/u1/sessions/history/durable-planning",
        messages=[{"id": "m1", "role": "user", "content": "Remember this durable project rule."}],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
        task_id="durable-planning-task",
    )
    SessionArchiveStore(tmp_path, tenant_id="t1").write_sync_archive(archive)

    planner.plan(archive)
    planner.plan(archive)

    assert extractor.calls == 1


def test_salience_gate_skips_ordinary_chat_before_model_call(tmp_path: Path) -> None:
    extractor = _CountingExtractor()
    planner = MemoryCommitPlanner(
        extractor=extractor,
        source_store=FileSystemSourceStore(tmp_path, tenant_id="t1"),
        index_store=InMemoryIndexStore(),
    )
    archive = SessionArchive(
        user_id="u1",
        session_id="ordinary-chat",
        archive_uri="memoryos://user/u1/sessions/history/ordinary-chat",
        messages=[{"id": "m1", "role": "user", "content": "hello"}],
        metadata={"tenant_id": "t1"},
    )
    SessionArchiveStore(tmp_path, tenant_id="t1").write_sync_archive(archive)

    result = planner.plan(archive)

    assert extractor.calls == 0
    assert result.operations == ()


def test_explicit_remember_rejects_missing_stable_identity(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    with pytest.raises(ValueError, match="identity"):
        client.remember(user_id="u1", content="PostgreSQL", memory_type="project_decision")


@pytest.mark.parametrize(
    ("memory_type", "title"),
    [
        ("profile", "profile"),
        ("preference", "preference"),
        ("project_rule", "project rule"),
        ("project_decision", "decision"),
    ],
)
def test_explicit_remember_rejects_generic_compatibility_identity(
    tmp_path: Path,
    memory_type: str,
    title: str,
) -> None:
    client = MemoryOSClient(str(tmp_path))

    with pytest.raises(ValueError, match="too generic for stable identity"):
        client.remember(
            user_id="u1",
            content="one specific durable fact",
            title=title,
            memory_type=memory_type,
            project_id="memoryos",
            constraint_polarity="FORBID" if memory_type == "project_rule" else "",
        )


def test_callerless_sdk_uses_client_tenant_instead_of_default(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    client.remember(
        user_id="u1",
        title="primary database",
        content="SQLite",
        memory_type="project_decision",
        project_id="memoryos",
    )

    results = client.search_context(
        "SQLite",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
    )

    assert results
    assert all(item["tenant_id"] == "tenant-a" for item in results)


def test_remote_egress_policy_blocks_medical_archive_by_default() -> None:
    prompts: list[str] = []
    provider = FakeMemoryModelProvider(
        json.dumps({"candidates": []}),
        prompts=prompts,
        is_remote=True,
    )
    archive = SessionArchive(
        user_id="u1",
        session_id="medical-egress",
        archive_uri="memoryos://user/u1/sessions/history/medical-egress",
        messages=[{"id": "m1", "role": "user", "content": "Remember this: my HIV diagnosis is private."}],
    )
    episode = SessionArchiveEpisodeAdapter().adapt(archive)

    result = LLMMemoryExtractorBackend(provider).extract_batch_with_context(
        archive,
        MemoryTypeRegistry().list(),
        existing_memories=(),
        episode=episode,
    )

    assert prompts == []
    assert result.accepted == ()
    assert "egress_denied" in result.security_flags
