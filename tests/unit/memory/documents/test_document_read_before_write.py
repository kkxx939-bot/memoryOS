from __future__ import annotations

import hashlib

import pytest

from memoryos.adapters.persistence.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from memoryos.memory.documents import (
    ABSENT,
    DocumentEditKind,
    MemoryCandidateKind,
    MemoryDocumentPlanner,
    MemoryEditProposal,
    PresentPath,
    RelatedDocumentCandidate,
    explicit_evidence_digest,
    new_document_id,
    render_new_document,
)


def _proposal() -> MemoryEditProposal:
    return MemoryEditProposal(
        candidate_kind=MemoryCandidateKind.TOPIC_NOTE,
        title="Read before write",
        subject="Read before write",
        body="Catalog hints must be verified against live Markdown.",
        evidence_refs=("event-1",),
    )


def test_related_catalog_candidate_is_hydrated_and_deduplicated_across_documents(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    document_id = new_document_id()
    relative = "knowledge/entities/memoryos.md"
    raw = render_new_document(
        document_id,
        "## Read before write\n\nCatalog hints must be verified against live Markdown.\n",
    )
    store.create("default", "u1", relative, raw, expected=ABSENT)
    digest = hashlib.sha256(raw).hexdigest()
    calls: list[int] = []

    def find_related(tenant, owner, proposal, limit):
        assert (tenant, owner, proposal.title) == ("default", "u1", "Read before write")
        calls.append(limit)
        return (
            RelatedDocumentCandidate(
                tenant_id=tenant,
                owner_user_id=owner,
                document_id=document_id,
                relative_path=relative,
                source_digest=digest,
                relevance=0.9,
            ),
        )

    plan = MemoryDocumentPlanner(store, related_document_finder=find_related).plan(
        _proposal(),
        tenant_id="default",
        owner_user_id="u1",
        idempotency_key="session:1",
        evidence_digest=explicit_evidence_digest("evidence"),
    )

    assert calls == [8]
    assert plan.edit_kind is DocumentEditKind.UPDATE
    assert plan.document_id == document_id
    assert plan.relative_path == relative
    assert isinstance(plan.expected_state, PresentPath)
    assert plan.expected_state.raw_sha256 == digest
    assert plan.after_bytes == raw


def test_stale_related_digest_uses_latest_live_state_as_write_authority(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    document_id = new_document_id()
    relative = "knowledge/entities/stale.md"
    raw = render_new_document(
        document_id,
        "## Read before write\n\nCatalog hints must be verified against live Markdown.\n",
    )
    store.create("default", "u1", relative, raw, expected=ABSENT)

    planner = MemoryDocumentPlanner(
        store,
        related_document_finder=lambda tenant, owner, proposal, limit: (
            RelatedDocumentCandidate(
                tenant_id=tenant,
                owner_user_id=owner,
                document_id=document_id,
                relative_path=relative,
                source_digest="f" * 64,
            ),
        ),
    )
    plan = planner.plan(
        _proposal(),
        tenant_id="default",
        owner_user_id="u1",
        idempotency_key="session:2",
        evidence_digest=explicit_evidence_digest("evidence"),
    )

    assert plan.edit_kind is DocumentEditKind.UPDATE
    assert plan.relative_path == relative
    assert plan.document_id == document_id
    assert isinstance(plan.expected_state, PresentPath)
    assert plan.expected_state.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert plan.after_bytes == raw


def test_missing_or_identity_mismatched_related_hint_is_ignored(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    live_id = new_document_id()
    mismatched_path = "knowledge/entities/mismatched.md"
    raw = render_new_document(
        live_id,
        "## Read before write\n\nCatalog hints must be verified against live Markdown.\n",
    )
    store.create("default", "u1", mismatched_path, raw, expected=ABSENT)
    candidates = (
        RelatedDocumentCandidate(
            tenant_id="default",
            owner_user_id="u1",
            document_id=new_document_id(),
            relative_path="knowledge/entities/missing.md",
            source_digest="a" * 64,
            relevance=1.0,
        ),
        RelatedDocumentCandidate(
            tenant_id="default",
            owner_user_id="u1",
            document_id=new_document_id(),
            relative_path=mismatched_path,
            source_digest=hashlib.sha256(raw).hexdigest(),
            relevance=0.9,
        ),
    )

    plan = MemoryDocumentPlanner(
        store,
        related_document_finder=lambda *_args: candidates,
    ).plan(
        _proposal(),
        tenant_id="default",
        owner_user_id="u1",
        idempotency_key="session:stale-path-or-id",
        evidence_digest=explicit_evidence_digest("evidence"),
    )

    assert plan.edit_kind is DocumentEditKind.CREATE
    assert plan.relative_path == "knowledge/topics/read-before-write.md"


def test_high_catalog_score_does_not_override_live_semantic_compatibility(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    document_id = new_document_id()
    relative = "knowledge/topics/unrelated.md"
    raw = render_new_document(
        document_id,
        "## Gardening\n\nTomatoes need regular watering and full sun.\n",
    )
    store.create("default", "u1", relative, raw, expected=ABSENT)
    candidate = RelatedDocumentCandidate(
        tenant_id="default",
        owner_user_id="u1",
        document_id=document_id,
        relative_path=relative,
        source_digest=hashlib.sha256(raw).hexdigest(),
        relevance=999.0,
    )

    plan = MemoryDocumentPlanner(
        store,
        related_document_finder=lambda *_args: (candidate,),
    ).plan(
        _proposal(),
        tenant_id="default",
        owner_user_id="u1",
        idempotency_key="session:semantic-revalidation",
        evidence_digest=explicit_evidence_digest("evidence"),
    )

    assert plan.edit_kind is DocumentEditKind.CREATE
    assert plan.relative_path == "knowledge/topics/read-before-write.md"


def test_related_live_semantics_require_token_or_heading_boundaries(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    document_id = new_document_id()
    relative = "knowledge/topics/reader.md"
    raw = render_new_document(
        document_id,
        "## Reader settings\n\nA screen reader uses a distinct configuration.\n",
    )
    store.create("default", "u1", relative, raw, expected=ABSENT)
    proposal = MemoryEditProposal(
        candidate_kind=MemoryCandidateKind.TOPIC_NOTE,
        title="Read",
        subject="Read",
        body="Read-before-write requires exact live bytes.",
        evidence_refs=("event-1",),
    )
    candidate = RelatedDocumentCandidate(
        tenant_id="default",
        owner_user_id="u1",
        document_id=document_id,
        relative_path=relative,
        source_digest=hashlib.sha256(raw).hexdigest(),
        relevance=999.0,
    )

    plan = MemoryDocumentPlanner(
        store,
        related_document_finder=lambda *_args: (candidate,),
    ).plan(
        proposal,
        tenant_id="default",
        owner_user_id="u1",
        idempotency_key="session:semantic-boundary",
        evidence_digest=explicit_evidence_digest("evidence"),
    )

    assert plan.edit_kind is DocumentEditKind.CREATE
    assert plan.relative_path == "knowledge/topics/read.md"


def test_related_candidate_receives_supplement_instead_of_splitting_routed_topic(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    document_id = new_document_id()
    relative = "knowledge/entities/memoryos.md"
    raw = render_new_document(
        document_id,
        "## Existing overview\n\nMemoryOS already has a document-native source.\n",
    )
    store.create("default", "u1", relative, raw, expected=ABSENT)
    digest = hashlib.sha256(raw).hexdigest()
    proposal = MemoryEditProposal(
        candidate_kind=MemoryCandidateKind.TOPIC_NOTE,
        title="Read before write",
        subject="MemoryOS",
        body="Catalog hints must be verified against live Markdown.",
        evidence_refs=("event-1",),
    )
    candidate = RelatedDocumentCandidate(
        tenant_id="default",
        owner_user_id="u1",
        document_id=document_id,
        relative_path=relative,
        source_digest=digest,
        relevance=0.99,
    )

    plan = MemoryDocumentPlanner(
        store,
        related_document_finder=lambda *_args: (candidate,),
    ).plan(
        proposal,
        tenant_id="default",
        owner_user_id="u1",
        idempotency_key="session:supplement",
        evidence_digest=explicit_evidence_digest("evidence"),
    )

    assert plan.edit_kind is DocumentEditKind.UPDATE
    assert plan.document_id == document_id
    assert plan.relative_path == relative
    assert isinstance(plan.expected_state, PresentPath)
    assert plan.expected_state.raw_sha256 == digest
    after_bytes = plan.after_bytes
    assert after_bytes is not None
    assert b"MemoryOS already has a document-native source." in after_bytes
    assert b"Catalog hints must be verified against live Markdown." in after_bytes
    assert plan.edit_summary.startswith("supplemented:")


def test_related_matching_section_is_corrected_in_place_from_latest_bytes(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    document_id = new_document_id()
    relative = "knowledge/entities/memoryos.md"
    raw = render_new_document(
        document_id,
        "## Read before write\n\nCatalog is trusted without a live read.\n\n"
        "## Stable section\n\nThis section must remain byte-visible.\n",
    )
    store.create("default", "u1", relative, raw, expected=ABSENT)
    digest = hashlib.sha256(raw).hexdigest()
    candidate = RelatedDocumentCandidate(
        tenant_id="default",
        owner_user_id="u1",
        document_id=document_id,
        relative_path=relative,
        source_digest=digest,
        relevance=1.0,
    )

    plan = MemoryDocumentPlanner(
        store,
        related_document_finder=lambda *_args: (candidate,),
    ).plan(
        _proposal(),
        tenant_id="default",
        owner_user_id="u1",
        idempotency_key="session:correction",
        evidence_digest=explicit_evidence_digest("evidence"),
    )

    assert plan.edit_kind is DocumentEditKind.UPDATE
    assert plan.document_id == document_id
    assert plan.relative_path == relative
    assert isinstance(plan.expected_state, PresentPath)
    assert plan.expected_state.raw_sha256 == digest
    after_bytes = plan.after_bytes
    assert after_bytes is not None
    assert b"Catalog is trusted without a live read." not in after_bytes
    assert after_bytes.count(b"## Read before write") == 1
    assert b"Catalog hints must be verified against live Markdown." in after_bytes
    assert b"This section must remain byte-visible." in after_bytes
    assert plan.edit_summary.startswith("corrected:")


def test_related_lookup_is_bounded_and_scope_checked(tmp_path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    candidate = RelatedDocumentCandidate(
        tenant_id="other",
        owner_user_id="u1",
        document_id=new_document_id(),
        relative_path="knowledge/topics/other.md",
        source_digest="a" * 64,
    )
    planner = MemoryDocumentPlanner(
        store,
        related_document_finder=lambda tenant, owner, proposal, limit: (candidate,),
    )
    with pytest.raises(PermissionError, match="trusted scope"):
        planner.plan(
            _proposal(),
            tenant_id="default",
            owner_user_id="u1",
            idempotency_key="session:3",
            evidence_digest=explicit_evidence_digest("evidence"),
        )

    planner = MemoryDocumentPlanner(
        store,
        max_related_documents=1,
        related_document_finder=lambda tenant, owner, proposal, limit: (candidate, candidate),
    )
    with pytest.raises(ValueError, match="exceeded its bound"):
        planner.plan(
            _proposal(),
            tenant_id="default",
            owner_user_id="u1",
            idempotency_key="session:4",
            evidence_digest=explicit_evidence_digest("evidence"),
        )
