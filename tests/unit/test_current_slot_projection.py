from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import memoryos.memory.canonical.slot_projection as slot_projection_module
from memoryos.contextdb.catalog import CatalogRecord
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore, vector_row_id
from memoryos.memory.canonical.event import canonical_digest
from memoryos.memory.canonical.slot_projection import (
    CurrentSlotProjection,
    CurrentSlotProjectionIntegrityError,
)
from memoryos.memory.canonical.state import (
    ClaimState,
    MemoryClaim,
    MemoryRevision,
    MemorySlot,
    TransitionProfile,
)
from memoryos.memory.canonical.visibility import CommittedCanonicalRead

_T1 = "2026-07-14T01:00:00+00:00"
_T2 = "2026-07-15T01:00:00+00:00"
_SLOT_URI = "memoryos://user/u1/memories/canonical/slots/slot-ice-cream"


class _CatalogStore:
    def __init__(self) -> None:
        self.records: dict[str, CatalogRecord] = {}
        self.tombstones: list[dict[str, Any]] = []

    def upsert_catalog(self, record: CatalogRecord) -> None:
        self.records[record.record_key] = record

    def apply_tombstone(
        self,
        *,
        tenant_id: str,
        record_key: str,
        reason: str,
        uri: str,
        source_revision: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self.records.pop(record_key, None)
        tombstone = {
            "record_key": record_key,
            "tenant_id": tenant_id,
            "reason": reason,
            "uri": uri,
            "source_revision": source_revision,
            "payload": payload,
        }
        self.tombstones.append(tombstone)
        return tombstone


class _DeleteOnlyCatalogStore:
    def __init__(self) -> None:
        self.records: dict[str, CatalogRecord] = {}
        self.deleted: list[str] = []

    def upsert_catalog(self, record: CatalogRecord) -> None:
        self.records[record.record_key] = record

    def delete_catalog(self, record_key: str, *, tenant_id: str) -> None:
        del tenant_id
        self.records.pop(record_key, None)
        self.deleted.append(record_key)


class _Repository:
    source_store: Any = object()
    relation_store: Any = None

    def __init__(self, slot: MemorySlot, claims: tuple[MemoryClaim, ...]) -> None:
        self.slot = slot
        self.claims = claims
        self.load_calls: list[str] = []

    def load_uri(self, slot_uri: str) -> tuple[MemorySlot, tuple[MemoryClaim, ...]]:
        self.load_calls.append(slot_uri)
        return self.slot, self.claims


class _Embedding:
    model_name = "test-current-slot"
    dimension = 2

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.texts.append(text)
        return [1.0, float(len(text) % 17 + 1)]


def _revision(
    revision: int,
    state: ClaimState,
    value: Any,
    *,
    at: str,
    previous: int | None = None,
) -> MemoryRevision:
    return MemoryRevision(
        revision=revision,
        state=state.value,
        value_fields={"canonical_value": value},
        evidence_refs=(),
        proposal_id=f"proposal-{revision}-{state.value.lower()}",
        relation="SUPERSEDES" if revision > 1 else "UNRELATED",
        epistemic_status="EXPLICIT",
        created_at=at,
        transaction_time=at,
        valid_from=at,
        previous_revision=previous,
    )


def _claim(
    claim_id: str,
    value: str,
    *,
    revisions: tuple[MemoryRevision, ...] | None = None,
) -> MemoryClaim:
    return MemoryClaim(
        claim_id=claim_id,
        uri=f"{_SLOT_URI}/claims/{claim_id}",
        slot_id="slot-ice-cream",
        canonical_value=value,
        profile=TransitionProfile.AUTHORITATIVE_STATE,
        revisions=revisions or (_revision(1, ClaimState.ACTIVE, value, at=_T1),),
    )


def _slot(
    active_claim_id: str | None,
    claim_ids: tuple[str, ...],
    *,
    revision: int = 1,
) -> MemorySlot:
    return MemorySlot(
        slot_id="slot-ice-cream",
        uri=_SLOT_URI,
        memory_type="preference",
        identity_fields={"subject": "food", "dimension": "ice_cream_flavors"},
        scope_keys=("tenant:t1", "principal:u1", "workspace:p1"),
        claim_ids=claim_ids,
        active_claim_id=active_claim_id,
        revision=revision,
    )


def _committed_slot(slot: MemorySlot, *, transaction_id: str) -> CommittedCanonicalRead:
    obj = slot.to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        scope={"project_id": "p1"},
    )
    return _committed(obj, kind="slot", revision=slot.revision, transaction_id=transaction_id)


def _committed_claim(claim: MemoryClaim, *, transaction_id: str) -> CommittedCanonicalRead:
    obj = claim.to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        memory_type="preference",
        scope={"project_id": "p1"},
    )
    return _committed(obj, kind="claim", revision=claim.latest_revision.revision, transaction_id=transaction_id)


def _committed(obj: Any, *, kind: str, revision: int, transaction_id: str) -> CommittedCanonicalRead:
    receipt_digest = canonical_digest([obj.uri, transaction_id, "receipt"])
    head_digest = canonical_digest([obj.uri, transaction_id, "head"])
    head = {
        "uri": obj.uri,
        "canonical_kind": kind,
        "tenant_id": "t1",
        "owner_user_id": "u1",
        "current_revision": revision,
        "current_transaction_id": transaction_id,
        "receipt_digest": receipt_digest,
        "head_digest": head_digest,
    }
    return CommittedCanonicalRead(
        object=obj,
        head=head,
        receipt={"receipt_digest": receipt_digest},
    )


def _projector(
    monkeypatch: pytest.MonkeyPatch,
    repository: _Repository,
    store: Any,
    committed: dict[str, CommittedCanonicalRead],
    *,
    vector_store: InMemoryVectorStore | None = None,
    embedding_provider: _Embedding | None = None,
) -> CurrentSlotProjection:
    def read(_source: Any, uri: str, _relations: Any) -> CommittedCanonicalRead:
        return committed[uri]

    monkeypatch.setattr(slot_projection_module, "read_committed_canonical", read)
    return CurrentSlotProjection(
        repository,
        store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )


def test_repeated_preference_has_one_current_slot_record(monkeypatch: pytest.MonkeyPatch) -> None:
    claim = _claim("claim-likes", "vanilla, chocolate, strawberry")
    slot = _slot(claim.claim_id, (claim.claim_id,))
    repository = _Repository(slot, (claim,))
    store = _CatalogStore()
    projector = _projector(
        monkeypatch,
        repository,
        store,
        {
            slot.uri: _committed_slot(slot, transaction_id="tx-1"),
            claim.uri: _committed_claim(claim, transaction_id="tx-1"),
        },
    )

    first = projector.project(slot.uri)
    repeated = projector.project(slot.uri)

    assert first.record_key == "slot:slot-ice-cream:current"
    assert repeated.record_key == first.record_key
    assert len(store.records) == 1
    record = store.records[first.record_key]
    assert record.record_kind == "current_slot"
    assert record.canonical_slot_id == slot.slot_id
    assert record.canonical_claim_id == claim.claim_id
    assert record.canonical_state == ClaimState.ACTIVE.value
    assert record.canonical_revision == 1
    assert record.tree_paths == ("memories/preferences/food/ice_cream_flavors",)
    assert record.metadata["identity_fields"] == dict(slot.identity_fields)
    assert record.metadata["canonical_value"] == "vanilla, chocolate, strawberry"
    assert "subject: food" in record.title
    assert "dimension: ice cream flavors" in record.l0_text
    assert "identity: dimension: ice cream flavors; subject: food" in record.l1_text
    assert repository.load_calls == [slot.uri, slot.uri]


def test_multi_value_preference_is_one_structured_value_and_one_active_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flavors = ["vanilla", "chocolate", "strawberry"]
    claim = _claim(
        "claim-flavor-set",
        "ice cream flavor set",
        revisions=(_revision(1, ClaimState.ACTIVE, flavors, at=_T1),),
    )
    slot = _slot(claim.claim_id, (claim.claim_id,))
    repository = _Repository(slot, (claim,))
    store = _CatalogStore()
    projector = _projector(
        monkeypatch,
        repository,
        store,
        {
            slot.uri: _committed_slot(slot, transaction_id="tx-multi"),
            claim.uri: _committed_claim(claim, transaction_id="tx-multi"),
        },
    )

    projected = projector.project(slot.uri).record

    assert projected is not None
    assert len(store.records) == 1
    assert slot.active_claim_id == claim.claim_id
    assert len(slot.claim_ids) == 1
    assert projected.metadata["canonical_value"] == flavors
    assert projected.canonical_claim_id == claim.claim_id


def test_active_claim_switch_updates_same_record_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    claim_b = _claim("claim-likes", "I like ice cream")
    slot_b = _slot(claim_b.claim_id, (claim_b.claim_id,))
    repository = _Repository(slot_b, (claim_b,))
    store = _CatalogStore()
    vectors = InMemoryVectorStore()
    embedding = _Embedding()
    committed = {
        slot_b.uri: _committed_slot(slot_b, transaction_id="tx-1"),
        claim_b.uri: _committed_claim(claim_b, transaction_id="tx-1"),
    }
    projector = _projector(
        monkeypatch,
        repository,
        store,
        committed,
        vector_store=vectors,
        embedding_provider=embedding,
    )

    first = projector.project(slot_b.uri).record
    assert first is not None
    assert tuple(vectors.rows) == (vector_row_id(first.tenant_id, first.record_key),)
    first_vector_metadata = vectors.get_vector_metadata(first.uri)
    assert first_vector_metadata is not None
    assert first_vector_metadata["canonical_claim_id"] == claim_b.claim_id

    superseded_b = _claim(
        "claim-likes",
        "I like ice cream",
        revisions=(
            _revision(1, ClaimState.ACTIVE, "I like ice cream", at=_T1),
            _revision(2, ClaimState.SUPERSEDED, "I like ice cream", at=_T2, previous=1),
        ),
    )
    claim_c = _claim(
        "claim-dislikes",
        "I do not like ice cream",
        revisions=(_revision(1, ClaimState.ACTIVE, "I do not like ice cream", at=_T2),),
    )
    slot_c = _slot(claim_c.claim_id, (superseded_b.claim_id, claim_c.claim_id), revision=2)
    repository.slot = slot_c
    repository.claims = (superseded_b, claim_c)
    committed.clear()
    committed.update(
        {
            slot_c.uri: _committed_slot(slot_c, transaction_id="tx-2"),
            claim_c.uri: _committed_claim(claim_c, transaction_id="tx-2"),
        }
    )

    second = projector.project(slot_c.uri).record

    assert second is not None
    assert len(store.records) == 1
    assert second.record_key == first.record_key
    assert second.uri == first.uri
    assert second.canonical_slot_id == first.canonical_slot_id
    assert second.canonical_claim_id == claim_c.claim_id
    assert second.metadata["canonical_value"] == "I do not like ice cream"
    assert second.projection_effect_hash != first.projection_effect_hash
    assert claim_b.claim_id not in second.l0_text
    assert tuple(vectors.rows) == (vector_row_id(second.tenant_id, second.record_key),)
    second_vector_metadata = vectors.get_vector_metadata(second.uri)
    assert second_vector_metadata is not None
    assert second_vector_metadata["canonical_claim_id"] == claim_c.claim_id
    assert "I do not like ice cream" in embedding.texts[-1]
    assert "I like ice cream" not in embedding.texts[-1]


def test_retracted_slot_tombstones_current_record(monkeypatch: pytest.MonkeyPatch) -> None:
    claim = _claim("claim-likes", "I like ice cream")
    slot = _slot(claim.claim_id, (claim.claim_id,))
    repository = _Repository(slot, (claim,))
    store = _CatalogStore()
    vectors = InMemoryVectorStore()
    embedding = _Embedding()
    committed = {
        slot.uri: _committed_slot(slot, transaction_id="tx-1"),
        claim.uri: _committed_claim(claim, transaction_id="tx-1"),
    }
    projector = _projector(
        monkeypatch,
        repository,
        store,
        committed,
        vector_store=vectors,
        embedding_provider=embedding,
    )
    projector.project(slot.uri)
    assert vectors.rows

    retracted = _claim(
        claim.claim_id,
        claim.canonical_value,
        revisions=(
            _revision(1, ClaimState.ACTIVE, claim.canonical_value, at=_T1),
            _revision(2, ClaimState.RETRACTED, claim.canonical_value, at=_T2, previous=1),
        ),
    )
    retired_slot = _slot(None, (retracted.claim_id,), revision=2)
    repository.slot = retired_slot
    repository.claims = (retracted,)
    committed.clear()
    committed[retired_slot.uri] = _committed_slot(retired_slot, transaction_id="tx-2")

    result = projector.project(retired_slot.uri)

    assert result.status == "tombstoned"
    assert store.records == {}
    assert vectors.rows == {}
    assert store.tombstones == [
        {
            "record_key": "slot:slot-ice-cream:current",
            "tenant_id": "t1",
            "reason": "canonical_claim_retracted",
            "uri": f"{_SLOT_URI}/serving/current",
            "source_revision": 2,
            "payload": {"record_kind": "current_slot"},
        }
    ]


def test_no_active_claim_can_use_explicit_delete_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    retracted = _claim(
        "claim-likes",
        "I like ice cream",
        revisions=(_revision(1, ClaimState.RETRACTED, "I like ice cream", at=_T1),),
    )
    slot = _slot(None, (retracted.claim_id,))
    repository = _Repository(slot, (retracted,))
    store = _DeleteOnlyCatalogStore()
    projector = _projector(
        monkeypatch,
        repository,
        store,
        {slot.uri: _committed_slot(slot, transaction_id="tx-1")},
    )

    result = projector.project(slot.uri)

    assert result.status == "deleted"
    assert store.deleted == ["slot:slot-ice-cream:current"]


def test_real_sqlite_upsert_and_durable_tombstone_are_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim("claim-likes", "I like ice cream")
    slot = _slot(claim.claim_id, (claim.claim_id,))
    repository = _Repository(slot, (claim,))
    store = SQLiteIndexStore(tmp_path / "catalog.db")
    committed = {
        slot.uri: _committed_slot(slot, transaction_id="tx-1"),
        claim.uri: _committed_claim(claim, transaction_id="tx-1"),
    }
    projector = _projector(monkeypatch, repository, store, committed)

    first = projector.project(slot.uri)
    repeated = projector.project(slot.uri)

    assert repeated.record_key == first.record_key
    stored = store.get_catalog(first.record_key, tenant_id="t1")
    assert first.record is not None and stored is not None
    projector.validate_record(
        stored,
        expected_slot=slot,
        expected_claim=claim,
        expected_head_digest=first.record.canonical_head_digest,
        expected_receipt_digest=first.record.receipt_digest,
        expected_effect_hash=first.record.projection_effect_hash,
        expected_claim_head_digest=str(first.record.metadata["claim_head_digest"]),
        expected_claim_receipt_digest=str(first.record.metadata["claim_receipt_digest"]),
        expected_tenant_id="t1",
        expected_owner_user_id="u1",
        expected_transaction_id="tx-1",
    )
    assert len(store.list_catalog(filters={"record_kind": "current_slot", "tenant_id": "t1"})) == 1

    retracted = _claim(
        claim.claim_id,
        claim.canonical_value,
        revisions=(
            _revision(1, ClaimState.ACTIVE, claim.canonical_value, at=_T1),
            _revision(2, ClaimState.RETRACTED, claim.canonical_value, at=_T2, previous=1),
        ),
    )
    retired_slot = _slot(None, (retracted.claim_id,), revision=2)
    repository.slot = retired_slot
    repository.claims = (retracted,)
    committed.clear()
    committed[retired_slot.uri] = _committed_slot(retired_slot, transaction_id="tx-2")

    retired = projector.project(retired_slot.uri)
    replayed = projector.project(retired_slot.uri)

    assert retired.status == replayed.status == "tombstoned"
    assert store.get_catalog(first.record_key, tenant_id="t1") is None
    assert store.get_pending_tombstones() == []


def test_tree_reclassification_does_not_change_canonical_identity_or_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim("claim-likes", "I like ice cream")
    slot = _slot(claim.claim_id, (claim.claim_id,))
    repository = _Repository(slot, (claim,))
    store = _CatalogStore()
    projector = _projector(
        monkeypatch,
        repository,
        store,
        {
            slot.uri: _committed_slot(slot, transaction_id="tx-1"),
            claim.uri: _committed_claim(claim, transaction_id="tx-1"),
        },
    )

    before = projector.project(slot.uri).record
    after = projector.project(
        slot.uri,
        tree_paths=("memories/preferences/dessert/flavors", "projects/p1"),
    ).record

    assert before is not None and after is not None
    assert after.tree_paths != before.tree_paths
    assert after.record_key == before.record_key
    assert after.canonical_slot_id == before.canonical_slot_id == slot.slot_id
    assert after.canonical_claim_id == before.canonical_claim_id == claim.claim_id
    assert after.canonical_slot_uri == before.canonical_slot_uri == slot.uri
    assert after.canonical_claim_uri == before.canonical_claim_uri == claim.uri
    assert after.projection_effect_hash == before.projection_effect_hash


def test_validation_fails_closed_on_identity_value_or_proof_tampering(monkeypatch: pytest.MonkeyPatch) -> None:
    claim = _claim("claim-likes", "I like ice cream")
    slot = _slot(claim.claim_id, (claim.claim_id,))
    repository = _Repository(slot, (claim,))
    store = _CatalogStore()
    projector = _projector(
        monkeypatch,
        repository,
        store,
        {
            slot.uri: _committed_slot(slot, transaction_id="tx-1"),
            claim.uri: _committed_claim(claim, transaction_id="tx-1"),
        },
    )
    record = projector.project(slot.uri).record
    assert record is not None

    def validate(candidate: CatalogRecord) -> None:
        projector.validate_record(
            candidate,
            expected_slot=slot,
            expected_claim=claim,
            expected_head_digest=record.canonical_head_digest,
            expected_receipt_digest=record.receipt_digest,
            expected_effect_hash=record.projection_effect_hash,
            expected_claim_head_digest=str(record.metadata["claim_head_digest"]),
            expected_claim_receipt_digest=str(record.metadata["claim_receipt_digest"]),
            expected_tenant_id="t1",
            expected_owner_user_id="u1",
            expected_transaction_id="tx-1",
        )

    validate(record)

    tampered_records = (
        replace(record, canonical_slot_id="other-slot"),
        replace(record, canonical_claim_id="other-claim"),
        replace(record, canonical_head_digest="bad-head"),
        replace(record, receipt_digest="bad-receipt"),
        replace(record, projection_effect_hash="bad-effect"),
        replace(record, metadata={**dict(record.metadata), "canonical_value": "tampered"}),
    )
    for tampered in tampered_records:
        with pytest.raises(CurrentSlotProjectionIntegrityError, match="validation failed"):
            validate(tampered)
