from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from memoryos.api.http.app import handle
from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.retrieval_contract import RetrievalOptions
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.catalog import CatalogRecord, CatalogRecordKind
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.query_plan import RetrievalQueryIntent
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import FileSystemSourceStore
from memoryos.memory.canonical.episode import SessionArchiveEpisodeAdapter
from memoryos.memory.canonical.event import canonical_digest
from memoryos.memory.canonical.evidence import EvidenceRef
from memoryos.memory.canonical.repository import CanonicalMemoryRepository
from memoryos.memory.canonical.scope import MemoryScope, ScopeRef, ScopeSelector, VisibilityPolicy
from memoryos.memory.canonical.semantic import Commitment, SpeechAct
from memoryos.memory.canonical.state import (
    ClaimState,
    MemoryClaim,
    MemoryRevision,
    MemorySlot,
    TransitionProfile,
)
from memoryos.operations.model.operation_action import OperationAction
from tests.support.canonical_transactions import (
    _plan,
    _proposal,
    _supplement_proposal,
)


def _remember_preference(
    client: MemoryOSClient,
    *,
    title: str,
    content: str,
) -> dict:
    return client.remember(
        user_id="u1",
        memory_type="preference",
        title=title,
        content=content,
        identity_fields={"subject": "food", "dimension": "ice_cream"},
    )


def _current(client: MemoryOSClient) -> list[dict]:
    return client.search_context(
        "",
        user_id="u1",
        context_type="memory",
        query_intent="CURRENT",
        limit=20,
    )


def test_legacy_slot_and_claim_uri_exact_filters_work_through_sdk_http_and_mcp(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    committed = client.remember(
        user_id="u1",
        memory_type="preference",
        title="Exact URI preference",
        content="I prefer vanilla ice cream",
        identity_fields={"subject": "food", "dimension": "ice_cream_flavor"},
    )
    assert committed["status"] == "COMMITTED"
    claim_uri = str(committed["uri"])
    slot_uri = claim_uri.rsplit("/claims/", 1)[0]

    legacy = {
        "query": "text with no projection lexical overlap",
        "user_id": "u1",
        "context_type": "memory",
        "slot_uris": [slot_uri],
        "query_intent": "CURRENT",
        "limit": 10,
    }
    sdk_current = client.search_context(
        str(legacy["query"]),
        user_id="u1",
        context_type="memory",
        slot_uris=[slot_uri],
        query_intent="CURRENT",
        limit=10,
    )
    http_current = handle("POST /context/search", client, legacy)["results"]
    mcp = MemoryOSMCPServer(
        client,
        config=MCPServerConfig(
            root=str(tmp_path),
            user_id="u1",
            tenant_id="default",
            adapter_id="codex",
            actor_id="codex",
        ),
    )
    mcp_current = mcp.call_tool(
        "memoryos_search",
        {
            "query": legacy["query"],
            "context_type": "memory",
            "slot_uris": [slot_uri],
            "query_intent": "CURRENT",
            "limit": 10,
        },
    )

    assert mcp_current["error"] is None, mcp_current
    for results in (sdk_current, http_current, mcp_current["results"]):
        assert len(results) == 1
        assert results[0]["metadata"]["record_kind"] == CatalogRecordKind.CURRENT_SLOT.value
        # CanonicalResolver preserves the historical public Claim URI while
        # the bounded Catalog identity remains the CurrentSlot record key.
        assert results[0]["uri"] == claim_uri
        assert results[0]["record_key"] == (
            f"slot:{results[0]['metadata']['canonical_slot_id']}:current"
        )
        assert results[0]["metadata"]["canonical_slot_uri"] == slot_uri
        assert results[0]["metadata"]["canonical_claim_uri"] == claim_uri

    history = client.search_context(
        "another query with no lexical overlap",
        user_id="u1",
        context_type="memory",
        claim_uris=[claim_uri],
        query_intent="HISTORY",
        limit=10,
    )
    assert history
    assert all(
        item["metadata"]["record_kind"] == CatalogRecordKind.CLAIM_REVISION.value
        for item in history
    )
    assert {item["metadata"]["canonical_claim_uri"] for item in history} == {claim_uri}
    assert all(item["canonical_validation_status"] == "validated_history" for item in history)


def test_current_relation_expansion_uses_canonical_identities_and_rebinds_through_catalog_acl(
    tmp_path,
) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    first = client.remember(
        user_id="u1",
        memory_type="preference",
        title="First related preference",
        content="I prefer vanilla ice cream",
        identity_fields={"subject": "food", "dimension": "primary_ice_cream_flavor"},
    )
    second = client.remember(
        user_id="u1",
        memory_type="preference",
        title="Second related preference",
        content="I prefer strawberry sorbet",
        identity_fields={"subject": "food", "dimension": "secondary_frozen_dessert"},
    )
    assert first["status"] == second["status"] == "COMMITTED"
    first_claim_uri = str(first["uri"])
    second_claim_uri = str(second["uri"])
    first_slot_uri = first_claim_uri.rsplit("/claims/", 1)[0]
    second_slot_uri = second_claim_uri.rsplit("/claims/", 1)[0]
    client.relation_store.add_relation(
        ContextRelation(
            source_uri=first_claim_uri,
            relation_type="related_preference",
            target_uri=second_claim_uri,
            weight=0.9,
            metadata={"tenant_id": "default", "owner_user_id": "u1"},
        )
    )

    results = client.search_context(
        "identity-only relation expansion",
        user_id="u1",
        options=RetrievalOptions(
            target_uris=(first_slot_uri,),
            context_types=(ContextType.MEMORY,),
            owner_user_id="u1",
            query_intent=RetrievalQueryIntent.CURRENT,
            relation_expansion=True,
            candidate_limit=20,
            final_limit=20,
        ),
    )

    by_slot_uri = {item["metadata"]["canonical_slot_uri"]: item for item in results}
    assert set(by_slot_uri) == {first_slot_uri, second_slot_uri}
    assert all(
        item["metadata"]["record_kind"] == CatalogRecordKind.CURRENT_SLOT.value
        for item in results
    )
    assert by_slot_uri[second_slot_uri]["metadata"]["score_components"]["relation_score"] > 0.0
    trace = client.recall_trace(str(client.last_recall_trace_id))
    assert trace["relation_candidates"] == 1
    assert trace["canonical_validated"] == 2


def test_repeated_preference_and_state_change_keep_one_current_slot(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    repository = CanonicalMemoryRepository(client.source_store, client.relation_store)

    first = _remember_preference(
        client,
        title="first preference evidence",
        content="I like ice cream",
    )
    assert first["status"] == "COMMITTED"
    first_current = _current(client)
    assert len(first_current) == 1
    slot_uri = str(first_current[0]["metadata"]["canonical_slot_uri"])
    first_claim_id = str(first_current[0]["metadata"]["canonical_claim_id"])
    slot, claims = repository.load_uri(slot_uri)
    first_claim = next(claim for claim in claims if claim.claim_id == first_claim_id)
    first_evidence_count = sum(len(revision.evidence_refs) for revision in first_claim.revisions)
    current_record_key = str(first_current[0]["record_key"])
    assert current_record_key == f"slot:{slot.slot_id}:current"

    repeated = _remember_preference(
        client,
        title="independent repeated preference evidence",
        content="I like ice cream",
    )
    assert repeated["status"] == "COMMITTED"
    repeated_current = _current(client)
    assert len(repeated_current) == 1
    assert repeated_current[0]["record_key"] == current_record_key
    assert repeated_current[0]["metadata"]["canonical_claim_id"] == first_claim_id
    slot_after_repeat, repeated_claims = repository.load_uri(slot_uri)
    assert slot_after_repeat.slot_id == slot.slot_id
    assert sum(claim.current.state == ClaimState.ACTIVE.value for claim in repeated_claims) == 1
    repeated_claim = next(claim for claim in repeated_claims if claim.claim_id == first_claim_id)
    assert sum(len(revision.evidence_refs) for revision in repeated_claim.revisions) > first_evidence_count

    changed = _remember_preference(
        client,
        title="authoritative preference change",
        content="I do not like ice cream anymore",
    )
    assert changed["status"] == "PENDING"
    pending = next(item for item in client.list_pending(user_id="u1") if item["uri"] == changed["uri"])
    reviewed = client.review_pending(
        user_id="u1",
        pending_uri=pending["uri"],
        decision="CONFIRM_AND_APPLY",
        expected_lifecycle_revision=pending["lifecycle_revision"],
        expected_proposal_fingerprint=pending["proposal_fingerprint"],
        command_id="confirm-preference-state-change",
    )
    assert reviewed["status"] == "resolved"
    changed_current = _current(client)
    assert len(changed_current) == 1
    assert changed_current[0]["record_key"] == current_record_key
    assert changed_current[0]["metadata"]["canonical_slot_id"] == slot.slot_id
    new_claim_id = str(changed_current[0]["metadata"]["canonical_claim_id"])
    assert new_claim_id != first_claim_id

    final_slot, final_claims = repository.load_uri(slot_uri)
    states = {claim.claim_id: claim.current.state for claim in final_claims}
    assert final_slot.slot_id == slot.slot_id
    assert final_slot.active_claim_id == new_claim_id
    assert states[first_claim_id] == ClaimState.SUPERSEDED.value
    assert states[new_claim_id] == ClaimState.ACTIVE.value
    assert sum(state == ClaimState.ACTIVE.value for state in states.values()) == 1

    claim_revision_rows = [
        record
        for record in client.index_store.list_catalog(  # type: ignore[attr-defined]
            filters={
                "tenant_id": "default",
                "record_kinds": ("claim_revision",),
                "canonical_slot_ids": (slot.slot_id,),
                "include_inactive": True,
            },
            limit=20,
        )
        if isinstance(record, CatalogRecord)
    ]
    old_active_rows = sorted(
        (
            row
            for row in claim_revision_rows
            if row.canonical_claim_id == first_claim_id and row.canonical_state == ClaimState.ACTIVE.value
        ),
        key=lambda row: row.canonical_revision,
    )
    assert old_active_rows
    old_superseded_row = next(
        row
        for row in claim_revision_rows
        if row.canonical_claim_id == first_claim_id and row.canonical_state == ClaimState.SUPERSEDED.value
    )
    new_active_row = next(
        row
        for row in claim_revision_rows
        if row.canonical_claim_id == new_claim_id and row.canonical_state == ClaimState.ACTIVE.value
    )
    old_claim_history = [*old_active_rows, old_superseded_row]
    assert all(
        old_claim_history[index].valid_to == old_claim_history[index + 1].valid_from
        for index in range(len(old_claim_history) - 1)
    )
    old_active_row = old_active_rows[-1]
    assert old_active_row.valid_to == old_superseded_row.valid_from
    old_claim = next(claim for claim in final_claims if claim.claim_id == first_claim_id)
    assert old_claim.revisions[0].valid_to is None

    as_of_before = client.search_context(
        "",
        user_id="u1",
        options=RetrievalOptions(
            context_types=(ContextType.MEMORY,),
            query_intent=RetrievalQueryIntent.AS_OF,
            valid_at=old_active_row.valid_from,
            candidate_limit=20,
            final_limit=20,
        ),
    )
    assert {
        (str(item["metadata"].get("canonical_claim_id") or ""), item["metadata"].get("canonical_state"))
        for item in as_of_before
    } == {(first_claim_id, ClaimState.ACTIVE.value)}

    as_of_after = client.search_context(
        "",
        user_id="u1",
        options=RetrievalOptions(
            context_types=(ContextType.MEMORY,),
            query_intent=RetrievalQueryIntent.AS_OF,
            valid_at=new_active_row.valid_from,
            candidate_limit=20,
            final_limit=20,
        ),
    )
    assert {
        (str(item["metadata"].get("canonical_claim_id") or ""), item["metadata"].get("canonical_state"))
        for item in as_of_after
    } == {(new_claim_id, ClaimState.ACTIVE.value)}

    current_rows = [
        record
        for record in client.index_store.list_catalog(  # type: ignore[attr-defined]
            filters={
                "tenant_id": "default",
                "record_kinds": ("current_slot",),
                "canonical_slot_ids": (slot.slot_id,),
                "include_inactive": True,
            },
            limit=20,
        )
        if isinstance(record, CatalogRecord)
    ]
    assert len(current_rows) == 1
    assert current_rows[0].record_key == current_record_key
    assert current_rows[0].canonical_claim_id == new_claim_id

    history = client.search_context(
        "",
        user_id="u1",
        context_type="memory",
        project_id="project-a",
        query_intent="HISTORY",
        limit=20,
    )
    history_claim_ids = {str(item["metadata"].get("canonical_claim_id") or "") for item in history}
    assert {first_claim_id, new_claim_id}.issubset(history_claim_ids)
    assert all(item["record_key"] != current_record_key for item in history)

    restarted = MemoryOSClient(str(tmp_path))
    restarted_current = _current(restarted)
    assert len(restarted_current) == 1
    assert restarted_current[0]["record_key"] == current_record_key
    assert restarted_current[0]["metadata"]["canonical_claim_id"] == new_claim_id


def test_natural_transaction_and_valid_time_use_real_claim_revision_projection(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    committed = client.remember(
        user_id="u1",
        memory_type="preference",
        title="Project database selection",
        content="PostgreSQL",
        identity_fields={"subject": "project", "dimension": "database"},
    )
    assert committed["status"] == "COMMITTED"

    rows = client.index_store.list_catalog(  # type: ignore[attr-defined]
        filters={
            "tenant_id": "default",
            "record_kinds": (CatalogRecordKind.CLAIM_REVISION.value,),
            "target_uris": (str(committed["uri"]),),
            "include_inactive": True,
        },
        limit=10,
    )
    assert len(rows) == 1 and isinstance(rows[0], CatalogRecord)
    revision = rows[0]
    assert revision.transaction_time
    assert revision.valid_from
    singapore = ZoneInfo("Asia/Singapore")
    transaction_day = datetime.fromisoformat(revision.transaction_time).astimezone(singapore)
    transaction_query = (
        f"{transaction_day.year}年{transaction_day.month}月{transaction_day.day}日系统新增了哪些记忆"
    )

    transaction_hits = client.search_context(
        transaction_query,
        user_id="u1",
        options=RetrievalOptions(
            context_types=(ContextType.MEMORY,),
            owner_user_id="u1",
            timezone="Asia/Singapore",
            candidate_limit=20,
            final_limit=20,
        ),
    )

    assert [item["metadata"]["canonical_claim_id"] for item in transaction_hits] == [
        revision.canonical_claim_id
    ]
    assert transaction_query not in transaction_hits[0]["text"]
    assert transaction_hits[0]["canonical_validation_status"] == "validated_history"
    transaction_trace = client.recall_trace(str(client.last_recall_trace_id))
    assert transaction_trace["query_plan"]["query_intent"] == RetrievalQueryIntent.HISTORY.value
    assert transaction_trace["structured_candidates"] >= 1
    assert transaction_trace["canonical_validated"] == 1

    valid_day = datetime.fromisoformat(revision.valid_from).astimezone(singapore) + timedelta(days=1)
    valid_query = f"{valid_day.year}年{valid_day.month}月{valid_day.day}日时项目使用什么数据库"
    valid_hits = client.search_context(
        valid_query,
        user_id="u1",
        options=RetrievalOptions(
            context_types=(ContextType.MEMORY,),
            owner_user_id="u1",
            timezone="Asia/Singapore",
            candidate_limit=20,
            final_limit=20,
        ),
    )

    assert [item["metadata"]["canonical_claim_id"] for item in valid_hits] == [revision.canonical_claim_id]
    assert valid_query not in valid_hits[0]["text"]
    assert valid_hits[0]["canonical_validation_status"] == "validated_history"
    valid_trace = client.recall_trace(str(client.last_recall_trace_id))
    assert valid_trace["query_plan"]["query_intent"] == RetrievalQueryIntent.AS_OF.value
    assert valid_trace["structured_candidates"] >= 1
    assert valid_trace["canonical_validated"] == 1


@pytest.mark.parametrize(
    "field_name",
    ("receipt_digest", "canonical_head_digest", "projection_effect_hash"),
)
def test_history_catalog_proof_tamper_fails_closed(tmp_path, field_name: str) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    committed = client.remember(
        user_id="u1",
        memory_type="project_rule",
        title="immutable history proof",
        content="Always run the complete verification gate",
        project_id="project-a",
        constraint_polarity="REQUIRE",
        identity_fields={"rule_topic": "verification_gate"},
    )
    assert committed["status"] == "COMMITTED"
    rows = client.index_store.list_catalog(  # type: ignore[attr-defined]
        filters={
            "tenant_id": "default",
            "record_kinds": ("claim_revision",),
            "target_uris": (str(committed["uri"]),),
            "include_inactive": True,
        },
        limit=10,
    )
    assert len(rows) == 1 and isinstance(rows[0], CatalogRecord)
    row = rows[0]
    before = client.search_context(
        "",
        user_id="u1",
        context_type="memory",
        project_id="project-a",
        query_intent="HISTORY",
        limit=10,
    )
    assert [item["record_key"] for item in before] == [row.record_key]

    tampered = {
        "receipt_digest": replace(row, receipt_digest="0" * 64),
        "canonical_head_digest": replace(row, canonical_head_digest="0" * 64),
        "projection_effect_hash": replace(row, projection_effect_hash="0" * 64),
    }[field_name]
    client.index_store.upsert_catalog(tampered)  # type: ignore[attr-defined]

    after = client.search_context(
        "",
        user_id="u1",
        context_type="memory",
        project_id="project-a",
        query_intent="HISTORY",
        limit=10,
    )
    assert after == []


def test_late_historical_revision_uses_public_unified_history_as_of_and_two_time_axes(
    tmp_path,
) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path), tenant_id="t1")
    initial_archive = SessionArchive(
        user_id="u1",
        session_id="late-history-initial",
        archive_uri="memoryos://user/u1/sessions/history/late-history-initial",
        messages=[
            {
                "id": "initial-message",
                "role": "user",
                "content": "The primary storage backend is SQLite.",
            }
        ],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
    )
    client.session_archive_store.write_sync_archive(initial_archive)
    initial_episode = SessionArchiveEpisodeAdapter().adapt(initial_archive)
    assert initial_episode.origin.primary_scope is not None
    scope = MemoryScope(
        ScopeSelector((initial_episode.origin.primary_scope,)),
        VisibilityPolicy("t1"),
        initial_episode.origin.scope_refs,
    )
    initial_proposal = _proposal(
        initial_episode,
        "late-history-initial",
        "SQLite",
        "confirmation",
        "confirmed",
    )
    identity, _transition, initial_plan = _plan(
        client.source_store,
        initial_episode,
        scope,
        initial_proposal,
    )
    client.committer.commit(
        "u1",
        initial_plan.to_context_operations(
            user_id="u1",
            tenant_id="t1",
            episode_id=initial_episode.episode_id,
        ),
    )
    client._process_memory_projections_or_raise()

    _slot, initial_claims = CanonicalMemoryRepository(
        client.source_store,
        client.relation_store,
    ).load(identity)
    active = next(claim for claim in initial_claims if claim.current.state == ClaimState.ACTIVE.value)
    late_archive = SessionArchive(
        user_id="u1",
        session_id="late-history-arrival",
        archive_uri="memoryos://user/u1/sessions/history/late-history-arrival",
        messages=[
            {
                "id": "late-message",
                "role": "user",
                "content": (
                    "The primary storage backend SQLite also had the historical rationale stable under load."
                ),
            }
        ],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
    )
    client.session_archive_store.write_sync_archive(late_archive)
    late_episode = SessionArchiveEpisodeAdapter().adapt(late_archive)
    late_proposal = _supplement_proposal(
        late_episode,
        "late-history-arrival",
        active,
        speech_act=SpeechAct.CONFIRMATION,
        commitment=Commitment.CONFIRMED,
    )
    late_proposal = replace(
        late_proposal,
        value_fields={"canonical_value": "SQLite"},
        field_evidence_refs={
            key: value
            for key, value in late_proposal.field_evidence_refs.items()
            if key != "value.rationale"
        },
        metadata={
            **dict(late_proposal.metadata),
            "effective_at": "2026-06-01T12:00:00+00:00",
            "relation_target_binding_validated": True,
        },
    )
    late_identity, _late_transition, late_plan = _plan(
        client.source_store,
        late_episode,
        scope,
        late_proposal,
    )
    assert late_identity.claim_id == identity.claim_id
    client.committer.commit(
        "u1",
        late_plan.to_context_operations(
            user_id="u1",
            tenant_id="t1",
            episode_id=late_episode.episode_id,
        ),
    )
    client._process_memory_projections_or_raise()

    rows = client.index_store.list_catalog(  # type: ignore[attr-defined]
        filters={
            "tenant_id": "t1",
            "record_kinds": (CatalogRecordKind.CLAIM_REVISION.value,),
            "target_uris": (identity.claim_uri,),
            "include_inactive": True,
        },
        limit=10,
    )
    by_revision = {row.canonical_revision: row for row in rows if isinstance(row, CatalogRecord)}
    assert set(by_revision) == {1, 2}
    current_row = by_revision[1]
    historical_row = by_revision[2]
    assert historical_row.canonical_state == ClaimState.PROPOSED.value
    assert historical_row.event_time == "2026-06-01T12:00:00+00:00"
    assert historical_row.transaction_time != historical_row.event_time
    assert "source revision: 2" in historical_row.l1_text
    assert historical_row.metadata["current_claim_revision"] == 1

    event_hits = client.search_context(
        "SQLite",
        user_id="u1",
        options=RetrievalOptions(
            context_types=(ContextType.MEMORY,),
            owner_user_id="u1",
            workspace_ids=("memoryos",),
            event_time_from="2026-06-01T00:00:00+00:00",
            event_time_to="2026-06-02T00:00:00+00:00",
            query_intent=RetrievalQueryIntent.HISTORY,
            candidate_limit=20,
            final_limit=20,
        ),
    )
    assert [
        (item["metadata"]["canonical_revision"], item["metadata"]["canonical_state"])
        for item in event_hits
    ] == [(2, ClaimState.PROPOSED.value)]

    transaction_point = datetime.fromisoformat(historical_row.transaction_time)
    transaction_hits = client.search_context(
        "SQLite",
        user_id="u1",
        options=RetrievalOptions(
            context_types=(ContextType.MEMORY,),
            owner_user_id="u1",
            workspace_ids=("memoryos",),
            transaction_time_from=transaction_point.isoformat(),
            transaction_time_to=(transaction_point + timedelta(microseconds=1)).isoformat(),
            query_intent=RetrievalQueryIntent.HISTORY,
            candidate_limit=20,
            final_limit=20,
        ),
    )
    assert [item["metadata"]["canonical_revision"] for item in transaction_hits] == [2]

    current_point = datetime.fromisoformat(current_row.valid_from) + timedelta(microseconds=1)
    as_of_hits = client.search_context(
        "SQLite",
        user_id="u1",
        options=RetrievalOptions(
            context_types=(ContextType.MEMORY,),
            owner_user_id="u1",
            workspace_ids=("memoryos",),
            valid_at=current_point.isoformat(),
            query_intent=RetrievalQueryIntent.AS_OF,
            candidate_limit=20,
            final_limit=20,
        ),
    )
    assert [
        (item["metadata"]["canonical_revision"], item["metadata"]["canonical_state"])
        for item in as_of_hits
    ] == [(1, ClaimState.ACTIVE.value)]

    source_before = canonical_digest(
        {
            "object": client.source_store.read_object(identity.claim_uri).to_dict(),
            "content": client.source_store.read_content(identity.claim_uri),
        }
    )
    client.index_store.clear()
    assert client.index_store.get_catalog(  # type: ignore[attr-defined]
        "claim:" + identity.claim_id + ":revision:1",
        tenant_id="t1",
    ) is None
    rebuilt = client.context_db.rebuild_index()
    assert rebuilt["canonical_projection"]["historical_restored"] == 1
    restored_rows = client.index_store.list_catalog(  # type: ignore[attr-defined]
        filters={
            "tenant_id": "t1",
            "record_kinds": (CatalogRecordKind.CLAIM_REVISION.value,),
            "target_uris": (identity.claim_uri,),
            "include_inactive": True,
        },
        limit=10,
    )
    assert {row.canonical_revision for row in restored_rows} == {1, 2}
    assert source_before == canonical_digest(
        {
            "object": client.source_store.read_object(identity.claim_uri).to_dict(),
            "content": client.source_store.read_content(identity.claim_uri),
        }
    )
    # Rebuild applies Retention after projection.  The late June revision is
    # old enough to become COLD on 2026-07-15, so it deliberately leaves FTS
    # and remains recallable through HISTORY's structured transaction axis.
    restored_history = client.search_context(
        "SQLite",
        user_id="u1",
        options=RetrievalOptions(
            context_types=(ContextType.MEMORY,),
            owner_user_id="u1",
            workspace_ids=("memoryos",),
            transaction_time_from=transaction_point.isoformat(),
            transaction_time_to=(transaction_point + timedelta(microseconds=1)).isoformat(),
            query_intent=RetrievalQueryIntent.HISTORY,
            candidate_limit=20,
            final_limit=20,
        ),
    )
    restored_revisions = {item["metadata"]["canonical_revision"] for item in restored_history}
    assert restored_revisions == {2}, client.recall_trace(client.last_recall_trace_id)
    restored_as_of = client.search_context(
        "SQLite",
        user_id="u1",
        options=RetrievalOptions(
            context_types=(ContextType.MEMORY,),
            owner_user_id="u1",
            workspace_ids=("memoryos",),
            valid_at=current_point.isoformat(),
            query_intent=RetrievalQueryIntent.AS_OF,
            candidate_limit=20,
            final_limit=20,
        ),
    )
    assert [item["metadata"]["canonical_revision"] for item in restored_as_of] == [1]


def test_late_non_current_revision_keeps_requested_value_across_sdk_http_and_mcp(
    tmp_path,
) -> None:  # noqa: ANN001
    from tests.support.canonical_retrieval import _scope, _write_committed_canonical_fixture

    client = MemoryOSClient(str(tmp_path), tenant_id="t1")
    assert isinstance(client.source_store, FileSystemSourceStore)
    scope = _scope(("workspace", "memoryos"), tenant_id="t1", principals=("u1",))
    subject = ScopeRef.from_dict(scope["canonical_subject"])
    slot_id = "late-distinct-value"
    claim_id = "late-distinct-value"
    slot_uri = f"memoryos://user/u1/memories/canonical/slots/{slot_id}"
    claim_uri = f"{slot_uri}/claims/{claim_id}"
    current_value = "current aggregate A"
    historical_value = "late non-current revision B"
    current_revision = MemoryRevision(
        revision=1,
        state=ClaimState.ACTIVE.value,
        value_fields={"canonical_value": current_value},
        evidence_refs=(EvidenceRef("current", None, "current-hash"),),
        proposal_id="current",
        relation="UNRELATED",
        epistemic_status="EXPLICIT",
        valid_from="2026-07-01T00:00:00+00:00",
        created_at="2026-07-01T00:00:00+00:00",
        transaction_time="2026-07-01T00:00:00+00:00",
    )
    late_revision = MemoryRevision(
        revision=2,
        state=ClaimState.ACTIVE.value,
        value_fields={"canonical_value": historical_value},
        evidence_refs=(EvidenceRef("late", None, "late-hash"),),
        proposal_id="late",
        relation="SUPPLEMENTS",
        epistemic_status="EXPLICIT",
        qualifiers={"non_current_historical": True},
        previous_revision=1,
        valid_from="2026-06-01T00:00:00+00:00",
        created_at="2026-07-15T00:00:00+00:00",
        transaction_time="2026-07-15T00:00:00+00:00",
    )
    initial_claim = MemoryClaim(
        claim_id,
        claim_uri,
        slot_id,
        current_value,
        TransitionProfile.AUTHORITATIVE_STATE,
        (current_revision,),
        canonical_subject_key=subject.key,
    )
    final_claim = MemoryClaim(
        claim_id,
        claim_uri,
        slot_id,
        current_value,
        TransitionProfile.AUTHORITATIVE_STATE,
        (current_revision, late_revision),
        canonical_subject_key=subject.key,
    )
    initial_slot = MemorySlot(
        slot_id=slot_id,
        uri=slot_uri,
        memory_type="project_rule",
        identity_fields={"rule_topic": "late distinct value"},
        scope_keys=("memoryos:workspace:memoryos",),
        claim_ids=(claim_id,),
        active_claim_id=claim_id,
        revision=1,
        canonical_subject_key=subject.key,
        canonical_subject=subject,
    )
    final_slot = replace(initial_slot, revision=2)
    initial_slot_object = initial_slot.to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        scope=scope,
    )
    initial_claim_object = initial_claim.to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        memory_type="project_rule",
        scope=scope,
    )
    final_slot_object = final_slot.to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        scope=scope,
    )
    final_claim_object = final_claim.to_context_object(
        tenant_id="t1",
        owner_user_id="u1",
        memory_type="project_rule",
        scope=scope,
    )
    for obj in (initial_slot_object, initial_claim_object, final_slot_object, final_claim_object):
        obj.metadata = {**dict(obj.metadata), "source_adapter_id": "codex"}
    _write_committed_canonical_fixture(
        client.source_store,
        [
            (initial_slot_object, ""),
            (initial_claim_object, ""),
        ],
        key="late-distinct-r1",
        queue_store=client.queue_store,
        finalize_outbox=True,
    )
    client._process_memory_projections_or_raise()
    _write_committed_canonical_fixture(
        client.source_store,
        [
            (final_slot_object, ""),
            (final_claim_object, ""),
        ],
        key="late-distinct-r2",
        action=OperationAction.UPDATE,
        queue_store=client.queue_store,
        finalize_outbox=True,
    )
    client._process_memory_projections_or_raise()

    request = {
        "query": historical_value,
        "user_id": "u1",
        "context_type": "memory",
        "project_id": "memoryos",
        "claim_uris": [claim_uri],
        "query_intent": "HISTORY",
        "limit": 10,
    }
    sdk_results = client.search_context(
        str(request["query"]),
        user_id="u1",
        context_type="memory",
        project_id="memoryos",
        claim_uris=[claim_uri],
        query_intent="HISTORY",
        limit=10,
    )
    assert sdk_results, client.recall_trace(str(client.last_recall_trace_id))
    http_results = handle("POST /context/search", client, request)["results"]
    mcp = MemoryOSMCPServer(
        client,
        config=MCPServerConfig(
            root=str(tmp_path),
            user_id="u1",
            tenant_id="t1",
            adapter_id="codex",
            actor_id="codex",
            allowed_workspace_ids=frozenset({"memoryos"}),
            authorized_scope_keys=frozenset({"memoryos:workspace:memoryos"}),
        ),
    )
    mcp_response = mcp.call_tool(
        "memoryos_search",
        {
            "query": request["query"],
            "context_type": "memory",
            "project_id": "memoryos",
            "claim_uris": [claim_uri],
            "query_intent": "HISTORY",
            "limit": 10,
        },
    )
    assert mcp_response["error"] is None, mcp_response

    def late_payload(results: list[dict]) -> tuple[str, object]:
        matches = [item for item in results if item["metadata"]["canonical_revision"] == 2]
        assert len(matches) == 1
        hit = matches[0]
        assert current_value not in repr((hit["content"], hit["metadata"]))
        return hit["content"], hit["metadata"]["canonical_value"]

    expected = (historical_value, historical_value)
    assert late_payload(sdk_results) == expected
    assert late_payload(http_results) == expected
    assert late_payload(mcp_response["results"]) == expected
