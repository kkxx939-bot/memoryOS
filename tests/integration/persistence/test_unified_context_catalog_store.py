from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest

from memoryos.contextdb.catalog import CatalogRecord, CatalogRecordKind
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.candidate_generator import CandidateGenerator
from memoryos.contextdb.retrieval.query_plan import RetrievalOptions, RetrievalQueryIntent
from memoryos.contextdb.retrieval.query_planner import QueryPlanner
from memoryos.contextdb.store.local_stores import InMemoryRelationStore
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore


def _session_record(
    record_key: str,
    *,
    uri: str = "memoryos://user/u1/sessions/history/s1",
    metadata: dict | None = None,
    source_revision: int = 1,
) -> CatalogRecord:
    timestamp = "2026-07-14T03:30:00+00:00"
    return CatalogRecord(
        record_key=record_key,
        uri=uri,
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_id="workspace-a",
        session_id="s1",
        adapter_id="codex",
        context_type="session",
        source_kind="tool_result",
        record_kind=CatalogRecordKind.TOOL_RESULT.value,
        primary_tree_path="timeline/2026/07/14",
        tree_paths=("timeline/2026/07/14", "sessions/s1", "resources/desktop"),
        created_at=timestamp,
        updated_at=timestamp,
        event_time=timestamp,
        ingested_at=timestamp,
        transaction_time=timestamp,
        title="Desktop report result",
        l0_text="Read a desktop report",
        l1_text=("Authorization: Bearer private-token-123456; opened /Users/gulf/Desktop/quarterly-report.txt"),
        l2_uri="memoryos://user/u1/sessions/history/s1/tool-results/1",
        source_uri="memoryos://user/u1/sessions/history/s1/tool-results/1",
        source_digest="digest-1",
        source_revision=source_revision,
        metadata={
            "summary": "quarterly report",
            "api_key": "sk-production-secret",
            "file_path": "/Users/gulf/Desktop/quarterly-report.txt",
            **(metadata or {}),
        },
    )


def _current_slot_record(
    record_key: str,
    *,
    owner_user_id: str,
    slot_id: str,
    claim_id: str,
    canonical_slot_uri: str,
    canonical_claim_uri: str,
    updated_at: str,
) -> CatalogRecord:
    return CatalogRecord(
        record_key=record_key,
        uri=f"{canonical_slot_uri}/serving/current/{slot_id}",
        tenant_id="tenant-a",
        owner_user_id=owner_user_id,
        context_type="memory",
        source_kind="canonical_current_slot",
        record_kind=CatalogRecordKind.CURRENT_SLOT.value,
        primary_tree_path="memories/preferences/food/flavor",
        tree_paths=("memories/preferences/food/flavor",),
        created_at=updated_at,
        updated_at=updated_at,
        event_time=updated_at,
        ingested_at=updated_at,
        transaction_time=updated_at,
        valid_from=updated_at,
        title=f"Current preference {slot_id}",
        l0_text="Current preference",
        l1_text=f"Current preference {claim_id}",
        source_uri=canonical_claim_uri,
        source_digest=f"digest-{claim_id}",
        source_revision=1,
        canonical_slot_id=slot_id,
        canonical_slot_uri=canonical_slot_uri,
        canonical_claim_id=claim_id,
        canonical_claim_uri=canonical_claim_uri,
        canonical_revision=1,
        canonical_state="ACTIVE",
        metadata={
            "scope": {
                "visibility": {
                    "tenant_id": "tenant-a",
                    "private": True,
                    "allowed_principal_ids": [owner_user_id],
                    "allowed_service_ids": [],
                }
            }
        },
    )


def test_exact_canonical_uri_identity_uses_indexes_and_applies_acl_before_limit(tmp_path) -> None:  # noqa: ANN001
    store = SQLiteIndexStore(tmp_path / "canonical-uri-exact.sqlite3")
    slot_uri = "memoryos://user/u1/memories/canonical/slots/stable-slot"
    claim_uri = f"{slot_uri}/claims/active-claim"
    authorized = _current_slot_record(
        "slot:authorized:current",
        owner_user_id="u1",
        slot_id="authorized",
        claim_id="active-claim",
        canonical_slot_uri=slot_uri,
        canonical_claim_uri=claim_uri,
        updated_at="2026-07-01T00:00:00+00:00",
    )
    unauthorized = tuple(
        _current_slot_record(
            f"slot:foreign-{index}:current",
            owner_user_id="u2",
            slot_id=f"foreign-{index}",
            claim_id=f"foreign-claim-{index}",
            canonical_slot_uri=slot_uri,
            canonical_claim_uri=f"{slot_uri}/claims/foreign-claim-{index}",
            updated_at=f"2026-07-15T00:{index:02d}:00+00:00",
        )
        for index in range(20)
    )
    store.upsert_catalog_batch((*unauthorized, authorized))
    filters = {
        "tenant_id": "tenant-a",
        "principal_owner_id": "u1",
        "record_kinds": (CatalogRecordKind.CURRENT_SLOT.value,),
        "target_identity_uris": (slot_uri,),
        "_identity_candidate_limit": 1,
    }

    selected = store.list_catalog(filters=filters, limit=1)
    plan = " ".join(store.explain_structured_query(filters, limit=1))

    assert [record.record_key for record in selected] == [authorized.record_key]
    assert "idx_contexts_tenant_canonical_slot_uri" in plan
    assert "context_acl_grants" in plan
    legacy_selected = store.list_legacy_catalog(filters=filters, limit=1)
    assert [record.record_key for record in legacy_selected] == [authorized.record_key]

    by_claim = store.list_catalog(
        filters={**filters, "target_identity_uris": (claim_uri,)},
        limit=1,
    )
    assert [record.record_key for record in by_claim] == [authorized.record_key]


def test_exact_identity_applies_as_of_lifecycle_path_and_time_before_branch_limit(tmp_path) -> None:  # noqa: ANN001
    store = SQLiteIndexStore(tmp_path / "canonical-uri-eligibility.sqlite3")
    slot_uri = "memoryos://user/u1/memories/canonical/slots/eligibility-slot"
    claim_uri = f"{slot_uri}/claims/eligibility-claim"
    base = _current_slot_record(
        "unused-current",
        owner_user_id="u1",
        slot_id="eligibility-slot",
        claim_id="eligibility-claim",
        canonical_slot_uri=slot_uri,
        canonical_claim_uri=claim_uri,
        updated_at="2026-07-01T00:00:00+00:00",
    )
    wanted_path = "memories/preferences/food/eligible"

    def revision(
        name: str,
        *,
        updated_at: str,
        lifecycle_state: str = "active",
        admission_status: str = "",
        serving_tier: str = "HOT",
        path: str = wanted_path,
        event_time: str = "2026-07-10T00:00:00+00:00",
        transaction_time: str = "2026-07-10T00:00:00+00:00",
        valid_from: str = "2026-07-01T00:00:00+00:00",
        valid_to: str = "2026-07-12T00:00:00+00:00",
        adapter_id: str = "codex",
        project_id: str = "project-a",
        run_mode: str = "context_reduction",
        scope_id: str = "eligible",
    ) -> CatalogRecord:
        return replace(
            base,
            record_key=f"claim:eligibility-claim:revision:{name}",
            uri=claim_uri,
            source_kind="canonical_claim_revision",
            record_kind=CatalogRecordKind.CLAIM_REVISION.value,
            primary_tree_path=path,
            tree_paths=(path,),
            updated_at=updated_at,
            event_time=event_time,
            transaction_time=transaction_time,
            valid_from=valid_from,
            valid_to=valid_to,
            adapter_id=adapter_id,
            lifecycle_state=lifecycle_state,
            serving_tier=serving_tier,
            canonical_revision=int(name) if name.isdigit() else 1,
            source_revision=int(name) if name.isdigit() else 1,
            source_digest=f"digest-{name}",
            metadata={
                **dict(base.metadata),
                "project_id": project_id,
                "connect": {"run_mode": run_mode},
                "scope": {
                    **dict(base.metadata["scope"]),
                    "applicability": {
                        "all_of": ({"kind": "workspace", "id": scope_id},),
                    },
                },
                **(
                    {"admission": {"decision": admission_status}}
                    if admission_status
                    else {}
                ),
            },
        )

    eligible = revision("1", updated_at="2026-07-10T00:00:00+00:00")
    newer_ineligible = (
        revision(
            "2",
            updated_at="2026-07-15T00:00:00+00:00",
            lifecycle_state="archived",
        ),
        revision(
            "3",
            updated_at="2026-07-16T00:00:00+00:00",
            valid_from="2026-07-14T00:00:00+00:00",
            valid_to="",
        ),
        revision(
            "4",
            updated_at="2026-07-17T00:00:00+00:00",
            path="memories/preferences/food/wrong-branch",
        ),
        revision(
            "5",
            updated_at="2026-07-18T00:00:00+00:00",
            event_time="2026-08-02T00:00:00+00:00",
        ),
        revision(
            "6",
            updated_at="2026-07-19T00:00:00+00:00",
            transaction_time="2026-08-02T00:00:00+00:00",
        ),
        revision(
            "7",
            updated_at="2026-07-20T00:00:00+00:00",
            admission_status="pending",
        ),
        revision(
            "8",
            updated_at="2026-07-21T00:00:00+00:00",
            serving_tier="ARCHIVED",
        ),
        revision(
            "9",
            updated_at="2026-07-22T00:00:00+00:00",
            adapter_id="other-adapter",
        ),
        revision(
            "10",
            updated_at="2026-07-23T00:00:00+00:00",
            project_id="project-b",
        ),
        revision(
            "11",
            updated_at="2026-07-24T00:00:00+00:00",
            run_mode="action_capable",
        ),
        revision(
            "12",
            updated_at="2026-07-25T00:00:00+00:00",
            scope_id="wrong-scope",
        ),
    )
    store.upsert_catalog_batch((*newer_ineligible, eligible))
    filters = {
        "tenant_id": "tenant-a",
        "principal_owner_id": "u1",
        "record_kinds": (CatalogRecordKind.CLAIM_REVISION.value,),
        "target_identity_uris": (slot_uri,),
        "target_paths": (wanted_path,),
        "event_time_from": "2026-07-01T00:00:00+00:00",
        "event_time_to": "2026-08-01T00:00:00+00:00",
        "transaction_time_from": "2026-07-01T00:00:00+00:00",
        "transaction_time_to": "2026-08-01T00:00:00+00:00",
        "valid_at": "2026-07-10T12:00:00+00:00",
        "adapter_id": "codex",
        "project_id": "project-a",
        "connect_filters": {"run_mode": "context_reduction"},
        "applicability_scope_keys": ("memoryos:workspace:eligible",),
        "_identity_candidate_limit": 1,
    }

    selected = store.list_catalog(filters=filters, limit=1)

    assert [record.record_key for record in selected] == [eligible.record_key]


def test_current_relation_expansion_uses_bounded_canonical_identities_and_catalog_acl(tmp_path) -> None:  # noqa: ANN001
    store = SQLiteIndexStore(tmp_path / "canonical-relation-identities.sqlite3")
    first_slot_uri = "memoryos://user/u1/memories/canonical/slots/first"
    first_claim_uri = f"{first_slot_uri}/claims/first-active"
    second_slot_uri = "memoryos://user/u1/memories/canonical/slots/second"
    second_claim_uri = f"{second_slot_uri}/claims/second-active"
    foreign_slot_uri = "memoryos://user/u2/memories/canonical/slots/foreign"
    foreign_claim_uri = f"{foreign_slot_uri}/claims/foreign-active"
    first = _current_slot_record(
        "slot:first:current",
        owner_user_id="u1",
        slot_id="first",
        claim_id="first-active",
        canonical_slot_uri=first_slot_uri,
        canonical_claim_uri=first_claim_uri,
        updated_at="2026-07-15T00:00:00+00:00",
    )
    second = _current_slot_record(
        "slot:second:current",
        owner_user_id="u1",
        slot_id="second",
        claim_id="second-active",
        canonical_slot_uri=second_slot_uri,
        canonical_claim_uri=second_claim_uri,
        updated_at="2026-07-15T00:01:00+00:00",
    )
    foreign = _current_slot_record(
        "slot:foreign:current",
        owner_user_id="u2",
        slot_id="foreign",
        claim_id="foreign-active",
        canonical_slot_uri=foreign_slot_uri,
        canonical_claim_uri=foreign_claim_uri,
        updated_at="2026-07-15T00:02:00+00:00",
    )
    store.upsert_catalog_batch((first, second, foreign))

    class RecordingRelations(InMemoryRelationStore):
        def __init__(self) -> None:
            super().__init__()
            self.lookups: list[tuple[str, int | None]] = []

        def relations_of(
            self,
            uri: str,
            *,
            tenant_id: str | None = None,
            owner_user_id: str | None = None,
            limit: int | None = None,
        ) -> list[ContextRelation]:
            self.lookups.append((uri, limit))
            return super().relations_of(
                uri,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                limit=limit,
            )

    relations = RecordingRelations()
    relations.add_relation(
        ContextRelation(
            source_uri=first_claim_uri,
            relation_type="related",
            target_uri=foreign_claim_uri,
            weight=1.0,
            metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
        )
    )
    relations.add_relation(
        ContextRelation(
            source_uri=first_claim_uri,
            relation_type="related",
            target_uri=second_claim_uri,
            weight=0.9,
            metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
        )
    )
    plan = QueryPlanner().plan(
        "identity-only expansion",
        options=RetrievalOptions(
            target_uris=(first_slot_uri,),
            context_types=(ContextType.MEMORY,),
            tenant_id="tenant-a",
            owner_user_id="u1",
            query_intent=RetrievalQueryIntent.CURRENT,
            relation_expansion=True,
            candidate_limit=20,
            final_limit=20,
        ),
    )

    generated = CandidateGenerator(store, relation_store=relations).generate(plan)

    assert [candidate.record_key for candidate in generated.branches["exact"]] == [first.record_key]
    assert [candidate.record_key for candidate in generated.branches["relation"]] == [second.record_key]
    assert generated.relation_candidates == 1
    assert [uri for uri, _limit in relations.lookups[:3]] == [
        first_claim_uri,
        first_slot_uri,
        first.uri,
    ]
    assert len(relations.lookups) <= CandidateGenerator.MAX_RELATION_IDENTITIES_PER_SEED
    assert all(
        limit is not None and 0 < limit <= CandidateGenerator.MAX_RELATIONS_PER_SEED
        for _uri, limit in relations.lookups
    )


def test_catalog_batch_supports_multi_record_uri_paths_time_and_sanitized_fts(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    first = _session_record("session:s1:tool:1")
    second = _session_record("session:s1:tool:2", metadata={"summary": "second quarterly report"})

    assert store.upsert_catalog_batch((first, second)) == 2
    assert [record.record_key for record in store.get_catalog_by_uri(first.uri, tenant_id="tenant-a")] == [
        first.record_key,
        second.record_key,
    ]
    selected = store.list_catalog(
        filters={
            "tenant_id": "tenant-a",
            "owner_user_id": "u1",
            "workspace_ids": ["workspace-a"],
            "context_types": ["session"],
            "source_kinds": ["tool_result"],
            "target_paths": ["resources/desktop"],
            "event_time_from": "2026-07-14T00:00:00+00:00",
            "event_time_to": "2026-07-15T00:00:00+00:00",
        },
        limit=10,
    )
    assert {record.record_key for record in selected} == {first.record_key, second.record_key}
    assert {
        hit.metadata["catalog_record_key"]
        for hit in store.search_catalog(
            "quarterly report",
            filters={"tenant_id": "tenant-a", "target_paths": ["resources/desktop"]},
            limit=10,
        )
    } == {first.record_key, second.record_key}

    with sqlite3.connect(store.path) as conn:
        catalog_text = " ".join(
            str(value)
            for value in conn.execute("SELECT title, l0_text, l1_text, metadata_json FROM contexts").fetchone()
        )
        fts_text = " ".join(
            str(value)
            for value in conn.execute(
                "SELECT title, content_text, metadata_text, search_terms FROM contexts_fts"
            ).fetchone()
        )
    combined = f"{catalog_text} {fts_text}"
    assert "private-token-123456" not in combined
    assert "sk-production-secret" not in combined
    assert "/Users/gulf" not in combined
    assert "quarterly-report.txt" in combined


def test_fts_rowid_map_and_tenant_record_auxiliary_updates_are_indexed(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "fts-rowid-map.sqlite3"
    store = SQLiteIndexStore(path)
    record = _session_record("session:s1:tool:indexed")
    store.upsert_catalog(record)
    store.upsert_catalog(replace(record, title="Updated indexed result"))

    with sqlite3.connect(path) as conn:
        fts_rows = conn.execute(
            "SELECT rowid, record_key FROM contexts_fts WHERE record_key = ?",
            (record.record_key,),
        ).fetchall()
        mapping = conn.execute(
            "SELECT fts_rowid FROM context_fts_map WHERE record_key = ?",
            (record.record_key,),
        ).fetchone()
        assert len(fts_rows) == 1
        assert mapping is not None and int(mapping[0]) == int(fts_rows[0][0])

        map_plan = " ".join(
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT fts_rowid FROM context_fts_map WHERE record_key = ?",
                (record.record_key,),
            )
        )
        path_plan = " ".join(
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN DELETE FROM context_path_acl WHERE tenant_id = ? AND record_key = ?",
                (record.tenant_id, record.record_key),
            )
        )
        acl_plan = " ".join(
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN DELETE FROM context_acl_grants WHERE tenant_id = ? AND record_key = ?",
                (record.tenant_id, record.record_key),
            )
        )
    assert "sqlite_autoindex_context_fts_map_1" in map_plan
    assert "sqlite_autoindex_context_path_acl_1" in path_plan
    assert "idx_context_acl_grants_record" in acl_plan

    # The rowid map is rebuildable startup state. Corrupting only that map
    # triggers an offline startup rebuild from the Catalog instead of an
    # unbounded online delete through the FTS UNINDEXED record_key column.
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM context_fts_map")
    restarted = SQLiteIndexStore(path)
    with sqlite3.connect(path) as conn:
        repaired = conn.execute(
            "SELECT fts_rowid FROM context_fts_map WHERE record_key = ?",
            (record.record_key,),
        ).fetchone()
        fts_rowid = conn.execute(
            "SELECT rowid FROM contexts_fts WHERE record_key = ?",
            (record.record_key,),
        ).fetchone()
    assert repaired is not None and fts_rowid is not None
    assert int(repaired[0]) == int(fts_rowid[0])

    assert restarted.delete_catalog(record.record_key, tenant_id=record.tenant_id)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM context_fts_map").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM contexts_fts").fetchone()[0] == 0


def test_multi_path_top_k_is_order_independent_and_does_not_starve_later_path(tmp_path) -> None:  # noqa: ANN001
    store = SQLiteIndexStore(tmp_path / "multi-path.sqlite3")
    old_timestamp = "2026-07-01T00:00:00+00:00"
    old_records = tuple(
        replace(
            _session_record(f"session:path-a:{ordinal:04d}"),
            uri=f"memoryos://user/u1/sessions/history/path-a/{ordinal}",
            primary_tree_path="projects/a",
            tree_paths=("projects/a",),
            created_at=old_timestamp,
            updated_at=old_timestamp,
            event_time=old_timestamp,
            ingested_at=old_timestamp,
            transaction_time=old_timestamp,
            title="needle old path a result",
            l0_text="needle old path a result",
            l1_text="needle " + ("low density filler " * 100),
        )
        for ordinal in range(300)
    )
    newest = replace(
        _session_record("session:path-b:newest"),
        uri="memoryos://user/u1/sessions/history/path-b/newest",
        primary_tree_path="projects/b",
        tree_paths=("projects/b",),
        updated_at="2026-07-15T00:00:00+00:00",
        event_time="2026-07-15T00:00:00+00:00",
        ingested_at="2026-07-15T00:00:00+00:00",
        transaction_time="2026-07-15T00:00:00+00:00",
        title="needle",
        l0_text="needle",
        l1_text="needle",
    )
    store.upsert_catalog_batch((*old_records, newest))

    forward = store.list_catalog(
        filters={"tenant_id": "tenant-a", "target_paths": ("projects/a", "projects/b")},
        limit=1,
    )
    reverse = store.list_catalog(
        filters={"tenant_id": "tenant-a", "target_paths": ("projects/b", "projects/a")},
        limit=1,
    )

    assert [record.record_key for record in forward] == [newest.record_key]
    assert [record.record_key for record in reverse] == [newest.record_key]
    lexical_forward = store.search_catalog(
        "needle",
        filters={"tenant_id": "tenant-a", "target_paths": ("projects/a", "projects/b")},
        limit=1,
    )
    lexical_reverse = store.search_catalog(
        "needle",
        filters={"tenant_id": "tenant-a", "target_paths": ("projects/b", "projects/a")},
        limit=1,
    )
    assert [hit.metadata["catalog_record_key"] for hit in lexical_forward] == [newest.record_key]
    assert [hit.metadata["catalog_record_key"] for hit in lexical_reverse] == [newest.record_key]


def test_natural_date_plans_filter_event_transaction_and_valid_time_in_sql(tmp_path) -> None:  # noqa: ANN001
    store = SQLiteIndexStore(tmp_path / "natural-date.sqlite3")
    timestamp = "2026-07-14T03:00:00+00:00"

    def record(
        key: str,
        text: str,
        *,
        event_time: str,
        transaction_time: str,
        valid_from: str = "",
        valid_to: str = "",
        path: str = "timeline/2026/07/14",
    ) -> CatalogRecord:
        return CatalogRecord(
            record_key=key,
            uri=f"memoryos://user/u1/catalog/{key}",
            tenant_id="tenant-a",
            owner_user_id="u1",
            context_type="session",
            source_kind="observation",
            record_kind=CatalogRecordKind.CONTEXT.value,
            primary_tree_path=path,
            tree_paths=(path,),
            created_at=timestamp,
            updated_at=timestamp,
            event_time=event_time,
            ingested_at=timestamp,
            transaction_time=transaction_time,
            valid_from=valid_from,
            valid_to=valid_to,
            title=text,
            l1_text=text,
            source_uri=f"memoryos://user/u1/evidence/{key}",
            source_digest=f"digest-{key}",
        )

    store.upsert_catalog_batch(
        (
            record(
                "event-inside",
                "Quarterly planning discussion",
                event_time="2026-07-14T03:00:00+00:00",
                transaction_time=timestamp,
            ),
            record(
                "event-outside",
                "Following-day planning discussion",
                event_time="2026-07-14T17:00:00+00:00",
                transaction_time=timestamp,
                path="timeline/2026/07/15",
            ),
            record(
                "transaction-inside",
                "PostgreSQL preference revision",
                event_time="2026-07-13T01:00:00+00:00",
                transaction_time="2026-07-14T03:00:00+00:00",
            ),
            record(
                "transaction-outside",
                "Later preference revision",
                event_time="2026-07-14T03:00:00+00:00",
                transaction_time="2026-07-14T17:00:00+00:00",
            ),
            record(
                "valid-inside",
                "The project database is PostgreSQL",
                event_time="2026-07-13T01:00:00+00:00",
                transaction_time=timestamp,
                valid_from="2026-07-01T00:00:00+00:00",
                valid_to="2026-08-01T00:00:00+00:00",
            ),
            record(
                "valid-outside",
                "The future project database is SQLite",
                event_time="2026-07-13T01:00:00+00:00",
                transaction_time=timestamp,
                valid_from="2026-07-14T00:00:00+00:00",
                valid_to="2026-08-01T00:00:00+00:00",
            ),
        )
    )
    planner = QueryPlanner(
        now_provider=lambda: datetime(2026, 7, 14, tzinfo=timezone.utc),
    )
    common = RetrievalOptions(
        tenant_id="tenant-a",
        owner_user_id="u1",
        timezone="Asia/Singapore",
    )

    def keys(query: str) -> set[str]:
        generated = CandidateGenerator(store).generate(planner.plan(query, options=common))
        return {candidate.record_key for branch in generated.branches.values() for candidate in branch}

    event_keys = keys("7月14日发生了什么")
    transaction_keys = keys("7月14日系统新增了哪些记忆")
    valid_keys = keys("7月14日时项目使用什么数据库")
    assert "event-inside" in event_keys
    assert "event-outside" not in event_keys
    assert "transaction-inside" in transaction_keys
    assert "transaction-outside" not in transaction_keys
    assert "valid-inside" in valid_keys
    assert "valid-outside" not in valid_keys


def test_ordinary_semantic_query_does_not_enable_structured_catalog_listing(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "semantic-not-list.sqlite3")
    store.upsert_catalog(
        replace(
            _session_record("session:s1:ordinary-semantic"),
            title="Unrelated quarterly report",
            l0_text="Unrelated quarterly report",
            l1_text="Unrelated quarterly report",
        )
    )
    plan = QueryPlanner().plan(
        "zzzz-no-lexical-match-984721",
        options=RetrievalOptions(
            tenant_id="tenant-a",
            owner_user_id="u1",
            workspace_ids=("workspace-a",),
            query_intent=RetrievalQueryIntent.OPEN_RECALL,
            candidate_limit=10,
            final_limit=10,
        ),
    )

    generated = CandidateGenerator(store).generate(plan)

    assert generated.branches["structured"] == ()
    assert generated.structured_candidates == 0
    assert all(not branch for branch in generated.branches.values())


def test_selective_fts_hit_suppresses_broader_temporal_structured_fallback(
    tmp_path,
    monkeypatch,
) -> None:  # noqa: ANN001
    store = SQLiteIndexStore(tmp_path / "selective-temporal-fts.sqlite3")
    marker = "selectiveneedle984721"
    store.upsert_catalog(
        replace(
            _session_record("session:s1:selective-temporal"),
            title=marker,
            l0_text=marker,
            l1_text=marker,
        )
    )

    def prohibit_temporal_listing(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("selective FTS hit must suppress the broader temporal list")

    monkeypatch.setattr(store, "list_catalog", prohibit_temporal_listing)
    plan = QueryPlanner().plan(
        marker,
        options=RetrievalOptions(
            tenant_id="tenant-a",
            owner_user_id="u1",
            workspace_ids=("workspace-a",),
            event_time_from="2026-07-14",
            event_time_to="2026-07-14",
            query_intent=RetrievalQueryIntent.OPEN_RECALL,
            candidate_limit=10,
            final_limit=10,
        ),
    )

    generated = CandidateGenerator(store).generate(plan)

    assert generated.structured_candidates == 0
    assert generated.exact_candidates == 0
    assert generated.fts_candidates == 1
    assert [item.record_key for item in generated.branches["lexical"]] == [
        "session:s1:selective-temporal"
    ]


def test_history_excludes_current_slots_before_the_sql_candidate_limit(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "history-record-kinds.sqlite3")
    current_slots = tuple(
        replace(
            _session_record(
                f"slot:current-{index}:current",
                uri=f"memoryos://user/u1/memories/canonical/slots/current-{index}/serving/current",
            ),
            context_type="memory",
            source_kind="canonical_current_slot",
            record_kind=CatalogRecordKind.CURRENT_SLOT.value,
            session_id="",
            canonical_slot_id=f"current-{index}",
            canonical_claim_id=f"claim-current-{index}",
            canonical_revision=1,
            canonical_state="ACTIVE",
        )
        for index in range(5)
    )
    ordinary_history = replace(
        _session_record("session:s1:message:history"),
        source_kind="message",
        record_kind=CatalogRecordKind.MESSAGE.value,
        title="ordinary historical event",
        l0_text="ordinary historical event",
        l1_text="ordinary historical event",
    )
    store.upsert_catalog_batch((*current_slots, ordinary_history))

    plan = QueryPlanner().plan(
        "",
        options=RetrievalOptions(
            tenant_id="tenant-a",
            owner_user_id="u1",
            workspace_ids=("workspace-a",),
            query_intent=RetrievalQueryIntent.HISTORY,
            candidate_limit=1,
            final_limit=1,
        ),
    )
    generated = CandidateGenerator(store).generate(plan)

    assert [candidate.record_key for candidate in generated.branches["structured"]] == [
        ordinary_history.record_key
    ]


def test_sqlite_persists_only_sanitized_primary_secondary_and_metadata_tree_paths(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    raw_primary = "memories/preferences/Users_gulf_Desktop_private.txt/ice_cream"
    raw_secondary = "projects/sk-abcdefghijk123456"
    record = CatalogRecord(
        record_key="slot:sensitive:current",
        uri="memoryos://user/u1/memories/canonical/slots/sensitive/serving/current",
        tenant_id="tenant-a",
        owner_user_id="u1",
        workspace_id="workspace-a",
        context_type="memory",
        source_kind="canonical_current_slot",
        record_kind=CatalogRecordKind.CURRENT_SLOT.value,
        primary_tree_path=raw_primary,
        tree_paths=(raw_primary, raw_secondary, "resources/repository"),
        title="safe current state",
        l0_text="safe state",
        l1_text="safe overview",
        source_uri="memoryos://user/u1/memories/canonical/slots/sensitive",
        source_digest="safe-digest",
        source_revision=1,
        canonical_slot_id="sensitive",
        canonical_claim_id="claim-sensitive",
        canonical_revision=1,
        canonical_state="ACTIVE",
        metadata={
            "primary_tree_path": raw_primary,
            "tree_paths": [raw_primary, raw_secondary],
            "nested": {"secondary_tree_paths": [raw_secondary]},
        },
    )

    store.upsert_catalog(record)
    stored = store.get_catalog(record.record_key, tenant_id="tenant-a")
    assert stored is not None
    assert stored.tree_paths == record.tree_paths
    assert store.list_catalog(
        filters={"tenant_id": "tenant-a", "target_paths": (raw_primary,)},
        limit=10,
    ) == [stored]
    assert store.list_catalog(
        filters={"tenant_id": "tenant-a", "target_paths": (raw_secondary,)},
        limit=10,
    ) == [stored]

    with sqlite3.connect(store.path) as conn:
        persisted = " ".join(
            str(value)
            for row in conn.execute(
                "SELECT primary_tree_path, metadata_json FROM contexts WHERE record_key = ?",
                (record.record_key,),
            )
            for value in row
        )
        persisted += " " + " ".join(
            str(row[0])
            for row in conn.execute(
                "SELECT path FROM context_paths WHERE record_key = ? ORDER BY path",
                (record.record_key,),
            )
        )
        persisted += " " + " ".join(
            str(value)
            for row in conn.execute(
                "SELECT metadata_text, search_terms FROM contexts_fts WHERE record_key = ?",
                (record.record_key,),
            )
            for value in row
        )
    assert "sk-abcdefghijk123456" not in persisted
    assert "Users_gulf" not in persisted
    assert "gulf_Desktop" not in persisted
    assert "resources/repository" in persisted


def test_current_slot_is_one_row_and_batch_constraint_failure_rolls_back(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "catalog.sqlite3")

    def current(record_key: str, slot_id: str, claim_id: str) -> CatalogRecord:
        return CatalogRecord(
            record_key=record_key,
            uri=f"memoryos://user/u1/memories/canonical/slots/{slot_id}",
            tenant_id="tenant-a",
            owner_user_id="u1",
            context_type="memory",
            source_kind="canonical_projection",
            record_kind=CatalogRecordKind.CURRENT_SLOT.value,
            canonical_slot_id=slot_id,
            canonical_claim_id=claim_id,
            canonical_state="ACTIVE",
            metadata={"memory_type": "preference", "canonical_value": claim_id},
        )

    store.upsert_catalog(current("slot:ice-cream:current", "ice-cream", "claim-a"))
    store.upsert_catalog(current("slot:ice-cream:current", "ice-cream", "claim-b"))
    rows = store.list_catalog(
        filters={"tenant_id": "tenant-a", "record_kind": CatalogRecordKind.CURRENT_SLOT.value},
        limit=10,
    )
    assert len(rows) == 1
    assert rows[0].canonical_claim_id == "claim-b"

    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_catalog_batch(
            (
                current("slot:conflict:a", "conflicting-slot", "claim-c"),
                current("slot:conflict:b", "conflicting-slot", "claim-d"),
            )
        )
    assert store.get_catalog("slot:conflict:a") is None
    assert store.get_catalog("slot:conflict:b") is None


def test_context_links_are_sanitized_indexed_and_deleted_with_either_endpoint(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "catalog-links.sqlite3")
    source = _session_record("session:source")
    target = replace(
        _session_record("session:target", source_revision=2),
        uri="memoryos://user/u1/sessions/history/s2",
        session_id="s2",
        source_uri="memoryos://user/u1/sessions/history/s2/tool-results/1",
        source_digest="digest-2",
    )
    store.upsert_catalog_batch((source, target))

    def insert_link() -> str:
        return store.upsert_context_link(
            tenant_id="tenant-a",
            source_record_key=source.record_key,
            source_uri=source.uri,
            relation_type="related_to",
            target_record_key=target.record_key,
            target_uri=target.uri,
            metadata={"Authorization": "Bearer private-link-token-123456"},
        )

    link_key = insert_link()
    with sqlite3.connect(store.path) as conn:
        row = conn.execute(
            "SELECT metadata_json FROM context_links WHERE link_key = ?",
            (link_key,),
        ).fetchone()
        indexes = {
            str(item[0])
            for item in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'context_links'"
            )
        }
    assert row is not None
    assert "private-link-token-123456" not in str(row[0])
    assert {"idx_context_links_source", "idx_context_links_target"} <= indexes

    assert store.delete_catalog(target.record_key, tenant_id="tenant-a") is True
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM context_links").fetchone()[0] == 0

    store.upsert_catalog(target)
    insert_link()
    assert store.delete_catalog(source.record_key, tenant_id="tenant-a") is True
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM context_links").fetchone()[0] == 0


def test_legacy_contexts_table_migrates_in_place_idempotently_and_preserves_search(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE contexts (
              uri TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              owner_user_id TEXT NOT NULL,
              context_type TEXT NOT NULL,
              project_id TEXT NOT NULL DEFAULT '',
              adapter_id TEXT NOT NULL DEFAULT '',
              admission_status TEXT NOT NULL DEFAULT '',
              claim_state TEXT NOT NULL DEFAULT '',
              slot_id TEXT NOT NULL DEFAULT '',
              memory_type TEXT NOT NULL DEFAULT '',
              scope_keys TEXT NOT NULL DEFAULT '[]',
              title TEXT NOT NULL,
              lifecycle_state TEXT NOT NULL,
              hotness REAL NOT NULL,
              semantic_hotness REAL NOT NULL,
              behavior_support_hotness REAL NOT NULL,
              metadata_json TEXT NOT NULL,
              content_text TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO contexts VALUES (
              ?, 'tenant-a', 'u1', 'resource', '', '', '', '', '', '', '[]',
              'Legacy report', 'active', 0, 0, 0, ?, ?, '2026-07-14T03:30:00+00:00'
            )
            """,
            (
                "memoryos://user/u1/resources/legacy-report",
                json.dumps({"summary": "legacy quarterly report", "password": "legacy-secret"}),
                "password=legacy-secret /Users/gulf/Desktop/legacy-report.txt",
            ),
        )
        conn.execute("CREATE TABLE contexts_fts(uri TEXT PRIMARY KEY, title, content_text, metadata_text)")
        conn.execute("PRAGMA user_version = 2")

    store = SQLiteIndexStore(path)
    migrated = store.get_catalog("memoryos://user/u1/resources/legacy-report", tenant_id="tenant-a")
    assert migrated is not None
    assert migrated.record_key == migrated.uri
    assert [hit.uri for hit in store.search("legacy report", filters={"tenant_id": "tenant-a"})] == [migrated.uri]
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 10
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(contexts)")}
        assert {"record_key", "event_time", "transaction_time", "canonical_slot_id", "serving_tier"} <= columns
        persisted = " ".join(
            str(value) for value in conn.execute("SELECT metadata_json, content_text FROM contexts").fetchone()
        )
        fts = " ".join(
            str(value) for value in conn.execute("SELECT content_text, metadata_text FROM contexts_fts").fetchone()
        )
    assert "legacy-secret" not in f"{persisted} {fts}"
    assert "/Users/gulf" not in f"{persisted} {fts}"

    restarted = SQLiteIndexStore(path)
    assert restarted.get_catalog(migrated.record_key, tenant_id="tenant-a") == migrated


def test_tombstone_queue_is_replayable_stale_safe_and_migration_state_resumes(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    record = _session_record("session:s1:tool:1", source_revision=2)
    store.upsert_catalog(record)

    stale = store.enqueue_tombstone(
        tenant_id="tenant-a",
        record_key=record.record_key,
        reason="stale-delete",
        source_revision=1,
    )
    assert stale["status"] == "PENDING"
    assert store.get_catalog(record.record_key) is not None
    stale_applied = store.mark_tombstone_applied(stale["tombstone_id"])
    assert stale_applied is not None and stale_applied["status"] == "STALE"
    assert store.get_catalog(record.record_key) is not None

    queued = store.enqueue_tombstone(
        tenant_id="tenant-a",
        record_key=record.record_key,
        reason="session-delete",
        source_revision=2,
    )
    failed = store.mark_tombstone_failed(queued["tombstone_id"], "password=do-not-log")
    assert failed is not None and failed["status"] == "FAILED" and failed["retry_count"] == 1
    assert "do-not-log" not in failed["last_error"]
    assert {item["tombstone_id"] for item in store.get_pending_tombstones()} == {queued["tombstone_id"]}
    applied = store.mark_tombstone_applied(queued["tombstone_id"])
    assert applied is not None and applied["status"] == "APPLIED"
    replayed = store.mark_tombstone_applied(queued["tombstone_id"])
    assert replayed is not None and replayed["status"] == "APPLIED"
    assert store.get_catalog(record.record_key) is None
    with pytest.raises(ValueError, match="not newer"):
        store.upsert_catalog(record)
    store.upsert_catalog(_session_record(record.record_key, source_revision=3))
    assert store.get_catalog(record.record_key) is not None

    cleaning = store.enqueue_tombstone(
        tenant_id="tenant-a",
        record_key=record.record_key,
        reason="two-phase-delete",
        source_revision=3,
    )
    begun = store.begin_tombstone_cleanup(str(cleaning["tombstone_id"]))
    assert begun is not None and begun["status"] == "CLEANING"
    assert store.get_catalog(record.record_key) is None
    assert str(cleaning["tombstone_id"]) in {str(item["tombstone_id"]) for item in store.get_pending_tombstones()}
    with pytest.raises(ValueError, match="in-progress tombstone"):
        store.upsert_catalog(_session_record(record.record_key, source_revision=4))
    finished = store.finish_tombstone_cleanup(str(cleaning["tombstone_id"]))
    assert finished is not None and finished["status"] == "APPLIED"
    store.upsert_catalog(_session_record(record.record_key, source_revision=4))
    assert store.get_catalog(record.record_key) is not None

    state = store.set_migration_state(
        "unified-context-v1",
        "BACKFILLING",
        "session:s1",
        {"processed": 10},
        tenant_id="tenant-a",
        batch_size=100,
    )
    assert state["checkpoint"] == "session:s1"
    shadow = store.set_migration_state(
        "unified-context-v1",
        "SHADOW_VALIDATING",
        "session:s1",
        {"processed": 10, "shadow_validation_epoch": "epoch-1"},
        tenant_id="tenant-a",
        batch_size=100,
    )
    assert shadow["state"] == "SHADOW_VALIDATING"
    proof = store.record_migration_equivalence_proof(
        "unified-context-v1",
        {
            "plane": "session_archive",
            "source_identity_digest": "source-digest",
            "evidence_digest": "archive-digest",
            "expected_count": 1,
            "actual_count": 0,
            "expected_digest": "expected-digest",
            "actual_digest": "actual-digest",
            "matched": False,
        },
        tenant_id="tenant-a",
    )
    assert proof["inserted"] is True
    summary = store.get_migration_equivalence_summary(
        "unified-context-v1",
        tenant_id="tenant-a",
        validation_epoch="epoch-1",
    )
    assert summary == {"sample_count": 1, "mismatch_count": 1}
    restarted = SQLiteIndexStore(store.path)
    restarted_state = restarted.get_migration_state("unified-context-v1", tenant_id="tenant-a")
    assert restarted_state is not None
    assert restarted_state["details_json"]["shadow_mismatch_count"] == 1


def test_required_structured_indexes_exist_and_are_selected(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    store.upsert_catalog(_session_record("session:s1:tool:1"))

    owner_plan = " ".join(
        store.explain_structured_query(
            {
                "tenant_id": "tenant-a",
                "owner_user_id": "u1",
                "context_type": "session",
            }
        )
    )
    event_plan = " ".join(
        store.explain_structured_query(
            {
                "tenant_id": "tenant-a",
                "event_time_from": "2026-07-14T00:00:00+00:00",
                "event_time_to": "2026-07-15T00:00:00+00:00",
            }
        )
    )
    updated_plan = " ".join(
        store.explain_structured_query(
            {
                "tenant_id": "tenant-a",
                "updated_at_from": "2026-07-14T00:00:00+00:00",
                "updated_at_to": "2026-07-15T00:00:00+00:00",
            }
        )
    )
    valid_plan = " ".join(
        store.explain_structured_query(
            {
                "tenant_id": "tenant-a",
                "valid_at": "2026-07-14T12:00:00+00:00",
            }
        )
    )
    tenant_path_plan = " ".join(
        store.explain_structured_query(
            {"tenant_id": "tenant-a", "target_paths": ("resources/desktop",)}
        )
    )
    owner_path_plan = " ".join(
        store.explain_structured_query(
            {
                "tenant_id": "tenant-a",
                "owner_user_id": "u1",
                "target_paths": ("resources/desktop",),
            }
        )
    )
    type_path_plan = " ".join(
        store.explain_structured_query(
            {
                "tenant_id": "tenant-a",
                "context_types": ("session",),
                "target_paths": ("resources/desktop",),
            }
        )
    )
    time_path_plan = " ".join(
        store.explain_structured_query(
            {
                "tenant_id": "tenant-a",
                "event_time_from": "2026-07-14T00:00:00+00:00",
                "event_time_to": "2026-07-15T00:00:00+00:00",
                "target_paths": ("resources/desktop",),
            }
        )
    )
    with sqlite3.connect(store.path) as conn:
        path_plan = " ".join(
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT record_key FROM context_paths "
                "WHERE tenant_id = ? AND path >= ? AND path < ? LIMIT 10",
                ("tenant-a", "resources/desktop", "resources/desktop/\uffff"),
            )
        )
        indexes = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name IN ('contexts', 'context_paths', 'context_path_closure', 'context_path_acl')"
            )
        }
        created_plan = " ".join(
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT record_key FROM contexts "
                "WHERE tenant_id = ? AND created_at >= ? AND created_at < ? LIMIT 10",
                ("tenant-a", "2026-07-14T00:00:00+00:00", "2026-07-15T00:00:00+00:00"),
            )
        )
        ingested_plan = " ".join(
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT record_key FROM contexts "
                "WHERE tenant_id = ? AND ingested_at >= ? AND ingested_at < ? LIMIT 10",
                ("tenant-a", "2026-07-14T00:00:00+00:00", "2026-07-15T00:00:00+00:00"),
            )
        )
    assert "idx_contexts_tenant_owner_type" in owner_plan
    assert "idx_contexts_tenant_event_time" in event_plan
    assert "idx_contexts_tenant_updated_at" in updated_plan
    assert "idx_contexts_tenant_valid_interval" in valid_plan
    assert "idx_context_path_closure_tenant_ancestor" in tenant_path_plan
    assert "idx_context_path_closure_owner_ancestor" in owner_path_plan
    assert "idx_context_path_closure_type_ancestor" in type_path_plan
    assert "idx_context_path_closure_ancestor_event" in time_path_plan
    assert all(
        "LIST SUBQUERY" in plan
        for plan in (tenant_path_plan, owner_path_plan, type_path_plan, time_path_plan)
    )
    assert all(
        "ancestor_path=?" in plan
        for plan in (tenant_path_plan, owner_path_plan, type_path_plan, time_path_plan)
    )
    assert "idx_contexts_tenant_created_at" in created_plan
    assert "idx_contexts_tenant_ingested_at" in ingested_plan
    assert "idx_context_paths_" in path_plan and "SCAN context_paths" not in path_plan
    assert {
        "idx_contexts_tenant_transaction_time",
        "idx_contexts_tenant_created_at",
        "idx_contexts_tenant_ingested_at",
        "idx_contexts_tenant_updated_at",
        "idx_contexts_tenant_valid_interval",
        "uq_contexts_current_slot",
        "idx_context_paths_owner_path",
        "idx_context_paths_type_path",
        "idx_context_paths_time_path",
        "idx_context_paths_path_time",
        "idx_context_paths_uri",
        "idx_context_path_closure_tenant_ancestor",
        "idx_context_path_closure_owner_ancestor",
        "idx_context_path_closure_type_ancestor",
        "idx_context_path_closure_ancestor_event",
    } <= indexes


def test_singapore_day_boundaries_use_catalog_utc_text_for_all_time_filters(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "catalog.sqlite3")

    def record(
        key: str,
        *,
        source_kind: str,
        event_time: str,
        transaction_time: str,
        updated_at: str | None = None,
        valid_from: str = "",
        valid_to: str = "",
    ) -> CatalogRecord:
        return CatalogRecord(
            record_key=key,
            uri=f"memoryos://user/u1/contexts/{key}",
            tenant_id="tenant-a",
            owner_user_id="u1",
            context_type="session",
            source_kind=source_kind,
            record_kind=CatalogRecordKind.CONTEXT.value,
            created_at=transaction_time,
            updated_at=updated_at or transaction_time,
            event_time=event_time,
            ingested_at=transaction_time,
            transaction_time=transaction_time,
            valid_from=valid_from,
            valid_to=valid_to,
            title=key,
            l0_text=key,
            l1_text=key,
            l2_uri=f"memoryos://user/u1/contexts/{key}",
            source_uri=f"memoryos://user/u1/contexts/{key}",
            source_digest=f"digest-{key}",
        )

    store.upsert_catalog_batch(
        (
            # Asia/Singapore 2026-07-14 is the half-open UTC interval
            # [2026-07-13T16:00:00, 2026-07-14T16:00:00).
            record(
                "event-before-transaction-start",
                source_kind="time_test",
                event_time="2026-07-13T15:59:59.999999Z",
                transaction_time="2026-07-14T00:00:00+08:00",
            ),
            record(
                "event-start-transaction-before",
                source_kind="time_test",
                event_time="2026-07-14T00:00:00+08:00",
                transaction_time="2026-07-13T15:59:59.999999Z",
            ),
            record(
                "event-last-transaction-end",
                source_kind="time_test",
                event_time="2026-07-14T23:59:59.999999+08:00",
                transaction_time="2026-07-15T00:00:00+08:00",
            ),
            record(
                "event-end-transaction-last",
                source_kind="time_test",
                event_time="2026-07-15T00:00:00+08:00",
                transaction_time="2026-07-14T23:59:59.999999+08:00",
            ),
            record(
                "updated-start-inclusive",
                source_kind="updated_test",
                event_time="2026-07-13T00:00:00Z",
                transaction_time="2026-07-13T00:00:00Z",
                updated_at="2026-07-14T00:00:00+08:00",
            ),
            record(
                "updated-end-exclusive",
                source_kind="updated_test",
                event_time="2026-07-13T00:00:00Z",
                transaction_time="2026-07-13T00:00:00Z",
                updated_at="2026-07-15T00:00:00+08:00",
            ),
            record(
                "valid-one-microsecond-after",
                source_kind="valid_test",
                event_time="2026-07-13T00:00:00Z",
                transaction_time="2026-07-13T00:00:00Z",
                valid_from="2026-07-13T00:00:00Z",
                valid_to="2026-07-14T00:00:00.000001+08:00",
            ),
            record(
                "valid-start-inclusive",
                source_kind="valid_test",
                event_time="2026-07-13T00:00:00Z",
                transaction_time="2026-07-13T00:00:00Z",
                valid_from="2026-07-14T00:00:00+08:00",
                valid_to="2026-07-15T00:00:00+08:00",
            ),
            record(
                "valid-end-exclusive",
                source_kind="valid_test",
                event_time="2026-07-13T00:00:00Z",
                transaction_time="2026-07-13T00:00:00Z",
                valid_from="2026-07-13T00:00:00Z",
                valid_to="2026-07-14T00:00:00+08:00",
            ),
            record(
                "valid-future",
                source_kind="valid_test",
                event_time="2026-07-13T00:00:00Z",
                transaction_time="2026-07-13T00:00:00Z",
                valid_from="2026-07-14T00:00:00.000001+08:00",
                valid_to="2026-07-15T00:00:00+08:00",
            ),
        )
    )

    planner = QueryPlanner()
    generator = CandidateGenerator(store)

    def selected(options: RetrievalOptions) -> set[str]:
        plan = planner.plan("", options=options)
        generated = generator.generate(plan)
        return {candidate.record_key for candidate in generated.branches["structured"]}

    common: dict[str, Any] = {
        "tenant_id": "tenant-a",
        "owner_user_id": "u1",
        "timezone": "Asia/Singapore",
        "query_intent": RetrievalQueryIntent.OPEN_RECALL,
        "candidate_limit": 100,
        "final_limit": 20,
    }
    event_options = RetrievalOptions(
        **common,
        source_kinds=("time_test",),
        event_time_from="2026-07-14",
        event_time_to="2026-07-14",
    )
    transaction_options = RetrievalOptions(
        **common,
        source_kinds=("time_test",),
        transaction_time_from="2026-07-14",
        transaction_time_to="2026-07-14",
    )
    valid_options = RetrievalOptions(
        **common,
        source_kinds=("valid_test",),
        valid_at="2026-07-14",
    )
    updated_options = RetrievalOptions(
        **common,
        source_kinds=("updated_test",),
        updated_at_from="2026-07-14",
        updated_at_to="2026-07-14",
    )

    assert event_options.event_time_from == "2026-07-13T16:00:00+00:00"
    assert event_options.event_time_to == "2026-07-14T16:00:00+00:00"
    assert selected(event_options) == {
        "event-start-transaction-before",
        "event-last-transaction-end",
    }
    assert selected(transaction_options) == {
        "event-before-transaction-start",
        "event-end-transaction-last",
    }
    assert selected(valid_options) == {
        "valid-one-microsecond-after",
        "valid-start-inclusive",
    }
    assert selected(updated_options) == {"updated-start-inclusive"}

    with sqlite3.connect(store.path) as conn:
        stored_event_start = conn.execute(
            "SELECT event_time FROM contexts WHERE record_key = ?",
            ("event-start-transaction-before",),
        ).fetchone()[0]
        stored_transaction_start = conn.execute(
            "SELECT transaction_time FROM contexts WHERE record_key = ?",
            ("event-before-transaction-start",),
        ).fetchone()[0]
        stored_valid_start = conn.execute(
            "SELECT valid_from FROM contexts WHERE record_key = ?",
            ("valid-start-inclusive",),
        ).fetchone()[0]
        stored_updated_start = conn.execute(
            "SELECT updated_at FROM contexts WHERE record_key = ?",
            ("updated-start-inclusive",),
        ).fetchone()[0]
    assert stored_event_start == event_options.event_time_from
    assert stored_transaction_start == transaction_options.transaction_time_from
    assert stored_valid_start == valid_options.valid_at
    assert stored_updated_start == updated_options.updated_at_from


def test_catalog_rejects_absolute_or_file_serving_uris() -> None:
    with pytest.raises(ValueError, match="logical memoryos URI"):
        _session_record("unsafe-uri", uri="file:///Users/u1/Desktop/private.txt")
    with pytest.raises(ValueError, match="logical memoryos URI"):
        CatalogRecord(
            record_key="unsafe-source",
            uri="memoryos://user/u1/catalog/unsafe-source",
            tenant_id="tenant-a",
            source_uri="/Users/u1/Desktop/private.txt",
        )


@pytest.mark.parametrize("context_type", ["resource", "skill"])
def test_public_retrieval_only_enumerates_unscoped_resources_and_skills(
    tmp_path,
    context_type: str,
) -> None:
    store = SQLiteIndexStore(tmp_path / f"{context_type}.sqlite3")
    base: dict[str, Any] = {
        "tenant_id": "tenant-a",
        "context_type": context_type,
        "source_kind": context_type,
        "record_kind": CatalogRecordKind.CONTEXT.value,
        "l0_text": "public lookup",
        "source_digest": "digest",
    }
    store.upsert_catalog_batch(
        (
            CatalogRecord(
                record_key=f"{context_type}:global",
                uri=f"memoryos://{context_type}s/global",
                title="global public lookup",
                **base,
            ),
            CatalogRecord(
                record_key=f"{context_type}:workspace",
                uri=f"memoryos://{context_type}s/workspace-private",
                title="workspace public lookup",
                metadata={
                    "scope": {
                        "applicability": {
                            "all_of": [
                                {
                                    "namespace": "memoryos",
                                    "kind": "workspace",
                                    "id": "private-workspace",
                                }
                            ]
                        }
                    }
                },
                **base,
            ),
        )
    )

    plan = QueryPlanner().plan(
        "",
        options=RetrievalOptions(
            tenant_id="tenant-a",
            context_types=(ContextType(context_type),),
            query_intent=RetrievalQueryIntent.OPEN_RECALL,
        ),
    )
    generated = CandidateGenerator(store).generate(plan)

    assert [item.record_key for item in generated.branches["structured"]] == [f"{context_type}:global"]


def test_conflicts_and_options_have_deterministic_default_states_with_explicit_override(
    tmp_path,
) -> None:
    store = SQLiteIndexStore(tmp_path / "states.sqlite3")

    def claim(state: str, index: int) -> CatalogRecord:
        return CatalogRecord(
            record_key=f"claim:{state.lower()}:{index}",
            uri=f"memoryos://user/u1/memories/canonical/claims/{state.lower()}-{index}",
            tenant_id="tenant-a",
            owner_user_id="u1",
            context_type="memory",
            source_kind="canonical_projection",
            record_kind=CatalogRecordKind.CLAIM_REVISION.value,
            canonical_slot_id="slot-1",
            canonical_claim_id=f"claim-{state.lower()}-{index}",
            canonical_revision=1,
            canonical_state=state,
                title=f"{state} claim",
                l0_text=f"{state} claim",
                source_digest=f"digest-{state}-{index}",
                metadata={
                    "scope": {
                        "visibility": {
                            "tenant_id": "tenant-a",
                            "private": True,
                            "allowed_principal_ids": ["u1"],
                            "allowed_service_ids": [],
                        }
                    }
                },
            )

    store.upsert_catalog_batch(
        (
            claim("ACTIVE", 1),
            claim("PROPOSED", 1),
            claim("CONFLICTED", 1),
            claim("SUPERSEDED", 1),
        )
    )
    generator = CandidateGenerator(store)

    def states(intent: RetrievalQueryIntent, *, explicit: tuple[str, ...] = ()) -> set[str]:
        metadata_filters = {"memory_states": explicit} if explicit else {}
        plan = QueryPlanner().plan(
            "",
            options=RetrievalOptions(
                tenant_id="tenant-a",
                owner_user_id="u1",
                context_types=(ContextType.MEMORY,),
                query_intent=intent,
                metadata_filters=metadata_filters,
            ),
        )
        return {
            str(item.metadata.get("canonical_state") or "") for item in generator.generate(plan).branches["structured"]
        }

    assert states(RetrievalQueryIntent.CONFLICTS) == {"CONFLICTED"}
    assert states(RetrievalQueryIntent.OPTIONS) == {"PROPOSED", "CONFLICTED"}
    assert states(RetrievalQueryIntent.CONFLICTS, explicit=("ACTIVE",)) == {"ACTIVE"}
