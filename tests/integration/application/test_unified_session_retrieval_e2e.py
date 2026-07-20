from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from infrastructure.context.candidate import CandidateGenerator
from infrastructure.context.query_planner import QueryPlanner
from infrastructure.context.retrieval.query_plan import (
    RetrievalOptions,
    RetrievalQueryIntent,
)
from infrastructure.store.model.catalog import CatalogRecord, CatalogRecordKind
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.sqlite.index_store import SQLiteIndexStore
from openApi.sdk.client import MemoryOSClient


def _session_options(
    *,
    record_kinds: tuple[str, ...] = (CatalogRecordKind.TOOL_RESULT.value,),
    final_limit: int = 10,
) -> RetrievalOptions:
    return RetrievalOptions(
        target_paths=("timeline/2026/07/14", "resources/desktop"),
        context_types=(ContextType.SESSION,),
        record_kinds=record_kinds,
        tenant_id="default",
        owner_user_id="u1",
        workspace_ids=("memoryOS",),
        event_time_from="2026-07-14",
        event_time_to="2026-07-14",
        timezone="Asia/Singapore",
        query_intent=RetrievalQueryIntent.OPEN_RECALL,
        candidate_limit=100,
        final_limit=final_limit,
    )


def test_tool_result_file_name_uses_unified_catalog_as_ordinary_session_context(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    result = client.commit_agent_session(
        user_id="u1",
        session_id="session-20260714",
        messages=[
            {
                "role": "user",
                "content": "请读取桌面文件",
                "occurred_at": "2026-07-14T00:30:00+08:00",
            }
        ],
        tool_results=[
            {
                "tool_name": "read_file",
                "output": "API_KEY=top-secret budget forecast",
                "path": "/Users/u1/Desktop/budget.xlsx",
                "occurred_at": "2026-07-14T00:31:00+08:00",
            }
        ],
        async_commit=False,
        project_id="memoryOS",
    )
    assert result.session_projection_status == "projected"

    hits = client.search_context(
        "budget.xlsx",
        options=_session_options(),
        user_id="u1",
        project_id="memoryOS",
    )

    assert len(hits) == 1
    hit = hits[0]
    metadata = dict(hit["metadata"])
    assert metadata["resource_name"] == "budget.xlsx"
    assert metadata["source_kind"] == "tool_result"
    assert hit["source_uri"] == "memoryos://user/u1/sessions/history/session-20260714"
    assert len(str(metadata["source_digest"])) == 64
    assert "top-secret" not in json.dumps(hit, ensure_ascii=False)
    assert "/Users/u1" not in json.dumps(hit, ensure_ascii=False)

    record = cast(SQLiteIndexStore, client.runtime.stores.index).get_catalog(
        str(metadata["catalog_record_key"]),
        tenant_id="default",
    )
    assert record is not None
    assert record.record_kind == CatalogRecordKind.TOOL_RESULT.value
    assert record.document_id == ""
    assert record.block_id == ""
    assert "timeline/2026/07/14" in record.tree_paths
    assert "sessions/session-20260714" in record.tree_paths
    assert "resources/desktop" in record.tree_paths

    trace = client.recall_trace(client.last_recall_trace_id)
    serialized_trace = json.dumps(trace, ensure_ascii=False)
    assert "top-secret" not in serialized_trace
    assert "/Users/u1" not in serialized_trace

def test_default_history_retrieves_ordinary_session_event_resource_and_tool_rows(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    result = client.commit_agent_session(
        user_id="u1",
        session_id="history-session-20260714",
        messages=[
            {
                "role": "user",
                "content": "history message marker",
                "occurred_at": "2026-07-14T10:00:00+08:00",
            }
        ],
        tool_results=[
            {
                "tool_name": "read_file",
                "output": "history tool marker",
                "path": "/Users/u1/Desktop/history-resource.txt",
                "occurred_at": "2026-07-14T10:01:00+08:00",
            }
        ],
        async_commit=False,
        project_id="memoryOS",
    )
    assert result.session_projection_status == "projected"

    def history(record_kind: str, query: str, path: str) -> list[dict]:
        return client.search_context(
            query,
            options=RetrievalOptions(
                target_paths=(path,),
                record_kinds=(record_kind,),
                tenant_id="default",
                owner_user_id="u1",
                workspace_ids=("memoryOS",),
                event_time_from="2026-07-14",
                event_time_to="2026-07-14",
                timezone="Asia/Singapore",
                query_intent=RetrievalQueryIntent.HISTORY,
                candidate_limit=50,
                final_limit=10,
            ),
            user_id="u1",
            project_id="memoryOS",
        )

    expected = {
        "session_root": history("session_root", "history-session-20260714", "timeline/2026/07/14"),
        "message": history("message", "history message marker", "timeline/2026/07/14"),
        "tool_result": history("tool_result", "history tool marker", "resources/desktop"),
        "resource_reference": history("resource_reference", "history-resource.txt", "resources/desktop"),
    }

    assert all(rows for rows in expected.values())
    for record_kind, rows in expected.items():
        assert all(dict(row["metadata"])["record_kind"] == record_kind for row in rows)


def test_public_session_event_time_alias_and_occurred_at_precedence_use_matching_timeline_days(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    result = client.commit_agent_session(
        user_id="u1",
        session_id="event-time-alias-public",
        messages=[
            {
                "role": "user",
                "content": "public alias-only marker",
                "event_time": "2026-07-14T23:30:00+08:00",
            },
            {
                "role": "assistant",
                "content": "public occurred-at-wins marker",
                "occurred_at": "2026-07-15T00:15:00+08:00",
                "event_time": "2026-07-14T22:00:00+08:00",
            },
        ],
        async_commit=False,
        project_id="memoryOS",
    )
    assert result.session_projection_status == "projected"

    def recalled(query: str, day: str) -> list[dict]:
        return client.search_context(
            query,
            options=RetrievalOptions(
                target_paths=(f"timeline/{day.replace('-', '/')}",),
                context_types=(ContextType.SESSION,),
                record_kinds=(CatalogRecordKind.MESSAGE.value,),
                tenant_id="default",
                owner_user_id="u1",
                workspace_ids=("memoryOS",),
                event_time_from=day,
                event_time_to=day,
                timezone="Asia/Singapore",
                query_intent=RetrievalQueryIntent.OPEN_RECALL,
                candidate_limit=20,
                final_limit=10,
            ),
            user_id="u1",
            project_id="memoryOS",
        )

    july_14 = recalled("public alias-only marker", "2026-07-14")
    july_15 = recalled("public occurred-at-wins marker", "2026-07-15")
    assert [row["text"] for row in july_14] == ["public alias-only marker"]
    assert [row["text"] for row in july_15] == ["public occurred-at-wins marker"]
    assert all(
        row["text"] != "public occurred-at-wins marker"
        for row in recalled("public occurred-at-wins marker", "2026-07-14")
    )


def test_natural_event_date_uses_real_session_projection_without_lexical_overlap(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    result = client.commit_agent_session(
        user_id="u1",
        session_id="natural-event-session",
        messages=[
            {
                "role": "user",
                "content": "Read the quarterly operations report",
                "occurred_at": "2026-07-14T10:00:00+08:00",
            }
        ],
        async_commit=False,
        project_id="memoryOS",
    )
    assert result.session_projection_status == "projected"

    recalled = client.search_context(
        "2026年7月14日发生了什么",
        options=RetrievalOptions(
            context_types=(ContextType.SESSION,),
            tenant_id="default",
            owner_user_id="u1",
            workspace_ids=("memoryOS",),
            timezone="Asia/Singapore",
            candidate_limit=20,
            final_limit=20,
        ),
        user_id="u1",
        project_id="memoryOS",
    )

    assert any(item["text"] == "Read the quarterly operations report" for item in recalled)
    assert all(
        item["source_uri"] == "memoryos://user/u1/sessions/history/natural-event-session"
        for item in recalled
    )
    trace = client.recall_trace(str(client.last_recall_trace_id))
    assert trace["query_plan"]["query_intent"] == RetrievalQueryIntent.OPEN_RECALL.value
    assert trace["query_plan"]["target_paths"] == []
    assert trace["query_plan"]["event_time_from"] == "2026-07-13T16:00:00+00:00"
    assert trace["query_plan"]["event_time_to"] == "2026-07-14T16:00:00+00:00"
    assert trace["structured_candidates"] > 0
    assert trace["fts_candidates"] == 0


def test_temporal_structured_fallback_marks_fts_unavailable_in_recall_trace(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    committed = client.commit_agent_session(
        user_id="u1",
        session_id="temporal-fts-unavailable",
        messages=[
            {
                "role": "user",
                "content": "Review the operational handoff",
                "occurred_at": "2026-07-14T11:00:00+08:00",
            }
        ],
        async_commit=False,
        project_id="memoryOS",
    )
    assert committed.session_projection_status == "projected"
    cast(SQLiteIndexStore, client.runtime.stores.index).fts_enabled = False

    recalled = client.search_context(
        "2026年7月14日发生了什么",
        options=RetrievalOptions(
            context_types=(ContextType.SESSION,),
            tenant_id="default",
            owner_user_id="u1",
            workspace_ids=("memoryOS",),
            timezone="Asia/Singapore",
            candidate_limit=20,
            final_limit=20,
        ),
        user_id="u1",
        project_id="memoryOS",
    )

    assert any(item["text"] == "Review the operational handoff" for item in recalled)
    trace = client.recall_trace(str(client.last_recall_trace_id))
    assert trace["structured_candidates"] > 0
    assert trace["fts_candidates"] == 0
    assert trace["degraded_modes"] == ["fts_unavailable"]
    assert all("fts_unavailable" in str(item["degraded_mode"]) for item in recalled)


def test_transaction_date_structured_candidate_applies_owner_before_sql_limit(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    query = "2026年7月14日系统新增了哪些记忆"

    def memory_document_record(
        key: str,
        *,
        owner_user_id: str,
        transaction_time: str,
        updated_at: str,
    ) -> CatalogRecord:
        document_id = key.replace(":", "-")
        uri = f"memoryos://user/{owner_user_id}/memories/documents/{document_id}"
        return CatalogRecord(
            record_key=key,
            uri=uri,
            tenant_id="default",
            owner_user_id=owner_user_id,
            workspace_id="memoryOS",
            context_type="memory",
            source_kind="markdown_memory_document",
            record_kind=CatalogRecordKind.MEMORY_DOCUMENT.value,
            primary_tree_path="memories/knowledge/topics/database",
            tree_paths=("memories/knowledge/topics/database",),
            created_at=transaction_time,
            updated_at=updated_at,
            event_time=transaction_time,
            ingested_at=transaction_time,
            transaction_time=transaction_time,
            title=f"Database memory document {key}",
            l0_text="Database memory",
            l1_text="A database note is stored in Markdown",
            l2_uri=uri,
            source_uri=uri,
            source_digest=f"digest-{document_id}",
            source_revision=1,
            document_id=document_id,
            document_kind="topic",
            document_revision=1,
            projection_generation=1,
            projection_effect_hash=f"digest-{document_id}",
            metadata={"relative_path": f"knowledge/topics/{document_id}.md"},
        )

    rows = (
        memory_document_record(
            "memory-document:allowed",
            owner_user_id="u1",
            transaction_time="2026-07-14T03:00:00+00:00",
            updated_at="2026-07-14T03:00:00+00:00",
        ),
        memory_document_record(
            "memory-document:outside-day",
            owner_user_id="u1",
            transaction_time="2026-07-14T17:00:00+00:00",
            updated_at="2026-07-14T17:00:00+00:00",
        ),
        memory_document_record(
            "memory-document:foreign-owner",
            owner_user_id="u2",
            transaction_time="2026-07-14T04:00:00+00:00",
            updated_at="2026-07-15T04:00:00+00:00",
        ),
    )
    index_store = cast(SQLiteIndexStore, client.runtime.stores.index)
    for row in rows:
        index_store.replace_memory_document_projection(
            row,
            (),
            None,
            tenant_id="default",
            owner_user_id=row.owner_user_id,
        )

    plan = QueryPlanner().plan(
        query,
        options=RetrievalOptions(
            context_types=(ContextType.MEMORY,),
            record_kinds=(CatalogRecordKind.MEMORY_DOCUMENT.value,),
            tenant_id="default",
            owner_user_id="u1",
            workspace_ids=("memoryOS",),
            timezone="Asia/Singapore",
            candidate_limit=1,
            final_limit=1,
        ),
    )
    generated = CandidateGenerator(index_store).generate(plan)

    assert plan.query_intent == RetrievalQueryIntent.HISTORY
    assert plan.transaction_time_from == "2026-07-13T16:00:00+00:00"
    assert plan.transaction_time_to == "2026-07-14T16:00:00+00:00"
    assert [candidate.record_key for candidate in generated.branches["structured"]] == [
        "memory-document:allowed"
    ]
    assert generated.structured_candidates == 1
    assert generated.fts_candidates == 0


def test_multiple_desktop_files_are_individually_recallable_and_broad_recall_is_bounded(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    names = [f"quarterly-{index:02d}.txt" for index in range(6)]
    markers = (
        "quartzalpha",
        "quartzbravo",
        "quartzcharlie",
        "quartzdelta",
        "quartzecho",
        "quartzfoxtrot",
    )
    client.commit_agent_session(
        user_id="u1",
        session_id="many-files-20260714",
        messages=[{"role": "user", "content": "读取这些文件"}],
        tool_results=[
            {
                "tool_name": "read_file",
                "output": f"quarterly result {marker}",
                "path": f"/Users/u1/Desktop/{name}",
                "occurred_at": f"2026-07-14T08:{index:02d}:00+08:00",
            }
            for index, (name, marker) in enumerate(zip(names, markers, strict=True))
        ],
        async_commit=False,
        project_id="memoryOS",
    )

    for name, marker in zip(names, markers, strict=True):
        exact = client.search_context(
            marker,
            options=_session_options(final_limit=5),
            user_id="u1",
            project_id="memoryOS",
        )
        assert any(dict(item["metadata"]).get("resource_name") == name for item in exact)

    first = client.search_context(
        "quarterly",
        options=_session_options(final_limit=20),
        user_id="u1",
        project_id="memoryOS",
    )
    second = client.search_context(
        "quarterly",
        options=_session_options(final_limit=20),
        user_id="u1",
        project_id="memoryOS",
    )

    assert [item["record_key"] for item in first] == [item["record_key"] for item in second]
    assert 1 <= len(first) <= 3  # configured per-resource-branch quota
    assert len({item["record_key"] for item in first}) == len(first)
    assert all(dict(item["metadata"])["resource_name"] in names for item in first)
