from __future__ import annotations

from pathlib import Path

from memoryos.contextdb.catalog import CatalogRecord, CatalogRecordKind
from memoryos.contextdb.session.context_projector import SessionContextProjector
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.vector_store import InMemoryVectorStore, vector_row_id
from memoryos.providers.embedding import HashingEmbeddingProvider


class _Catalog:
    def __init__(self) -> None:
        self.records: dict[str, CatalogRecord] = {}

    def upsert_catalog_batch(self, records: tuple[CatalogRecord, ...], *, tenant_id: str) -> None:
        for record in records:
            assert record.tenant_id == tenant_id
            self.records[record.record_key] = record

    def upsert_catalog(self, record: CatalogRecord, *, tenant_id: str) -> None:
        assert record.tenant_id == tenant_id
        self.records[record.record_key] = record


def _written_archive(tmp_path: Path) -> SessionArchive:
    archive = SessionArchive(
        user_id="u1",
        session_id="session-20260714",
        archive_uri="memoryos://user/u1/sessions/history/session-20260714",
        created_at="2026-07-14T00:30:00+08:00",
        metadata={
            "tenant_id": "tenant-a",
            "timezone": "Asia/Singapore",
            "project_id": "memoryOS",
            "adapter_id": "coding-agent",
        },
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
                "output": "API_KEY=top-secret first sheet",
                "path": "/Users/u1/Desktop/budget.xlsx",
                "occurred_at": "2026-07-14T00:31:00+08:00",
            },
            {
                "tool_name": "read_file",
                "output": "Authorization: Bearer secret-token-123 second file",
                "resource_uri": "file:///Users/u1/Desktop/roadmap.md",
                "occurred_at": "2026-07-14T00:32:00+08:00",
            },
        ],
    )
    SessionArchiveStore(tmp_path, tenant_id="tenant-a").write_sync_archive(archive)
    return archive


def test_tool_result_file_names_project_to_catalog_as_session_records(tmp_path: Path) -> None:
    archive = _written_archive(tmp_path)
    catalog = _Catalog()

    result = SessionContextProjector(catalog).project(archive)

    assert result.projected == len(catalog.records)
    tool_results = [
        record for record in catalog.records.values() if record.record_kind == CatalogRecordKind.TOOL_RESULT.value
    ]
    assert {record.metadata["resource_name"] for record in tool_results} == {"budget.xlsx", "roadmap.md"}
    assert all(record.context_type == "session" for record in tool_results)
    assert all(record.document_id == "" and record.block_id == "" for record in tool_results)
    assert all("timeline/2026/07/14" in record.tree_paths for record in tool_results)
    assert all("resources/desktop" in record.tree_paths for record in tool_results)
    assert all(record.source_uri == archive.archive_uri for record in tool_results)
    assert all(len(record.source_digest) == 64 for record in tool_results)


def test_projection_sanitizes_secrets_paths_and_keeps_atomic_rows_out_of_vector(tmp_path: Path) -> None:
    archive = _written_archive(tmp_path)
    catalog = _Catalog()

    SessionContextProjector(catalog).project(archive)

    serving_text = "\n".join(
        f"{record.title}\n{record.l0_text}\n{record.l1_text}\n{record.metadata}" for record in catalog.records.values()
    )
    assert "top-secret" not in serving_text
    assert "secret-token-123" not in serving_text
    assert "/Users/u1" not in serving_text
    assert "budget.xlsx" in serving_text
    assert "roadmap.md" in serving_text
    atomics = [
        record
        for record in catalog.records.values()
        if record.record_kind in {CatalogRecordKind.MESSAGE.value, CatalogRecordKind.TOOL_RESULT.value}
    ]
    assert atomics
    assert all(record.metadata["vector_eligible"] is False for record in atomics)


def test_projection_is_idempotent_and_has_one_primary_bounded_path_set(tmp_path: Path) -> None:
    archive = _written_archive(tmp_path)
    catalog = _Catalog()
    projector = SessionContextProjector(catalog)

    first = projector.project(archive)
    second = projector.project(archive)

    assert first.record_keys == second.record_keys
    assert len(catalog.records) == first.projected
    assert all(record.primary_tree_path == record.tree_paths[0] for record in catalog.records.values())
    assert all(len(record.tree_paths) <= 8 for record in catalog.records.values())


def test_timeline_path_uses_event_time_and_caller_timezone(tmp_path: Path) -> None:
    archive = _written_archive(tmp_path)
    catalog = _Catalog()

    SessionContextProjector(catalog).project(archive)

    # 2026-07-14 00:31 +08:00 is still July 14 for the caller but July 13 UTC.
    tool = next(
        record for record in catalog.records.values() if record.record_kind == CatalogRecordKind.TOOL_RESULT.value
    )
    assert tool.event_time.startswith("2026-07-13T16:31:00")
    assert "timeline/2026/07/14" in tool.tree_paths


def test_session_summary_timeline_uses_episode_start_not_archive_write_time(tmp_path: Path) -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="cross-day-session",
        archive_uri="memoryos://user/u1/sessions/history/cross-day-session",
        created_at="2026-07-15T09:00:00+08:00",
        metadata={"tenant_id": "tenant-a", "timezone": "Asia/Singapore"},
        messages=[
            {
                "role": "user",
                "content": "the event happened before the archive was written",
                "occurred_at": "2026-07-14T23:55:00+08:00",
            }
        ],
        used_contexts=[{"title": "context without event time", "source_uri": "memoryos://resource/context-a"}],
        used_skills=[{"skill_name": "skill-without-event-time"}],
    )
    SessionArchiveStore(tmp_path, tenant_id="tenant-a").write_sync_archive(archive)

    records = SessionContextProjector(_Catalog()).build_records(archive)
    summaries = [
        record
        for record in records
        if record.record_kind
        in {
            CatalogRecordKind.SESSION_ROOT.value,
            CatalogRecordKind.SESSION_L0.value,
            CatalogRecordKind.SESSION_L1.value,
        }
    ]

    assert len(summaries) == 3
    assert all(record.event_time == "2026-07-14T15:55:00+00:00" for record in summaries)
    assert all("timeline/2026/07/14" in record.tree_paths for record in summaries)
    assert all("timeline/2026/07/15" not in record.tree_paths for record in summaries)
    references = [
        record
        for record in records
        if record.record_kind
        in {
            CatalogRecordKind.USED_CONTEXT.value,
            CatalogRecordKind.USED_SKILL.value,
        }
    ]
    assert len(references) == 2
    assert all(record.event_time == "2026-07-14T15:55:00+00:00" for record in references)
    assert all("timeline/2026/07/14" in record.tree_paths for record in references)
    assert all("timeline/2026/07/15" not in record.tree_paths for record in references)


def test_reference_timeline_uses_its_explicit_event_time_instead_of_episode_start() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="reference-time-session",
        archive_uri="memoryos://user/u1/sessions/history/reference-time-session",
        created_at="2026-07-15T09:00:00+08:00",
        metadata={"tenant_id": "tenant-a", "timezone": "Asia/Singapore"},
        messages=[
            {
                "role": "user",
                "content": "session starts on July 14",
                "occurred_at": "2026-07-14T09:00:00+08:00",
            }
        ],
        used_contexts=[
            {
                "title": "context explicitly used on July 13",
                "source_uri": "memoryos://resource/context-explicit",
                "event_time": "2026-07-13T23:30:00+08:00",
            }
        ],
    )

    record = next(
        item
        for item in SessionContextProjector(_Catalog()).build_records(archive)
        if item.record_kind == CatalogRecordKind.USED_CONTEXT.value
    )

    assert record.event_time == "2026-07-13T15:30:00+00:00"
    assert "timeline/2026/07/13" in record.tree_paths
    assert "timeline/2026/07/14" not in record.tree_paths


def test_event_time_alias_and_occurred_at_precedence_keep_segments_on_one_timeline_day() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="event-time-alias-session",
        archive_uri="memoryos://user/u1/sessions/history/event-time-alias-session",
        created_at="2026-07-15T09:00:00+08:00",
        metadata={"tenant_id": "tenant-a", "timezone": "Asia/Singapore"},
        messages=[
            {
                "role": "user",
                "content": "event_time is the only occurrence timestamp",
                "event_time": "2026-07-14T23:30:00+08:00",
            },
            {
                "role": "assistant",
                "content": "occurred_at wins when both aliases are present",
                "occurred_at": "2026-07-15T00:15:00+08:00",
                "event_time": "2026-07-14T22:00:00+08:00",
            },
        ],
    )

    records = SessionContextProjector(_Catalog()).build_records(archive)
    messages = {
        record.l1_text: record
        for record in records
        if record.record_kind == CatalogRecordKind.MESSAGE.value
    }
    alias_only = messages["event_time is the only occurrence timestamp"]
    occurred_wins = messages["occurred_at wins when both aliases are present"]

    assert alias_only.event_time == "2026-07-14T15:30:00+00:00"
    assert [path for path in alias_only.tree_paths if path.startswith("timeline/")] == [
        "timeline/2026/07/14"
    ]
    assert occurred_wins.event_time == "2026-07-14T16:15:00+00:00"
    assert [path for path in occurred_wins.tree_paths if path.startswith("timeline/")] == [
        "timeline/2026/07/15"
    ]

    segments = [
        record
        for record in records
        if record.record_kind == CatalogRecordKind.SEMANTIC_SEGMENT.value
    ]
    assert len(segments) == 2
    assert [
        [path for path in segment.tree_paths if path.startswith("timeline/")]
        for segment in segments
    ] == [["timeline/2026/07/14"], ["timeline/2026/07/15"]]


def test_session_vectors_use_only_sanitized_policy_eligible_records(tmp_path: Path) -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="session-vector",
        archive_uri="memoryos://user/u1/sessions/history/session-vector",
        created_at="2026-07-14T09:00:00+08:00",
        metadata={"tenant_id": "tenant-a", "timezone": "Asia/Singapore"},
        messages=[
            {
                "role": "user",
                "content": "read the desktop plan",
                "occurred_at": "2026-07-14T09:00:00+08:00",
            }
        ],
        tool_results=[
            {
                "tool_name": "read_file",
                "output": "API_KEY=vector-secret launch plan",
                "path": "/Users/u1/Desktop/launch-plan.md",
                "important": True,
                "occurred_at": "2026-07-14T09:01:00+08:00",
            }
        ],
        observations=[
            {
                "raw_text": "Authorization: Bearer observation-secret reusable finding",
                "important": True,
                "occurred_at": "2026-07-14T09:02:00+08:00",
            }
        ],
    )
    SessionArchiveStore(tmp_path, tenant_id="tenant-a").write_sync_archive(archive)
    catalog = _Catalog()
    vectors = InMemoryVectorStore()

    class RecordingEmbedding(HashingEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.texts: list[str] = []

        def embed(self, text: str) -> list[float]:
            self.texts.append(text)
            return super().embed(text)

    embedding = RecordingEmbedding()
    result = SessionContextProjector(
        catalog,
        vector_store=vectors,
        embedding_provider=embedding,
        vectorize_important_events=True,
    ).project(archive)

    vector_record_kinds = {
        catalog.records[str(metadata["catalog_record_key"])].record_kind
        for _uri, metadata in ((uri, vectors.get_vector_metadata(uri) or {}) for uri in vectors.rows)
    }
    assert CatalogRecordKind.SESSION_ROOT.value in vector_record_kinds
    assert CatalogRecordKind.SEMANTIC_SEGMENT.value in vector_record_kinds
    assert CatalogRecordKind.OBSERVATION.value in vector_record_kinds
    assert CatalogRecordKind.RESOURCE_REFERENCE.value in vector_record_kinds
    assert CatalogRecordKind.MESSAGE.value not in vector_record_kinds
    assert CatalogRecordKind.TOOL_RESULT.value not in vector_record_kinds
    assert result.vectors_projected == len(vectors.rows) == len(embedding.texts)
    embedded_text = "\n".join(embedding.texts)
    assert "vector-secret" not in embedded_text
    assert "observation-secret" not in embedded_text
    assert "/Users/u1" not in embedded_text
    assert all("content" not in metadata for _embedding, metadata in vectors.rows.values())
    required_filter_fields = {
        "catalog_record_key",
        "tenant_id",
        "owner_user_id",
        "workspace_id",
        "session_id",
        "adapter_id",
        "context_type",
        "source_kind",
        "record_kind",
        "lifecycle_state",
        "primary_tree_path",
        "tree_paths",
        "scope_keys",
        "created_at",
        "updated_at",
        "event_time",
        "ingested_at",
        "transaction_time",
        "source_uri",
        "source_digest",
        "source_revision",
        "projection_effect_hash",
        "serving_tier",
        "projection_status",
    }
    for _embedding, metadata in vectors.rows.values():
        assert required_filter_fields <= metadata.keys()
        assert metadata["tenant_id"] == "tenant-a"
        assert metadata["owner_user_id"] == "u1"
        assert metadata["session_id"] == "session-vector"
        assert metadata["source_manifest_digest"] == archive.manifest_digest
        assert metadata["projection_effect_hash"] == archive.manifest_digest
        assert metadata["source_revision"] == 1


def test_same_session_manifests_keep_catalog_identity_and_vector_ownership_distinct(
    tmp_path: Path,
) -> None:
    """A stale manifest tombstone must be unable to delete its replacement vector."""

    store = SessionArchiveStore(tmp_path, tenant_id="tenant-a")
    first = SessionArchive(
        user_id="u1",
        session_id="session-shared",
        archive_uri="memoryos://user/u1/sessions/history/session-shared",
        created_at="2026-07-14T09:00:00+08:00",
        metadata={
            "tenant_id": "tenant-a",
            "timezone": "Asia/Singapore",
            "workspace_id": "project-a",
            "adapter_id": "coding-agent",
        },
        messages=[
            {
                "role": "user",
                "content": "first manifest context",
                "occurred_at": "2026-07-14T09:00:00+08:00",
            }
        ],
    )
    store.write_sync_archive(first)
    catalog = _Catalog()
    vectors = InMemoryVectorStore()
    projector = SessionContextProjector(
        catalog,
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )

    first_result = projector.project(first)
    root_uri = f"{first.archive_uri}/context/root"
    first_root_key = next(key for key in first_result.record_keys if catalog.records[key].uri == root_uri)
    first_vector = vectors.get_vector_metadata(vector_row_id("tenant-a", first_root_key))
    assert first_vector is not None
    assert first_root_key == str(first_vector["catalog_record_key"])

    second = SessionArchive(
        user_id="u1",
        session_id=first.session_id,
        archive_uri=first.archive_uri,
        created_at="2026-07-14T09:05:00+08:00",
        metadata=dict(first.metadata),
        messages=[
            {
                "role": "user",
                "content": "second manifest replacement context",
                "occurred_at": "2026-07-14T09:05:00+08:00",
            }
        ],
    )
    store.write_sync_archive(second)
    second_result = projector.project(second)
    second_root_key = next(key for key in second_result.record_keys if catalog.records[key].uri == root_uri)
    second_vector = vectors.get_vector_metadata(vector_row_id("tenant-a", second_root_key))

    assert first.manifest_digest != second.manifest_digest
    assert set(first_result.record_keys).isdisjoint(second_result.record_keys)
    assert first_root_key in catalog.records
    assert second_vector is not None
    assert second_root_key == str(second_vector["catalog_record_key"])
    assert second_root_key in catalog.records
    assert second_root_key != first_root_key
    assert second_vector["source_manifest_digest"] == second.manifest_digest
    assert second_vector["projection_effect_hash"] == second.manifest_digest
    assert second_vector["source_revision"] == 1
    assert second_vector["session_id"] == first.session_id
    assert second_vector["workspace_id"] == "project-a"
    assert second_vector["adapter_id"] == "coding-agent"
    assert second_vector["source_manifest_digest"] != first_vector["source_manifest_digest"]


def test_shared_vector_backend_keeps_same_public_uri_tenant_scoped(tmp_path: Path) -> None:
    public_uri = "memoryos://user/shared/sessions/history/same"
    archives: list[SessionArchive] = []
    for tenant_id in ("tenant-a", "tenant-b"):
        archive = SessionArchive(
            user_id="shared",
            session_id="same",
            archive_uri=public_uri,
            created_at="2026-07-14T09:00:00+00:00",
            metadata={"tenant_id": tenant_id, "timezone": "UTC"},
            messages=[
                {
                    "role": "user",
                    "content": f"private context for {tenant_id}",
                    "occurred_at": "2026-07-14T09:00:00+00:00",
                }
            ],
        )
        SessionArchiveStore(tmp_path / tenant_id, tenant_id=tenant_id).write_sync_archive(archive)
        archives.append(archive)

    catalog = _Catalog()
    vectors = InMemoryVectorStore()
    projector = SessionContextProjector(
        catalog,
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )
    for archive in archives:
        projector.project(archive)

    roots = [
        record for record in catalog.records.values() if record.record_kind == CatalogRecordKind.SESSION_ROOT.value
    ]
    assert {record.tenant_id for record in roots} == {"tenant-a", "tenant-b"}
    row_ids = {record.tenant_id: vector_row_id(record.tenant_id, record.record_key) for record in roots}
    assert row_ids["tenant-a"] != row_ids["tenant-b"]
    assert vectors.get_vector_metadata(public_uri) is None, "ambiguous public URI must fail closed"
    for record in roots:
        metadata = vectors.get_vector_metadata(row_ids[record.tenant_id])
        assert metadata is not None
        assert metadata["tenant_id"] == record.tenant_id
        assert metadata["catalog_record_key"] == record.record_key
        assert metadata["public_uri"] == f"{public_uri}/context/root"

    vectors.delete_vector(row_ids["tenant-a"])
    assert vectors.get_vector_metadata(row_ids["tenant-a"]) is None
    assert vectors.get_vector_metadata(row_ids["tenant-b"]) is not None


def test_important_resource_vector_requires_explicit_projection_policy(tmp_path: Path) -> None:
    archive = _written_archive(tmp_path)
    archive.tool_results[0]["important"] = True
    SessionArchiveStore(tmp_path, tenant_id="tenant-a").write_sync_archive(archive)
    catalog = _Catalog()
    vectors = InMemoryVectorStore()
    SessionContextProjector(
        catalog,
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
        vectorize_important_events=False,
    ).project(archive)

    vector_kinds = {str(metadata.get("record_kind") or "") for _embedding, metadata in vectors.rows.values()}
    assert vector_kinds == {
        CatalogRecordKind.SESSION_ROOT.value,
        CatalogRecordKind.SEMANTIC_SEGMENT.value,
    }
