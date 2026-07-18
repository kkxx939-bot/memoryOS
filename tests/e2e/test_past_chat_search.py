from __future__ import annotations

from datetime import datetime, timezone

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.query_plan import (
    RetrievalOptions,
    RetrievalQueryIntent,
)
from memoryos.security.trusted_context import (
    AUTHORITATIVE_REMEMBER,
    READ_CONTEXT,
    TrustedRequestContext,
)

_MARKER = "PastChatJavaRaftMarker"
_PROJECT_MARKER = "MemoryOSCrossSourceMarker"
_RECORD_KINDS = (
    "session_root",
    "session_l0",
    "session_l1",
    "message",
    "semantic_segment",
    "memory_document",
    "memory_block",
)
_DOCUMENT_KINDS = ("episode", "topic", "entity")


class _FixedPromptDateTime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: ANN001, ANN206 - mirrors datetime.now.
        fixed = cls(2026, 7, 17, 4, 0, tzinfo=timezone.utc)
        return fixed if tz is None else fixed.astimezone(tz)


def _caller() -> TrustedRequestContext:
    return TrustedRequestContext(
        tenant_id="default",
        user_id="u1",
        actor_kind="user",
        actor_id="u1",
        capabilities=frozenset({READ_CONTEXT, AUTHORITATIVE_REMEMBER}),
    )


def _options() -> RetrievalOptions:
    return RetrievalOptions(
        context_types=(ContextType.SESSION, ContextType.MEMORY),
        record_kinds=_RECORD_KINDS,
        document_kinds=_DOCUMENT_KINDS,
        tenant_id="default",
        owner_user_id="u1",
        query_intent=RetrievalQueryIntent.OPEN_RECALL,
        timezone="Asia/Singapore",
        candidate_limit=100,
        final_limit=100,
        token_budget=4_096,
    )


def _project_all(client: MemoryOSClient) -> None:
    while client.queue_store.stats(queue_name="memory_projection").get("pending", 0):
        result = client.memory_projection_worker.process_pending(limit=20)
        assert not result.failed


def _source_uris(items: list[dict]) -> set[str]:
    return {
        source_uri
        for item in items
        if (source_uri := str(item.get("source_uri") or item.get("archive_uri") or ""))
    }


def test_fixed_past_chat_query_uses_one_catalog_chain_for_memory_and_session(
    tmp_path,
    monkeypatch,
) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "memoryos.application.context.query_planner.datetime",
        _FixedPromptDateTime,
    )
    client = MemoryOSClient(str(tmp_path))
    caller = _caller()
    remembered = [
        client.remember(
            f"{_MARKER}: Java 分布式方案的 episode 记录采用 Raft。",
            occurred_at="2026-07-11T00:20:00+08:00",
            target_hint="episode:Java distributed episode",
            caller=caller,
        ),
        client.remember(
            f"{_MARKER}: Java 分布式方案的 topic 记录包含幂等消息。",
            target_hint="topic:Java distributed topic",
            caller=caller,
        ),
        client.remember(
            f"{_MARKER}: Java 分布式方案的 entity 记录是 RaftCluster。",
            target_hint="entity:RaftCluster",
            caller=caller,
        ),
    ]
    assert {item["document_kind"] for item in remembered} == {
        "episode",
        "topic",
        "entity",
    }
    _project_all(client)

    local_day_session = client.commit_agent_session(
        user_id="u1",
        session_id="java-distributed-local-11",
        messages=[
            {
                "role": "user",
                "content": f"{_MARKER}: 我们讨论了 Java 分布式方案、Raft 和幂等消息。",
                # 00:30 on July 11 in Asia/Singapore.
                "occurred_at": "2026-07-10T16:30:00+00:00",
            }
        ],
        async_commit=False,
    )
    next_day_session = client.commit_agent_session(
        user_id="u1",
        session_id="java-distributed-local-12",
        messages=[
            {
                "role": "user",
                "content": f"{_MARKER}: 我们在本地 12 号继续讨论 Java 分布式缓存。",
                # 00:30 on July 12 in Asia/Singapore.
                "occurred_at": "2026-07-11T16:30:00+00:00",
            }
        ],
        async_commit=False,
    )
    assert local_day_session.session_projection_status == "projected"
    assert next_day_session.session_projection_status == "projected"

    broad_query = f"{_MARKER} Java 分布式"
    archive_results = client.archive_search(
        broad_query,
        user_id="u1",
        limit=20,
        timezone_name="Asia/Singapore",
        caller=caller,
    )
    archive_trace = client.recall_trace(str(client.last_recall_trace_id), caller=caller)
    assembled = client.assemble_context(
        broad_query,
        options=_options(),
        user_id="u1",
        caller=caller,
    )

    expected_memory_sources = {item["document_uri"] for item in remembered}
    expected_session_sources = {
        "memoryos://user/u1/sessions/history/java-distributed-local-11",
        "memoryos://user/u1/sessions/history/java-distributed-local-12",
    }
    expected_sources = expected_memory_sources | expected_session_sources
    archive_sources = _source_uris(archive_results)
    assembled_sources = set(assembled["source_uris"])
    assert expected_sources <= archive_sources
    assert expected_sources <= assembled_sources
    assert archive_sources == assembled_sources
    assert archive_trace["query_plan"] == {
        key: value
        for key, value in assembled["query_plan"].items()
        if key != "semantic_query"
    }
    assert assembled["query_plan"]["timezone"] == "Asia/Singapore"
    assert assembled["query_plan"]["document_kinds"] == list(_DOCUMENT_KINDS)
    hydrated_session_l2 = [
        item
        for item in assembled["contexts"]
        if item.get("selected_layer") == "L2"
        and dict(item.get("metadata", {})).get("record_kind")
        in {"session_root", "session_l1"}
    ]
    assert hydrated_session_l2
    assert any(_MARKER in str(item.get("content") or "") for item in hydrated_session_l2)
    assert 0 < assembled["metrics"]["source_reads"] <= (
        assembled["query_plan"]["candidate_limit"] + 8
    )

    past_chat_query = (
        f"我想看一下我11号和你讨论java的分布式方案 {_MARKER}，你还记得吗？"
    )
    dated_archive = client.archive_search(
        past_chat_query,
        user_id="u1",
        limit=20,
        timezone_name="Asia/Singapore",
        caller=caller,
    )
    dated_trace = client.recall_trace(str(client.last_recall_trace_id), caller=caller)
    dated_assembled = client.assemble_context(
        past_chat_query,
        options=_options(),
        user_id="u1",
        caller=caller,
    )

    dated_sessions = {
        str(item["session_id"])
        for item in dated_archive
        if item.get("session_id")
    }
    assert dated_sessions == {"java-distributed-local-11"}
    assert (
        "memoryos://user/u1/sessions/history/java-distributed-local-12"
        not in _source_uris(dated_archive)
    )
    assert dated_trace["query_plan"] == {
        key: value
        for key, value in dated_assembled["query_plan"].items()
        if key != "semantic_query"
    }
    assert dated_assembled["query_plan"]["timezone"] == "Asia/Singapore"
    assert dated_assembled["query_plan"]["event_time_from"] == "2026-07-10T16:00:00+00:00"
    assert dated_assembled["query_plan"]["event_time_to"] == "2026-07-11T16:00:00+00:00"


def test_project_phrase_recalls_entity_topic_episode_and_session_without_memory_project_path(
    tmp_path,
) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    caller = _caller()
    remembered = [
        client.remember(
            f"{_PROJECT_MARKER}: MemoryOS 项目的 episode 记录了 Markdown 重构发布。",
            occurred_at="2026-07-11T09:00:00+08:00",
            target_hint="episode:MemoryOS Markdown refactor",
            caller=caller,
        ),
        client.remember(
            f"{_PROJECT_MARKER}: MemoryOS 项目的 topic 采用文档优先检索。",
            target_hint="topic:MemoryOS architecture",
            caller=caller,
        ),
        client.remember(
            f"{_PROJECT_MARKER}: MemoryOS 项目的 entity 是 MemoryOS。",
            target_hint="entity:MemoryOS",
            caller=caller,
        ),
    ]
    assert {item["document_kind"] for item in remembered} == {
        "episode",
        "topic",
        "entity",
    }
    assert all(not str(item["relative_path"]).startswith("projects/") for item in remembered)
    _project_all(client)

    committed = client.commit_agent_session(
        user_id="u1",
        session_id="memoryos-cross-source-session",
        messages=[
            {
                "role": "user",
                "content": f"{_PROJECT_MARKER}: 我们讨论了 MemoryOS 项目的文档主链方案。",
                "occurred_at": "2026-07-11T10:00:00+08:00",
            }
        ],
        async_commit=False,
    )
    assert committed.session_projection_status == "projected"

    assembled = client.assemble_context(
        f"MemoryOS 项目 {_PROJECT_MARKER} 当前采用什么方案",
        options=_options(),
        user_id="u1",
        caller=caller,
    )
    expected_sources = {item["document_uri"] for item in remembered} | {
        "memoryos://user/u1/sessions/history/memoryos-cross-source-session"
    }
    assert expected_sources <= set(assembled["source_uris"])
