from __future__ import annotations

import asyncio
import json

from memoryos.api.http.app import MemoryOSASGI
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.connect import ConnectMetadata
from memoryos.memory.extraction import FakeMemoryModelProvider, LLMMemoryExtractorBackend


def test_finalize_extract_commit_and_cross_agent_project_recall(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    result = client.commit_agent_session(
        user_id="u1",
        session_id="claude-native",
        session_key="stable-session",
        project_id="project-a",
        messages=[{"id": "m1", "role": "user", "content": "Project rule: must run pytest before merge."}],
        connect_metadata=ConnectMetadata.default_agent("claude_code").to_dict(),
        async_commit=True,
    )
    assert result.done is True
    shared = client.search_context(
        "pytest",
        user_id="u1",
        project_id="project-a",
        search_scope="project_rules",
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )
    isolated = client.search_context(
        "pytest",
        user_id="u1",
        project_id="project-b",
        search_scope="project_rules",
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
    )
    assert any("pytest" in item["text"] for item in shared)
    assert isolated == []
    assembled = client.assemble_context(
        "pytest", user_id="u1", project_id="project-a", search_scope="project_rules", token_budget=100
    )
    assert "pytest" in assembled["packed_context"]
    assert client.recall_trace(assembled["trace_id"])["selected"]


def test_worker_queued_to_committed_is_idempotent(tmp_path) -> None:  # noqa: ANN001
    from memoryos.workers.session_commit_worker import SessionCommitWorker

    client = MemoryOSClient(str(tmp_path))
    queued = client.commit_agent_session(
        user_id="u1",
        session_id="s1",
        messages=[{"role": "user", "content": "Remember: I prefer concise output."}],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
        async_commit=False,
    )
    first = SessionCommitWorker(client.session_commit_service).process_pending()
    second = SessionCommitWorker(client.session_commit_service).process_pending()
    assert queued.status == "queued"
    assert first["committed"] == 1
    assert second["claimed"] == 0


def test_user_preference_is_recalled_across_projects(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    client.remember(
        user_id="u1",
        memory_type="preference",
        title="Response style",
        content="I prefer concise engineering responses.",
    )
    project_a = client.search_context(
        "concise",
        user_id="u1",
        project_id="project-a",
        search_scope="user_preferences",
    )
    project_b = client.search_context(
        "concise",
        user_id="u1",
        project_id="project-b",
        search_scope="user_preferences",
    )
    assert project_a and project_b
    assert project_a[0]["uri"] == project_b[0]["uri"]


def test_runtime_injected_llm_extractor_enters_operation_plane(tmp_path) -> None:  # noqa: ANN001
    provider = FakeMemoryModelProvider(
        response='{"candidates":[{"proposal_id":"p-sqlite","memory_type":"project_decision","identity_fields":{"decision_topic":"primary storage backend"},"value_fields":{"canonical_value":"SQLite"},"semantic":{"speech_act":"confirmation","commitment":"confirmed","temporal_scope":"current","relation_to_existing":"unrelated"},"epistemic_status":"EXPLICIT","suggested_scope_refs":[{"namespace":"memoryos","kind":"workspace","id":"p1"}],"evidence_refs":[{"event_id":"message:0"}],"field_evidence_refs":{"identity.decision_topic":[{"event_id":"message:0"}],"value.canonical_value":[{"event_id":"message:0"}],"semantic.speech_act":[{"event_id":"message:0"}],"semantic.commitment":[{"event_id":"message:0"}],"semantic.temporal_scope":[{"event_id":"message:0"}],"semantic.relation_to_existing":[{"event_id":"message:0"}],"transition":[{"event_id":"message:0"}]},"confidence":0.9,"source_role":"user"}]}'
    )
    client = MemoryOSClient(str(tmp_path), memory_extractor=LLMMemoryExtractorBackend(provider))
    client.commit_agent_session(
        user_id="u1",
        session_id="s1",
        project_id="p1",
        messages=[{"role": "user", "content": "We decided to use SQLite as the primary storage backend."}],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
        async_commit=True,
    )
    results = client.search_context("SQLite", user_id="u1", project_id="p1", search_scope="project_decisions")
    assert results
    assert client.health()["memory_extractor"] == "ready"


def test_http_session_append_finalize_recall_flow(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path), mode="server")
    app = MemoryOSASGI(client)

    async def post(path: str, payload: dict) -> dict:
        sent = []
        incoming = iter([{"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}])

        async def receive():  # noqa: ANN202
            return next(incoming)

        async def send(message):  # noqa: ANN001, ANN202
            sent.append(message)

        await app({"type": "http", "method": "POST", "path": path, "headers": []}, receive, send)
        assert sent[0]["status"] == 200
        return json.loads(sent[1]["body"])

    event = asyncio.run(
        post(
            "/v1/sessions/events",
            {
                "event_id": "e1",
                "event_type": "PROMPT_SUBMIT",
                "adapter_id": "openclaw",
                "user_id": "u1",
                "project_id": "p1",
                "session_id": "native",
                "prompt": "Project rule: must run ruff before merge.",
            },
        )
    )
    finalized = asyncio.run(post(f"/v1/sessions/{event['session_key']}/finalize", {}))
    assert finalized["done"] is True
    assert client.search_context("ruff", user_id="u1", project_id="p1", search_scope="project_rules")
