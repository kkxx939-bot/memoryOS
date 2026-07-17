from __future__ import annotations

import asyncio
import json

from memoryos.api.http.app import MemoryOSASGI
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.trusted_context import ATTEST_USER_INPUT, DEFAULT_AGENT_CAPABILITIES, TrustedRequestContext
from memoryos.connect import ConnectMetadata
from memoryos.memory.extraction import FakeMemoryModelProvider, LLMMemoryExtractorBackend


def _project_rule_response(event_id: str, text: str, project_id: str, *, proposal_id: str) -> str:
    atomic = {"event_id": event_id, "span_start": 0, "span_end": len(text)}
    semantic = {
        "speech_act": "confirmation",
        "commitment": "confirmed",
        "temporal_scope": "current",
        "relation_to_existing": "unrelated",
        "utterance_mode": "directive",
        "attribution": "source_actor",
        "durability": "durable",
        "modal_force": "require",
        "atomicity": "atomic",
    }
    rule_topic = text.split("must ", 1)[-1].rstrip(".")
    identity = {"rule_topic": rule_topic}
    values = {"canonical_value": "required", "rule": text}
    bindings = {
        **{f"identity.{key}": [atomic] for key in identity},
        **{f"value.{key}": [atomic] for key in values},
        **{f"semantic.{key}": [atomic] for key in semantic},
        "transition": [atomic],
    }
    return json.dumps(
        {
            "candidates": [
                {
                    "proposal_id": proposal_id,
                    "memory_type": "project_rule",
                    "identity_fields": identity,
                    "value_fields": values,
                    "semantic": semantic,
                    "epistemic_status": "EXPLICIT",
                    "suggested_scope_refs": [{"namespace": "memoryos", "kind": "workspace", "id": project_id}],
                    "evidence_refs": [atomic],
                    "atomic_evidence_ref": atomic,
                    "field_evidence_refs": bindings,
                    "confidence": 0.95,
                    "source_role": "user",
                }
            ]
        }
    )


def test_finalize_extract_commit_and_cross_agent_project_recall(tmp_path) -> None:  # noqa: ANN001
    text = "Project rule: must run pytest before merge."
    provider = FakeMemoryModelProvider(_project_rule_response("m1", text, "project-a", proposal_id="p-pytest"))
    client = MemoryOSClient(str(tmp_path), memory_extractor=LLMMemoryExtractorBackend(provider))
    result = client.commit_agent_session(
        user_id="u1",
        session_id="claude-native",
        session_key="stable-session",
        project_id="project-a",
        messages=[{"id": "m1", "role": "user", "content": text}],
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
    text = "We decided to use SQLite as the primary storage backend."
    atomic = {"event_id": "message:0", "span_start": 0, "span_end": len(text)}
    semantic = {
        "speech_act": "confirmation",
        "commitment": "confirmed",
        "temporal_scope": "current",
        "relation_to_existing": "unrelated",
        "utterance_mode": "assertion",
        "attribution": "source_actor",
        "durability": "durable",
        "modal_force": "none",
        "atomicity": "atomic",
    }
    bindings = {
        "identity.decision_topic": [atomic],
        "value.canonical_value": [atomic],
        **{f"semantic.{field_name}": [atomic] for field_name in semantic},
        "transition": [atomic],
    }
    provider = FakeMemoryModelProvider(
        response=json.dumps(
            {
                "candidates": [
                    {
                        "proposal_id": "p-sqlite",
                        "memory_type": "project_decision",
                        "identity_fields": {"decision_topic": "primary storage backend"},
                        "value_fields": {"canonical_value": "SQLite"},
                        "semantic": semantic,
                        "epistemic_status": "EXPLICIT",
                        "suggested_scope_refs": [{"namespace": "memoryos", "kind": "workspace", "id": "p1"}],
                        "evidence_refs": [atomic],
                        "atomic_evidence_ref": atomic,
                        "field_evidence_refs": bindings,
                        "confidence": 0.9,
                        "source_role": "user",
                    }
                ]
            }
        )
    )
    client = MemoryOSClient(str(tmp_path), memory_extractor=LLMMemoryExtractorBackend(provider))
    client.commit_agent_session(
        user_id="u1",
        session_id="s1",
        project_id="p1",
        messages=[{"role": "user", "content": text}],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
        async_commit=True,
    )
    results = client.search_context("SQLite", user_id="u1", project_id="p1", search_scope="project_decisions")
    assert results
    assert client.health()["memory_extractor"] == "ready"


def test_http_session_append_finalize_recall_flow(tmp_path) -> None:  # noqa: ANN001
    text = "Project rule: must run ruff before merge."
    provider = FakeMemoryModelProvider(_project_rule_response("e1", text, "p1", proposal_id="p-ruff"))
    client = MemoryOSClient(
        str(tmp_path),
        mode="server",
        memory_extractor=LLMMemoryExtractorBackend(provider),
    )
    app = MemoryOSASGI(
        client,
        api_token="test-token",
        trusted_context=TrustedRequestContext(
            tenant_id="default",
            user_id="u1",
            actor_kind="agent",
            actor_id="openclaw",
            capabilities=DEFAULT_AGENT_CAPABILITIES | frozenset({ATTEST_USER_INPUT}),
            allowed_workspace_ids=frozenset({"p1"}),
        ),
    )

    async def post(path: str, payload: dict) -> dict:
        sent = []
        incoming = iter([{"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}])

        async def receive():  # noqa: ANN202
            return next(incoming)

        async def send(message):  # noqa: ANN001, ANN202
            sent.append(message)

        await app(
            {
                "type": "http",
                "method": "POST",
                "path": path,
                "headers": [(b"authorization", b"Bearer test-token")],
            },
            receive,
            send,
        )
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
                "prompt": text,
            },
        )
    )
    finalized = asyncio.run(post(f"/v1/sessions/{event['session_key']}/finalize", {}))
    assert finalized["done"] is True
    assert client.search_context("ruff", user_id="u1", project_id="p1", search_scope="project_rules")
