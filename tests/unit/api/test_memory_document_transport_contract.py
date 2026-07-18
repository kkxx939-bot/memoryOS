from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from memoryos.api.cli.agent_hook_transport import AgentHookTransportClient
from memoryos.api.http.app import MemoryOSASGI
from memoryos.api.mcp.schemas import TOOL_INPUT_SCHEMAS
from memoryos.api.memory_contract import (
    MEMORY_COMMAND_REQUEST_SCHEMAS,
    MEMORY_COMMAND_RESPONSE_SCHEMAS,
    memory_request_schema,
    validate_memory_request,
    validate_memory_response,
)
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.sdk.http_client import HTTPMemoryOSClient
from memoryos.memory.documents.layout import user_memory_root
from memoryos.security.trusted_context import (
    AUTHORITATIVE_FORGET,
    AUTHORITATIVE_REMEMBER,
    TrustedRequestContext,
)


class _CaptureHTTPClient(HTTPMemoryOSClient):
    def __init__(self) -> None:
        super().__init__("http://memoryos.invalid")
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((method, path, payload))
        document = {
            "document_uri": "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "document_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "document_kind": "topic",
            "relative_path": "topics/database.md",
            "document_revision": 1,
            "source_digest": "a" * 64,
            "changed": True,
            "edit_summary": "test",
            "projection_status": "ENQUEUED",
        }
        if path == "/v1/memories/forget":
            return {**document, "mode": "SOFT_FORGET", "recoverable": True}
        if path == "/v1/memories/review":
            return {
                **document,
                "proposal_id": "proposal-1",
                "status": "REJECTED",
                "proposed_source_digest": "b" * 64,
                "proposed_diff_digest": "c" * 64,
            }
        if path == "/v1/memories/review/preview":
            return {
                "proposal_id": "proposal-1",
                "status": "PENDING",
                "document_uri": document["document_uri"],
                "document_id": document["document_id"],
                "document_kind": document["document_kind"],
                "relative_path": document["relative_path"],
                "source_digest": document["source_digest"],
                "proposed_source_digest": "b" * 64,
                "proposed_diff_digest": "c" * 64,
                "proposed_diff": "-old\n+new",
                "edit_summary": "preview",
            }
        if path == "/v1/memories/merge/propose":
            return {
                "proposal_id": "proposal-consolidation",
                "status": "PENDING",
                "document_uri": document["document_uri"],
                "document_id": document["document_id"],
                "document_kind": document["document_kind"],
                "relative_path": document["relative_path"],
                "source_digest": document["source_digest"],
                "proposed_source_digest": "b" * 64,
                "proposed_diff_digest": "c" * 64,
                "proposed_diff": "-target\n+target plus source",
                "edit_summary": "consolidation preview",
                "workflow_kind": "CONSOLIDATION",
                "consolidation_sources": [
                    {
                        "document_uri": document["document_uri"] + "-source",
                        "document_id": "source-1",
                        "relative_path": "knowledge/topics/source.md",
                        "source_digest": "a" * 64,
                        "size": 10,
                    }
                ],
            }
        if path in {"/v1/memories/merge", "/v1/memories/merge/resume"}:
            return {
                "saga_id": "mdconsolidate_1",
                "status": "AWAITING_TARGET_PROJECTION",
                "target_document_id": document["document_id"],
                "target_projection_generation": 1,
                "target_projection_confirmed": False,
                "soft_forgotten_document_ids": [],
                "pending_document_ids": ["source-1"],
            }
        if path.startswith("/v1/memories/history?"):
            return {
                "document_uri": document["document_uri"],
                "document_id": document["document_id"],
                "document_kind": document["document_kind"],
                "relative_path": document["relative_path"],
                "revisions": [],
            }
        return document


def test_mcp_memory_tools_use_the_transport_neutral_request_schemas() -> None:
    mapping = {
        "memoryos_adopt_memory_document": "adopt",
        "memoryos_remember": "remember",
        "memoryos_edit_memory_document": "edit",
        "memoryos_rename_memory_document": "rename",
        "memoryos_merge_memory_documents": "merge",
        "memoryos_propose_memory_consolidation": "merge_propose",
        "memoryos_resume_memory_consolidation": "merge_resume",
        "memoryos_forget": "forget",
        "memoryos_memory_history": "history",
        "memoryos_restore_memory_revision": "restore",
        "memoryos_review_memory_edit": "review",
        "memoryos_preview_memory_edit": "review_preview",
    }
    for tool_name, operation in mapping.items():
        assert TOOL_INPUT_SCHEMAS[tool_name] == memory_request_schema(operation)
        assert TOOL_INPUT_SCHEMAS[tool_name]["additionalProperties"] is False
    assert "edit" in MEMORY_COMMAND_REQUEST_SCHEMAS["rename"]["properties"]


def test_document_command_results_expose_only_document_native_fields() -> None:
    required = set(MEMORY_COMMAND_RESPONSE_SCHEMAS["remember"]["required"])
    assert required == {
        "document_uri",
        "document_id",
        "document_kind",
        "relative_path",
        "document_revision",
        "source_digest",
        "changed",
        "edit_summary",
        "projection_status",
    }
    for operation, schema in MEMORY_COMMAND_REQUEST_SCHEMAS.items():
        assert "user_id" not in schema["properties"], operation
        assert "tenant_id" not in schema["properties"], operation


def test_request_validator_rejects_removed_and_cross_mode_fields() -> None:
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_memory_request("remember", {"content": "value", "title": "old shape"})
    with pytest.raises(ValueError, match="whole-document"):
        validate_memory_request(
            "forget",
            {
                "document_uri": "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "mode": "HARD_ERASE",
                "section_anchor": "private",
            },
        )
    with pytest.raises(ValueError, match="requires corrected_edit"):
        validate_memory_request("review", {"proposal_id": "p1", "decision": "CORRECT"})
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_memory_request(
            "adopt",
            {
                "relative_path": "knowledge/topics/external.md",
                "expected_raw_sha256": "a" * 64,
                "assigned_document_id": "memdoc_" + "b" * 64,
            },
        )


def test_response_validator_json_normalizes_and_rejects_schema_drift() -> None:
    payload = {
        "document_uri": "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "document_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "document_kind": "topic",
        "relative_path": "knowledge/topics/database.md",
        "document_revision": 2,
        "source_digest": "a" * 64,
        "changed": True,
        "edit_summary": "hard erase",
        "projection_status": "ERASED",
        "mode": "HARD_ERASE",
        "recoverable": False,
        "pending_backends": ("derived.catalog", "derived.recall_traces"),
        "independent_evidence_retained": (
            "memoryos://user/u1/sessions/history/session-1",
        ),
    }

    normalized = validate_memory_response("forget", payload)

    assert normalized["pending_backends"] == ["derived.catalog", "derived.recall_traces"]
    assert normalized["independent_evidence_retained"] == [
        "memoryos://user/u1/sessions/history/session-1"
    ]
    assert isinstance(normalized["pending_backends"], list)
    with pytest.raises(ValueError, match="requires fields"):
        validate_memory_response("remember", {"document_uri": payload["document_uri"]})
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_memory_response(
            "remember",
            {key: value for key, value in payload.items() if key in MEMORY_COMMAND_RESPONSE_SCHEMAS["remember"]["properties"]}
            | {"legacy_status": "old"},
        )
    with pytest.raises(TypeError, match="non-JSON"):
        validate_memory_response(
            "forget",
            {**payload, "pending_backends": {"derived.catalog"}},
        )


def test_http_client_uses_exact_document_operation_paths_and_payloads() -> None:
    client = _CaptureHTTPClient()
    uri = "memoryos://user/u1/memory/documents/01ARZ3NDEKTSV4RRFFQ69G5FAV"
    digest = "a" * 64

    client.remember("SQLite", target_hint="topic:database")
    client.adopt_memory_document("knowledge/topics/external.md", digest)
    client.edit_memory_document(uri, "PostgreSQL", digest)
    client.rename_memory_document(
        uri,
        "knowledge/topics/postgresql.md",
        digest,
        edit="PostgreSQL renamed and edited",
    )
    client.merge_memory_documents(
        uri,
        "Merged body",
        digest,
        [{"document_uri": uri + "-source", "expected_digest": digest}],
    )
    client.propose_memory_consolidation(
        uri,
        "Proposed merged body",
        digest,
        [{"document_uri": uri + "-source", "expected_digest": digest}],
    )
    client.resume_memory_consolidation("mdconsolidate_1")
    client.forget(uri, mode="SOFT_FORGET", expected_digest=digest)
    client.restore_memory_revision(uri, 1, "")
    client.review_memory_edit("proposal-1", "REJECT")
    client.preview_memory_edit("proposal-1")

    assert [path for _method, path, _payload in client.calls] == [
        "/v1/memories/remember",
        "/v1/memories/adopt",
        "/v1/memories/edit",
        "/v1/memories/rename",
        "/v1/memories/merge",
        "/v1/memories/merge/propose",
        "/v1/memories/merge/resume",
        "/v1/memories/forget",
        "/v1/memories/restore",
        "/v1/memories/review",
        "/v1/memories/review/preview",
    ]
    assert client.calls[0][2] == {
        "content": "SQLite",
        "occurred_at": None,
        "target_hint": "topic:database",
        "expected_document_digest": None,
    }
    assert client.calls[1][2] == {
        "relative_path": "knowledge/topics/external.md",
        "expected_raw_sha256": digest,
    }
    assert client.calls[2][2] == {
        "document_uri": uri,
        "edit": "PostgreSQL",
        "expected_digest": digest,
    }
    assert client.calls[3][2] == {
        "document_uri": uri,
        "new_relative_path": "knowledge/topics/postgresql.md",
        "expected_digest": digest,
        "edit": "PostgreSQL renamed and edited",
    }


def test_agent_hook_remote_dispatch_preserves_rename_edit_and_consolidation_proposal() -> None:
    class _Remote:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def rename_memory_document(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("rename", kwargs))
            return {"status": "renamed"}

        def merge_memory_documents(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("merge", kwargs))
            return {"status": "AWAITING_TARGET_PROJECTION"}

        def propose_memory_consolidation(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("merge_propose", kwargs))
            return {"status": "PENDING"}

    remote = _Remote()
    transport = AgentHookTransportClient.__new__(AgentHookTransportClient)
    transport.remote = remote  # type: ignore[assignment]
    transport.server = None
    rename = {
        "document_uri": "memoryos://user/u1/memory/documents/doc-1",
        "new_relative_path": "knowledge/topics/new.md",
        "expected_digest": "a" * 64,
        "edit": "new body",
    }
    proposal = {
        "target_document_uri": rename["document_uri"],
        "merged_edit": "merged body",
        "expected_target_digest": "a" * 64,
        "source_documents": [
            {
                "document_uri": rename["document_uri"] + "-source",
                "expected_digest": "b" * 64,
            }
        ],
    }

    assert transport.call_tool("memoryos_rename_memory_document", rename) == {
        "status": "renamed"
    }
    assert transport.call_tool("memoryos_merge_memory_documents", proposal) == {
        "status": "AWAITING_TARGET_PROJECTION"
    }
    assert transport.call_tool("memoryos_propose_memory_consolidation", proposal) == {
        "status": "PENDING"
    }
    assert remote.calls == [
        ("rename", rename),
        ("merge", proposal),
        ("merge_propose", proposal),
    ]


def test_http_adopt_route_binds_identity_outside_the_payload() -> None:
    class _Ready:
        def require_ready(self) -> None:
            return None

    class _Client:
        tenant_id = "tenant-a"
        readiness = _Ready()
        agent_session_service = object()

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def adopt_memory_document(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            return {
                "document_uri": "memoryos://user/alice/memory/documents/memdoc_AAAAAAAAAAAAAAAA",
                "document_id": "memdoc_AAAAAAAAAAAAAAAA",
                "document_kind": "topic",
                "relative_path": "knowledge/topics/external.md",
                "document_revision": 1,
                "source_digest": "a" * 64,
                "changed": True,
                "edit_summary": "adopt",
                "projection_status": "ENQUEUED",
            }

    client = _Client()
    caller = TrustedRequestContext(
        tenant_id="tenant-a",
        user_id="alice",
        actor_kind="user",
        actor_id="alice",
        capabilities=frozenset({AUTHORITATIVE_REMEMBER}),
    )
    app = MemoryOSASGI(client, trusted_context=caller)  # type: ignore[arg-type]
    digest = "a" * 64

    response = app._dispatch(
        "POST",
        "/v1/memories/adopt",
        {
            "relative_path": "knowledge/topics/external.md",
            "expected_raw_sha256": digest,
        },
        {},
        caller,
    )

    assert response["document_uri"].startswith("memoryos://user/alice/")
    assert client.calls == [
        {
            "relative_path": "knowledge/topics/external.md",
            "expected_raw_sha256": digest,
            "tenant_id": "tenant-a",
            "caller": caller,
        }
    ]
    with pytest.raises(ValueError, match="unsupported fields"):
        app._dispatch(
            "POST",
            "/v1/memories/adopt",
            {
                "relative_path": "knowledge/topics/external.md",
                "expected_raw_sha256": digest,
                "owner_user_id": "mallory",
            },
            {},
            caller,
        )


def test_http_routes_preserve_rename_edit_and_copy_on_write_consolidation_contracts() -> None:
    class _Ready:
        def require_ready(self) -> None:
            return None

    class _Client:
        tenant_id = "tenant-a"
        readiness = _Ready()
        agent_session_service = object()

        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def rename_memory_document(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("rename", kwargs))
            return {
                "document_uri": kwargs["document_uri"],
                "document_id": "memdoc_AAAAAAAAAAAAAAAA",
                "document_kind": "topic",
                "relative_path": kwargs["new_relative_path"],
                "document_revision": 2,
                "source_digest": "b" * 64,
                "changed": True,
                "edit_summary": "rename and edit memory document",
                "projection_status": "ENQUEUED",
            }

        def propose_memory_consolidation(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("merge_propose", kwargs))
            source = kwargs["source_documents"][0]
            return {
                "proposal_id": "proposal-consolidation",
                "status": "PENDING",
                "document_uri": kwargs["target_document_uri"],
                "document_id": "memdoc_AAAAAAAAAAAAAAAA",
                "document_kind": "topic",
                "relative_path": "knowledge/topics/target.md",
                "source_digest": kwargs["expected_target_digest"],
                "proposed_source_digest": "b" * 64,
                "proposed_diff_digest": "c" * 64,
                "proposed_diff": "-old\n+merged",
                "edit_summary": "consolidation preview",
                "workflow_kind": "CONSOLIDATION",
                "consolidation_sources": [
                    {
                        "document_uri": source["document_uri"],
                        "document_id": "memdoc_BBBBBBBBBBBBBBBB",
                        "relative_path": "knowledge/topics/source.md",
                        "source_digest": source["expected_digest"],
                        "size": 10,
                    }
                ],
            }

        def merge_memory_documents(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("merge", kwargs))
            return {
                "saga_id": "memsaga_" + "e" * 64,
                "status": "AWAITING_TARGET_PROJECTION",
                "target_document_id": "memdoc_AAAAAAAAAAAAAAAA",
                "target_projection_generation": 2,
                "target_projection_confirmed": False,
                "soft_forgotten_document_ids": [],
                "pending_document_ids": ["memdoc_BBBBBBBBBBBBBBBB"],
            }

    caller = TrustedRequestContext(
        tenant_id="tenant-a",
        user_id="alice",
        actor_kind="user",
        actor_id="alice",
        capabilities=frozenset({AUTHORITATIVE_REMEMBER, AUTHORITATIVE_FORGET}),
    )
    client = _Client()
    app = MemoryOSASGI(client, trusted_context=caller)  # type: ignore[arg-type]
    uri = "memoryos://user/alice/memory/documents/memdoc_AAAAAAAAAAAAAAAA"
    rename_request = {
        "document_uri": uri,
        "new_relative_path": "knowledge/topics/renamed.md",
        "expected_digest": "a" * 64,
        "edit": "renamed body",
    }
    proposal_request = {
        "target_document_uri": uri,
        "merged_edit": "merged target body",
        "expected_target_digest": "a" * 64,
        "source_documents": [
            {
                "document_uri": "memoryos://user/alice/memory/documents/memdoc_BBBBBBBBBBBBBBBB",
                "expected_digest": "d" * 64,
            }
        ],
    }

    renamed = app._dispatch("POST", "/v1/memories/rename", rename_request, {}, caller)
    merged = app._dispatch("POST", "/v1/memories/merge", proposal_request, {}, caller)
    proposed = app._dispatch(
        "POST",
        "/v1/memories/merge/propose",
        proposal_request,
        {},
        caller,
    )

    assert renamed["edit_summary"] == "rename and edit memory document"
    assert merged["status"] == "AWAITING_TARGET_PROJECTION"
    assert proposed["workflow_kind"] == "CONSOLIDATION"
    assert client.calls == [
        ("rename", {**rename_request, "tenant_id": "tenant-a", "caller": caller}),
        ("merge", {**proposal_request, "tenant_id": "tenant-a", "caller": caller}),
        (
            "merge_propose",
            {**proposal_request, "tenant_id": "tenant-a", "caller": caller},
        ),
    ]


def test_local_sdk_adopt_uses_trusted_caller_root(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    caller = TrustedRequestContext(
        tenant_id="tenant-a",
        user_id="alice",
        actor_kind="user",
        actor_id="alice",
        capabilities=frozenset({AUTHORITATIVE_REMEMBER}),
    )
    relative_path = "knowledge/topics/sdk-note.md"
    raw = b"# SDK-created file\n\nNo hidden identity yet.\n"
    path = user_memory_root(tmp_path, caller.tenant_id, caller.user_id) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)

    result = client.adopt_memory_document(
        relative_path,
        hashlib.sha256(raw).hexdigest(),
        caller=caller,
    )

    assert result["document_uri"].startswith("memoryos://user/alice/memory/documents/")
    assert result["document_revision"] == 1
    assert result["projection_status"] == "ENQUEUED"
    assert client.queue_store.stats(queue_name="memory_projection")["pending"] == 1

    renamed = client.rename_memory_document(
        result["document_uri"],
        "knowledge/entities/sdk-note.md",
        result["source_digest"],
        edit="SDK rename and edit body",
        caller=caller,
    )

    assert renamed["document_id"] == result["document_id"]
    assert renamed["relative_path"] == "knowledge/entities/sdk-note.md"
    assert renamed["source_digest"] != result["source_digest"]
    assert renamed["edit_summary"] == "rename and edit memory document"
