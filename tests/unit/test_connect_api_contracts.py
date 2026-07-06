from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from memoryos.api.http.app import handle
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.connect import CapabilityProfile, ConnectMetadata, PipelineMode
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.context_assembler import ContextAssembler
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.source_store import IndexHit
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.skill.tool_registry import ToolRegistry


class FakeContextDB:
    def __init__(self, hits: list[IndexHit] | None = None) -> None:
        self.hits = hits or []
        self.search_calls: list[dict] = []
        self.committed: list[tuple[SessionArchive, bool]] = []

    def search(self, query: str, *, owner_user_id=None, context_type=None, limit: int = 10):  # noqa: ANN001
        self.search_calls.append(
            {"query": query, "owner_user_id": owner_user_id, "context_type": context_type, "limit": limit}
        )
        return self.hits[:limit]

    def read_object(self, uri: str) -> ContextObject:
        raise FileNotFoundError(uri)

    def commit_session(self, archive: SessionArchive, *, async_commit: bool = True):  # noqa: ANN201
        self.committed.append((archive, async_commit))
        return {"status": "done"}


class FailingEngine:
    def __init__(self) -> None:
        self.called = False

    def process(self, request, policies=None):  # noqa: ANN001, ANN201
        self.called = True
        raise AssertionError("PredictionEngine must not be called")


class FailingExecutor:
    def __init__(self) -> None:
        self.called = False

    def execute(self, decision, action_context):  # noqa: ANN001, ANN201
        self.called = True
        raise AssertionError("ActionExecutor must not be called")


class ReturningEngine:
    def __init__(self, result) -> None:  # noqa: ANN001
        self.result = result
        self.called = False

    def process(self, request, policies=None):  # noqa: ANN001, ANN201
        self.called = True
        return self.result


def _client(context_db: FakeContextDB | None = None) -> Any:
    client: Any = object.__new__(MemoryOSClient)
    client.context_db = context_db or FakeContextDB()
    client.engine = FailingEngine()
    client.executor = FailingExecutor()
    return client


def _request(connect_metadata: dict | None = None) -> PredictionRequest:
    return PredictionRequest(
        user_id="u1",
        episode_id="s1",
        observation="hello",
        available_actions=["ask_user"],
        connect_metadata=connect_metadata or {},
    )


def test_prediction_request_connect_metadata_compatibility_and_frozen() -> None:
    legacy = _request()
    modern = _request({"adapter_id": "codex"})

    assert legacy.connect_metadata == {}
    assert modern.connect_metadata == {"adapter_id": "codex"}
    with pytest.raises(FrozenInstanceError):
        legacy.user_id = "u2"  # type: ignore[misc]


def test_session_archive_metadata_manifest_defaults() -> None:
    archive = SessionArchive(user_id="u1", session_id="s1", archive_uri="memoryos://user/u1/sessions/history/s1")
    metadata_archive = SessionArchive(
        user_id="u1",
        session_id="s2",
        archive_uri="memoryos://user/u1/sessions/history/s2",
        metadata={"connect": {"adapter_id": "codex"}},
    )

    assert archive.metadata == {}
    assert metadata_archive.manifest()["metadata"] == {"connect": {"adapter_id": "codex"}}


def test_context_object_preserves_connect_metadata_roundtrip() -> None:
    connect = ConnectMetadata.default_agent("codex").to_dict()
    obj = ContextObject(
        uri="memoryos://user/u1/memories/anchors/m1",
        context_type=ContextType.MEMORY,
        title="Memory",
        metadata={"connect": connect},
    )

    assert ContextObject.from_dict(obj.to_dict()).metadata["connect"] == connect


def test_context_reduction_sdk_does_not_call_prediction_or_executor() -> None:
    context_db = FakeContextDB(
        [
            IndexHit(
                uri="memoryos://user/u1/memories/anchors/m1",
                score=1.0,
                context_type="memory",
                title="MemoryOS MCP server",
                metadata={"connect": ConnectMetadata.default_agent("codex").to_dict()},
            )
        ]
    )
    client = _client(context_db)

    results = client.search_context("MCP", user_id="u1", limit=1, connect_metadata={"adapter_id": "codex"})
    assembled = client.assemble_context("MCP", user_id="u1", token_budget=200, limit=1)
    client.commit_agent_session(
        user_id="u1",
        session_id="s1",
        messages=[{"role": "user", "content": "remember this"}],
        used_contexts=results,
        tool_results=[],
        connect_metadata={"adapter_id": "codex"},
        async_commit=False,
    )

    assert results[0]["uri"] == "memoryos://user/u1/memories/anchors/m1"
    assert assembled["source_uris"] == ["memoryos://user/u1/memories/anchors/m1"]
    assert assembled["packed_context"]
    assert context_db.search_calls[0]["limit"] == 50
    archive, async_commit = context_db.committed[0]
    assert async_commit is False
    assert archive.metadata["connect"]["adapter_id"] == "codex"
    assert archive.predictions == []
    assert archive.action_results == []
    assert client.engine.called is False
    assert client.executor.called is False


def test_connect_filters_from_metadata_only_uses_explicit_simple_fields() -> None:
    client = _client()
    full_metadata = ConnectMetadata(
        adapter_id="codex",
        run_mode=PipelineMode.ACTION_CAPABLE,
        world_domain="digital",
        source_kind="terminal",
        modality=("text", "tool"),
        capabilities=CapabilityProfile(can_predict_behavior=True, can_execute_action=True),
        extra={"workspace": "/repo"},
    ).to_dict()

    assert client._connect_filters_from_metadata(None) == {}
    assert client._connect_filters_from_metadata({}) == {}
    assert client._connect_filters_from_metadata({"adapter_id": "codex"}) == {"adapter_id": "codex"}
    assert client._connect_filters_from_metadata(full_metadata) == {
        "connect_type": "agent",
        "adapter_id": "codex",
        "run_mode": "action_capable",
        "world_domain": "digital",
        "source_kind": "terminal",
    }
    assert "capabilities" not in client._connect_filters_from_metadata(full_metadata)
    assert "modality" not in client._connect_filters_from_metadata(full_metadata)
    assert "extra" not in client._connect_filters_from_metadata(full_metadata)


def test_search_and_assemble_context_apply_connect_filters_without_behavior_or_action() -> None:
    claude_metadata = ConnectMetadata(adapter_id="claude_code", source_kind="terminal").to_dict()
    codex_metadata = ConnectMetadata(adapter_id="codex", source_kind="terminal").to_dict()
    context_db = FakeContextDB(
        [
            IndexHit(
                uri="memoryos://user/u1/memories/anchors/claude",
                score=2.0,
                context_type="memory",
                title="Claude terminal context",
                metadata={"connect": claude_metadata},
            ),
            IndexHit(
                uri="memoryos://user/u1/memories/anchors/codex",
                score=1.0,
                context_type="memory",
                title="Codex terminal context",
                metadata={"connect": codex_metadata},
            ),
        ]
    )
    client = _client(context_db)

    assert client._connect_filters_from_metadata(claude_metadata) == {
        "connect_type": "agent",
        "adapter_id": "claude_code",
        "run_mode": "context_reduction",
        "world_domain": "digital",
        "source_kind": "terminal",
    }
    search_results = client.search_context("terminal", connect_metadata=claude_metadata)
    assembled = client.assemble_context("terminal", connect_metadata=claude_metadata, token_budget=500)
    unfiltered_results = client.search_context("terminal")

    assert [item["uri"] for item in search_results] == ["memoryos://user/u1/memories/anchors/claude"]
    assert assembled["source_uris"] == ["memoryos://user/u1/memories/anchors/claude"]
    assert {item["uri"] for item in unfiltered_results} == {
        "memoryos://user/u1/memories/anchors/claude",
        "memoryos://user/u1/memories/anchors/codex",
    }
    assert client.engine.called is False
    assert client.executor.called is False


def test_search_context_adapter_id_filter_does_not_apply_default_fields() -> None:
    codex_metadata = ConnectMetadata(adapter_id="codex", source_kind="terminal").to_dict()
    context_db = FakeContextDB(
        [
            IndexHit(
                uri="memoryos://user/u1/memories/anchors/codex",
                score=1.0,
                context_type="memory",
                title="Codex terminal context",
                metadata={"connect": codex_metadata},
            )
        ]
    )
    client = _client(context_db)

    codex_results = client.search_context("terminal", connect_metadata={"adapter_id": "codex"})
    claude_results = client.search_context("terminal", connect_metadata={"adapter_id": "claude_code"})
    unfiltered_results = client.search_context("terminal")

    assert [item["uri"] for item in codex_results] == ["memoryos://user/u1/memories/anchors/codex"]
    assert claude_results == []
    assert [item["uri"] for item in unfiltered_results] == ["memoryos://user/u1/memories/anchors/codex"]
    assert client.engine.called is False
    assert client.executor.called is False


def test_assemble_context_passes_explicit_connect_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class RecordingAssembler:
        def __init__(self, context_db: FakeContextDB) -> None:
            self.context_db = context_db

        def assemble(
            self,
            query: str,
            *,
            user_id: str | None = None,
            token_budget: int = 2000,
            context_types: list[object] | None = None,
            limit: int = 20,
            connect_metadata: dict[str, Any] | None = None,
            connect_filters: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            calls.append(
                {
                    "query": query,
                    "user_id": user_id,
                    "token_budget": token_budget,
                    "context_types": context_types,
                    "limit": limit,
                    "connect_metadata": connect_metadata,
                    "connect_filters": connect_filters,
                }
            )
            return {
                "query": query,
                "token_budget": token_budget,
                "contexts": [],
                "packed_context": "",
                "source_uris": [],
                "dropped_contexts": [],
                "connect_metadata": dict(connect_metadata or {}),
            }

    monkeypatch.setattr("memoryos.api.sdk.client.ContextAssembler", RecordingAssembler)
    client = _client()

    assembled = client.assemble_context("terminal", connect_metadata={"adapter_id": "codex"}, token_budget=500)

    assert assembled["connect_metadata"]["adapter_id"] == "codex"
    assert calls[0]["connect_filters"] == {"adapter_id": "codex"}
    assert client.engine.called is False
    assert client.executor.called is False


def test_context_assembler_connect_filter_simple_fields() -> None:
    claude_connect = ConnectMetadata(adapter_id="claude_code", source_kind="terminal").to_dict()
    codex_connect = ConnectMetadata(adapter_id="codex", source_kind="terminal").to_dict()
    context_db = FakeContextDB(
        [
            IndexHit(
                uri="memoryos://user/u1/memories/anchors/claude",
                score=2.0,
                context_type="memory",
                title="Claude context",
                metadata={"connect": claude_connect},
            ),
            IndexHit(
                uri="memoryos://user/u1/memories/anchors/codex",
                score=1.0,
                context_type="memory",
                title="Codex context",
                metadata={"connect": codex_connect},
            ),
        ]
    )
    assembler = ContextAssembler(cast(Any, context_db))

    matched = assembler.search(
        "context",
        connect_filters={
            "connect_type": "agent",
            "adapter_id": "claude_code",
            "run_mode": "context_reduction",
            "world_domain": "digital",
            "source_kind": "terminal",
        },
    )
    adapter_miss = assembler.search("context", connect_filters={"adapter_id": "openclaw"})
    domain_miss = assembler.search("context", connect_filters={"world_domain": "physical"})
    ignored_complex_filters = assembler.search(
        "context",
        connect_filters={"capabilities": {"can_predict_behavior": True}, "modality": ["text"], "extra": {"x": "y"}},
    )

    assert [item["uri"] for item in matched] == ["memoryos://user/u1/memories/anchors/claude"]
    assert adapter_miss == []
    assert domain_miss == []
    assert {item["uri"] for item in ignored_complex_filters} == {
        "memoryos://user/u1/memories/anchors/claude",
        "memoryos://user/u1/memories/anchors/codex",
    }


def test_context_assembler_connect_filter_overfetches_before_limit_slice() -> None:
    claude_connect = ConnectMetadata(adapter_id="claude_code", source_kind="terminal").to_dict()
    codex_connect = ConnectMetadata(adapter_id="codex", source_kind="terminal").to_dict()
    context_db = FakeContextDB(
        [
            IndexHit(
                uri="memoryos://user/u1/memories/anchors/claude",
                score=2.0,
                context_type="memory",
                title="Claude context",
                metadata={"connect": claude_connect},
            ),
            IndexHit(
                uri="memoryos://user/u1/memories/anchors/codex",
                score=1.0,
                context_type="memory",
                title="Codex context",
                metadata={"connect": codex_connect},
            ),
        ]
    )
    assembler = ContextAssembler(cast(Any, context_db))

    matched = assembler.search("context", limit=1, connect_filters={"adapter_id": "codex"})

    assert [item["uri"] for item in matched] == ["memoryos://user/u1/memories/anchors/codex"]
    assert context_db.search_calls[0]["limit"] == 50


def test_assemble_context_empty_results_are_stable() -> None:
    assembled = _client(FakeContextDB()).assemble_context("missing", token_budget=100)

    assert assembled["contexts"] == []
    assert assembled["packed_context"] == ""
    assert assembled["source_uris"] == []
    assert assembled["dropped_contexts"] == []


def test_predict_rejects_explicit_context_reduction_metadata() -> None:
    client = _client()

    with pytest.raises(ValueError):
        client.predict(_request(ConnectMetadata.default_agent("codex").to_dict()))
    assert client.engine.called is False


def test_predict_rejects_missing_behavior_capability() -> None:
    metadata = ConnectMetadata(
        run_mode=PipelineMode.ACTION_CAPABLE,
        capabilities=CapabilityProfile(can_predict_behavior=False),
    ).to_dict()
    client = _client()

    with pytest.raises(ValueError):
        client.predict(_request(metadata))
    assert client.engine.called is False


def test_predict_without_connect_metadata_keeps_legacy_behavior() -> None:
    client = _client()
    client.engine = ReturningEngine({"ok": True})

    assert client.predict(_request()) == {"ok": True}
    assert client.engine.called is True


def test_process_observation_rejects_missing_execute_capability_before_executor() -> None:
    metadata = ConnectMetadata(
        run_mode=PipelineMode.ACTION_CAPABLE,
        capabilities=CapabilityProfile(can_predict_behavior=True, can_execute_action=False),
    ).to_dict()
    client = _client()

    with pytest.raises(PermissionError):
        client.process_observation(_request(metadata))
    assert client.executor.called is False


class FakeExternalClient:
    def __init__(self) -> None:
        self.predict_called = False
        self.search_called = False

    def predict(self, request, policies=None):  # noqa: ANN001, ANN201
        self.predict_called = True

        class Result:
            def to_dict(self) -> dict:
                return {"episode_id": request.episode_id}

        return Result()

    def search_context(self, query, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.search_called = True
        return [{"uri": "memoryos://x", "score": 1.0}]

    def assemble_context(self, query, **kwargs):  # noqa: ANN001, ANN003, ANN201
        return {"query": query, "packed_context": "ctx"}

    def commit_agent_session(self, **kwargs) -> None:  # noqa: ANN003
        self.committed = kwargs


def test_http_routes_and_errors() -> None:
    client = FakeExternalClient()

    typed_client = cast(MemoryOSClient, client)

    assert handle("POST /predict", typed_client, {"request": _request().__dict__}) == {"episode_id": "s1"}
    assert handle("POST /context/search", typed_client, {"query": "memoryOS"})["results"][0]["uri"] == "memoryos://x"
    assert handle("POST /context/assemble", typed_client, {"query": "memoryOS"})["packed_context"] == "ctx"
    assert handle("POST /sessions/commit", typed_client, {"user_id": "u1", "session_id": "s1"}) == {"status": "accepted"}
    assert client.predict_called is True
    assert client.search_called is True
    with pytest.raises(ValueError, match="requires non-empty string field: query"):
        handle("POST /context/search", typed_client, {})


def test_mcp_routes_and_unknown_tool() -> None:
    server = MemoryOSMCPServer(cast(MemoryOSClient, FakeExternalClient()))

    assert server.call_tool("memoryos_predict", {"request": _request().__dict__})["error"]["code"] == "PERMISSION_DENIED"
    assert server.call_tool("memoryos_search_context", {"query": "memoryOS"})["results"]
    assert server.call_tool("memoryos_assemble_context", {"query": "memoryOS"})["packed_context"] == "ctx"
    assert server.call_tool("memoryos_commit_session", {"user_id": "u1", "session_id": "s1"})["status"] == "accepted"
    assert server.call_tool("unknown", {})["error"]["code"] == "VALIDATION_ERROR"


def test_tool_registry_metadata_and_legacy_behavior() -> None:
    registry = ToolRegistry()
    registry.register("echo", lambda args: {"ok": True, **args})
    registry.register(
        "robot.move",
        lambda args: {"ok": True, **args},
        input_schema={"type": "object", "required": ["direction"], "properties": {"direction": {"type": "string"}}},
        metadata={
            "allowed_connect_types": ["embodied"],
            "allowed_adapter_ids": ["reachy_mini"],
            "requires_confirmation": True,
            "risk_level": "medium",
        },
    )

    assert registry.can_execute("echo")
    assert registry.execute("echo", {"value": 1}) == {"ok": True, "value": 1}
    metadata = registry.metadata("robot.move")
    metadata["risk_level"] = "low"
    assert registry.metadata("robot.move")["risk_level"] == "medium"
    assert registry.execute("robot.move", {"direction": "left"})["ok"] is True
