"""本地单用户入口不再暴露多租户认证能力。"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from foundation.identity import LocalUserContext
from openApi.http.app import MemoryOSASGI, create_app
from openApi.http.config import HTTPServerConfig
from openApi.mcp.config import MCPServerConfig
from openApi.mcp.schemas import tool_definitions
from openApi.retrieval_contract import parse_retrieval_options, retrieval_options_json_schema
from openApi.sdk.client import MemoryOSClient
from openApi.sdk.http_client import HTTPMemoryOSClient


def test_local_context_uses_one_internal_storage_namespace() -> None:
    context = LocalUserContext(user_id="gulf", adapter_id="codex", workspace_id="project-a")

    assert context.tenant_id == "default"
    assert context.bind_read_workspace() == "project-a"
    assert context.retrieval_scope_keys() == frozenset(
        {"memoryos:principal:gulf", "memoryos:workspace:project-a"}
    )


def test_public_retrieval_contract_does_not_accept_tenant() -> None:
    assert "tenant_id" not in retrieval_options_json_schema()["properties"]
    with pytest.raises(ValueError, match="tenant_id is not a public retrieval option"):
        parse_retrieval_options({"tenant_id": "tenant-a"})


def test_http_and_mcp_configs_have_no_access_token_or_capability_grants(tmp_path: Path) -> None:
    http = HTTPServerConfig(root=str(tmp_path))
    mcp = MCPServerConfig(root=str(tmp_path), user_id="gulf")

    assert not hasattr(http, "api_token")
    assert not hasattr(mcp, "capabilities")
    assert not hasattr(mcp, "authorized_scope_keys")
    assert "memoryos_remember" in {item["name"] for item in tool_definitions(mcp)}


def test_http_config_rejects_non_loopback_listener(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        HTTPServerConfig(root=str(tmp_path), host="0.0.0.0")


def test_public_entrypoints_do_not_restore_multi_user_parameters() -> None:
    prohibited = {
        "api_token",
        "capabilities",
        "tenant_id",
        "trusted_context",
    }
    entrypoints = (
        MemoryOSClient,
        MemoryOSClient.search_context,
        MemoryOSClient.assemble_context,
        MemoryOSClient.remember,
        MemoryOSClient.commit_agent_session,
        MemoryOSClient.archive_search,
        HTTPMemoryOSClient,
        MemoryOSASGI,
        create_app,
    )

    for entrypoint in entrypoints:
        assert prohibited.isdisjoint(inspect.signature(entrypoint).parameters)
    assert "user_id" not in inspect.signature(HTTPMemoryOSClient).parameters


def test_legacy_security_packages_are_removed() -> None:
    root = Path(__file__).resolve().parents[3]

    assert not (root / "memoryos" / "security").exists()
    assert not (root / "foundation" / "security").exists()
    assert not (root / "memory" / "security").exists()
    assert not (root / "openApi" / "trusted_context.py").exists()
