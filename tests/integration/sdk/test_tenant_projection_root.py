from __future__ import annotations

from pathlib import Path

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.trusted_context import AUTHORITATIVE_REMEMBER, TrustedRequestContext
from memoryos.memory.documents.layout import user_memory_root


def _caller(tenant_id: str) -> TrustedRequestContext:
    return TrustedRequestContext(
        tenant_id=tenant_id,
        user_id="u1",
        actor_kind="user",
        actor_id="u1",
        capabilities=frozenset({AUTHORITATIVE_REMEMBER}),
    )


def test_nondefault_tenants_write_to_isolated_markdown_roots(tmp_path: Path) -> None:
    tenant_a = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    tenant_b = MemoryOSClient(str(tmp_path), tenant_id="tenant-b")

    result_a = tenant_a.remember(
        "tenant A uses SQLite",
        target_hint="topic:primary database",
        caller=_caller("tenant-a"),
    )
    result_b = tenant_b.remember(
        "tenant B uses PostgreSQL",
        target_hint="topic:primary database",
        caller=_caller("tenant-b"),
    )

    root_a = user_memory_root(tmp_path, "tenant-a", "u1")
    root_b = user_memory_root(tmp_path, "tenant-b", "u1")
    raw_a = (root_a / result_a["relative_path"]).read_text()
    raw_b = (root_b / result_b["relative_path"]).read_text()
    assert "SQLite" in raw_a and "PostgreSQL" not in raw_a
    assert "PostgreSQL" in raw_b and "SQLite" not in raw_b
    assert result_a["source_digest"] != result_b["source_digest"]
    assert root_a != root_b
