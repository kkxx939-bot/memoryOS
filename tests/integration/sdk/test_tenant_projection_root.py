from __future__ import annotations

from pathlib import Path

from memoryos.api.sdk.client import MemoryOSClient


def test_nondefault_tenant_projector_and_retriever_share_only_injected_store(tmp_path: Path) -> None:
    tenant_a = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    tenant_b = MemoryOSClient(str(tmp_path), tenant_id="tenant-b")

    result_a = tenant_a.remember(
        user_id="u1",
        title="primary database",
        content="tenant A uses SQLite",
        memory_type="project_decision",
        project_id="memoryos",
        tenant_id="tenant-a",
    )
    result_b = tenant_b.remember(
        user_id="u1",
        title="primary database",
        content="tenant B uses PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        tenant_id="tenant-b",
    )
    claim_a = result_a["uri"]
    claim_b = result_b["uri"]

    store_a = tenant_a.context_db.projection_store
    store_b = tenant_b.context_db.projection_store
    assert store_a is not None and store_b is not None
    assert store_a.root == tmp_path / "tenants" / "tenant-a"
    assert store_b.root == tmp_path / "tenants" / "tenant-b"
    record_a = store_a.load_current(claim_a)
    record_b = store_b.load_current(claim_b)
    assert record_a is not None and record_b is not None
    assert record_a.input_effect_hash != record_b.input_effect_hash
    assert store_a.load_current(claim_b) is None
    assert store_b.load_current(claim_a) is None
    resolver_a = tenant_a._context_assembler().unified_retrieval.resolver
    resolver_b = tenant_b._context_assembler().unified_retrieval.resolver
    assert resolver_a.projection_store is store_a
    assert resolver_b.projection_store is store_b
    assert not (tmp_path / "system" / "projection-state").exists()

    recalled_a = tenant_a.search_context(
        "SQLite",
        user_id="u1",
        project_id="memoryos",
        tenant_id="tenant-a",
        query_intent="CURRENT",
    )
    recalled_b = tenant_b.search_context(
        "PostgreSQL",
        user_id="u1",
        project_id="memoryos",
        tenant_id="tenant-b",
        query_intent="CURRENT",
    )
    assert recalled_a[0]["uri"] == claim_a
    assert "SQLite" in recalled_a[0]["text"]
    assert recalled_b[0]["uri"] == claim_b
    assert "PostgreSQL" in recalled_b[0]["text"]
    assert tenant_a.search_context(
        "PostgreSQL",
        user_id="u1",
        project_id="memoryos",
        tenant_id="tenant-a",
        query_intent="CURRENT",
    ) == []
