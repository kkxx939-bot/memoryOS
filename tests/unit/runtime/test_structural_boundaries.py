from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_runtime_build_and_start_are_separate(tmp_path) -> None:
    from foundation.readiness import RuntimeReadinessState
    from runtime import RuntimeBuilder, RuntimeConfig

    runtime = RuntimeBuilder(RuntimeConfig(root=str(tmp_path))).build()

    assert runtime.readiness.state is RuntimeReadinessState.RECOVERING
    assert runtime.stores.source is not None
    assert runtime.memory.command_service is not None
    assert not hasattr(runtime, "source_store")
    assert not hasattr(runtime, "memory_command_service")

    report = runtime.start()

    assert report.ready is True
    assert runtime.readiness.state is RuntimeReadinessState.READY


def test_runtime_config_does_not_hold_live_dependencies(tmp_path) -> None:
    from runtime import RuntimeConfig

    config = RuntimeConfig(root=str(tmp_path))

    for name in ("source_store", "vector_store", "embedding", "reranker", "memory_extractor"):
        assert not hasattr(config, name)


def test_client_initializes_with_runtime_container(tmp_path) -> None:
    from openApi.http.app import MemoryOSASGI
    from openApi.sdk.client import MemoryOSClient
    from runtime.config import RuntimeConfig
    from tests.support.runtime import build_test_runtime

    client = MemoryOSClient(str(tmp_path / "client"))
    container = build_test_runtime(RuntimeConfig(root=str(tmp_path / "runtime")))
    app = MemoryOSASGI(client)

    assert client.runtime.context.facade is not None
    assert not hasattr(client, "context_db")
    assert client.runtime.context.administration_service is not None
    assert client.runtime.context.lifecycle_service is not None
    assert client.runtime.policy.engine is not None
    assert client.runtime.agent.session_service is not None
    assert app.sessions is client.runtime.agent.session_service
    assert container.context.facade is not None
    assert not hasattr(container, "context_db")
    assert container.context.administration_service is not None
    assert container.context.lifecycle_service is not None
    assert container.agent.session_service is not None
    assert container.memory.consolidator.saga_store is container.memory.consolidation_store
    assert container.memory.command_service.consolidator is container.memory.consolidator
    assert client.runtime.memory.consolidator is client.runtime.memory.command_service.consolidator


def test_memory_public_api_is_document_owned() -> None:
    from memory.commit import DocumentCommitResult, MemoryDocumentCommitter
    from memory.core import MemoryDocument, MemoryEditProposal

    root = Path(__file__).resolve().parents[3]
    assert MemoryDocument.__name__ == "MemoryDocument"
    assert MemoryDocumentCommitter.__name__ == "MemoryDocumentCommitter"
    assert MemoryEditProposal.__name__ == "MemoryEditProposal"
    assert DocumentCommitResult.__name__ == "DocumentCommitResult"
    assert not (root / "memoryos" / "memory").exists()
    assert not (root / "memoryos" / "runtime").exists()


def test_contextdb_boundary_imports() -> None:
    from infrastructure.context.facade import ContextDB
    from infrastructure.context.selection import ContextSelector
    from infrastructure.store.contracts import IndexStore, SourceStore
    from memory.commit import SessionCommitService
    from pre.session import SessionArchive
    from transaction.commit.recovery import RecoveryService

    assert ContextDB.__name__ == "ContextDB"
    assert ContextSelector.__name__ == "ContextSelector"
    assert IndexStore.__name__ == "IndexStore"
    assert RecoveryService.__name__ == "RecoveryService"
    assert SessionArchive.__name__ == "SessionArchive"
    assert SessionCommitService.__name__ == "SessionCommitService"
    assert SourceStore.__name__ == "SourceStore"


def test_online_retrieval_exports_only_the_unified_product_boundary() -> None:
    import infrastructure.context.retrieval as retrieval
    import infrastructure.context.retrieval.hybrid_search as internal_hybrid

    assert not hasattr(retrieval, "HybridSearch")
    assert not hasattr(retrieval, "HierarchicalRetriever")
    assert internal_hybrid.__all__ == []
    root = Path(__file__).resolve().parents[3]
    assert not (root / "memoryos" / "contextdb" / "retrieval" / "hierarchical_retriever.py").exists()


def test_product_retrieval_modules_contain_no_global_scan_or_snapshot_call() -> None:
    root = Path(__file__).resolve().parents[3]
    product_modules = (
        "openApi/sdk/client.py",
        "openApi/http/app.py",
        "openApi/mcp/tools.py",
        "infrastructure/context/query_service.py",
        "infrastructure/context/orchestrator.py",
        "infrastructure/context/candidate/generator.py",
        "infrastructure/context/candidate/vector.py",
        "infrastructure/context/candidate/relation.py",
    )
    forbidden_calls = (
        ".list_objects(",
        ".vector_uris(",
        ".glob(",
        ".rglob(",
    )

    for relative_path in product_modules:
        source = (root / relative_path).read_text(encoding="utf-8")
        for forbidden_call in forbidden_calls:
            assert forbidden_call not in source, f"{relative_path} contains {forbidden_call}"

def test_provider_boundary_imports() -> None:
    from infrastructure.context.retrieval.embedding import EmbeddingProvider
    from infrastructure.model import ChatRequest, ModelProvider
    from infrastructure.store.contracts.vector import VectorStore

    assert ModelProvider.__name__ == "ModelProvider"
    assert ChatRequest.__name__ == "ChatRequest"
    assert EmbeddingProvider.__name__ == "EmbeddingProvider"
    assert VectorStore.__name__ == "VectorStore"


def test_context_assembly_smoke_benchmark_runs() -> None:
    root = Path(__file__).resolve().parents[3]
    script = root / "tests" / "benchmark" / "smoke" / "context_assembly_smoke.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
