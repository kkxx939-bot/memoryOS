from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_client_initializes_with_runtime_container(tmp_path) -> None:
    from memoryos.api.http.app import MemoryOSASGI
    from memoryos.api.sdk.client import MemoryOSClient
    from memoryos.runtime import RuntimeConfig, build_runtime_container

    client = MemoryOSClient(str(tmp_path / "client"))
    container = build_runtime_container(RuntimeConfig(root=str(tmp_path / "runtime")))
    app = MemoryOSASGI(client)

    assert client.context_db is not None
    assert client.engine is not None
    assert client.agent_session_service is not None
    assert app.sessions is client.agent_session_service
    assert container.context_db is not None
    assert container.agent_session_service is not None


def test_memory_has_single_lifecycle_and_service_paths() -> None:
    from memoryos.memory.lifecycle import MemoryCoolingPolicy
    from memoryos.memory.service import MemoryUpdater

    root = Path(__file__).resolve().parents[3]
    assert MemoryCoolingPolicy.__name__ == "MemoryCoolingPolicy"
    assert MemoryUpdater.__name__ == "MemoryUpdater"
    assert not (root / "memoryos" / "memory" / "update" / "__init__.py").exists()
    assert not (root / "memoryos" / "memory" / "update" / "memory_cooling.py").exists()
    assert not (root / "memoryos" / "memory" / "update" / "memory_updater.py").exists()


def test_contextdb_boundary_imports() -> None:
    from memoryos.contextdb.context_db import ContextDB
    from memoryos.contextdb.resource import ResourceImporter
    from memoryos.contextdb.retrieval import ContextSelector
    from memoryos.contextdb.session import SessionArchive, SessionCommitService
    from memoryos.contextdb.skill import SkillRegistry
    from memoryos.contextdb.store import IndexStore, SourceStore
    from memoryos.contextdb.transaction import RecoveryService

    assert ContextDB.__name__ == "ContextDB"
    assert ContextSelector.__name__ == "ContextSelector"
    assert IndexStore.__name__ == "IndexStore"
    assert RecoveryService.__name__ == "RecoveryService"
    assert ResourceImporter.__name__ == "ResourceImporter"
    assert SessionArchive.__name__ == "SessionArchive"
    assert SessionCommitService.__name__ == "SessionCommitService"
    assert SkillRegistry.__name__ == "SkillRegistry"
    assert SourceStore.__name__ == "SourceStore"


def test_online_retrieval_exports_only_the_unified_product_boundary() -> None:
    import memoryos.contextdb.retrieval as retrieval
    import memoryos.contextdb.retrieval.hybrid_search as internal_hybrid
    import memoryos.memory.canonical as canonical
    from memoryos.contextdb.retrieval.hierarchical_retriever import OfflineHierarchicalRetriever
    from memoryos.memory.canonical.retrieval import OfflineCanonicalMemoryRetriever

    assert not hasattr(retrieval, "HybridSearch")
    assert not hasattr(retrieval, "HierarchicalRetriever")
    assert internal_hybrid.__all__ == []
    assert not hasattr(canonical, "CanonicalMemoryRetriever")
    assert canonical.OfflineCanonicalMemoryRetriever is OfflineCanonicalMemoryRetriever

    try:
        OfflineCanonicalMemoryRetriever(None, None, offline_admin=False)  # type: ignore[arg-type]
    except PermissionError:
        pass
    else:  # pragma: no cover - protects a security boundary, not a normal branch.
        raise AssertionError("offline canonical snapshot reader accepted an online caller")

    try:
        OfflineHierarchicalRetriever(None, offline_admin=False)  # type: ignore[arg-type]
    except PermissionError:
        pass
    else:  # pragma: no cover - protects a security boundary, not a normal branch.
        raise AssertionError("offline hierarchical reader accepted an online caller")


def test_product_retrieval_modules_contain_no_global_scan_or_snapshot_call() -> None:
    root = Path(__file__).resolve().parents[3]
    product_modules = (
        "memoryos/api/sdk/client.py",
        "memoryos/api/http/app.py",
        "memoryos/api/mcp/tools.py",
        "memoryos/contextdb/retrieval/context_assembler.py",
        "memoryos/contextdb/retrieval/service.py",
        "memoryos/contextdb/retrieval/orchestrator.py",
        "memoryos/contextdb/retrieval/candidate_generator.py",
        "memoryos/contextdb/retrieval/canonical_resolver.py",
    )
    forbidden_calls = (
        "capture_committed_canonical_snapshot(",
        ".list_objects(",
        ".vector_uris(",
        ".glob(",
        ".rglob(",
    )

    for relative_path in product_modules:
        source = (root / relative_path).read_text(encoding="utf-8")
        for forbidden_call in forbidden_calls:
            assert forbidden_call not in source, f"{relative_path} contains {forbidden_call}"

    offline_source = (root / "memoryos/memory/canonical/retrieval.py").read_text(encoding="utf-8")
    assert "class OfflineCanonicalMemoryRetriever" in offline_source
    assert "capture_committed_canonical_snapshot(" in offline_source


def test_provider_boundary_imports() -> None:
    from memoryos.contextdb.store.vector_store import VectorStore
    from memoryos.providers import ChatProvider, EmbeddingProvider, HashingEmbeddingProvider
    from memoryos.providers.llm import ChatRequest

    assert ChatProvider.__name__ == "ChatProvider"
    assert ChatRequest.__name__ == "ChatRequest"
    assert EmbeddingProvider.__name__ == "EmbeddingProvider"
    assert HashingEmbeddingProvider.__name__ == "HashingEmbeddingProvider"
    assert VectorStore.__name__ == "VectorStore"


def test_example_main_runs_without_import_errors() -> None:
    root = Path(__file__).resolve().parents[3]

    subprocess.run([sys.executable, str(root / "examples" / "main.py")], check=True, cwd=root)


def test_context_assembly_smoke_benchmark_runs() -> None:
    root = Path(__file__).resolve().parents[3]
    script = root / "benchmark" / "smoke" / "context_assembly_smoke.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
