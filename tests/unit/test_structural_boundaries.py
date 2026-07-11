from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_client_initializes_with_runtime_container(tmp_path) -> None:
    from memoryos.api.sdk.client import MemoryOSClient
    from memoryos.runtime import RuntimeConfig, build_runtime_container

    client = MemoryOSClient(str(tmp_path / "client"))
    container = build_runtime_container(RuntimeConfig(root=str(tmp_path / "runtime")))

    assert client.context_db is not None
    assert client.engine is not None
    assert container.context_db is not None


def test_memory_has_single_lifecycle_and_service_paths() -> None:
    from memoryos.memory.lifecycle import MemoryCoolingPolicy
    from memoryos.memory.service import MemoryUpdater

    root = Path(__file__).resolve().parents[2]
    assert MemoryCoolingPolicy.__name__ == "MemoryCoolingPolicy"
    assert MemoryUpdater.__name__ == "MemoryUpdater"
    assert not (root / "memoryos" / "memory" / "update" / "__init__.py").exists()
    assert not (root / "memoryos" / "memory" / "update" / "memory_cooling.py").exists()
    assert not (root / "memoryos" / "memory" / "update" / "memory_updater.py").exists()


def test_contextdb_boundary_imports() -> None:
    from memoryos.contextdb.context_db import ContextDB
    from memoryos.contextdb.resource import ResourceImporter
    from memoryos.contextdb.retrieval import ContextSelector, HybridSearch
    from memoryos.contextdb.session import SessionArchive, SessionCommitService
    from memoryos.contextdb.skill import SkillRegistry
    from memoryos.contextdb.store import IndexStore, SourceStore
    from memoryos.contextdb.transaction import RecoveryService

    assert ContextDB.__name__ == "ContextDB"
    assert ContextSelector.__name__ == "ContextSelector"
    assert HybridSearch.__name__ == "HybridSearch"
    assert IndexStore.__name__ == "IndexStore"
    assert RecoveryService.__name__ == "RecoveryService"
    assert ResourceImporter.__name__ == "ResourceImporter"
    assert SessionArchive.__name__ == "SessionArchive"
    assert SessionCommitService.__name__ == "SessionCommitService"
    assert SkillRegistry.__name__ == "SkillRegistry"
    assert SourceStore.__name__ == "SourceStore"


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
    root = Path(__file__).resolve().parents[2]

    subprocess.run([sys.executable, str(root / "examples" / "main.py")], check=True, cwd=root)


def test_context_assembly_smoke_benchmark_runs() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "benchmark" / "smoke" / "context_assembly_smoke.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
