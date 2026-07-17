"""Embedding protocol ownership and compatibility boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

from tests.support.import_graph import production_imports

ROOT = Path(__file__).resolve().parents[3]

_PROVIDER_COMPATIBILITY_EXPORTS = {
    "memoryos/providers/__init__.py",
    "memoryos/providers/embedding/__init__.py",
    "memoryos/providers/embedding/base.py",
}


def test_contextdb_does_not_import_provider_implementations() -> None:
    violations = [
        f"{edge.source.relative_to(ROOT)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if edge.source.relative_to(ROOT).as_posix().startswith("memoryos/contextdb/")
        and (
            edge.target == "memoryos.providers"
            or edge.target.startswith("memoryos.providers.")
        )
    ]
    assert violations == []


def test_production_type_imports_use_embedding_protocol_owner() -> None:
    violations: list[str] = []
    for path in sorted((ROOT / "memoryos").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in _PROVIDER_COMPATIBILITY_EXPORTS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if node.module != "memoryos.providers.embedding" and not node.module.startswith(
                "memoryos.providers.embedding."
            ):
                continue
            if any(alias.name == "EmbeddingProvider" for alias in node.names):
                violations.append(f"{relative}:{node.lineno} -> {node.module}")
    assert violations == []


def test_historical_embedding_protocol_exports_keep_object_identity() -> None:
    from memoryos.contextdb.retrieval import EmbeddingProvider as RetrievalPackageProtocol
    from memoryos.contextdb.retrieval.embedding import EmbeddingProvider
    from memoryos.providers import EmbeddingProvider as ProviderPackageProtocol
    from memoryos.providers.embedding import EmbeddingProvider as EmbeddingPackageProtocol
    from memoryos.providers.embedding.base import EmbeddingProvider as HistoricalBaseProtocol

    assert (
        EmbeddingProvider
        is RetrievalPackageProtocol
        is ProviderPackageProtocol
        is EmbeddingPackageProtocol
        is HistoricalBaseProtocol
    )
