"""向量协议只能有一个所有者，生产代码不得内置伪向量实现。"""

from __future__ import annotations

from pathlib import Path

from tests.support.import_graph import production_imports

ROOT = Path(__file__).resolve().parents[3]


def test_context_does_not_import_provider_implementations() -> None:
    violations = [
        f"{edge.source.relative_to(ROOT)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if edge.source.relative_to(ROOT).as_posix().startswith("infrastructure/context/")
        and (
            edge.target == "memoryos.providers"
            or edge.target.startswith("memoryos.providers.")
        )
    ]
    assert violations == []


def test_production_does_not_import_removed_embedding_providers() -> None:
    violations = [
        f"{edge.source.relative_to(ROOT)}:{edge.line} [{edge.kind}] -> {edge.target}"
        for edge in production_imports(ROOT)
        if edge.target == "memoryos.providers.embedding"
        or edge.target.startswith("memoryos.providers.embedding.")
    ]
    assert violations == []


def test_embedding_protocol_has_one_owner_and_no_builtin_implementation() -> None:
    from infrastructure.context.retrieval import EmbeddingProvider as RetrievalPackageProtocol
    from infrastructure.context.retrieval.embedding import EmbeddingProvider

    assert EmbeddingProvider is RetrievalPackageProtocol
    assert not (ROOT / "memoryos" / "providers" / "embedding").exists()
