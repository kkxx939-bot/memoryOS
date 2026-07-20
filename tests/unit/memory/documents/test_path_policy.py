from __future__ import annotations

import pytest

from memory.core.model import MemoryDocumentKind
from memory.core.structure.path_policy import MemoryDocumentPathPolicy

DOCUMENT_ID = "memdoc_0123456789ABCDEF"


@pytest.mark.parametrize(
    ("relative_path", "kind"),
    [
        ("MEMORY.md", MemoryDocumentKind.ROOT_INDEX),
        ("profile.md", MemoryDocumentKind.PROFILE),
        ("preferences.md", MemoryDocumentKind.PREFERENCES),
        ("knowledge/MEMORY.md", MemoryDocumentKind.KNOWLEDGE_INDEX),
        ("knowledge/open-loops.md", MemoryDocumentKind.OPEN_LOOPS),
        ("knowledge/entities/memoryos.md", MemoryDocumentKind.ENTITY),
        ("knowledge/topics/file-memory.md", MemoryDocumentKind.TOPIC),
        ("knowledge/episodes/2026-07-17-design.md", MemoryDocumentKind.EPISODE),
        ("experiences/recovery.md", MemoryDocumentKind.EXPERIENCE),
    ],
)
def test_controlled_paths_derive_document_kind(relative_path: str, kind: MemoryDocumentKind) -> None:
    assert MemoryDocumentPathPolicy.normalize_relative_path(relative_path) == relative_path
    assert MemoryDocumentPathPolicy.kind_for(relative_path) is kind


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        "/profile.md",
        "../profile.md",
        "knowledge/../profile.md",
        "knowledge/topics/./topic.md",
        "knowledge/topics//topic.md",
        "knowledge\\topics\\topic.md",
        "knowledge/topics/topic.md\x00",
        "projects/memoryos.md",
        "knowledge/topics/.hidden.md",
        "knowledge/topics/nested/topic.md",
        "knowledge/topics/topic.txt",
        "knowledge/topics/MEMORY.md",
        "knowledge/open-loops.md/child.md",
    ],
)
def test_unsafe_or_out_of_layout_paths_are_rejected(relative_path: str) -> None:
    with pytest.raises(ValueError):
        MemoryDocumentPathPolicy.normalize_relative_path(relative_path)


def test_decomposed_unicode_path_is_rejected_and_casefold_collision_is_stable() -> None:
    decomposed = "knowledge/topics/cafe\u0301.md"

    with pytest.raises(ValueError, match="NFC"):
        MemoryDocumentPathPolicy.normalize_relative_path(decomposed)
    assert MemoryDocumentPathPolicy.collision_key("knowledge/topics/straße.md") == (
        MemoryDocumentPathPolicy.collision_key("knowledge/topics/STRASSE.md")
    )


def test_tenant_owner_segments_and_stable_document_uri_are_validated() -> None:
    uri = MemoryDocumentPathPolicy.document_uri("alice", DOCUMENT_ID)

    assert uri == f"memoryos://user/alice/memory/documents/{DOCUMENT_ID}"
    assert MemoryDocumentPathPolicy.parse_document_uri(uri) == ("alice", DOCUMENT_ID)
    for unsafe in ("../alice", "a/b", "a\\b", "a\x00b"):
        with pytest.raises(ValueError):
            MemoryDocumentPathPolicy.document_uri(unsafe, DOCUMENT_ID)
    with pytest.raises(ValueError):
        MemoryDocumentPathPolicy.parse_document_uri(f"memoryos://user/alice/memory/records/{DOCUMENT_ID}")
