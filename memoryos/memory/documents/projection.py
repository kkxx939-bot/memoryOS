"""Bounded rebuildable L0/L1 projection of live Markdown documents."""

from __future__ import annotations

import re
from dataclasses import dataclass

from memoryos.core.ids import stable_hash
from memoryos.memory.documents.frontmatter import parse_front_matter
from memoryos.memory.documents.model import MemoryDocumentKind
from memoryos.memory.documents.path_policy import MemoryDocumentPathPolicy

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class MemoryBlockProjection:
    block_id: str
    document_id: str
    heading_path: tuple[str, ...]
    occurrence: int
    text: str
    source_digest: str
    projection_generation: int


@dataclass(frozen=True)
class MemoryDocumentProjection:
    tenant_id: str
    owner_user_id: str
    document_id: str
    document_kind: MemoryDocumentKind
    relative_path: str
    source_digest: str
    document_revision: int
    projection_generation: int
    title: str
    l0_text: str
    l1_text: str
    blocks: tuple[MemoryBlockProjection, ...]


class MemoryDocumentProjector:
    def __init__(
        self,
        *,
        max_blocks: int = 128,
        max_block_chars: int = 8_000,
        max_summary_chars: int = 2_000,
        max_front_matter_bytes: int = 32 * 1024,
        max_front_matter_depth: int = 12,
    ) -> None:
        if min(max_blocks, max_block_chars, max_summary_chars) <= 0:
            raise ValueError("projection limits must be positive")
        self.max_blocks = max_blocks
        self.max_block_chars = max_block_chars
        self.max_summary_chars = max_summary_chars
        self.max_front_matter_bytes = max_front_matter_bytes
        self.max_front_matter_depth = max_front_matter_depth

    def project(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        relative_path: str,
        raw_bytes: bytes,
        source_digest: str,
        document_revision: int,
        projection_generation: int,
    ) -> MemoryDocumentProjection:
        relative = MemoryDocumentPathPolicy.normalize_relative_path(relative_path)
        parsed = parse_front_matter(
            raw_bytes,
            max_header_bytes=self.max_front_matter_bytes,
            max_depth=self.max_front_matter_depth,
        )
        document_id = parsed.document_id
        blocks = self._blocks(
            document_id=document_id,
            body=parsed.body,
            source_digest=source_digest,
            projection_generation=projection_generation,
        )
        title = next((path[-1] for block in blocks if (path := block.heading_path)), relative.rsplit("/", 1)[-1])
        summaries = [block.text.strip() for block in blocks if block.text.strip()]
        l1_text = "\n\n".join(summaries)[: self.max_summary_chars]
        l0_text = " | ".join(
            dict.fromkeys(block.heading_path[-1] for block in blocks if block.heading_path)
        )[: self.max_summary_chars]
        return MemoryDocumentProjection(
            tenant_id=MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id"),
            owner_user_id=MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id"),
            document_id=document_id,
            document_kind=MemoryDocumentPathPolicy.kind_for(relative),
            relative_path=relative,
            source_digest=source_digest,
            document_revision=document_revision,
            projection_generation=projection_generation,
            title=title,
            l0_text=l0_text,
            l1_text=l1_text,
            blocks=blocks,
        )

    def _blocks(
        self,
        *,
        document_id: str,
        body: str,
        source_digest: str,
        projection_generation: int,
    ) -> tuple[MemoryBlockProjection, ...]:
        heading_stack: list[str] = []
        block_heading: tuple[str, ...] = ()
        block_lines: list[str] = []
        drafts: list[tuple[tuple[str, ...], str]] = []

        def flush() -> None:
            text = "\n".join(block_lines).strip()
            if text or block_heading:
                drafts.append((block_heading, text[: self.max_block_chars]))
            block_lines.clear()

        for line in body.splitlines():
            match = _HEADING.match(line)
            if match:
                flush()
                level = len(match.group(1))
                heading = " ".join(match.group(2).split())[:240]
                del heading_stack[level - 1 :]
                while len(heading_stack) < level - 1:
                    heading_stack.append("")
                heading_stack.append(heading)
                block_heading = tuple(item for item in heading_stack if item)
                continue
            block_lines.append(line)
        flush()
        occurrences: dict[tuple[str, ...], int] = {}
        projections: list[MemoryBlockProjection] = []
        for heading_path, text in drafts[: self.max_blocks]:
            occurrence = occurrences.get(heading_path, 0)
            occurrences[heading_path] = occurrence + 1
            block_id = f"memblk_{stable_hash((document_id, heading_path, occurrence, source_digest), 32)}"
            projections.append(
                MemoryBlockProjection(
                    block_id=block_id,
                    document_id=document_id,
                    heading_path=heading_path,
                    occurrence=occurrence,
                    text=text,
                    source_digest=source_digest,
                    projection_generation=projection_generation,
                )
            )
        return tuple(projections)


__all__ = ["MemoryBlockProjection", "MemoryDocumentProjection", "MemoryDocumentProjector"]
