"""Read-only stable-URI overlay over live Markdown source bytes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from memoryos.memory.documents.frontmatter import parse_front_matter
from memoryos.memory.documents.model import PresentPath
from memoryos.memory.documents.path_policy import MemoryDocumentPathPolicy
from memoryos.memory.documents.store import DocumentConflictError, MemoryDocumentStore


@dataclass(frozen=True)
class MemoryDocumentContextView:
    uri: str
    tenant_id: str
    owner_user_id: str
    document_id: str
    relative_path: str
    source_digest: str
    raw_bytes: bytes
    markdown: str


class MemoryDocumentContextOverlay:
    def __init__(
        self,
        store: MemoryDocumentStore,
        *,
        max_front_matter_bytes: int = 32 * 1024,
        max_front_matter_depth: int = 12,
    ) -> None:
        self.store = store
        self.max_front_matter_bytes = max_front_matter_bytes
        self.max_front_matter_depth = max_front_matter_depth

    def read(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_uri: str,
        relative_path: str,
        expected_source_digest: str,
    ) -> MemoryDocumentContextView:
        uri_owner, document_id = MemoryDocumentPathPolicy.parse_document_uri(document_uri)
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        if uri_owner != owner:
            raise PermissionError("document URI owner does not match trusted caller")
        relative = MemoryDocumentPathPolicy.normalize_relative_path(relative_path)
        state = self.store.read_state(tenant, owner, relative)
        if not isinstance(state, PresentPath) or state.raw_sha256 != expected_source_digest:
            raise DocumentConflictError("catalog path or digest is stale relative to live Markdown")
        raw = self.store.read_raw(tenant, owner, document_id=document_id)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_source_digest:
            raise DocumentConflictError("catalog candidate is stale relative to live Markdown")
        parsed = parse_front_matter(
            raw,
            max_header_bytes=self.max_front_matter_bytes,
            max_depth=self.max_front_matter_depth,
        )
        if parsed.document_id != document_id:
            raise DocumentConflictError("live Markdown document_id does not match the stable URI")
        return MemoryDocumentContextView(
            uri=document_uri,
            tenant_id=tenant,
            owner_user_id=owner,
            document_id=document_id,
            relative_path=relative,
            source_digest=digest,
            raw_bytes=raw,
            markdown=parsed.body,
        )


__all__ = ["MemoryDocumentContextOverlay", "MemoryDocumentContextView"]
