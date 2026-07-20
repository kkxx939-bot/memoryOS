"""从统一 Catalog 获取有边界的相关记忆文档提示。"""

from __future__ import annotations

from collections.abc import Sequence

from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.model.catalog import CatalogRecordKind
from memory.core import MemoryEditProposal, validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.execute import RelatedDocumentCandidate


def find_related_memory_documents(
    index_store: IndexStore,
    *,
    tenant_id: str,
    owner_user_id: str,
    proposal: MemoryEditProposal,
    limit: int,
) -> tuple[RelatedDocumentCandidate, ...]:
    """返回有限 Catalog 提示，调用方仍需使用实时正文重新校验。"""

    tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
    owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
    maximum = max(1, min(int(limit), 8))
    search_catalog = getattr(index_store, "search_catalog", None)
    if not callable(search_catalog):
        return ()
    semantic_query = "\n".join(
        value.strip() for value in (proposal.title, proposal.subject, proposal.body) if value and value.strip()
    )[:4_096]
    raw_hits = search_catalog(
        semantic_query,
        tenant_id=tenant,
        filters={
            "tenant_id": tenant,
            "owner_user_id": owner,
            "record_kinds": (CatalogRecordKind.MEMORY_DOCUMENT.value,),
            "lifecycle_state": "active",
        },
        limit=maximum,
    )
    if not isinstance(raw_hits, Sequence) or isinstance(raw_hits, str | bytes):
        raise TypeError("related memory Catalog search returned an invalid result")
    hits = tuple(raw_hits)
    if len(hits) > maximum:
        raise RuntimeError("related memory Catalog search exceeded its bound")
    candidates: list[RelatedDocumentCandidate] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        metadata = dict(getattr(hit, "metadata", {}) or {})
        if (
            str(metadata.get("tenant_id") or "") != tenant
            or str(metadata.get("owner_user_id") or "") != owner
            or str(metadata.get("record_kind") or "") != CatalogRecordKind.MEMORY_DOCUMENT.value
            or str(metadata.get("lifecycle_state") or "") != "active"
        ):
            raise PermissionError("related memory Catalog search crossed its bounded scope")
        try:
            document_id = validate_document_id(str(metadata.get("document_id") or ""))
            relative_path = MemoryDocumentPathPolicy.normalize_relative_path(str(metadata.get("relative_path") or ""))
        except ValueError:
            continue
        source_digest = str(metadata.get("source_digest") or "")
        if len(source_digest) != 64 or any(character not in "0123456789abcdef" for character in source_digest):
            continue
        identity = (document_id, relative_path)
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(
            RelatedDocumentCandidate(
                tenant_id=tenant,
                owner_user_id=owner,
                document_id=document_id,
                relative_path=relative_path,
                source_digest=source_digest,
                relevance=float(getattr(hit, "score", 0.0) or 0.0),
            )
        )
    return tuple(candidates)


__all__ = ["find_related_memory_documents"]
