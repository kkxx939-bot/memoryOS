"""Catalog 与索引命中到统一候选模型的转换。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from infrastructure.context.retrieval.fusion import RetrievalCandidate
from infrastructure.store.contracts.index import IndexHit, IndexStore
from infrastructure.store.model.catalog import CatalogRecord, CatalogRecordKind


class CandidateMapper:
    """集中维护候选字段映射，避免各召回后端复制身份和版本字段。"""

    def __init__(self, index_store: IndexStore) -> None:
        self.index_store = index_store

    def from_hit(self, hit: IndexHit) -> RetrievalCandidate:
        metadata = dict(hit.metadata or {})
        record_key = str(metadata.get("catalog_record_key") or hit.uri)
        getter = getattr(self.index_store, "get_catalog", None)
        if callable(getter):
            record = getter(
                tenant_id=str(metadata.get("tenant_id") or "default"),
                record_key=record_key,
            )
            if isinstance(record, CatalogRecord):
                scores = dict(metadata.get("retrieval_scores", {}) or {})
                score = float(scores.get("lexical") or scores.get("identity") or hit.score or 0.0)
                branch = "exact" if float(scores.get("identity") or 0.0) > 0 else "lexical"
                return self.from_record(record, branch=branch, score=score, extra_metadata=metadata)
        scores = dict(metadata.get("retrieval_scores", {}) or {})
        lexical = float(scores.get("lexical") or scores.get("identity") or hit.score or 0.0)
        record_kind = str(metadata.get("record_kind") or CatalogRecordKind.CONTEXT.value)
        return RetrievalCandidate(
            record_key=record_key,
            uri=hit.uri,
            title=hit.title,
            context_type=hit.context_type,
            text=hit.title,
            source_uri=hit.uri,
            record_kind=record_kind,
            source_kind=str(metadata.get("source_kind") or "context"),
            tenant_id=str(metadata.get("tenant_id") or "default"),
            owner_user_id=str(metadata.get("owner_user_id") or ""),
            session_id=str(metadata.get("session_id") or ""),
            workspace_id=str(metadata.get("workspace_id") or metadata.get("project_id") or ""),
            document_id=str(metadata.get("document_id") or ""),
            block_id=str(metadata.get("block_id") or ""),
            document_kind=str(metadata.get("document_kind") or ""),
            document_revision=int(metadata.get("document_revision") or 0),
            projection_generation=int(metadata.get("projection_generation") or 0),
            archive_digest=str(metadata.get("archive_digest") or ""),
            manifest_digest=str(metadata.get("manifest_digest") or ""),
            source_digest=str(metadata.get("source_digest") or ""),
            event_time=str(metadata.get("event_time") or ""),
            metadata=metadata,
            branch_scores={"lexical": lexical},
        )

    @staticmethod
    def from_record(
        record: CatalogRecord,
        *,
        branch: str,
        score: float,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> RetrievalCandidate:
        metadata = {
            **dict(record.metadata),
            **dict(extra_metadata or {}),
            "tenant_id": record.tenant_id,
            "owner_user_id": record.owner_user_id,
            "workspace_id": record.workspace_id,
            "document_id": record.document_id,
            "block_id": record.block_id,
            "document_kind": record.document_kind,
            "document_revision": record.document_revision,
            "projection_generation": record.projection_generation,
            "projection_effect_hash": record.projection_effect_hash,
            "catalog_record_key": record.record_key,
            "record_kind": record.record_kind,
            "serving_tier": record.serving_tier,
        }
        return RetrievalCandidate(
            record_key=record.record_key,
            uri=record.uri,
            title=record.title,
            context_type=record.context_type,
            source_kind=record.source_kind,
            record_kind=record.record_kind,
            text="",
            l0_text=record.l0_text,
            l1_text=record.l1_text,
            l2_uri=record.l2_uri,
            source_uri=record.source_uri,
            source_digest=record.source_digest,
            tenant_id=record.tenant_id,
            owner_user_id=record.owner_user_id,
            session_id=record.session_id,
            workspace_id=record.workspace_id,
            document_id=record.document_id,
            block_id=record.block_id,
            document_kind=record.document_kind,
            document_revision=record.document_revision,
            projection_generation=record.projection_generation,
            archive_digest=str(record.metadata.get("archive_digest") or ""),
            manifest_digest=str(record.metadata.get("manifest_digest") or ""),
            event_time=record.event_time,
            hotness=max(record.hotness, record.semantic_hotness, record.behavior_support_hotness),
            metadata=metadata,
            branch_scores={branch: score},
        )


__all__ = ["CandidateMapper"]
