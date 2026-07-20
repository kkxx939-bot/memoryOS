"""生成 Catalog、Vector 和文档链接等可重建投影。"""

from __future__ import annotations

import math
import posixpath
import re
from collections.abc import Mapping
from typing import Any, cast

from infrastructure.context.projection.memory_document import MemoryDocumentProjection
from infrastructure.store.contracts.vector import vector_row_id
from infrastructure.store.model.catalog import (
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    catalog_vector_metadata,
)
from infrastructure.store.model.context.context_relation import ContextRelation
from infrastructure.store.model.context.context_type import ContextType
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.worker.projection.model import (
    CatalogBatchScanner,
    CatalogLister,
)

_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]\n]{1,500}\]\(([^)\n]{1,2000})\)")


class ProjectionCatalogMixin:
    """负责投影记录构造、向量准备、关系更新和队列载荷校验。"""

    def _records(
        self: Any,
        projection: MemoryDocumentProjection,
    ) -> tuple[CatalogRecord, tuple[CatalogRecord, ...]]:
        uri = MemoryDocumentPathPolicy.document_uri(projection.owner_user_id, projection.document_id)
        tree_path = self._tree_path(projection.relative_path)
        common: dict[str, Any] = {
            "tenant_id": projection.tenant_id,
            "owner_user_id": projection.owner_user_id,
            "context_type": ContextType.MEMORY.value,
            "source_kind": "markdown_memory_document",
            "lifecycle_state": "active",
            "primary_tree_path": tree_path,
            "tree_paths": (tree_path,),
            "source_uri": uri,
            "source_digest": projection.source_digest,
            "source_revision": projection.document_revision,
            "document_id": projection.document_id,
            "document_kind": projection.document_kind.value,
            "document_revision": projection.document_revision,
            "projection_generation": projection.projection_generation,
            "projection_effect_hash": projection.source_digest,
            "projection_status": CatalogProjectionStatus.PROJECTED.value,
            "metadata": {
                "relative_path": projection.relative_path,
                "source_authority": "live_markdown",
            },
        }
        document = CatalogRecord(
            record_key=f"memory-document:{projection.owner_user_id}:{projection.document_id}",
            uri=uri,
            record_kind=CatalogRecordKind.MEMORY_DOCUMENT.value,
            title=projection.title,
            l0_text=projection.l0_text,
            l1_text=projection.l1_text,
            l2_uri=uri,
            **common,
        )
        blocks = tuple(
            CatalogRecord(
                record_key=f"memory-block:{projection.owner_user_id}:{block.block_id}",
                uri=f"{uri}/blocks/{block.block_id}",
                record_kind=CatalogRecordKind.MEMORY_BLOCK.value,
                parent_uri=uri,
                title=" / ".join(block.heading_path) or projection.title,
                l0_text=" / ".join(block.heading_path),
                l1_text=block.text,
                l2_uri=uri,
                block_id=block.block_id,
                metadata={
                    **cast(dict[str, Any], common["metadata"]),
                    "heading_path": list(block.heading_path),
                    "occurrence": block.occurrence,
                },
                **{key: value for key, value in common.items() if key != "metadata"},
            )
            for block in projection.blocks
        )
        return document, blocks

    def _prepare_vector_rows(
        self: Any,
        records: tuple[CatalogRecord, ...],
    ) -> tuple[tuple[str, list[float], dict[str, Any]], ...]:
        if self.vector_store is None or self.embedding_provider is None:
            return ()
        prepared: list[tuple[str, list[float], dict[str, Any]]] = []
        for record in records:
            text = "\n".join(part for part in (record.title, record.l0_text, record.l1_text) if part)
            if not text:
                continue
            embedding = [float(value) for value in self.embedding_provider.embed(text)]
            if not embedding or any(not math.isfinite(value) for value in embedding):
                raise ValueError("memory document embedding provider returned an invalid vector")
            metadata = {
                **catalog_vector_metadata(record),
                "record_key": record.record_key,
                "public_uri": record.uri,
                "embedding_model": str(getattr(self.embedding_provider, "model_name", "")),
                "schema_version": "memory_document_vector_v1",
            }
            prepared.append((vector_row_id(record.tenant_id, record.record_key), embedding, metadata))
        return tuple(prepared)

    def _replace_document_links(self: Any, document_record: CatalogRecord, raw_bytes: bytes) -> None:
        if self.relation_store is None:
            return
        with self.erasure_store.owner_relation_lock(
            document_record.tenant_id,
            document_record.owner_user_id,
        ):
            self._replace_document_links_locked(document_record, raw_bytes)

    def _replace_document_links_locked(self: Any, document_record: CatalogRecord, raw_bytes: bytes) -> None:
        assert self.relation_store is not None
        uri = document_record.uri
        while self.relation_store.delete_projection_relations(
            uri,
            tenant_id=document_record.tenant_id,
            catalog_record_key=document_record.record_key,
            limit=1_000,
        ):
            pass
        body = raw_bytes.decode("utf-8", errors="strict")
        targets_by_path = {
            str(record.metadata.get("relative_path") or ""): record
            for record in self._owner_document_records(
                document_record.tenant_id,
                document_record.owner_user_id,
            )
        }
        base = posixpath.dirname(str(document_record.metadata.get("relative_path") or ""))
        seen: set[str] = set()
        for match in _MARKDOWN_LINK.finditer(body):
            destination = match.group(1).strip().split("#", 1)[0]
            if not destination or "://" in destination or destination.startswith(("#", "/")):
                continue
            try:
                relative = MemoryDocumentPathPolicy.normalize_relative_path(
                    posixpath.normpath(posixpath.join(base, destination))
                )
            except ValueError:
                continue
            target_record = targets_by_path.get(relative)
            if target_record is None:
                continue
            target_uri = target_record.uri
            if target_uri == uri or target_uri in seen:
                continue
            target_document_id = target_record.document_id
            if (
                self.erasure_store.load(
                    document_record.tenant_id,
                    document_record.owner_user_id,
                    target_document_id,
                )
                is not None
            ):
                continue
            target_barrier = self.control_store.load_publication_barrier(
                document_record.tenant_id,
                document_record.owner_user_id,
                target_document_id,
            )
            target_state = (
                self._projection_state(
                    document_record.tenant_id,
                    document_record.owner_user_id,
                    target_document_id,
                )
                or {}
            )
            if str(target_state.get("deletion_status") or ""):
                continue
            if target_barrier is not None:
                target_control = self.control_store.load_control(
                    document_record.tenant_id,
                    document_record.owner_user_id,
                    target_document_id,
                )
                if not self._control_authorizes_restored_serving(
                    target_control,
                    target_barrier,
                    target_state,
                ):
                    continue
            seen.add(target_uri)
            self.relation_store.add_relation(
                ContextRelation(
                    source_uri=uri,
                    relation_type="links_to",
                    target_uri=target_uri,
                    metadata={
                        "tenant_id": document_record.tenant_id,
                        "owner_user_id": document_record.owner_user_id,
                        "catalog_record_key": document_record.record_key,
                        "projection_generation": document_record.projection_generation,
                        "source_digest": document_record.source_digest,
                    },
                ),
                tenant_id=document_record.tenant_id,
            )

    def _existing_projection_uris(
        self: Any,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> dict[str, str]:
        list_catalog = getattr(self.catalog_store, "list_catalog", None)
        if not callable(list_catalog):
            return {}
        records = cast(CatalogLister, list_catalog)(
            tenant_id=tenant_id,
            filters={"owner_user_id": owner_user_id, "document_id": document_id, "include_inactive": True},
            limit=1_000,
        )
        return {str(record.record_key): str(record.uri) for record in records}

    def _owner_document_records(
        self: Any,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[CatalogRecord, ...]:
        scanner = getattr(self.catalog_store, "scan_catalog_batch", None)
        if not callable(scanner):
            list_catalog = getattr(self.catalog_store, "list_catalog", None)
            if not callable(list_catalog):
                raise RuntimeError("Catalog store has no bounded projection scanner")
            listed_records = cast(CatalogLister, list_catalog)(
                tenant_id=tenant_id,
                filters={
                    "owner_user_id": owner_user_id,
                    "record_kind": CatalogRecordKind.MEMORY_DOCUMENT.value,
                    "include_inactive": True,
                },
                limit=1_000,
            )
            return tuple(listed_records)
        scan_batch = cast(CatalogBatchScanner, scanner)
        records: list[CatalogRecord] = []
        cursor = ""
        while True:
            batch = scan_batch(
                tenant_id=tenant_id,
                after_record_key=cursor,
                filters={
                    "owner_user_id": owner_user_id,
                    "record_kind": CatalogRecordKind.MEMORY_DOCUMENT.value,
                    "include_inactive": True,
                },
                limit=256,
            )
            if not batch:
                return tuple(records)
            records.extend(batch)
            next_cursor = str(batch[-1].record_key)
            if next_cursor <= cursor:
                raise RuntimeError("Catalog document scan did not advance")
            cursor = next_cursor

    def _remove_obsolete(
        self: Any,
        tenant_id: str,
        record_keys: tuple[str, ...],
        existing_uris: Mapping[str, str],
    ) -> None:
        for record_key in record_keys:
            if self.vector_store is not None:
                self.vector_store.delete_vector(vector_row_id(tenant_id, record_key))
                # 合规写入器使用确定性 row ID；同时按精确元数据身份删除，可以
                # 覆盖物理键未遵循该约定的历史或外部数据行。
                deleted = self.vector_store.delete_by_filter(
                    {
                        "tenant_id": tenant_id,
                        "catalog_record_key": record_key,
                    }
                )
                if not isinstance(deleted, int) or isinstance(deleted, bool) or deleted < 0:
                    raise TypeError("vector delete-by-filter returned an invalid deletion count")
            # 投影创建的关系携带精确 Catalog key，用它限定删除范围。
            if self.relation_store is not None:
                uri = str(existing_uris.get(record_key) or "")
                if not uri:
                    continue
                while self.relation_store.delete_projection_relations(
                    uri,
                    tenant_id=tenant_id,
                    catalog_record_key=record_key,
                    limit=1_000,
                ):
                    pass

    @staticmethod
    def _tree_path(relative_path: str) -> str:
        relative = MemoryDocumentPathPolicy.normalize_relative_path(relative_path)
        if relative == "MEMORY.md":
            return "memories/root"
        if relative == "profile.md":
            return "memories/profile"
        if relative == "preferences.md":
            return "memories/preferences"
        if relative == "knowledge/MEMORY.md":
            return "memories/knowledge"
        if relative == "knowledge/open-loops.md":
            return "memories/knowledge/open-loops"
        stem = relative.removesuffix(".md")
        return f"memories/{stem}"

__all__ = ["ProjectionCatalogMixin"]
