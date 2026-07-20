"""上下文 URI 的精确分层读取。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from foundation.identity import LocalUserContext
from infrastructure.context.contracts import ContextObjectReader
from infrastructure.context.layers.memory_document_overlay import MemoryDocumentContextOverlay
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_uri import ContextURI
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from sanitization.context_projection import ContextProjectionSanitizer


class ContextExactReader:
    """按 URI、固定存储命名空间和本地用户读取 L0、L1 或 L2 内容。"""

    def __init__(
        self,
        *,
        source_store: SourceStore | None,
        index_store: IndexStore | None,
        context_reader: ContextObjectReader,
        document_overlay: MemoryDocumentContextOverlay | None,
        require_exact_read_scope: Callable[[str, Any, LocalUserContext], None],
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.context_reader = context_reader
        self.document_overlay = document_overlay
        self.require_exact_read_scope = require_exact_read_scope
        self.sanitizer = ContextProjectionSanitizer()

    def read(
        self,
        uri: str,
        *,
        layer: str,
        tenant_id: str,
        caller: LocalUserContext | None,
    ) -> dict[str, Any]:
        parsed = ContextURI.parse(uri)
        if len(parsed.segments) == 4 and parsed.segments[1:3] == ("memory", "documents"):
            return self._memory_document(uri, layer=layer, tenant_id=tenant_id, caller=caller)
        obj = self.context_reader.read_object(uri)
        if caller is not None:
            self.require_exact_read_scope(uri, obj, caller)
        requested_layer = layer.upper()
        layer_uri = {
            "L0": obj.layers.l0_uri,
            "L1": obj.layers.l1_uri,
            "L2": obj.layers.l2_uri or obj.uri,
        }.get(requested_layer)
        if not layer_uri:
            raise FileNotFoundError(f"layer unavailable: {layer}")
        if caller is not None:
            layer_parsed = ContextURI.parse(layer_uri)
            if layer_parsed.authority != parsed.authority or layer_parsed.user_id != parsed.user_id:
                raise FileNotFoundError(uri)
        if self.source_store is None:
            raise FileNotFoundError(uri)
        content = self.source_store.read_content(layer_uri)
        return self._public_result(
            object_payload=obj.to_dict(),
            title=obj.title,
            metadata=dict(obj.metadata or {}),
            source_kind=str(obj.metadata.get("source_kind") or obj.context_type.value),
            layer=requested_layer,
            content=content,
        )

    def _memory_document(
        self,
        uri: str,
        *,
        layer: str,
        tenant_id: str,
        caller: LocalUserContext | None,
    ) -> dict[str, Any]:
        if self.index_store is None:
            raise FileNotFoundError(uri)
        owner_user_id, document_id = MemoryDocumentPathPolicy.parse_document_uri(uri)
        if caller is not None and owner_user_id != caller.user_id:
            raise FileNotFoundError(uri)
        records = self.index_store.get_catalog_by_uri(
            tenant_id=tenant_id,
            uri=uri,
            limit=2,
        )
        document_records = [
            record
            for record in records
            if record.record_kind == "memory_document" and record.document_id == document_id
        ]
        if len(document_records) != 1:
            raise FileNotFoundError(uri)
        record = document_records[0]
        if record.owner_user_id != owner_user_id:
            raise FileNotFoundError(uri)
        requested_layer = layer.upper()
        if requested_layer == "L0":
            content = record.l0_text
        elif requested_layer == "L1":
            content = record.l1_text
        elif requested_layer == "L2":
            if self.document_overlay is None:
                raise FileNotFoundError(uri)
            view = self.document_overlay.read(
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                document_uri=uri,
                relative_path=str(record.metadata.get("relative_path") or ""),
                expected_source_digest=record.source_digest,
            )
            content = view.markdown
        else:
            raise FileNotFoundError(f"layer unavailable: {layer}")
        return self._public_result(
            object_payload=record.to_dict(),
            title=record.title,
            metadata=dict(record.metadata),
            source_kind=record.source_kind or "memory_document",
            layer=requested_layer,
            content=content,
        )

    def _public_result(
        self,
        *,
        object_payload: dict[str, Any],
        title: str,
        metadata: dict[str, Any],
        source_kind: str,
        layer: str,
        content: str,
    ) -> dict[str, Any]:
        """清洗精确回源结果，避免公开读取绕过 Serving 出口策略。"""

        safe = self.sanitizer.sanitize(
            title=title,
            l0_text=content if layer == "L0" else "",
            l1_text=content if layer != "L0" else "",
            metadata=metadata,
            source_kind=source_kind,
        )
        safe_object = {
            **object_payload,
            "title": safe.title,
            "metadata": safe.metadata,
        }
        if "l0_text" in safe_object:
            safe_object["l0_text"] = safe.l0_text
        if "l1_text" in safe_object:
            safe_object["l1_text"] = safe.l1_text
        public = self.sanitizer.sanitize_trace(
            {
                "object": safe_object,
                "layer": layer,
                "content": safe.l0_text if layer == "L0" else safe.l1_text,
            }
        )
        if not isinstance(public, dict):
            raise ValueError("exact context sanitization produced an invalid payload")
        return public


__all__ = ["ContextExactReader"]
