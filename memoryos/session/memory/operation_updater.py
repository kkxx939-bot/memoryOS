from __future__ import annotations

from typing import Any

from .merge_ops import merge_op_factory
from .models import MemoryItem
from .schema import memory_type_spec, render_template
from ...storage.memory_store import MemoryStore
from .operation_resolver import ResolvedMemoryOperation


class MemoryOperationUpdater:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def apply_upsert(
        self,
        resolved: ResolvedMemoryOperation,
        user_id: str,
        source: str,
        metadata_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        operation = resolved.operation
        spec = memory_type_spec(operation.memory_type)
        merge_op = merge_op_factory(spec.merge_op)
        metadata_patch = dict(metadata_patch or {})
        if resolved.page_id is not None:
            metadata_patch["page_id"] = resolved.page_id

        if resolved.is_edit and resolved.target:
            metadata_patch["source"] = metadata_patch.get("source", source)
            current = self.store.resolve_memory(resolved.target, user_id)
            patch_content = self._render_content(spec.content_template, resolved.fields, current.get("content", ""))
            merged_content = merge_op.apply(current.get("content"), patch_content)
            merged_title = self._merge_scalar(current.get("title"), resolved.fields.get("title"), spec.merge_op)
            merged_tags = sorted(
                set([*current.get("tags", []), *resolved.fields.get("tags", [])])
            )
            metadata_patch = self._metadata_with_links(
                metadata_patch,
                current.get("links", []),
                resolved.links,
                source_uri=resolved.target,
            )
            result = self.store.update_memory(
                resolved.target,
                user_id=user_id,
                title=merged_title,
                text=merged_content,
                tags=merged_tags,
                metadata_patch=metadata_patch,
            )
            self._sync_backlinks(user_id, result["uri"], resolved.links)
            return result

        rendered_content = self._render_content(spec.content_template, resolved.fields, "")
        item = MemoryItem(
            user_id=user_id,
            memory_type=operation.memory_type,
            title=str(resolved.fields.get("title") or operation.title),
            text=rendered_content,
            tags=list(resolved.fields.get("tags") or operation.tags),
            source=source,
            confidence=float(resolved.fields.get("confidence", operation.confidence)),
            evidence_count=int(metadata_patch.pop("evidence_count", 1)),
            positive_count=int(metadata_patch.pop("positive_count", 1)),
            negative_count=int(metadata_patch.pop("negative_count", 0)),
        )
        self.store.add_memory(item)
        if resolved.links:
            metadata_patch = self._metadata_with_links(
                metadata_patch,
                [],
                resolved.links,
                source_uri=item.path or "",
            )
        if metadata_patch:
            result = self.store.update_memory(item.path or "", user_id=user_id, metadata_patch=metadata_patch)
            self._sync_backlinks(user_id, result["uri"], resolved.links)
            return result
        result = {
            "uri": item.path,
            "operation": "create",
            "before": None,
            "after": {"metadata": self.store.resolve_memory(item.path or "", user_id), "content": item.text},
        }
        self._sync_backlinks(user_id, result["uri"], resolved.links)
        return result

    def _merge_scalar(self, current_value: Any, patch_value: Any, merge_op: str) -> Any:
        if merge_op == "immutable" and current_value not in {None, ""}:
            return current_value
        return patch_value if patch_value not in {None, ""} else current_value

    def _render_content(self, template: str, fields: dict[str, Any], current_content: str) -> str:
        return render_template(
            template,
            {
                "title": fields.get("title", ""),
                "content": fields.get("content", ""),
                "tags": " ".join(str(tag) for tag in fields.get("tags", [])),
                "current_content": current_content,
            },
        ).strip()

    def _metadata_with_links(
        self,
        metadata_patch: dict[str, Any],
        existing_links: Any,
        new_links: list[dict[str, Any]],
        source_uri: str,
    ) -> dict[str, Any]:
        if not new_links:
            return metadata_patch
        metadata_patch["links"] = self._merge_links(existing_links, new_links, source_uri=source_uri)
        return metadata_patch

    def _merge_links(
        self,
        existing_links: Any,
        new_links: list[dict[str, Any]],
        source_uri: str,
    ) -> list[dict[str, Any]]:
        merged = []
        seen = set()
        for link in [*(existing_links if isinstance(existing_links, list) else []), *new_links]:
            target = str(link.get("to") or "").strip()
            if not target:
                continue
            key = (source_uri, target, str(link.get("link_type", "related_to")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "from": source_uri,
                    "to": target,
                    "link_type": str(link.get("link_type", "related_to")),
                    "description": str(link.get("description", "")),
                    "weight": max(0.0, min(1.0, float(link.get("weight", 0.5) or 0.5))),
                }
            )
        return merged

    def _sync_backlinks(self, user_id: str, source_uri: str, links: list[dict[str, Any]]) -> None:
        for link in links:
            target = str(link.get("to") or "").strip()
            if not target:
                continue
            try:
                current = self.store.resolve_memory(target, user_id)
            except FileNotFoundError:
                continue
            backlink = {
                "from": source_uri,
                "to": target,
                "link_type": str(link.get("link_type", "related_to")),
                "description": str(link.get("description", "")),
                "weight": max(0.0, min(1.0, float(link.get("weight", 0.5) or 0.5))),
            }
            backlinks = self._merge_links(current.get("backlinks", []), [backlink], source_uri=source_uri)
            self.store.update_memory(target, user_id=user_id, metadata_patch={"backlinks": backlinks})
