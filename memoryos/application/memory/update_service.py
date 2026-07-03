from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from memoryos.application.memory.extractor import MemoryOperation
from memoryos.application.memory.operation_resolver import MemoryOperationResolver, ResolvedMemoryOperation
from memoryos.application.memory.operation_updater import MemoryOperationUpdater
from memoryos.application.memory.schema import memory_type_spec
from memoryos.domain.memory.memory_item import utc_now
from memoryos.domain.memory.update_policy import normalize_operation_for_policy
from memoryos.infrastructure.repositories.memory_repository import MemoryStore


@dataclass(frozen=True)
class MemoryUpdateContext:
    user_id: str
    source: str
    diff_id: str
    diff_path: Path | None = None
    explicit_user_intent: bool = False
    day: str | None = None


class MemoryUpdateService:
    def __init__(self, store: MemoryStore, min_evidence_days: int = 3) -> None:
        self.store = store
        self.min_evidence_days = min_evidence_days
        self.resolver = MemoryOperationResolver(store)
        self.updater = MemoryOperationUpdater(store)

    def apply(self, operations: list[MemoryOperation], context: MemoryUpdateContext) -> dict:
        self.store.init(context.user_id)
        diff_operations: dict[str, list[dict]] = {"adds": [], "updates": [], "deletes": [], "ignores": []}
        for operation in operations:
            self._apply_one(operation, context, diff_operations)
        diff = self._diff(context.diff_id, diff_operations)
        if context.diff_path:
            context.diff_path.parent.mkdir(parents=True, exist_ok=True)
            context.diff_path.write_text(self._json(diff), encoding="utf-8")
        return diff

    def empty_diff(self, diff_id: str) -> dict:
        return self._diff(diff_id, {"adds": [], "updates": [], "deletes": [], "ignores": []}, committed_at=None)

    def operation_record(self, operation: MemoryOperation) -> dict:
        return {
            "action": operation.action,
            "target": operation.target,
            "memory_type": operation.memory_type,
            "title": operation.title,
            "text": operation.text,
            "tags": operation.tags,
            "confidence": operation.confidence,
            "rationale": operation.rationale,
            "page_id": operation.page_id,
            "links": operation.links,
        }

    def _apply_one(self, operation: MemoryOperation, context: MemoryUpdateContext, diff_operations: dict[str, list]) -> None:
        if operation.action == "ignore":
            diff_operations["ignores"].append(self.operation_record(operation))
            return

        action, reason = normalize_operation_for_policy(
            operation.action,
            operation.memory_type,
            explicit_user_intent=context.explicit_user_intent or self._has_explicit_user_intent(operation),
        )
        if action == "ignore":
            record = self.operation_record(operation)
            record["reason"] = reason
            diff_operations["ignores"].append(record)
            return

        if action == "update":
            self._apply_update(operation, context, diff_operations)
            return

        if action == "delete":
            self._apply_delete(operation, context, diff_operations)
            return

        spec = memory_type_spec(operation.memory_type)
        if spec.operation_mode == "evidence_then_aggregate" and reason:
            self._store_evidence_first(operation, context, diff_operations, reason)
            self._aggregate_if_ready(operation, context, diff_operations)
            return

        if operation.memory_type == "profile":
            self._apply_profile(operation, context, diff_operations)
            return

        if operation.memory_type == "preference":
            self._apply_preference(operation, context, diff_operations)
            return

        if operation.memory_type == "event":
            self._apply_event(operation, context, diff_operations)
            return

        if spec.operation_mode == "append_then_aggregate":
            self._store_append_then_aggregate(operation, context, diff_operations)
            return

        self._apply_add(operation, context, diff_operations)

    def _apply_update(self, operation: MemoryOperation, context: MemoryUpdateContext, diff_operations: dict[str, list]) -> None:
        if not operation.target:
            record = self.operation_record(operation)
            record["reason"] = "update operation missing target"
            diff_operations["ignores"].append(record)
            return
        resolved = self.resolver.resolve(operation, context.user_id)
        update = self.updater.apply_upsert(
            resolved,
            user_id=context.user_id,
            source=context.source,
        )
        diff_operations["updates"].append(
            {
                "uri": update["uri"],
                "memory_type": operation.memory_type,
                "before": update["before"],
                "after": update["after"],
                "rationale": operation.rationale,
            }
        )

    def _apply_delete(self, operation: MemoryOperation, context: MemoryUpdateContext, diff_operations: dict[str, list]) -> None:
        if not operation.target:
            record = self.operation_record(operation)
            record["reason"] = "delete operation missing target"
            diff_operations["ignores"].append(record)
            return
        deletion = self.store.delete_memory(operation.target, user_id=context.user_id)
        diff_operations["deletes"].append(
            {
                "uri": deletion["uri"],
                "memory_type": operation.memory_type,
                "metadata": deletion["metadata"],
                "deleted_content": deletion["deleted_content"],
                "rationale": operation.rationale,
            }
        )

    def _apply_profile(self, operation: MemoryOperation, context: MemoryUpdateContext, diff_operations: dict[str, list]) -> None:
        current = self.store.search("User Profile", context.user_id, memory_type="profile", limit=1)
        compact_text = operation.text.strip()
        if current:
            body = str(current[0].get("content", "")).strip()
            if compact_text and compact_text not in body:
                compact_text = self._compact_profile_text(body, compact_text)
            elif body:
                compact_text = body
        result = self.store.upsert_profile(context.user_id, compact_text, mode="replace")
        bucket = "adds" if result["operation"] == "create" else "updates"
        diff_operations[bucket].append(
            {
                "uri": result["uri"],
                "memory_type": "profile",
                "before": result["before"],
                "after": result["after"],
                "rationale": operation.rationale,
            }
        )

    def _apply_preference(self, operation: MemoryOperation, context: MemoryUpdateContext, diff_operations: dict[str, list]) -> None:
        existing = self._existing_topic_memory(context.user_id, operation, memory_type="preference")
        if existing:
            merged_text = self._merge_topic_text(existing, operation)
            patch_operation = MemoryOperation(
                action="update",
                memory_type=operation.memory_type,
                title=existing.get("title") or operation.title,
                text=merged_text,
                tags=sorted(set([*existing.get("tags", []), *operation.tags])),
                confidence=operation.confidence,
                target=existing["path"],
                rationale=operation.rationale,
            )
            metadata_patch = {
                "evidence_count": int(existing.get("evidence_count", 1)) + 1,
                "positive_count": int(existing.get("positive_count", 1)) + 1,
                "source": context.source,
            }
            update = self.updater.apply_upsert(
                self.resolver.resolve(patch_operation, context.user_id),
                user_id=context.user_id,
                source=context.source,
                metadata_patch=metadata_patch,
            )
            diff_operations["updates"].append(
                {
                    "uri": update["uri"],
                    "memory_type": "preference",
                    "before": update["before"],
                    "after": update["after"],
                    "rationale": operation.rationale or "Patched existing preference topic.",
                }
            )
            return
        self._apply_add(operation, context, diff_operations)

    def _apply_event(self, operation: MemoryOperation, context: MemoryUpdateContext, diff_operations: dict[str, list]) -> None:
        event = self.store.record_event(
            user_id=context.user_id,
            event_type=self._event_type(operation),
            text=operation.text,
            day=context.day,
            tags=operation.tags,
        )
        diff_operations["adds"].append(
            {
                "uri": event["event_uri"],
                "memory_type": "event",
                "after": operation.text,
                "rationale": operation.rationale,
                "daily_log": event["daily_log"],
                "source_memory_type": operation.memory_type,
            }
        )

    def _store_evidence_first(
        self,
        operation: MemoryOperation,
        context: MemoryUpdateContext,
        diff_operations: dict[str, list],
        reason: str,
    ) -> None:
        evidence_tags = sorted(set([*operation.tags, operation.memory_type, f"{operation.memory_type}_evidence"]))
        event = self.store.record_event(
            user_id=context.user_id,
            event_type=f"{operation.memory_type}_evidence",
            text=operation.text,
            day=context.day,
            tags=evidence_tags,
        )
        diff_operations["adds"].append(
            {
                "uri": event["event_uri"],
                "memory_type": "event",
                "after": operation.text,
                "rationale": operation.rationale,
                "source_memory_type": operation.memory_type,
                "policy_reason": reason,
                "daily_log": event["daily_log"],
            }
        )

    def _store_append_then_aggregate(
        self,
        operation: MemoryOperation,
        context: MemoryUpdateContext,
        diff_operations: dict[str, list],
    ) -> None:
        self._apply_add(operation, context, diff_operations)
        similar = self._topic_memories(context.user_id, operation, operation.memory_type, limit=20)
        raw_similar = [item for item in similar if "aggregated" not in {str(tag) for tag in item.get("tags", [])}]
        existing = self._existing_rolling_aggregate(similar)
        if len(raw_similar) < 3:
            return
        evidence_count = sum(int(item.get("evidence_count", 1)) for item in raw_similar)
        aggregate_text = self._rolling_aggregate_text(operation.memory_type, raw_similar)
        aggregate_tags = sorted(set([*operation.tags, operation.memory_type, "aggregated"]))
        update = None
        if existing:
            aggregate_update = MemoryOperation(
                action="update",
                memory_type=operation.memory_type,
                title=existing.get("title") or operation.title,
                text=aggregate_text,
                tags=aggregate_tags,
                confidence=operation.confidence,
                target=existing["path"],
                rationale=f"Aggregated rolling {operation.memory_type} evidence.",
            )
            update = self.updater.apply_upsert(
                self.resolver.resolve(aggregate_update, context.user_id),
                user_id=context.user_id,
                source=f"{context.source}:aggregate",
                metadata_patch={
                    "evidence_count": evidence_count,
                    "positive_count": evidence_count,
                    "negative_count": 0,
                    "source": f"{context.source}:aggregate",
                },
            )
        if update:
            diff_operations["updates"].append(
                {
                    "uri": update["uri"],
                    "memory_type": operation.memory_type,
                    "before": update["before"],
                    "after": update["after"],
                    "rationale": f"Aggregated rolling {operation.memory_type} evidence.",
                }
            )
            return
        aggregate_operation = MemoryOperation(
            action="add",
            memory_type=operation.memory_type,
            title=f"Aggregated {operation.title}",
            text=aggregate_text,
            tags=aggregate_tags,
            confidence=operation.confidence,
            rationale=f"Created aggregated rolling {operation.memory_type} summary.",
        )
        created = self.updater.apply_upsert(
            ResolvedMemoryOperation(
                operation=aggregate_operation,
                target=None,
                fields={
                    "title": aggregate_operation.title,
                    "content": aggregate_operation.text,
                    "tags": aggregate_operation.tags,
                    "confidence": aggregate_operation.confidence,
                },
            ),
            user_id=context.user_id,
            source=f"{context.source}:aggregate",
            metadata_patch={
                "evidence_count": evidence_count,
                "positive_count": evidence_count,
                "negative_count": 0,
            },
        )
        diff_operations["adds"].append(
            {
                "uri": created["uri"],
                "memory_type": operation.memory_type,
                "after": aggregate_text,
                "rationale": f"Created aggregated rolling {operation.memory_type} summary.",
            }
        )

    def _apply_add(self, operation: MemoryOperation, context: MemoryUpdateContext, diff_operations: dict[str, list]) -> None:
        created = self.updater.apply_upsert(
            self.resolver.resolve(operation, context.user_id),
            user_id=context.user_id,
            source=context.source,
        )
        diff_operations["adds"].append(
            {
                "uri": created["uri"],
                "memory_type": operation.memory_type,
                "after": operation.text,
                "rationale": operation.rationale,
            }
        )

    def _aggregate_if_ready(
        self,
        operation: MemoryOperation,
        context: MemoryUpdateContext,
        diff_operations: dict[str, list],
    ) -> None:
        evidence = self._matching_evidence(context.user_id, operation)
        days = self._distinct_days(evidence)
        if len(days) < self.min_evidence_days:
            return
        aggregate_text = self._aggregate_text(operation, evidence, days)
        existing = self._existing_aggregate(context.user_id, operation)
        metadata_patch = {
            "evidence_count": len(evidence),
            "positive_count": len(evidence),
            "negative_count": 0,
            "source": f"{context.source}:aggregate",
        }
        if existing:
            aggregate_update = MemoryOperation(
                action="update",
                memory_type=operation.memory_type,
                title=operation.title,
                text=aggregate_text,
                tags=sorted(set([*operation.tags, operation.memory_type, "aggregated"])),
                confidence=operation.confidence,
                target=existing["path"],
                rationale="Aggregated repeated evidence into existing memory.",
            )
            update = self.updater.apply_upsert(
                self.resolver.resolve(aggregate_update, context.user_id),
                user_id=context.user_id,
                source=f"{context.source}:aggregate",
                metadata_patch=metadata_patch,
            )
            diff_operations["updates"].append(
                {
                    "uri": update["uri"],
                    "memory_type": operation.memory_type,
                    "before": update["before"],
                    "after": update["after"],
                    "rationale": "Aggregated repeated evidence into existing memory.",
                    "evidence_days": days,
                }
            )
            return
        aggregate_operation = MemoryOperation(
            action="add",
            memory_type=operation.memory_type,
            title=operation.title,
            text=aggregate_text,
            tags=sorted(set([*operation.tags, operation.memory_type, "aggregated"])),
            confidence=operation.confidence,
            rationale="Aggregated repeated evidence into a durable memory.",
        )
        created = self.updater.apply_upsert(
            ResolvedMemoryOperation(
                operation=aggregate_operation,
                target=None,
                fields={
                    "title": aggregate_operation.title,
                    "content": aggregate_operation.text,
                    "tags": aggregate_operation.tags,
                    "confidence": aggregate_operation.confidence,
                },
            ),
            user_id=context.user_id,
            source=f"{context.source}:aggregate",
            metadata_patch=metadata_patch,
        )
        diff_operations["adds"].append(
            {
                "uri": created["uri"],
                "memory_type": operation.memory_type,
                "after": aggregate_text,
                "rationale": "Aggregated repeated evidence into a durable memory.",
                "evidence_days": days,
            }
        )

    def _matching_evidence(self, user_id: str, operation: MemoryOperation) -> list[dict[str, Any]]:
        evidence_tag = f"{operation.memory_type}_evidence"
        rows = self.store.hybrid_search(operation.text, user_id=user_id, memory_type="event", limit=50)
        matches = []
        for row in rows:
            tags = {str(tag) for tag in row.get("tags", [])}
            if evidence_tag in tags:
                matches.append(row)
        return matches

    def _existing_aggregate(self, user_id: str, operation: MemoryOperation) -> dict[str, Any] | None:
        rows = self.store.hybrid_search(operation.text, user_id=user_id, memory_type=operation.memory_type, limit=5)
        return rows[0] if rows else None

    def _existing_topic_memory(self, user_id: str, operation: MemoryOperation, memory_type: str) -> dict[str, Any] | None:
        candidates = self._topic_memories(user_id, operation, memory_type, limit=8)
        if not candidates:
            return None
        topic_tokens = self._topic_tokens(operation)
        for item in candidates:
            item_tokens = self._topic_tokens_from_text(
                f"{item.get('title', '')} {' '.join(item.get('tags', []))} {item.get('abstract', '')} {item.get('content', '')}"
            )
            if topic_tokens & item_tokens:
                return item
        return candidates[0]

    def _existing_rolling_aggregate(self, memories: list[dict[str, Any]]) -> dict[str, Any] | None:
        for item in memories:
            tags = {str(tag) for tag in item.get("tags", [])}
            if "aggregated" in tags:
                return item
        return None

    def _topic_memories(self, user_id: str, operation: MemoryOperation, memory_type: str, limit: int) -> list[dict[str, Any]]:
        query = " ".join([operation.title, operation.text, *operation.tags]).strip()
        return self.store.hybrid_search(query, user_id=user_id, memory_type=memory_type, limit=limit)

    def _merge_topic_text(self, existing: dict[str, Any], operation: MemoryOperation) -> str:
        body = str(existing.get("content", "")).strip()
        if operation.text.strip() in body:
            return body
        return body + f"\n\n## Update {utc_now()}\n\n{operation.text.strip()}\n"

    def _compact_profile_text(self, current_body: str, new_text: str) -> str:
        current_lines = [line.strip() for line in current_body.splitlines() if line.strip() and not line.startswith("#")]
        new_lines = [line.strip() for line in new_text.splitlines() if line.strip() and not line.startswith("#")]
        seen = set()
        compacted = []
        for line in [*current_lines, *new_lines]:
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            compacted.append(line)
        return "\n".join(compacted[-20:]).strip()

    def _rolling_aggregate_text(self, memory_type: str, items: list[dict[str, Any]]) -> str:
        lines = [
            f"Rolling {memory_type} summary.",
            "",
            "## Evidence",
        ]
        for item in items[:12]:
            lines.append(f"- {item.get('updated_at', '')[:10]} {item.get('path')}: {item.get('abstract')}")
        return "\n".join(lines).strip()

    def _aggregate_text(self, operation: MemoryOperation, evidence: list[dict[str, Any]], days: list[str]) -> str:
        lines = [
            operation.text.strip(),
            "",
            "## Evidence",
            f"- Distinct days: {len(days)} ({', '.join(days[:8])})",
            f"- Evidence count: {len(evidence)}",
        ]
        for item in evidence[:8]:
            lines.append(f"- {item.get('updated_at', '')[:10]} {item.get('path')}: {item.get('abstract')}")
        return "\n".join(lines).strip()

    def _distinct_days(self, evidence: list[dict[str, Any]]) -> list[str]:
        days = []
        for item in evidence:
            tags = [str(tag) for tag in item.get("tags", [])]
            day = next((tag for tag in tags if self._looks_like_day(tag)), "")
            if not day:
                day = str(item.get("updated_at", ""))[:10]
            if day and day not in days:
                days.append(day)
        return sorted(days)

    def _event_type(self, operation: MemoryOperation) -> str:
        if operation.tags:
            return str(operation.tags[0])
        return operation.memory_type

    def _has_explicit_user_intent(self, operation: MemoryOperation) -> bool:
        tags = {str(tag) for tag in operation.tags}
        return "explicit_user_intent" in tags or "user_confirmed" in tags

    def _topic_tokens(self, operation: MemoryOperation) -> set[str]:
        return self._topic_tokens_from_text(" ".join([operation.title, operation.text, *operation.tags]))

    def _topic_tokens_from_text(self, text: str) -> set[str]:
        normalized = text.lower()
        tokens = set()
        current = []
        for ch in normalized:
            if ch.isalnum():
                current.append(ch)
            else:
                if current:
                    tokens.add("".join(current))
                    current = []
                if "\u4e00" <= ch <= "\u9fff":
                    tokens.add(ch)
        if current:
            tokens.add("".join(current))
        stopwords = {
            "user",
            "prefers",
            "preference",
            "likes",
            "wants",
            "usually",
            "memory",
            "the",
            "and",
            "when",
            "用户",
            "喜欢",
            "偏好",
            "希望",
            "通常",
        }
        return {token for token in tokens if token and token not in stopwords}

    def _looks_like_day(self, value: str) -> bool:
        if len(value) != 10:
            return False
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
        return True

    def _diff(self, diff_id: str, operations: dict[str, list], committed_at: str | None = "") -> dict:
        if committed_at == "":
            committed_at = utc_now()
        return {
            "diff_id": diff_id,
            "committed_at": committed_at,
            "operations": operations,
            "summary": {
                "total_adds": len(operations["adds"]),
                "total_updates": len(operations["updates"]),
                "total_deletes": len(operations["deletes"]),
                "total_ignores": len(operations["ignores"]),
            },
        }

    def _json(self, payload: dict) -> str:
        import json

        return json.dumps(payload, ensure_ascii=False, indent=2)
