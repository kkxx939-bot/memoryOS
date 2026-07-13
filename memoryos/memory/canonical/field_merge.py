"""Schema-driven materialization of complete canonical revision fields."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from memoryos.memory.canonical.event import canonical_digest, canonical_json
from memoryos.memory.schema import FieldMergeMode, MemoryTypeSchema


class FieldMergeError(ValueError):
    """An incoming partial state cannot be deterministically materialized."""


def _payload(value: Any) -> Any:
    serializer = getattr(value, "to_dict", None)
    return serializer() if callable(serializer) else value


@dataclass(frozen=True)
class FieldMergeDecision:
    field_name: str
    mode: str
    outcome: str
    before_digest: str
    incoming_digest: str
    after_digest: str

    def to_dict(self) -> dict[str, str]:
        return {
            "field_name": self.field_name,
            "mode": self.mode,
            "outcome": self.outcome,
            "before_digest": self.before_digest,
            "incoming_digest": self.incoming_digest,
            "after_digest": self.after_digest,
        }


@dataclass(frozen=True)
class FieldMergeResult:
    value_fields: dict[str, Any]
    field_evidence_refs: dict[str, tuple[Any, ...]]
    changed_fields: tuple[str, ...]
    unchanged_fields: tuple[str, ...]
    rejected_fields: tuple[str, ...]
    decisions: tuple[FieldMergeDecision, ...]
    merge_digest: str


class FieldMerger:
    """Apply finite MergeRule semantics and return one complete state."""

    VERSION = "canonical_field_merge_v1"

    def merge(
        self,
        schema: MemoryTypeSchema,
        current_value_fields: Mapping[str, Any],
        current_field_evidence: Mapping[str, tuple[Any, ...]],
        incoming_value_fields: Mapping[str, Any],
        incoming_field_evidence: Mapping[str, tuple[Any, ...]],
        *,
        relation: str,
        review_authority: bool,
    ) -> FieldMergeResult:
        current = dict(current_value_fields)
        incoming = dict(incoming_value_fields)
        undeclared = (set(current) | set(incoming)) - set(schema.allowed_value_fields())
        if undeclared:
            raise FieldMergeError(
                "canonical field merge contains fields outside the memory schema: " + ",".join(sorted(undeclared))
            )
        evidence = {str(key): tuple(value) for key, value in current_field_evidence.items()}
        merged = dict(current)
        decisions: list[FieldMergeDecision] = []
        changed: list[str] = []
        unchanged: list[str] = []
        rejected: list[str] = []

        for field_name in sorted(set(current) | set(incoming)):
            mode = schema.field_merge_rules.get(field_name, FieldMergeMode.REPLACE)
            before = current.get(field_name)
            has_incoming = field_name in incoming
            proposed = incoming.get(field_name)
            if not has_incoming:
                outcome, after = "preserved_missing", before
            elif field_name not in current:
                outcome, after = "initialized", self._incoming_value(mode, proposed, before=None)
            elif canonical_json(before) == canonical_json(proposed):
                outcome, after = "unchanged_equal", before
            elif mode == FieldMergeMode.IMMUTABLE:
                rejected.append(field_name)
                outcome, after = "rejected_immutable", before
            elif mode == FieldMergeMode.REPLACE:
                if self._destructive(before, proposed) and not review_authority:
                    rejected.append(field_name)
                    outcome, after = "rejected_replace_authority", before
                else:
                    outcome, after = "replaced", proposed
            elif mode == FieldMergeMode.APPEND_UNIQUE:
                outcome, after = "appended_unique", self._append_unique(before, proposed)
            elif mode == FieldMergeMode.PATCH_TEXT:
                try:
                    after = self._patch_text(before, proposed, review_authority=review_authority)
                except FieldMergeError:
                    rejected.append(field_name)
                    outcome, after = "rejected_patch_expression", before
                else:
                    outcome = "patched_text"
            else:  # pragma: no cover - Enum keeps this closed.
                raise FieldMergeError(f"unsupported merge mode: {mode}")

            changed_now = canonical_json(after) != canonical_json(before)
            if changed_now and field_name not in rejected:
                refs = tuple(incoming_field_evidence.get(f"value.{field_name}", ()))
                if not refs:
                    rejected.append(field_name)
                    outcome, after, changed_now = "rejected_missing_evidence", before, False
                else:
                    merged[field_name] = after
                    if mode == FieldMergeMode.APPEND_UNIQUE:
                        combined = (*evidence.get(f"value.{field_name}", ()), *refs)
                        evidence[f"value.{field_name}"] = tuple(
                            {canonical_json(_payload(item)): item for item in combined}.values()
                        )
                    else:
                        evidence[f"value.{field_name}"] = refs
                    changed.append(field_name)
            if not changed_now:
                unchanged.append(field_name)
                if field_name in current:
                    merged[field_name] = before
            decisions.append(
                FieldMergeDecision(
                    field_name=field_name,
                    mode=mode.value,
                    outcome=outcome,
                    before_digest=canonical_digest(before),
                    incoming_digest=canonical_digest(proposed) if has_incoming else canonical_digest(None),
                    after_digest=canonical_digest(after),
                )
            )

        # Semantic/transition evidence describes this new revision, while
        # unchanged value provenance remains bound to the prior evidence.
        for field_name, refs in incoming_field_evidence.items():
            if not str(field_name).startswith("value."):
                evidence[str(field_name)] = tuple(refs)

        if rejected:
            raise FieldMergeError("canonical field merge rejected fields: " + ",".join(sorted(dict.fromkeys(rejected))))
        payload = {
            "schema_version": self.VERSION,
            "memory_type": schema.memory_type.value,
            "relation": str(relation),
            "review_authority": bool(review_authority),
            "value_fields": merged,
            "field_evidence": {key: [_payload(ref) for ref in refs] for key, refs in sorted(evidence.items())},
            "decisions": [decision.to_dict() for decision in decisions],
        }
        return FieldMergeResult(
            value_fields=merged,
            field_evidence_refs=evidence,
            changed_fields=tuple(dict.fromkeys(changed)),
            unchanged_fields=tuple(dict.fromkeys(unchanged)),
            rejected_fields=(),
            decisions=tuple(decisions),
            merge_digest=canonical_digest(payload),
        )

    def _incoming_value(self, mode: FieldMergeMode, value: Any, *, before: Any) -> Any:
        if mode == FieldMergeMode.PATCH_TEXT:
            # Initial creation is a value initialization, not a patch.
            if before is None and not isinstance(value, Mapping):
                return value
            return self._patch_text(before or "", value, review_authority=True)
        if mode == FieldMergeMode.APPEND_UNIQUE:
            return self._append_unique([], value)
        return value

    def _append_unique(self, before: Any, incoming: Any) -> list[Any]:
        old_values = (
            list(before) if isinstance(before, list | tuple) else ([] if before is None or before == "" else [before])
        )
        new_values = list(incoming) if isinstance(incoming, list | tuple) else [incoming]
        result: list[Any] = []
        seen: set[str] = set()
        for value in (*old_values, *new_values):
            identity = canonical_json(value)
            if identity in seen:
                continue
            seen.add(identity)
            result.append(value)
        return result

    def _patch_text(self, before: Any, patch: Any, *, review_authority: bool) -> str:
        if not isinstance(before, str) or not isinstance(patch, Mapping):
            raise FieldMergeError("PATCH_TEXT requires a structured patch expression")
        operation = str(patch.get("op") or "").casefold()
        if operation == "append":
            text = patch.get("text")
            separator = str(patch.get("separator", "\n"))
            if not isinstance(text, str) or not text:
                raise FieldMergeError("PATCH_TEXT append requires text")
            return before + (separator if before else "") + text
        if operation == "splice":
            if patch.get("base_digest") != canonical_digest(before):
                raise FieldMergeError("PATCH_TEXT splice base digest mismatch")
            start, end, text = patch.get("start"), patch.get("end"), patch.get("text")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not isinstance(text, str)
                or start < 0
                or end < start
                or end > len(before)
            ):
                raise FieldMergeError("PATCH_TEXT splice range is invalid")
            return before[:start] + text + before[end:]
        if operation == "replace" and review_authority:
            value = patch.get("value")
            if not isinstance(value, str):
                raise FieldMergeError("PATCH_TEXT replace requires a string value")
            return value
        raise FieldMergeError("PATCH_TEXT expression is unsupported or unauthorized")

    def _destructive(self, before: Any, incoming: Any) -> bool:
        present = before is not None and before != "" and before != () and before != []
        if not present or canonical_json(before) == canonical_json(incoming):
            return False
        if incoming is None or incoming == "" or incoming == () or incoming == []:
            return True
        if isinstance(before, Mapping) and isinstance(incoming, Mapping):
            return not set(before).issubset(incoming)
        if isinstance(before, list | tuple) and isinstance(incoming, list | tuple):
            incoming_values = {canonical_json(item) for item in incoming}
            return any(canonical_json(item) not in incoming_values for item in before)
        return False
