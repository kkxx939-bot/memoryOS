from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SearchReplaceBlock:
    search: str
    replace: str


class MergeOp(Protocol):
    def apply(self, current_value: Any, patch_value: Any) -> Any:
        """Return the merged value."""


class ReplaceOp:
    def apply(self, current_value: Any, patch_value: Any) -> Any:
        if patch_value is None or patch_value == "":
            return current_value
        return patch_value


class ImmutableOp:
    def apply(self, current_value: Any, patch_value: Any) -> Any:
        if current_value in {None, ""}:
            return patch_value
        return current_value


class SumOp:
    def apply(self, current_value: Any, patch_value: Any) -> Any:
        if patch_value is None or patch_value == "":
            return current_value
        if current_value is None or current_value == "":
            return patch_value
        try:
            if isinstance(current_value, float) or isinstance(patch_value, float):
                return float(current_value) + float(patch_value)
            return int(current_value) + int(patch_value)
        except (TypeError, ValueError):
            return current_value


class PatchOp:
    def apply(self, current_value: Any, patch_value: Any) -> Any:
        if current_value is None:
            return self._initial_value(patch_value)
        current = str(current_value)
        blocks = self._blocks(patch_value)
        if blocks:
            merged = current
            for block in blocks:
                if not block.search or block.search not in merged:
                    continue
                merged = merged.replace(block.search, block.replace, 1)
            return merged
        if patch_value is None or patch_value == "":
            return current_value
        patch = str(patch_value).strip()
        if not patch or patch in current:
            return current_value
        return current.rstrip() + "\n\n" + patch + "\n"

    def _initial_value(self, patch_value: Any) -> str:
        blocks = self._blocks(patch_value)
        if blocks:
            return blocks[0].replace
        return "" if patch_value is None else str(patch_value)

    def _blocks(self, patch_value: Any) -> list[SearchReplaceBlock]:
        if isinstance(patch_value, dict) and isinstance(patch_value.get("blocks"), list):
            blocks = []
            for raw in patch_value["blocks"]:
                if isinstance(raw, dict):
                    blocks.append(
                        SearchReplaceBlock(
                            search=str(raw.get("search", "")),
                            replace=str(raw.get("replace", "")),
                        )
                    )
            return blocks
        if isinstance(patch_value, list):
            blocks = []
            for raw in patch_value:
                if isinstance(raw, dict):
                    blocks.append(
                        SearchReplaceBlock(
                            search=str(raw.get("search", "")),
                            replace=str(raw.get("replace", "")),
                        )
                    )
            return blocks
        return []


def merge_op_factory(name: str) -> MergeOp:
    normalized = name.strip().lower()
    if normalized == "replace":
        return ReplaceOp()
    if normalized == "sum":
        return SumOp()
    if normalized == "immutable":
        return ImmutableOp()
    return PatchOp()
