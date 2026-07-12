"""操作提交里的重做日志。"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from memoryos.core.ids import require_safe_path_segment
from memoryos.operations.model.context_operation import ContextOperation


class RedoIntegrityError(RuntimeError):
    """The durable redo phase does not match the current SourceStore effect."""


@dataclass(frozen=True)
class RedoEntry:
    operation: ContextOperation
    phase: str
    source_effect: dict | None = None
    relation_manifest: dict | None = None

    @property
    def operation_id(self) -> str:
        return self.operation.operation_id

    @property
    def target_uri(self) -> str | None:
        return self.operation.target_uri

    @property
    def user_id(self) -> str:
        return self.operation.user_id


class RedoLog:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.redo_dir = self.root / "system" / "redo"

    def begin(
        self,
        operation: ContextOperation,
        phase: str = "begin",
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> Path:
        operation_id = require_safe_path_segment(operation.operation_id, "operation_id")
        self.redo_dir.mkdir(parents=True, exist_ok=True)
        path = self.redo_dir / f"{operation_id}.json"
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        payload = {**operation.to_dict(), "redo_phase": phase}
        if source_effect is not None:
            payload["redo_source_effect"] = source_effect
        if relation_manifest is not None:
            payload["redo_relation_manifest"] = relation_manifest
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def advance(
        self,
        operation: ContextOperation,
        phase: str,
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> Path:
        operation_id = require_safe_path_segment(operation.operation_id, "operation_id")
        if source_effect is None or relation_manifest is None:
            path = self.redo_dir / f"{operation_id}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                stored_effect = payload.get("redo_source_effect")
                if source_effect is None and isinstance(stored_effect, dict):
                    source_effect = stored_effect
                stored_manifest = payload.get("redo_relation_manifest")
                if relation_manifest is None and isinstance(stored_manifest, dict):
                    relation_manifest = stored_manifest
        return self.begin(
            operation,
            phase=phase,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
        )

    def commit(self, operation_id: str) -> None:
        operation_id = require_safe_path_segment(operation_id, "operation_id")
        path = self.redo_dir / f"{operation_id}.json"
        if path.exists():
            path.unlink()

    def pending(self) -> list[ContextOperation]:
        return [entry.operation for entry in self.pending_entries()]

    def pending_entries(self) -> list[RedoEntry]:
        if not self.redo_dir.exists():
            return []
        entries: list[RedoEntry] = []
        for path in sorted(self.redo_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            source_effect = payload.get("redo_source_effect")
            relation_manifest = payload.get("redo_relation_manifest")
            entries.append(
                RedoEntry(
                    operation=ContextOperation.from_dict(payload),
                    phase=str(payload.get("redo_phase", "started")),
                    source_effect=dict(source_effect) if isinstance(source_effect, dict) else None,
                    relation_manifest=(
                        dict(relation_manifest) if isinstance(relation_manifest, dict) else None
                    ),
                )
            )
        return entries
