"""操作提交里的重做日志。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from memoryos.core.durable_io import atomic_write_json
from memoryos.core.durable_io.quarantine import QuarantineRecord, quarantine_control_file
from memoryos.core.ids import require_safe_path_segment
from memoryos.core.integrity import canonical_digest
from memoryos.operations.model.context_operation import ContextOperation


class RedoIntegrityError(RuntimeError):
    """The durable redo phase does not match the current SourceStore effect."""


class RedoControlFileError(RedoIntegrityError):
    """One or more corrupt redo files were moved out of the live scan path."""

    def __init__(self, records: list[QuarantineRecord]) -> None:
        super().__init__("corrupt redo control file quarantined")
        self.records = records


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
    SCHEMA_VERSION = "transaction_redo_v1"
    PHASES = {
        "begin",
        "started",
        "tombstones_enqueued",
        "source_written",
        "index_written",
        "audit_written",
        "diff_written",
        "head_published",
        "committed",
    }

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
        if phase not in self.PHASES:
            raise ValueError(f"unsupported redo phase: {phase}")
        self.redo_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.redo_dir / f"{operation_id}.json"
        payload = {
            **operation.to_dict(),
            "control_schema_version": self.SCHEMA_VERSION,
            "redo_operation_id": operation.operation_id,
            "redo_user_id": operation.user_id,
            "redo_tenant_id": str(operation.payload.get("tenant_id") or "default"),
            "redo_phase": phase,
        }
        if source_effect is not None:
            payload["redo_source_effect"] = source_effect
        if relation_manifest is not None:
            payload["redo_relation_manifest"] = relation_manifest
        payload["redo_digest"] = canonical_digest(payload)
        atomic_write_json(path, payload, artifact_root=self.root)
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
                payload = self._read_payload(path)
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
        if path.is_symlink():
            raise RedoIntegrityError("redo control file cannot be a symbolic link")
        if path.exists():
            path.unlink()
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    def pending(self) -> list[ContextOperation]:
        return [entry.operation for entry in self.pending_entries()]

    def pending_entries(self) -> list[RedoEntry]:
        if not self.redo_dir.exists():
            return []
        entries: list[RedoEntry] = []
        quarantined: list[QuarantineRecord] = []
        for path in sorted(self.redo_dir.glob("*.json")):
            try:
                payload = self._read_payload(path)
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                quarantined.append(
                    quarantine_control_file(
                        self.root,
                        path,
                        kind="redo",
                        error=exc,
                        identifiers={"file_id": path.stem},
                    )
                )
                continue
            source_effect = payload.get("redo_source_effect")
            relation_manifest = payload.get("redo_relation_manifest")
            entries.append(
                RedoEntry(
                    operation=ContextOperation.from_dict(payload),
                    phase=str(payload.get("redo_phase", "started")),
                    source_effect=dict(source_effect) if isinstance(source_effect, dict) else None,
                    relation_manifest=(dict(relation_manifest) if isinstance(relation_manifest, dict) else None),
                )
            )
        if quarantined:
            raise RedoControlFileError(quarantined)
        return entries

    def _read_payload(self, path: Path) -> dict:
        if path.is_symlink():
            raise ValueError("redo control file cannot be a symbolic link")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("redo control file must be a JSON object")
        digest = payload.get("redo_digest")
        core = {key: value for key, value in payload.items() if key != "redo_digest"}
        if (
            payload.get("control_schema_version") != self.SCHEMA_VERSION
            or not isinstance(digest, str)
            or digest != canonical_digest(core)
            or payload.get("redo_phase") not in self.PHASES
        ):
            raise ValueError("redo control file integrity check failed")
        operation = ContextOperation.from_dict(payload)
        if (
            path.stem != operation.operation_id
            or payload.get("redo_operation_id") != operation.operation_id
            or payload.get("redo_user_id") != operation.user_id
            or payload.get("redo_tenant_id") != str(operation.payload.get("tenant_id") or "default")
        ):
            raise ValueError("redo control file crosses its operation boundary")
        source_effect = payload.get("redo_source_effect")
        relation_manifest = payload.get("redo_relation_manifest")
        if source_effect is not None and not isinstance(source_effect, dict):
            raise ValueError("redo Source effect must be an object")
        if relation_manifest is not None and not isinstance(relation_manifest, dict):
            raise ValueError("redo Relation manifest must be an object")
        return payload
