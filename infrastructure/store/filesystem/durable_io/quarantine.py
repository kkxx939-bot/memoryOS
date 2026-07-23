"""在租户范围内隔离损坏的耐久控制文件。"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foundation.clock import utc_now
from foundation.integrity import canonical_digest
from infrastructure.store.filesystem.durable_io.atomic_json import atomic_write_json


@dataclass(frozen=True)
class QuarantineRecord:
    kind: str
    original_relative_path: str
    quarantined_relative_path: str
    metadata_relative_path: str
    error_type: str
    quarantined_at: str
    identifiers: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "control_file_quarantine_v1",
            "status": "quarantined",
            "kind": self.kind,
            "original_relative_path": self.original_relative_path,
            "quarantined_relative_path": self.quarantined_relative_path,
            "metadata_relative_path": self.metadata_relative_path,
            "error_type": self.error_type,
            "quarantined_at": self.quarantined_at,
            "identifiers": dict(self.identifiers),
        }


def quarantine_control_file(
    artifact_root: Path,
    path: Path,
    *,
    kind: str,
    error: BaseException,
    identifiers: dict[str, object] | None = None,
) -> QuarantineRecord:
    root = artifact_root.expanduser().resolve()
    requested = path.expanduser()
    if not requested.is_absolute():
        requested = requested.absolute()
    # 只解析父目录。解析末级路径会跟随伪造的控制文件符号链接，
    # 从而错误隔离其目标，而不是有问题的目录项。
    source = requested.parent.resolve() / requested.name
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError("control file is outside the tenant artifact root") from exc
    safe_identifiers = {
        str(key): str(value) for key, value in dict(identifiers or {}).items() if value not in {None, ""}
    }
    identity = canonical_digest(
        {
            "kind": kind,
            "relative_path": relative.as_posix(),
            "identifiers": safe_identifiers,
        }
    )[:24]
    quarantine_root = root / "system" / "quarantine" / kind
    quarantine_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        quarantine_root.chmod(0o700)
    except OSError:
        pass
    destination = quarantine_root / f"{identity}-{uuid.uuid4().hex}.original"
    metadata_path = destination.with_suffix(".json")
    if not source.exists() and not source.is_symlink():
        raise FileNotFoundError(path.name)
    os.replace(source, destination)
    try:
        destination.chmod(0o600)
    except OSError:
        pass
    record = QuarantineRecord(
        kind=kind,
        original_relative_path=relative.as_posix(),
        quarantined_relative_path=destination.relative_to(root).as_posix(),
        metadata_relative_path=metadata_path.relative_to(root).as_posix(),
        error_type=type(error).__name__,
        quarantined_at=utc_now(),
        identifiers=safe_identifiers,
    )
    payload = record.to_dict()
    payload["record_digest"] = canonical_digest(payload)
    atomic_write_json(metadata_path, payload, artifact_root=root)
    return record


def list_quarantine_records(artifact_root: Path) -> list[dict[str, Any]]:
    root = artifact_root / "system" / "quarantine"
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            records.append(
                {
                    "status": "quarantined",
                    "kind": "quarantine_metadata",
                    "error_type": "UnreadableQuarantineMetadata",
                    "metadata_relative_path": path.relative_to(artifact_root).as_posix(),
                }
            )
            continue
        digest = payload.get("record_digest") if isinstance(payload, dict) else None
        core = (
            {key: value for key, value in payload.items() if key != "record_digest"}
            if isinstance(payload, dict)
            else {}
        )
        if not isinstance(digest, str) or digest != canonical_digest(core):
            records.append(
                {
                    "status": "quarantined",
                    "kind": "quarantine_metadata",
                    "error_type": "CorruptQuarantineMetadata",
                    "metadata_relative_path": path.relative_to(artifact_root).as_posix(),
                }
            )
            continue
        records.append(payload)
    return records


__all__ = ["QuarantineRecord", "list_quarantine_records", "quarantine_control_file"]
