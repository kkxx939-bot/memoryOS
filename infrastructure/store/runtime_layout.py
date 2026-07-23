"""进程共享的租户运行目录布局。"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from foundation.ids import require_safe_path_segment
from infrastructure.store.filesystem.durable_io import atomic_create_bytes

RUNTIME_LAYOUT_SCHEMA = "context_runtime_v1"


class UnsupportedRuntimeLayout(RuntimeError):
    pass


class RuntimeResetRequired(UnsupportedRuntimeLayout):
    pass


def tenant_runtime_root(root: str | Path, tenant_id: str) -> Path:
    base = Path(root).expanduser().resolve(strict=False)
    tenant = require_safe_path_segment(tenant_id, "tenant_id")
    return base if tenant == "default" else base / "tenants" / tenant


@dataclass(frozen=True)
class RuntimeLayout:
    root: Path
    tenant_id: str

    @classmethod
    def open(cls, root: str | Path, *, tenant_id: str) -> RuntimeLayout:
        raw = str(root)
        if not raw or any(marker in raw for marker in ("$", "${", "*", "?", "[", "]")):
            raise UnsupportedRuntimeLayout("runtime root must be one explicit path")
        candidate = Path(raw).expanduser().absolute()
        for existing in (candidate, *candidate.parents):
            if existing.is_symlink():
                raise UnsupportedRuntimeLayout("runtime root cannot traverse a symbolic link")
        resolved = candidate.resolve(strict=False)
        tenant = require_safe_path_segment(tenant_id, "tenant_id")
        return cls(root=resolved, tenant_id=tenant)

    @property
    def tenant_root(self) -> Path:
        return tenant_runtime_root(self.root, self.tenant_id)

    @property
    def marker_path(self) -> Path:
        return self.tenant_root / "system" / "runtime-layout.json"

    def initialize_or_validate(self) -> dict[str, object]:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise UnsupportedRuntimeLayout("runtime root must be a real directory")
        try:
            self.root.chmod(0o700)
        except OSError:
            pass
        marker = self.marker_path
        if marker.is_symlink():
            raise UnsupportedRuntimeLayout("runtime layout marker cannot be a symbolic link")
        if marker.exists():
            return self._read_marker(marker)
        if self._contains_unversioned_state():
            raise RuntimeResetRequired(
                "runtime contains data without context_runtime_v1 marker; explicit reset is required"
            )
        payload: dict[str, object] = {
            "schema": RUNTIME_LAYOUT_SCHEMA,
            "tenant_id": self.tenant_id,
            "source_layout": "tenant-scoped runtime state",
        }
        encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode()
        atomic_create_bytes(marker, encoded, artifact_root=self.tenant_root)
        return payload

    def _read_marker(self, marker: Path) -> dict[str, object]:
        try:
            descriptor = os.open(marker, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise UnsupportedRuntimeLayout("runtime layout marker is unsafe") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise UnsupportedRuntimeLayout("runtime layout marker must be a regular file")
            if metadata.st_nlink != 1:
                raise UnsupportedRuntimeLayout("runtime layout marker cannot be hard-linked")
            raw = b""
            while True:
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    break
                raw += chunk
                if len(raw) > 64 * 1024:
                    raise UnsupportedRuntimeLayout("runtime layout marker is too large")
        finally:
            os.close(descriptor)
        try:
            payload = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UnsupportedRuntimeLayout("runtime layout marker is invalid") from exc
        if not isinstance(payload, dict) or payload.get("schema") != RUNTIME_LAYOUT_SCHEMA:
            raise UnsupportedRuntimeLayout("runtime layout schema is unsupported")
        if payload.get("tenant_id") != self.tenant_id:
            raise UnsupportedRuntimeLayout("runtime layout tenant binding does not match")
        return payload

    def _contains_unversioned_state(self) -> bool:
        tenant_root = self.tenant_root
        if not tenant_root.exists():
            return False
        allowed_empty_parents = {tenant_root / "system"}
        for child in tenant_root.iterdir():
            if child in allowed_empty_parents and child.is_dir() and not any(child.iterdir()):
                continue
            return True
        return False


__all__ = [
    "RUNTIME_LAYOUT_SCHEMA",
    "RuntimeLayout",
    "RuntimeResetRequired",
    "UnsupportedRuntimeLayout",
    "tenant_runtime_root",
]
