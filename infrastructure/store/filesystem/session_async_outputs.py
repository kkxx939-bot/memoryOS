"""会话异步派生输出的 generation 发布与完整性校验。

异步输出不是不可变证据本体。它们按 task_id 写入独立 generation，只有完整
manifest 写入后才允许原子切换 current head；损坏的控制文件会被隔离。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from foundation.clock import utc_now
from foundation.ids import require_safe_path_segment
from foundation.integrity import canonical_digest, canonical_json
from infrastructure.store.filesystem.durable_io.quarantine import quarantine_control_file
from infrastructure.store.filesystem.file_lock import open_private_lock
from infrastructure.store.filesystem.session_archive_io import SessionArchiveFileIO
from infrastructure.store.filesystem.session_archive_layout import SessionArchiveLayout
from memory.commit.evidence.errors import (
    AsyncOutputIntegrityError,
    EvidenceArchiveConflictError,
)
from pre.session import SessionArchive

try:  # pragma: no cover - 生产使用的 POSIX 平台均提供 fcntl。
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

ASYNC_OUTPUT_MANIFEST_SCHEMA_VERSION = "session_async_output_manifest_v2"
ASYNC_OUTPUT_HEAD_SCHEMA_VERSION = "session_async_output_head_v2"

ASYNC_OUTPUT_FILES = (
    "abstract.md",
    "overview.md",
    "behavior_diff.json",
    "action_policy_diff.json",
    "context_diff.json",
    "commit_group_status.json",
)


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class SessionAsyncOutputStore:
    """管理会话异步输出的并发发布、读取、验证和隔离。"""

    def __init__(
        self,
        layout: SessionArchiveLayout,
        files: SessionArchiveFileIO,
        *,
        test_hook: Callable[[], Callable[[str, str], None] | None],
    ) -> None:
        self.layout = layout
        self.files = files
        self._test_hook = test_hook
        self.last_error = ""
        self._fallback_locks: dict[str, threading.RLock] = {}
        self._fallback_guard = threading.Lock()

    def write_outputs(
        self,
        archive_uri: str,
        abstract: str,
        overview: str,
        behavior_diff: dict,
        action_policy_diff: dict,
        context_diff: dict,
        tenant_id: str | None = None,
        commit_group_status: dict[str, Any] | None = None,
        complete: bool = True,
        task_id: str | None = None,
        created_at: str | None = None,
    ) -> Path:
        """写入 task generation，并在完整时按时间顺序发布 current head。"""

        directory = self.layout.directory(archive_uri, tenant_id=tenant_id)
        resolved_task_id = require_safe_path_segment(
            task_id or context_diff.get("task_id"),
            "async output task_id",
        )
        resolved_tenant = tenant_id or self.layout.tenant_id
        json_payloads = {
            "behavior_diff.json": behavior_diff,
            "action_policy_diff.json": action_policy_diff,
            "context_diff.json": context_diff,
            "commit_group_status.json": commit_group_status or {},
        }
        for filename, payload in json_payloads.items():
            if not isinstance(payload, dict):
                raise ValueError(f"{filename} must contain a JSON object")
            if filename == "commit_group_status.json":
                if complete and payload.get("task_id") != resolved_task_id:
                    raise ValueError("commit group status does not match async output task")
            elif payload.get("task_id") != resolved_task_id:
                raise ValueError(f"{filename} does not match async output task")
        resolved_created_at = str(created_at or (commit_group_status or {}).get("created_at") or utc_now())
        file_bytes = {
            "abstract.md": abstract.encode("utf-8"),
            "overview.md": overview.encode("utf-8"),
            **{filename: canonical_json(payload).encode("utf-8") for filename, payload in json_payloads.items()},
        }
        async_root = directory / "async_outputs"
        generation = async_root / resolved_task_id
        with self._output_lock(async_root):
            existing_head = self._read_head_optional(async_root / "current.json")
            if existing_head and existing_head.get("task_id") == resolved_task_id:
                existing = self._read_generation(directory, existing_head, resolved_task_id)
                desired_digests = {filename: _digest_bytes(content) for filename, content in file_bytes.items()}
                existing_digests = {
                    filename: str(details.get("digest") or "")
                    for filename, details in dict(existing["manifest"].get("files", {}) or {}).items()
                    if isinstance(details, dict)
                }
                if desired_digests != existing_digests:
                    raise EvidenceArchiveConflictError(
                        "published async output task cannot be overwritten with different bytes"
                    )
                return generation
            generation.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                generation.chmod(0o700)
            except OSError:
                pass
            for filename in ASYNC_OUTPUT_FILES:
                self.files.write_bytes_atomic(generation / filename, file_bytes[filename])
                self._notify(f"after_{filename}", resolved_task_id)
            self._notify("after_files", resolved_task_id)
            if not complete:
                return generation
            files = {
                filename: {
                    "digest": _digest_bytes(file_bytes[filename]),
                    "size": len(file_bytes[filename]),
                }
                for filename in ASYNC_OUTPUT_FILES
            }
            manifest_core = {
                "schema_version": ASYNC_OUTPUT_MANIFEST_SCHEMA_VERSION,
                "archive_uri": archive_uri,
                "task_id": resolved_task_id,
                "tenant_id": resolved_tenant,
                "files": files,
                "complete": True,
                "created_at": resolved_created_at,
            }
            manifest = {**manifest_core, "manifest_digest": canonical_digest(manifest_core)}
            self.files.write_immutable_json(generation / "manifest.json", manifest)
            self._notify("after_manifest", resolved_task_id)
            current_key = (
                (
                    str(existing_head.get("created_at") or ""),
                    str(existing_head.get("task_id") or ""),
                )
                if existing_head
                else ("", "")
            )
            requested_key = (resolved_created_at, resolved_task_id)
            if existing_head and requested_key < current_key:
                return generation
            head_core = {
                "schema_version": ASYNC_OUTPUT_HEAD_SCHEMA_VERSION,
                "archive_uri": archive_uri,
                "tenant_id": resolved_tenant,
                "task_id": resolved_task_id,
                "created_at": resolved_created_at,
                "manifest_relative_path": f"{resolved_task_id}/manifest.json",
                "manifest_digest": manifest["manifest_digest"],
            }
            head = {**head_core, "head_digest": canonical_digest(head_core)}
            self._notify("before_current", resolved_task_id)
            self.files.write_head(async_root / "current.json", head)
            self._notify("after_current", resolved_task_id)
        return generation

    def outputs_done_for_task(self, archive: SessionArchive) -> bool:
        """验证 current 是否完整指向指定 archive/task 的 generation。"""

        directory = self.layout.directory(
            archive.archive_uri,
            tenant_id=self.layout.archive_tenant(archive),
        )
        try:
            self._read_generation_for_archive(directory, archive)
        except FileNotFoundError:
            self.last_error = ""
            return False
        except AsyncOutputIntegrityError as exc:
            self.last_error = type(exc).__name__
            self._quarantine_controls(directory, archive, exc)
            return False
        self.last_error = ""
        return True

    def read_outputs(self, archive: SessionArchive) -> dict[str, Any]:
        """读取并完整校验指定归档的当前异步输出。"""

        directory = self.layout.directory(
            archive.archive_uri,
            tenant_id=self.layout.archive_tenant(archive),
        )
        return self._read_generation_for_archive(directory, archive)

    @contextmanager
    def _output_lock(self, async_root: Path) -> Iterator[None]:
        self.files.secure_directory(async_root)
        lock_path = async_root / ".publish.lock"
        if lock_path.is_symlink():
            raise AsyncOutputIntegrityError("async output lock cannot be a symbolic link")
        if fcntl is None:  # pragma: no cover
            key = str(lock_path)
            with self._fallback_guard:
                lock = self._fallback_locks.setdefault(key, threading.RLock())
            with lock:
                yield
            return
        descriptor = open_private_lock(lock_path, root=self.layout.root)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read_generation_for_archive(
        self,
        directory: Path,
        archive: SessionArchive,
    ) -> dict[str, Any]:
        head_path = directory / "async_outputs" / "current.json"
        head = self._read_head_optional(head_path)
        if head is None:
            raise FileNotFoundError(head_path.name)
        tenant_id = self.layout.archive_tenant(archive)
        if (
            head.get("archive_uri") != archive.archive_uri
            or head.get("tenant_id") != tenant_id
            or head.get("task_id") != archive.task_id
        ):
            raise FileNotFoundError(archive.task_id)
        return self._read_generation(directory, head, archive.task_id)

    def _read_head_optional(self, path: Path) -> dict[str, Any] | None:
        if path.is_symlink():
            raise AsyncOutputIntegrityError("async output head cannot be a symbolic link")
        if not path.exists():
            return None
        try:
            head = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AsyncOutputIntegrityError("async output head is unreadable") from exc
        if not isinstance(head, dict):
            raise AsyncOutputIntegrityError("async output head must be a JSON object")
        core = {key: value for key, value in head.items() if key != "head_digest"}
        if head.get("schema_version") != ASYNC_OUTPUT_HEAD_SCHEMA_VERSION or head.get(
            "head_digest"
        ) != canonical_digest(core):
            raise AsyncOutputIntegrityError("async output head digest is corrupt")
        task_id = require_safe_path_segment(head.get("task_id"), "async output head task_id")
        if head.get("manifest_relative_path") != f"{task_id}/manifest.json":
            raise AsyncOutputIntegrityError("async output head manifest path is invalid")
        return head

    def _read_generation(
        self,
        directory: Path,
        head: dict[str, Any],
        task_id: str,
    ) -> dict[str, Any]:
        safe_task = require_safe_path_segment(task_id, "async output task_id")
        generation = directory / "async_outputs" / safe_task
        manifest_path = generation / "manifest.json"
        if generation.is_symlink() or manifest_path.is_symlink():
            raise AsyncOutputIntegrityError("async output generation cannot be a symbolic link")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AsyncOutputIntegrityError("async output manifest is unreadable") from exc
        if not isinstance(manifest, dict):
            raise AsyncOutputIntegrityError("async output manifest must be a JSON object")
        core = {key: value for key, value in manifest.items() if key != "manifest_digest"}
        if (
            manifest.get("schema_version") != ASYNC_OUTPUT_MANIFEST_SCHEMA_VERSION
            or manifest.get("manifest_digest") != canonical_digest(core)
            or manifest.get("manifest_digest") != head.get("manifest_digest")
            or manifest.get("archive_uri") != head.get("archive_uri")
            or manifest.get("tenant_id") != head.get("tenant_id")
            or manifest.get("task_id") != safe_task
            or manifest.get("complete") is not True
        ):
            raise AsyncOutputIntegrityError("async output manifest identity or digest is corrupt")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != set(ASYNC_OUTPUT_FILES):
            raise AsyncOutputIntegrityError("async output manifest file set is incomplete")
        result: dict[str, Any] = {"head": head, "manifest": manifest}
        for filename in ASYNC_OUTPUT_FILES:
            details = files.get(filename)
            if not isinstance(details, dict):
                raise AsyncOutputIntegrityError("async output file proof is invalid")
            path = generation / filename
            if path.is_symlink():
                raise AsyncOutputIntegrityError("async output file cannot be a symbolic link")
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise AsyncOutputIntegrityError("async output file is missing") from exc
            if details.get("digest") != _digest_bytes(raw) or details.get("size") != len(raw):
                raise AsyncOutputIntegrityError("async output file digest is corrupt")
            if filename.endswith(".json"):
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise AsyncOutputIntegrityError("async output JSON is corrupt") from exc
                if not isinstance(payload, dict) or payload.get("task_id") != safe_task:
                    raise AsyncOutputIntegrityError("async output file task identity is mixed")
                if filename == "commit_group_status.json" and (
                    payload.get("archive_uri") != head.get("archive_uri")
                    or payload.get("tenant_id") != head.get("tenant_id")
                ):
                    raise AsyncOutputIntegrityError("async output commit group identity is mixed")
                result[filename.removesuffix(".json")] = payload
            else:
                result[filename.removesuffix(".md")] = raw.decode("utf-8")
        return result

    def _quarantine_controls(
        self,
        directory: Path,
        archive: SessionArchive,
        error: BaseException,
    ) -> None:
        tenant_id = self.layout.archive_tenant(archive)
        artifact_root = self.layout.root if tenant_id == "default" else self.layout.root / "tenants" / tenant_id
        controls = [
            directory / "async_outputs" / "current.json",
            directory / "async_outputs" / archive.task_id / "manifest.json",
        ]
        for path in controls:
            if path.exists() or path.is_symlink():
                quarantine_control_file(
                    artifact_root,
                    path,
                    kind="async_output",
                    error=error,
                    identifiers={
                        "task_id": archive.task_id,
                        "archive_uri_digest": canonical_digest(archive.archive_uri),
                    },
                )

    def _notify(self, stage: str, task_id: str) -> None:
        hook = self._test_hook()
        if hook is not None:
            hook(stage, task_id)


__all__ = ["SessionAsyncOutputStore"]
