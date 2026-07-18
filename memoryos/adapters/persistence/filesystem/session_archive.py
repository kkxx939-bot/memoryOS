"""Content-addressed, immutable session evidence archives."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.session.errors import (
    AsyncOutputIntegrityError,
    EvidenceArchiveConflictError,
    EvidenceArchiveIntegrityError,
)
from memoryos.contextdb.session.evidence_encoder import session_evidence_encoder
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.clock import utc_now
from memoryos.core.durable_io.quarantine import quarantine_control_file
from memoryos.core.file_lock import open_private_lock
from memoryos.core.ids import require_safe_path_segment
from memoryos.core.integrity import canonical_digest, canonical_json, canonicalize

try:  # pragma: no cover - production POSIX platforms provide fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

ARCHIVE_MANIFEST_SCHEMA_VERSION = "session_archive_manifest_v2"
ARCHIVE_HEAD_SCHEMA_VERSION = "session_archive_head_v2"
ASYNC_OUTPUT_MANIFEST_SCHEMA_VERSION = "session_async_output_manifest_v1"
ASYNC_OUTPUT_HEAD_SCHEMA_VERSION = "session_async_output_head_v1"


def canonical_digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class SessionArchiveStore:
    _MAX_ENUMERATION_ENTRIES = 10_000
    _COLLECTION_FILES = {
        "messages": "messages.jsonl",
        "observations": "observations.jsonl",
        "predictions": "predictions.jsonl",
        "action_results": "action_results.jsonl",
        "feedback": "feedback.jsonl",
        "used_contexts": "used_contexts.json",
        "used_skills": "used_skills.json",
        "tool_results": "tool_results.jsonl",
    }

    _ASYNC_FILES = (
        "abstract.md",
        "overview.md",
        "memory_diff.json",
        "behavior_diff.json",
        "action_policy_diff.json",
        "context_diff.json",
        "commit_group_status.json",
    )

    def __init__(
        self,
        root: str | Path,
        tenant_id: str = "default",
        *,
        test_hook: Callable[[str, str], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.tenant_id = tenant_id
        self.test_hook = test_hook
        self.last_async_output_error = ""
        self._fallback_locks: dict[str, threading.RLock] = {}
        self._fallback_guard = threading.Lock()

    def write_sync_archive(self, archive: SessionArchive) -> Path:
        tenant_id = self._archive_tenant(archive)
        self._materialize_archive_tenant(archive, tenant_id)
        directory = self._dir(archive.archive_uri, tenant_id=tenant_id)
        head_path = directory / "commit_head.json"
        if head_path.is_symlink():
            raise EvidenceArchiveIntegrityError(
                "session archive head cannot be a symbolic link"
            )
        self._secure_directory(directory)

        collections: dict[str, str] = {}
        for name in self._COLLECTION_FILES:
            payload = canonicalize(getattr(archive, name))
            digest = canonical_digest(payload)
            self._write_immutable_json(directory / "evidence" / "objects" / f"{digest}.json", payload)
            collections[name] = digest

        event_refs = []
        for event in session_evidence_encoder().encode(archive):
            payload = event.payload
            self._write_immutable_json(
                directory / "evidence" / "events" / f"{event.event_digest}.json",
                payload,
            )
            event_refs.append(event.manifest_reference())

        archive_core = {
            "schema_version": archive.schema_version,
            "tenant_id": tenant_id,
            "user_id": archive.user_id,
            "session_id": archive.session_id,
            "archive_uri": archive.archive_uri,
            "created_at": archive.created_at,
            "metadata": archive.metadata,
            "event_digests": [item["event_digest"] for item in event_refs],
            "collection_digests": collections,
        }
        archive_digest = canonical_digest(archive_core)
        manifest_core = {
            "schema_version": ARCHIVE_MANIFEST_SCHEMA_VERSION,
            "archive_schema_version": archive.schema_version,
            "archive_digest": archive_digest,
            "tenant_id": tenant_id,
            "task_id": archive.task_id,
            "user_id": archive.user_id,
            "session_id": archive.session_id,
            "archive_uri": archive.archive_uri,
            "created_at": archive.created_at,
            "metadata": archive.metadata,
            "collections": collections,
            "events": event_refs,
        }
        manifest_digest = canonical_digest(manifest_core)
        manifest = {**manifest_core, "manifest_digest": manifest_digest}
        manifest_path = directory / "evidence" / "manifests" / f"{manifest_digest}.json"
        self._write_immutable_json(manifest_path, manifest)

        manifest_uri = self._manifest_uri(archive.archive_uri, manifest_digest)
        archive.archive_digest = archive_digest
        archive.manifest_digest = manifest_digest
        archive.manifest_uri = manifest_uri
        self._write_head(
            head_path,
            {
                "schema_version": ARCHIVE_HEAD_SCHEMA_VERSION,
                "archive_uri": archive.archive_uri,
                "tenant_id": tenant_id,
                "user_id": archive.user_id,
                "archive_digest": archive_digest,
                "manifest_digest": manifest_digest,
                "manifest_uri": manifest_uri,
            },
        )
        return directory

    def write_async_outputs(
        self,
        archive_uri: str,
        abstract: str,
        overview: str,
        memory_diff: dict,
        behavior_diff: dict,
        action_policy_diff: dict,
        context_diff: dict,
        tenant_id: str | None = None,
        commit_group_status: dict[str, Any] | None = None,
        complete: bool = True,
        task_id: str | None = None,
        created_at: str | None = None,
    ) -> Path:
        directory = self._dir(archive_uri, tenant_id=tenant_id)
        resolved_task_id = require_safe_path_segment(
            task_id or memory_diff.get("task_id"),
            "async output task_id",
        )
        resolved_tenant = tenant_id or self.tenant_id
        json_payloads = {
            "memory_diff.json": memory_diff,
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
        resolved_created_at = str(
            created_at
            or (commit_group_status or {}).get("created_at")
            or utc_now()
        )
        file_bytes = {
            "abstract.md": abstract.encode("utf-8"),
            "overview.md": overview.encode("utf-8"),
            **{
                filename: canonical_json(payload).encode("utf-8")
                for filename, payload in json_payloads.items()
            },
        }
        async_root = directory / "async_outputs"
        generation = async_root / resolved_task_id
        with self._async_output_lock(async_root):
            existing_head = self._read_async_head_optional(async_root / "current.json")
            if existing_head and existing_head.get("task_id") == resolved_task_id:
                existing = self._read_async_generation(directory, existing_head, resolved_task_id)
                desired_digests = {
                    filename: canonical_digest_bytes(content)
                    for filename, content in file_bytes.items()
                }
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
            for filename in self._ASYNC_FILES:
                self._write_bytes_atomic(generation / filename, file_bytes[filename])
                self._notify_async(f"after_{filename}", resolved_task_id)
            self._notify_async("after_files", resolved_task_id)
            if not complete:
                return generation
            files = {
                filename: {
                    "digest": canonical_digest_bytes(file_bytes[filename]),
                    "size": len(file_bytes[filename]),
                }
                for filename in self._ASYNC_FILES
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
            self._write_immutable_json(generation / "manifest.json", manifest)
            self._notify_async("after_manifest", resolved_task_id)
            current_key = (
                str(existing_head.get("created_at") or ""),
                str(existing_head.get("task_id") or ""),
            ) if existing_head else ("", "")
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
            self._notify_async("before_current", resolved_task_id)
            self._write_head(async_root / "current.json", head)
            self._notify_async("after_current", resolved_task_id)
        return generation

    def async_outputs_done_for_task(self, archive: SessionArchive) -> bool:
        directory = self._dir(archive.archive_uri, tenant_id=self._archive_tenant(archive))
        try:
            self._read_async_generation_for_archive(directory, archive)
        except FileNotFoundError:
            self.last_async_output_error = ""
            return False
        except AsyncOutputIntegrityError as exc:
            self.last_async_output_error = type(exc).__name__
            self._quarantine_async_output_controls(directory, archive, exc)
            return False
        self.last_async_output_error = ""
        return True

    def read_async_outputs(self, archive: SessionArchive) -> dict[str, Any]:
        directory = self._dir(archive.archive_uri, tenant_id=self._archive_tenant(archive))
        return self._read_async_generation_for_archive(directory, archive)

    def read_archive(
        self,
        archive_uri: str,
        *,
        tenant_id: str | None = None,
        manifest_digest: str | None = None,
    ) -> SessionArchive:
        effective_tenant = tenant_id or self.tenant_id
        parsed_uri = ContextURI.parse(archive_uri)
        directory = self._dir(archive_uri, tenant_id=tenant_id)
        head_path = directory / "commit_head.json"
        if not manifest_digest and head_path.is_symlink():
            raise EvidenceArchiveIntegrityError(
                "session archive head cannot be a symbolic link"
            )
        head = {} if manifest_digest else dict(self._read_json(head_path) or {})
        if head and str(head.get("schema_version") or "") != ARCHIVE_HEAD_SCHEMA_VERSION:
            raise EvidenceArchiveIntegrityError("unsupported session archive head schema")
        if head and str(head.get("archive_uri") or "") != archive_uri:
            raise EvidenceArchiveIntegrityError("session archive head URI mismatch")
        if head and str(head.get("tenant_id") or "") != effective_tenant:
            raise EvidenceArchiveIntegrityError("session archive head tenant mismatch")
        if head and str(head.get("user_id") or "") != str(parsed_uri.user_id or ""):
            raise EvidenceArchiveIntegrityError("session archive head user mismatch")
        selected = manifest_digest or str(head.get("manifest_digest") or "")
        if not selected:
            raise EvidenceArchiveIntegrityError("session archive head has no manifest digest")
        archive = self._read_v2_archive(
            directory,
            archive_uri,
            selected,
            tenant_id=effective_tenant,
        )
        if head and str(head.get("archive_digest") or "") != archive.archive_digest:
            raise EvidenceArchiveIntegrityError("session archive head aggregate digest mismatch")
        return archive

    def read_archive_from_commit_head(
        self,
        head_path: Path,
        *,
        tenant_id: str,
        user_id: str,
    ) -> SessionArchive:
        """Read one enumerated archive only after proving its head path identity."""

        if head_path.is_symlink():
            raise EvidenceArchiveIntegrityError(
                "archive commit head cannot be a symbolic link"
            )
        try:
            head = self._read_json(head_path)
        except EvidenceArchiveIntegrityError as exc:
            raise EvidenceArchiveIntegrityError(
                f"archive commit head is unreadable: {head_path}"
            ) from exc
        if not isinstance(head, dict):
            raise EvidenceArchiveIntegrityError("archive commit head must be a JSON object")
        if str(head.get("schema_version") or "") != ARCHIVE_HEAD_SCHEMA_VERSION:
            raise EvidenceArchiveIntegrityError("archive commit head schema is invalid")
        if str(head.get("tenant_id") or "") != tenant_id:
            raise EvidenceArchiveIntegrityError("archive commit head tenant mismatch")
        if str(head.get("user_id") or "") != user_id:
            raise EvidenceArchiveIntegrityError("archive commit head user mismatch")
        archive_uri = str(head.get("archive_uri") or "")
        try:
            parsed_uri = ContextURI.parse(archive_uri)
        except ValueError as exc:
            raise EvidenceArchiveIntegrityError("archive commit head URI is invalid") from exc
        if parsed_uri.user_id != user_id:
            raise EvidenceArchiveIntegrityError("archive commit head URI user mismatch")
        expected_path = self._dir(archive_uri, tenant_id=tenant_id) / "commit_head.json"
        if head_path.is_symlink() or head_path.resolve() != expected_path.resolve():
            raise EvidenceArchiveIntegrityError("archive commit head path identity mismatch")
        archive = self.read_archive(archive_uri, tenant_id=tenant_id)
        if archive.user_id != user_id:
            raise EvidenceArchiveIntegrityError("archive manifest user mismatch")
        return archive

    def list_archives(
        self,
        *,
        tenant_id: str | None = None,
        after_archive_uri: str = "",
        limit: int = 256,
    ) -> tuple[SessionArchive, ...]:
        """Enumerate immutable archive heads through a bounded tenant tree.

        This is a recovery-only source scan, not an online retrieval path.  It
        deliberately walks only the fixed ``users/*/sessions/history/*``
        layout, rejects aliases, and caps every directory fan-out before
        sorting so an unexpected tree cannot turn startup into an unbounded
        filesystem crawl.
        """

        effective_tenant = tenant_id or self.tenant_id
        if effective_tenant != self.tenant_id:
            raise PermissionError("Session archive enumeration crossed the bound tenant")
        maximum = int(limit)
        if maximum <= 0 or maximum > 1_000:
            raise ValueError("Session archive enumeration limit must be between 1 and 1000")
        cursor = str(after_archive_uri or "")
        if cursor:
            parsed_cursor = ContextURI.parse(cursor)
            if (
                parsed_cursor.authority != "user"
                or len(parsed_cursor.segments) != 4
                or parsed_cursor.segments[1:3] != ("sessions", "history")
            ):
                raise ValueError("Session archive cursor is not an archive URI")

        users_root = self.root.resolve() / "tenants" / effective_tenant / "users"
        if not users_root.exists():
            return ()
        users = self._bounded_child_directories(users_root, label="Session user root")
        candidates: list[tuple[Path, str]] = []
        for user_root in users:
            user_id = require_safe_path_segment(user_root.name, "Session archive user_id")
            history_root = user_root / "sessions" / "history"
            if not history_root.exists():
                continue
            for session_root in self._bounded_child_directories(
                history_root,
                label="Session history root",
            ):
                head_path = session_root / "commit_head.json"
                if not head_path.exists() and not head_path.is_symlink():
                    continue
                candidates.append((head_path, user_id))
                if len(candidates) > self._MAX_ENUMERATION_ENTRIES:
                    raise EvidenceArchiveIntegrityError(
                        "Session archive tree exceeded its enumeration bound"
                    )
        archives = sorted(
            (
                self.read_archive_from_commit_head(
                    head_path,
                    tenant_id=effective_tenant,
                    user_id=user_id,
                )
                for head_path, user_id in candidates
            ),
            key=lambda archive: archive.archive_uri,
        )
        return tuple(archive for archive in archives if archive.archive_uri > cursor)[:maximum]

    def _bounded_child_directories(self, parent: Path, *, label: str) -> tuple[Path, ...]:
        if parent.is_symlink() or not parent.is_dir():
            raise EvidenceArchiveIntegrityError(f"{label} is unsafe")
        children: list[Path] = []
        for child in parent.iterdir():
            if child.is_symlink():
                raise EvidenceArchiveIntegrityError(f"{label} contains a symbolic link")
            if not child.is_dir():
                raise EvidenceArchiveIntegrityError(f"{label} contains a non-directory entry")
            children.append(child)
            if len(children) > self._MAX_ENUMERATION_ENTRIES:
                raise EvidenceArchiveIntegrityError(f"{label} exceeded its enumeration bound")
        return tuple(sorted(children, key=lambda path: path.name))

    def read_archive_at_manifest(
        self,
        archive_uri: str,
        manifest_digest: str,
        *,
        tenant_id: str | None = None,
    ) -> SessionArchive:
        return self.read_archive(archive_uri, tenant_id=tenant_id, manifest_digest=manifest_digest)

    def archive_exists(self, archive_uri: str, *, tenant_id: str | None = None) -> bool:
        directory = self._dir(archive_uri, tenant_id=tenant_id)
        path = directory / "commit_head.json"
        if path.is_symlink():
            raise EvidenceArchiveIntegrityError(
                "session archive head cannot be a symbolic link"
            )
        return path.exists()

    def archive_tenant(self, archive: SessionArchive) -> str:
        """Resolve the tenant path used by both archive writes and idempotent reads."""

        return self._archive_tenant(archive)

    def read_event(
        self,
        archive_uri: str,
        event_digest: str,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        directory = self._dir(archive_uri, tenant_id=tenant_id)
        path = directory / "evidence" / "events" / f"{event_digest}.json"
        payload = self._read_json(path)
        claimed = str(payload.get("event_digest") or "")
        body = {key: value for key, value in payload.items() if key != "event_digest"}
        if claimed != event_digest or canonical_digest(body) != event_digest:
            raise EvidenceArchiveIntegrityError(f"immutable event digest mismatch: {event_digest}")
        return payload

    def current_manifest(self, archive_uri: str, *, tenant_id: str | None = None) -> dict[str, Any]:
        directory = self._dir(archive_uri, tenant_id=tenant_id)
        head = self._read_json(directory / "commit_head.json")
        digest = str(head.get("manifest_digest") or "")
        if not digest:
            raise EvidenceArchiveIntegrityError("session archive head has no manifest digest")
        return self._read_manifest(directory, digest)

    def _read_v2_archive(
        self,
        directory: Path,
        archive_uri: str,
        manifest_digest: str,
        *,
        tenant_id: str,
    ) -> SessionArchive:
        manifest = self._read_manifest(directory, manifest_digest)
        if str(manifest.get("schema_version") or "") != ARCHIVE_MANIFEST_SCHEMA_VERSION:
            raise EvidenceArchiveIntegrityError("unsupported session archive manifest schema")
        if str(manifest.get("archive_uri")) != archive_uri:
            raise EvidenceArchiveIntegrityError("session archive manifest URI mismatch")
        if str(manifest.get("tenant_id") or "") != tenant_id:
            raise EvidenceArchiveIntegrityError("session archive manifest tenant mismatch")
        if str(manifest.get("user_id") or "") != str(ContextURI.parse(archive_uri).user_id or ""):
            raise EvidenceArchiveIntegrityError("session archive manifest user mismatch")
        for event_ref in manifest.get("events", []) or []:
            self.read_event(
                archive_uri,
                str(event_ref["event_digest"]),
                tenant_id=str(manifest.get("tenant_id") or self.tenant_id),
            )
        collection_refs = dict(manifest.get("collections", {}) or {})
        if set(collection_refs) != set(self._COLLECTION_FILES):
            raise EvidenceArchiveIntegrityError("session archive manifest collections are incomplete")
        collections = {
            name: self._read_content_object(directory, str(digest)) for name, digest in collection_refs.items()
        }
        if any(not isinstance(payload, list) for payload in collections.values()):
            raise EvidenceArchiveIntegrityError("session archive collection must be a list")
        archive = SessionArchive(
            user_id=str(manifest["user_id"]),
            session_id=str(manifest["session_id"]),
            archive_uri=archive_uri,
            messages=list(collections.get("messages", []) or []),
            observations=list(collections.get("observations", []) or []),
            predictions=list(collections.get("predictions", []) or []),
            action_results=list(collections.get("action_results", []) or []),
            feedback=list(collections.get("feedback", []) or []),
            used_contexts=list(collections.get("used_contexts", []) or []),
            used_skills=list(collections.get("used_skills", []) or []),
            tool_results=list(collections.get("tool_results", []) or []),
            metadata=dict(manifest.get("metadata", {}) or {}),
            task_id=str(manifest["task_id"]),
            created_at=str(manifest.get("created_at", "")),
            schema_version=str(manifest.get("archive_schema_version") or "session_archive_v2"),
            archive_digest=str(manifest.get("archive_digest") or ""),
            manifest_digest=manifest_digest,
            manifest_uri=self._manifest_uri(archive_uri, manifest_digest),
        )
        expected_archive_digest = canonical_digest(
            {
                "schema_version": archive.schema_version,
                "tenant_id": str(manifest.get("tenant_id") or self.tenant_id),
                "user_id": archive.user_id,
                "session_id": archive.session_id,
                "archive_uri": archive.archive_uri,
                "created_at": archive.created_at,
                "metadata": archive.metadata,
                "event_digests": [str(item["event_digest"]) for item in manifest.get("events", []) or []],
                "collection_digests": dict(manifest.get("collections", {}) or {}),
            }
        )
        if archive.archive_digest != expected_archive_digest:
            raise EvidenceArchiveIntegrityError("session archive aggregate digest mismatch")
        self._materialize_archive_tenant(archive, tenant_id)
        return archive

    def _read_manifest(self, directory: Path, digest: str) -> dict[str, Any]:
        path = directory / "evidence" / "manifests" / f"{digest}.json"
        manifest = self._read_json(path)
        claimed = str(manifest.get("manifest_digest") or "")
        body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
        if claimed != digest or canonical_digest(body) != digest:
            raise EvidenceArchiveIntegrityError(f"immutable manifest digest mismatch: {digest}")
        return manifest

    def _read_content_object(self, directory: Path, digest: str) -> Any:
        path = directory / "evidence" / "objects" / f"{digest}.json"
        payload = self._read_json(path)
        if canonical_digest(payload) != digest:
            raise EvidenceArchiveIntegrityError(f"immutable archive object digest mismatch: {digest}")
        return payload

    def _write_immutable_json(self, path: Path, payload: Any) -> None:
        self._write_create_only(path, canonical_json(payload).encode("utf-8"), compare_existing=True)

    def _write_create_only(self, path: Path, payload: bytes, *, compare_existing: bool) -> None:
        self._secure_directory(path.parent)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
                os.chmod(path, 0o600)
                self._fsync_directory(path.parent)
            except FileExistsError:
                if path.is_symlink():
                    raise EvidenceArchiveConflictError(
                        f"immutable evidence path cannot be a symbolic link: {path}"
                    ) from None
                if compare_existing and path.read_bytes() != payload:
                    raise EvidenceArchiveConflictError(
                        f"immutable evidence path contains different content: {path}"
                    ) from None
        finally:
            temporary.unlink(missing_ok=True)

    def _write_head(self, path: Path, payload: dict[str, Any]) -> None:
        if path.is_symlink():
            raise EvidenceArchiveIntegrityError(
                "session archive head cannot be a symbolic link"
            )
        self._secure_directory(path.parent)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                os.chmod(temporary, 0o600)
                handle.write(canonical_json(payload))
                handle.flush()
                os.fsync(handle.fileno())
            if path.is_symlink():
                raise EvidenceArchiveIntegrityError(
                    "session archive head cannot be a symbolic link"
                )
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            self._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_bytes_atomic(self, path: Path, payload: bytes) -> None:
        if path.is_symlink():
            raise EvidenceArchiveIntegrityError(
                "session archive output cannot be a symbolic link"
            )
        self._secure_directory(path.parent)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if path.is_symlink():
                raise EvidenceArchiveIntegrityError(
                    "session archive output cannot be a symbolic link"
                )
            os.replace(temporary, path)
            try:
                path.chmod(0o600)
            except OSError:
                pass
            self._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _async_output_lock(self, async_root: Path) -> Iterator[None]:
        self._secure_directory(async_root)
        lock_path = async_root / ".publish.lock"
        if lock_path.is_symlink():
            raise AsyncOutputIntegrityError(
                "async output lock cannot be a symbolic link"
            )
        if fcntl is None:  # pragma: no cover
            key = str(lock_path)
            with self._fallback_guard:
                lock = self._fallback_locks.setdefault(key, threading.RLock())
            with lock:
                yield
            return
        descriptor = open_private_lock(lock_path, root=self.root)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read_async_generation_for_archive(
        self,
        directory: Path,
        archive: SessionArchive,
    ) -> dict[str, Any]:
        head_path = directory / "async_outputs" / "current.json"
        head = self._read_async_head_optional(head_path)
        if head is None:
            raise FileNotFoundError(head_path.name)
        tenant_id = self._archive_tenant(archive)
        if (
            head.get("archive_uri") != archive.archive_uri
            or head.get("tenant_id") != tenant_id
            or head.get("task_id") != archive.task_id
        ):
            raise FileNotFoundError(archive.task_id)
        return self._read_async_generation(directory, head, archive.task_id)

    def _read_async_head_optional(self, path: Path) -> dict[str, Any] | None:
        if path.is_symlink():
            raise AsyncOutputIntegrityError(
                "async output head cannot be a symbolic link"
            )
        if not path.exists():
            return None
        try:
            head = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AsyncOutputIntegrityError("async output head is unreadable") from exc
        if not isinstance(head, dict):
            raise AsyncOutputIntegrityError("async output head must be a JSON object")
        core = {key: value for key, value in head.items() if key != "head_digest"}
        if (
            head.get("schema_version") != ASYNC_OUTPUT_HEAD_SCHEMA_VERSION
            or head.get("head_digest") != canonical_digest(core)
        ):
            raise AsyncOutputIntegrityError("async output head digest is corrupt")
        task_id = require_safe_path_segment(head.get("task_id"), "async output head task_id")
        if head.get("manifest_relative_path") != f"{task_id}/manifest.json":
            raise AsyncOutputIntegrityError("async output head manifest path is invalid")
        return head

    def _read_async_generation(
        self,
        directory: Path,
        head: dict[str, Any],
        task_id: str,
    ) -> dict[str, Any]:
        safe_task = require_safe_path_segment(task_id, "async output task_id")
        generation = directory / "async_outputs" / safe_task
        manifest_path = generation / "manifest.json"
        if generation.is_symlink() or manifest_path.is_symlink():
            raise AsyncOutputIntegrityError(
                "async output generation cannot be a symbolic link"
            )
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
        if not isinstance(files, dict) or set(files) != set(self._ASYNC_FILES):
            raise AsyncOutputIntegrityError("async output manifest file set is incomplete")
        result: dict[str, Any] = {"head": head, "manifest": manifest}
        for filename in self._ASYNC_FILES:
            details = files.get(filename)
            if not isinstance(details, dict):
                raise AsyncOutputIntegrityError("async output file proof is invalid")
            path = generation / filename
            if path.is_symlink():
                raise AsyncOutputIntegrityError(
                    "async output file cannot be a symbolic link"
                )
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise AsyncOutputIntegrityError("async output file is missing") from exc
            if (
                details.get("digest") != canonical_digest_bytes(raw)
                or details.get("size") != len(raw)
            ):
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

    def _quarantine_async_output_controls(
        self,
        directory: Path,
        archive: SessionArchive,
        error: BaseException,
    ) -> None:
        artifact_root = (
            self.root
            if self._archive_tenant(archive) == "default"
            else self.root / "tenants" / self._archive_tenant(archive)
        )
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
                    identifiers={"task_id": archive.task_id, "archive_uri_digest": canonical_digest(archive.archive_uri)},
                )

    def _notify_async(self, stage: str, task_id: str) -> None:
        if self.test_hook is not None:
            self.test_hook(stage, task_id)

    def _fsync_directory(self, directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _manifest_uri(self, archive_uri: str, manifest_digest: str) -> str:
        return f"{archive_uri}#manifest={manifest_digest}"

    def _dir(self, archive_uri: str, *, tenant_id: str | None = None) -> Path:
        return ContextURI.parse(archive_uri).to_source_path(self.root, tenant_id=tenant_id or self.tenant_id)

    def _archive_tenant(self, archive: SessionArchive) -> str:
        metadata = dict(archive.metadata or {})
        scope = dict(metadata.get("scope", {}) or {})
        direct = str(metadata.get("tenant_id") or "")
        scoped = str(scope.get("tenant_id") or "")
        if direct and scoped and direct != scoped:
            raise EvidenceArchiveIntegrityError("session archive metadata has conflicting tenants")
        claimed = direct or scoped
        if claimed and claimed != self.tenant_id:
            raise EvidenceArchiveIntegrityError(
                "session archive tenant does not match the bound archive store"
            )
        return self.tenant_id

    def _materialize_archive_tenant(self, archive: SessionArchive, tenant_id: str) -> None:
        metadata = dict(archive.metadata or {})
        scope = dict(metadata.get("scope", {}) or {})
        claimed = tuple(
            str(value)
            for value in (metadata.get("tenant_id"), scope.get("tenant_id"))
            if value not in (None, "")
        )
        if any(value != tenant_id for value in claimed):
            raise EvidenceArchiveIntegrityError("session archive metadata tenant mismatch")
        metadata["tenant_id"] = tenant_id
        if "scope" in metadata:
            scope["tenant_id"] = tenant_id
            metadata["scope"] = scope
        archive.metadata = metadata

    def _write_json(self, path: Path, payload: Any) -> None:
        self._write_bytes_atomic(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    def _secure_directory(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        current = directory
        root = self.root.expanduser().resolve()
        while current == root or root in current.resolve().parents:
            try:
                current.chmod(0o700)
            except OSError:
                pass
            if current.resolve() == root:
                break
            current = current.parent

    def _read_json(self, path: Path) -> Any:
        if path.is_symlink():
            raise EvidenceArchiveIntegrityError(
                f"evidence archive path cannot be a symbolic link: {path}"
            )
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise EvidenceArchiveIntegrityError(f"missing evidence archive object: {path}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvidenceArchiveIntegrityError(f"invalid evidence archive JSON: {path}") from exc
