"""Content-addressed, immutable session evidence archives."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical.event import canonical_digest, canonical_json, canonicalize

ARCHIVE_MANIFEST_SCHEMA_VERSION = "session_archive_manifest_v2"
ARCHIVE_HEAD_SCHEMA_VERSION = "session_archive_head_v2"


class EvidenceArchiveError(ValueError):
    """Base class for observable evidence archive failures."""


class EvidenceArchiveConflictError(EvidenceArchiveError):
    """A content-addressed path already exists with different bytes."""


class EvidenceArchiveIntegrityError(EvidenceArchiveError):
    """Immutable evidence no longer matches its recorded digest."""


class SessionArchiveStore:
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

    def __init__(self, root: str | Path, tenant_id: str = "default") -> None:
        self.root = Path(root)
        self.tenant_id = tenant_id

    def write_sync_archive(self, archive: SessionArchive) -> Path:
        tenant_id = self._archive_tenant(archive)
        directory = self._dir(archive.archive_uri, tenant_id=tenant_id)
        directory.mkdir(parents=True, exist_ok=True)

        collections: dict[str, str] = {}
        for name in self._COLLECTION_FILES:
            payload = canonicalize(getattr(archive, name))
            digest = canonical_digest(payload)
            self._write_immutable_json(directory / "evidence" / "objects" / f"{digest}.json", payload)
            collections[name] = digest

        # Import locally to avoid a module cycle: episode imports SessionArchive.
        from memoryos.memory.canonical.episode import SessionArchiveEpisodeAdapter

        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        event_refs = []
        for event in episode.events:
            payload = event.to_dict()
            self._write_immutable_json(
                directory / "evidence" / "events" / f"{event.digest}.json",
                payload,
            )
            event_refs.append(
                {
                    "event_id": event.event_id,
                    "event_digest": event.digest,
                    "event_type": event.event_type,
                    "category": str(event.metadata.get("category", "")),
                    "occurred_at": event.occurred_at,
                    "ingested_at": event.ingested_at,
                    "sequence": event.sequence,
                }
            )

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
            directory / "commit_head.json",
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
    ) -> Path:
        directory = self._dir(archive_uri, tenant_id=tenant_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".abstract.md").write_text(abstract, encoding="utf-8")
        (directory / ".overview.md").write_text(overview, encoding="utf-8")
        self._write_json(directory / "memory_diff.json", memory_diff)
        self._write_json(directory / "behavior_diff.json", behavior_diff)
        self._write_json(directory / "action_policy_diff.json", action_policy_diff)
        self._write_json(directory / "context_diff.json", context_diff)
        if commit_group_status is not None:
            self._write_json(directory / "commit_group_status.json", commit_group_status)
        done_path = directory / ".done"
        if complete:
            done_path.write_text("done\n", encoding="utf-8")
        else:
            done_path.unlink(missing_ok=True)
        return directory

    def async_outputs_done_for_task(self, archive: SessionArchive) -> bool:
        directory = self._dir(archive.archive_uri, tenant_id=self._archive_tenant(archive))
        if not (directory / ".done").exists():
            return False
        try:
            payload = json.loads((directory / "memory_diff.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("task_id") == archive.task_id

    def read_archive(
        self,
        archive_uri: str,
        *,
        tenant_id: str | None = None,
        manifest_digest: str | None = None,
    ) -> SessionArchive:
        directory = self._dir(archive_uri, tenant_id=tenant_id)
        head_path = directory / "commit_head.json"
        head = {} if manifest_digest else dict(self._read_json(head_path) or {})
        if head and str(head.get("schema_version") or "") != ARCHIVE_HEAD_SCHEMA_VERSION:
            raise EvidenceArchiveIntegrityError("unsupported session archive head schema")
        if head and str(head.get("archive_uri") or "") != archive_uri:
            raise EvidenceArchiveIntegrityError("session archive head URI mismatch")
        selected = manifest_digest or str(head.get("manifest_digest") or "")
        if not selected:
            raise EvidenceArchiveIntegrityError("session archive head has no manifest digest")
        archive = self._read_v2_archive(directory, archive_uri, selected)
        if head and str(head.get("archive_digest") or "") != archive.archive_digest:
            raise EvidenceArchiveIntegrityError("session archive head aggregate digest mismatch")
        return archive

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
        return (directory / "commit_head.json").exists()

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

    def _read_v2_archive(self, directory: Path, archive_uri: str, manifest_digest: str) -> SessionArchive:
        manifest = self._read_manifest(directory, manifest_digest)
        if str(manifest.get("schema_version") or "") != ARCHIVE_MANIFEST_SCHEMA_VERSION:
            raise EvidenceArchiveIntegrityError("unsupported session archive manifest schema")
        if str(manifest.get("archive_uri")) != archive_uri:
            raise EvidenceArchiveIntegrityError("session archive manifest URI mismatch")
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
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
                self._fsync_directory(path.parent)
            except FileExistsError:
                if compare_existing and path.read_bytes() != payload:
                    raise EvidenceArchiveConflictError(
                        f"immutable evidence path contains different content: {path}"
                    ) from None
        finally:
            temporary.unlink(missing_ok=True)

    def _write_head(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        self._fsync_directory(path.parent)

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
        return str(metadata.get("tenant_id") or scope.get("tenant_id") or self.tenant_id)

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise EvidenceArchiveIntegrityError(f"missing evidence archive object: {path}") from exc
        except json.JSONDecodeError as exc:
            raise EvidenceArchiveIntegrityError(f"invalid evidence archive JSON: {path}") from exc
