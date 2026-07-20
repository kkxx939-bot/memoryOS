"""内容寻址、不可变会话证据归档的文件系统编排器。

该类负责证据 collection、event、manifest 与 commit head 的领域一致性；
异步派生输出、目录布局和原子文件操作分别由独立组件负责。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from foundation.ids import require_safe_path_segment
from foundation.integrity import canonical_digest, canonicalize
from infrastructure.store.contracts.session_evidence import SessionEvidenceEncoder
from infrastructure.store.filesystem.session_archive_io import SessionArchiveFileIO
from infrastructure.store.filesystem.session_archive_layout import SessionArchiveLayout
from infrastructure.store.filesystem.session_async_outputs import SessionAsyncOutputStore
from infrastructure.store.model.context.context_uri import ContextURI
from memory.commit.evidence.errors import EvidenceArchiveIntegrityError
from pre.session import SessionArchive

ARCHIVE_MANIFEST_SCHEMA_VERSION = "session_archive_manifest_v2"
ARCHIVE_HEAD_SCHEMA_VERSION = "session_archive_head_v2"


class SessionArchiveStore:
    """持久化不可变会话证据，并通过摘要校验所有读取结果。"""

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

    def __init__(
        self,
        root: str | Path,
        tenant_id: str = "default",
        *,
        evidence_encoder: SessionEvidenceEncoder,
        test_hook: Callable[[str, str], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.tenant_id = tenant_id
        self.evidence_encoder = evidence_encoder
        self.test_hook = test_hook
        self._layout = SessionArchiveLayout(self.root, tenant_id)
        self._files = SessionArchiveFileIO(self.root)
        self._async_outputs = SessionAsyncOutputStore(
            self._layout,
            self._files,
            test_hook=lambda: self.test_hook,
        )

    @property
    def last_async_output_error(self) -> str:
        """返回最近一次异步输出完整性检查的错误类型。"""

        return self._async_outputs.last_error

    def write_sync_archive(self, archive: SessionArchive) -> Path:
        """写入不可变证据对象与 manifest，最后原子发布 commit head。"""

        tenant_id = self._layout.archive_tenant(archive)
        self._layout.materialize_archive_tenant(archive, tenant_id)
        directory = self._dir(archive.archive_uri, tenant_id=tenant_id)
        head_path = directory / "commit_head.json"
        if head_path.is_symlink():
            raise EvidenceArchiveIntegrityError("session archive head cannot be a symbolic link")
        self._files.secure_directory(directory)

        collections: dict[str, str] = {}
        for name in self._COLLECTION_FILES:
            payload = canonicalize(getattr(archive, name))
            digest = canonical_digest(payload)
            self._files.write_immutable_json(
                directory / "evidence" / "objects" / f"{digest}.json",
                payload,
            )
            collections[name] = digest

        event_refs = []
        for event in self.evidence_encoder.encode(archive):
            self._files.write_immutable_json(
                directory / "evidence" / "events" / f"{event.event_digest}.json",
                event.payload,
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
        self._files.write_immutable_json(
            directory / "evidence" / "manifests" / f"{manifest_digest}.json",
            manifest,
        )

        manifest_uri = self._layout.manifest_uri(archive.archive_uri, manifest_digest)
        archive.archive_digest = archive_digest
        archive.manifest_digest = manifest_digest
        archive.manifest_uri = manifest_uri
        self._files.write_head(
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
        """把异步派生输出交给独立 generation 发布组件。"""

        return self._async_outputs.write_outputs(
            archive_uri,
            abstract,
            overview,
            memory_diff,
            behavior_diff,
            action_policy_diff,
            context_diff,
            tenant_id,
            commit_group_status,
            complete,
            task_id,
            created_at,
        )

    def async_outputs_done_for_task(self, archive: SessionArchive) -> bool:
        """判断指定 task 的异步输出是否完整发布且校验通过。"""

        return self._async_outputs.outputs_done_for_task(archive)

    def read_async_outputs(self, archive: SessionArchive) -> dict[str, Any]:
        """读取并验证指定归档的异步派生输出。"""

        return self._async_outputs.read_outputs(archive)

    def read_archive(
        self,
        archive_uri: str,
        *,
        tenant_id: str | None = None,
        manifest_digest: str | None = None,
    ) -> SessionArchive:
        """通过 commit head 或明确 manifest 摘要读取不可变归档。"""

        effective_tenant = tenant_id or self.tenant_id
        parsed_uri = ContextURI.parse(archive_uri)
        directory = self._dir(archive_uri, tenant_id=tenant_id)
        head_path = directory / "commit_head.json"
        if not manifest_digest and head_path.is_symlink():
            raise EvidenceArchiveIntegrityError("session archive head cannot be a symbolic link")
        head = {} if manifest_digest else dict(self._files.read_json(head_path) or {})
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
        """证明枚举出的 head 路径身份后再读取对应归档。"""

        if head_path.is_symlink():
            raise EvidenceArchiveIntegrityError("archive commit head cannot be a symbolic link")
        try:
            head = self._files.read_json(head_path)
        except EvidenceArchiveIntegrityError as exc:
            raise EvidenceArchiveIntegrityError(f"archive commit head is unreadable: {head_path}") from exc
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
        """沿固定租户目录树有界枚举归档 head，仅供恢复和管理流程使用。"""

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
        candidates: list[tuple[Path, str]] = []
        for user_root in self._bounded_child_directories(users_root, label="Session user root"):
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
                    raise EvidenceArchiveIntegrityError("Session archive tree exceeded its enumeration bound")
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

    def read_archive_at_manifest(
        self,
        archive_uri: str,
        manifest_digest: str,
        *,
        tenant_id: str | None = None,
    ) -> SessionArchive:
        """读取明确指定的不可变 manifest，而不依赖当前 head。"""

        return self.read_archive(
            archive_uri,
            tenant_id=tenant_id,
            manifest_digest=manifest_digest,
        )

    def archive_exists(self, archive_uri: str, *, tenant_id: str | None = None) -> bool:
        """检查安全 commit head 是否存在。"""

        path = self._dir(archive_uri, tenant_id=tenant_id) / "commit_head.json"
        if path.is_symlink():
            raise EvidenceArchiveIntegrityError("session archive head cannot be a symbolic link")
        return path.exists()

    def archive_tenant(self, archive: SessionArchive) -> str:
        """返回写入与幂等读取共同使用的已验证租户。"""

        return self._layout.archive_tenant(archive)

    def read_event(
        self,
        archive_uri: str,
        event_digest: str,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """按摘要读取单个不可变事件，并验证事件自声明摘要。"""

        directory = self._dir(archive_uri, tenant_id=tenant_id)
        payload = self._files.read_json(directory / "evidence" / "events" / f"{event_digest}.json")
        claimed = str(payload.get("event_digest") or "")
        body = {key: value for key, value in payload.items() if key != "event_digest"}
        if claimed != event_digest or canonical_digest(body) != event_digest:
            raise EvidenceArchiveIntegrityError(f"immutable event digest mismatch: {event_digest}")
        return payload

    def current_manifest(self, archive_uri: str, *, tenant_id: str | None = None) -> dict[str, Any]:
        """读取当前 commit head 指向的不可变 manifest。"""

        directory = self._dir(archive_uri, tenant_id=tenant_id)
        head = self._files.read_json(directory / "commit_head.json")
        digest = str(head.get("manifest_digest") or "")
        if not digest:
            raise EvidenceArchiveIntegrityError("session archive head has no manifest digest")
        return self._read_manifest(directory, digest)

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
            manifest_uri=self._layout.manifest_uri(archive_uri, manifest_digest),
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
        self._layout.materialize_archive_tenant(archive, tenant_id)
        return archive

    def _read_manifest(self, directory: Path, digest: str) -> dict[str, Any]:
        manifest = self._files.read_json(directory / "evidence" / "manifests" / f"{digest}.json")
        claimed = str(manifest.get("manifest_digest") or "")
        body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
        if claimed != digest or canonical_digest(body) != digest:
            raise EvidenceArchiveIntegrityError(f"immutable manifest digest mismatch: {digest}")
        return manifest

    def _read_content_object(self, directory: Path, digest: str) -> Any:
        payload = self._files.read_json(directory / "evidence" / "objects" / f"{digest}.json")
        if canonical_digest(payload) != digest:
            raise EvidenceArchiveIntegrityError(f"immutable archive object digest mismatch: {digest}")
        return payload

    def _dir(self, archive_uri: str, *, tenant_id: str | None = None) -> Path:
        """保留给归档恢复用例使用的真实路径解析入口。"""

        return self._layout.directory(archive_uri, tenant_id=tenant_id)


__all__ = ["SessionArchiveStore"]
