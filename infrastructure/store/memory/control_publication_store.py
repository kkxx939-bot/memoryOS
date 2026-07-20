"""删除发布屏障、血缘查询和文档控制清理操作。"""

from __future__ import annotations

import os
from dataclasses import replace

from infrastructure.store.filesystem.durable_io import (
    atomic_write_json,
)
from infrastructure.store.filesystem.durable_io.atomic_file import _open_control_parent
from infrastructure.store.memory.control_commit_store import ControlCommitStoreMixin
from infrastructure.store.memory.control_common import (
    _EVENT_SCHEMA,
    _MAX_LINEAGE_EVENTS,
    _MAX_PUBLICATION_BARRIERS,
    DocumentControlIntegrityError,
    DocumentDeletionStatus,
    DocumentIntentStatus,
)
from infrastructure.store.memory.control_common import (
    is_hex as _is_hex,
)
from infrastructure.store.memory.control_intent import (
    DocumentCommitIntent,
    deletion_event_digest,
)
from infrastructure.store.memory.control_record import (
    DocumentPublicationBarrier,
)
from memory.core.model import (
    AbsentPath,
    DocumentEditKind,
    PresentPath,
)
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy


class ControlPublicationStoreMixin(ControlCommitStoreMixin):
    """发布屏障操作；文件和意图读取能力由组合存储的兄弟 Mixin 提供。"""

    def write_publication_barrier(
        self,
        barrier: DocumentPublicationBarrier,
    ) -> DocumentPublicationBarrier:
        """在可重建服务状态之外发布单调递进的删除屏障。"""

        current = self.load_publication_barrier(
            barrier.tenant_id,
            barrier.owner_user_id,
            barrier.document_id,
        )
        if current is not None:
            current_identity = (
                current.relative_path_digest,
                current.deletion_generation,
                current.deletion_event_digest,
                current.status,
            )
            requested_identity = (
                barrier.relative_path_digest,
                barrier.deletion_generation,
                barrier.deletion_event_digest,
                barrier.status,
            )
            if current.status is DocumentDeletionStatus.HARD_ERASED:
                same_erasure_identity = (
                    barrier.relative_path_digest == current.relative_path_digest
                    and barrier.deletion_event_digest == current.deletion_event_digest
                    and barrier.status is current.status
                )
                if not same_erasure_identity or barrier.deletion_generation < current.deletion_generation:
                    raise DocumentControlIntegrityError("hard-erased document publication barrier is immutable")
                if barrier.deletion_generation == current.deletion_generation:
                    return current
            if barrier.deletion_generation < current.deletion_generation:
                raise DocumentControlIntegrityError("document publication barrier generation regressed")
            if barrier.deletion_generation == current.deletion_generation:
                if requested_identity != current_identity:
                    raise DocumentControlIntegrityError(
                        "document publication barrier conflicts at the current generation"
                    )
                return current
        atomic_write_json(
            self._publication_barrier_path(
                barrier.tenant_id,
                barrier.owner_user_id,
                barrier.document_id,
            ),
            barrier.to_dict(),
            artifact_root=self._artifact_root(barrier.tenant_id),
        )
        durable = self.load_publication_barrier(
            barrier.tenant_id,
            barrier.owner_user_id,
            barrier.document_id,
        )
        if durable is None:  # pragma: no cover - durable publication cannot disappear cooperatively.
            raise DocumentControlIntegrityError("document publication barrier disappeared after publication")
        return durable

    def scrub_hard_erasure_path(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        *,
        expected_relative_path_digest: str,
        expected_deletion_event_digest: str,
        updated_at: str,
    ) -> DocumentPublicationBarrier:
        """所有硬删除后端确认后，移除语义路径。"""

        current = self.load_publication_barrier(tenant_id, owner_user_id, document_id)
        if current is None:
            raise DocumentControlIntegrityError("hard-erasure publication barrier is missing")
        if (
            current.status is not DocumentDeletionStatus.HARD_ERASED
            or current.relative_path_digest != expected_relative_path_digest
            or current.deletion_event_digest != expected_deletion_event_digest
        ):
            raise DocumentControlIntegrityError("hard-erasure publication barrier identity changed")
        if not current.relative_path:
            return current
        scrubbed = replace(current, relative_path="", updated_at=updated_at)
        atomic_write_json(
            self._publication_barrier_path(tenant_id, owner_user_id, document_id),
            scrubbed.to_dict(),
            artifact_root=self._artifact_root(tenant_id),
        )
        durable = self.load_publication_barrier(tenant_id, owner_user_id, document_id)
        if durable is None or durable != scrubbed:
            raise DocumentControlIntegrityError("hard-erasure path scrub was not durable")
        return durable

    def ensure_soft_forget_barrier(
        self,
        intent: DocumentCommitIntent,
    ) -> DocumentPublicationBarrier:
        """在实时字节解除链接之前，为 DELETE 意图建立屏障。"""

        if intent.edit_kind is not DocumentEditKind.DELETE or len(intent.effects) != 1:
            raise ValueError("soft-forget publication barrier requires one DELETE intent")
        effect = intent.effects[0]
        if not isinstance(effect.before, PresentPath) or not isinstance(effect.after, AbsentPath):
            raise ValueError("soft-forget publication barrier requires PRESENT to ABSENT")
        digest = deletion_event_digest(
            event_id=intent.event_id,
            document_id=intent.document_id,
            before_raw_digest=effect.before.raw_sha256,
            projection_generation=intent.projection_generation,
        )
        return self.write_publication_barrier(
            DocumentPublicationBarrier(
                tenant_id=intent.tenant_id,
                owner_user_id=intent.owner_user_id,
                document_id=intent.document_id,
                relative_path=effect.relative_path,
                deletion_generation=intent.projection_generation,
                deletion_event_digest=digest,
                status=DocumentDeletionStatus.SOFT_FORGOTTEN,
                updated_at=intent.updated_at,
            )
        )

    def load_publication_barrier(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> DocumentPublicationBarrier | None:
        path = self._publication_barrier_path(tenant_id, owner_user_id, document_id)
        payload = self._read_json(path, tenant_id)
        if payload is None:
            return None
        barrier = DocumentPublicationBarrier.from_dict(payload)
        if (barrier.tenant_id, barrier.owner_user_id, barrier.document_id) != (
            tenant_id,
            owner_user_id,
            document_id,
        ):
            raise DocumentControlIntegrityError(
                "document publication barrier path identity does not match its payload"
            )
        return barrier

    def publication_barriers(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[DocumentPublicationBarrier, ...]:
        """读取离线或全量重建使用的有界受保护屏障集合。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        directory = self._owner_root(tenant, owner) / "publication-barriers"
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant))
        try:
            names = sorted(os.listdir(descriptor))
        finally:
            os.close(descriptor)
        if len(names) > _MAX_PUBLICATION_BARRIERS:
            raise DocumentControlIntegrityError("document publication barrier count exceeds its bound")
        barriers: list[DocumentPublicationBarrier] = []
        for name in names:
            if not name.endswith(".json") or "/" in name:
                raise DocumentControlIntegrityError(
                    "document publication barrier directory contains an unexpected artifact"
                )
            try:
                document_id = validate_document_id(name.removesuffix(".json"))
            except ValueError as exc:
                raise DocumentControlIntegrityError(
                    "document publication barrier filename is invalid"
                ) from exc
            barrier = self.load_publication_barrier(tenant, owner, document_id)
            if barrier is None:  # pragma: no cover - stable directory snapshot cannot lose a cooperative file.
                raise DocumentControlIntegrityError("document publication barrier disappeared during scan")
            barriers.append(barrier)
        return tuple(barriers)

    def lineage_references(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> tuple[str, ...]:
        """返回一个文档的有界、无正文证据引用。

        文档专属事件目录就是耐久血缘索引。硬删除在移除控制产物前读取它，
        让调用方可以披露独立 Session 证据而不保留文档正文。
        """

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        directory = self._owner_root(tenant, owner) / "events" / identifier
        if not directory.exists():
            return ()
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant))
        try:
            names = sorted(os.listdir(descriptor))
        finally:
            os.close(descriptor)
        if len(names) > _MAX_LINEAGE_EVENTS:
            raise DocumentControlIntegrityError("document lineage exceeds its bounded event limit")
        references: set[str] = set()
        for name in names:
            prefix, separator, suffix = name.partition("-")
            event_id = suffix.removesuffix(".json")
            if (
                not separator
                or len(prefix) != 20
                or not prefix.isdigit()
                or not name.endswith(".json")
                or not event_id.startswith("memchg_")
                or not _is_hex(event_id.removeprefix("memchg_"), 64)
            ):
                raise DocumentControlIntegrityError("document lineage contains an unexpected event artifact")
            payload = self._read_json(directory / name, tenant)
            if (
                payload is None
                or payload.get("schema") != _EVENT_SCHEMA
                or payload.get("tenant_id") != tenant
                or payload.get("owner_user_id") != owner
                or payload.get("document_id") != identifier
            ):
                raise DocumentControlIntegrityError("document lineage event identity is invalid")
            reference = str(payload.get("evidence_reference") or "")
            if reference:
                references.add(reference)
        return tuple(sorted(references))

    def purge_document(self, tenant_id: str, owner_user_id: str, document_id: str) -> int:
        """实时正文删除后，耐久移除文档提交元数据。

        删除墓碑保存在这些路径之外。未完成的提交可能仍持有实时 CAS，因此
        绝不丢弃。无正文接管凭证会与墓碑一起保留，防止硬删除后的既有身份
        被再次使用。
        """

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        matching: list[DocumentCommitIntent] = []
        for name in self._json_names(self._owner_root(tenant, owner) / "intents", tenant):
            intent = self.load_intent(tenant, owner, name.removesuffix(".json"))
            if intent is not None and intent.document_id == identifier:
                matching.append(intent)
        unfinished = [
            intent.intent_id
            for intent in matching
            if intent.status not in {DocumentIntentStatus.COMPLETED, DocumentIntentStatus.CONFLICTED}
        ]
        if unfinished:
            raise DocumentControlIntegrityError("cannot purge a document with an unfinished commit intent")

        removed = 0
        for intent in matching:
            removed += self._unlink_regular_if_present(
                self._intent_path(tenant, owner, intent.intent_id),
                tenant,
            )
            removed += self._unlink_regular_if_present(
                self._conflict_path(tenant, owner, intent.intent_id),
                tenant,
            )
        removed += self._purge_event_directory(tenant, owner, identifier)
        removed += self._unlink_regular_if_present(
            self._control_path(tenant, owner, identifier),
            tenant,
        )
        return removed
