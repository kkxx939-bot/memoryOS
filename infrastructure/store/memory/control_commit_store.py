"""提交意图、变更事件和当前控制快照的持久化操作。"""

from __future__ import annotations

import os
from dataclasses import replace

from infrastructure.store.filesystem.durable_io import (
    ImmutableArtifactConflictError,
    atomic_create_json,
    atomic_write_json,
)
from infrastructure.store.filesystem.durable_io.atomic_file import _open_control_parent
from infrastructure.store.memory.control_common import (
    _EVENT_SCHEMA,
    _MAX_DOCUMENT_CONTROLS,
    _MAX_LINEAGE_EVENTS,
    DocumentControlIntegrityError,
    DocumentIntentStatus,
)
from infrastructure.store.memory.control_common import (
    validate_prefixed_digest as _validate_prefixed_digest,
)
from infrastructure.store.memory.control_files import ControlFileMixin
from infrastructure.store.memory.control_intent import (
    DocumentCommitIntent,
)
from infrastructure.store.memory.control_record import (
    DocumentControlRecord,
)
from memory.core.model import (
    DocumentChangeEvent,
    DocumentEditKind,
    PresentPath,
)
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy


class ControlCommitStoreMixin(ControlFileMixin):
    """提交控制操作；文件能力由 ControlFileMixin 提供。"""

    def prepare_intent(self, intent: DocumentCommitIntent) -> DocumentCommitIntent:
        path = self._intent_path(intent.tenant_id, intent.owner_user_id, intent.intent_id)
        try:
            atomic_create_json(path, intent.to_dict(), artifact_root=self._artifact_root(intent.tenant_id))
        except ImmutableArtifactConflictError:
            # 并发的幂等准备者可能已经抢先完成只创建一次的发布。
            pass
        durable = self.load_intent(intent.tenant_id, intent.owner_user_id, intent.intent_id)
        if durable is None:  # pragma: no cover - create-only publication cannot disappear cooperatively.
            raise DocumentControlIntegrityError("prepared document intent disappeared after publication")
        return durable

    def load_intent(self, tenant_id: str, owner_user_id: str, intent_id: str) -> DocumentCommitIntent | None:
        path = self._intent_path(tenant_id, owner_user_id, intent_id)
        payload = self._read_json(path, tenant_id)
        if payload is None:
            return None
        intent = DocumentCommitIntent.from_dict(payload)
        if intent.intent_id != intent_id or intent.tenant_id != tenant_id or intent.owner_user_id != owner_user_id:
            raise DocumentControlIntegrityError("document intent path identity does not match its payload")
        return intent

    def update_intent(
        self,
        intent: DocumentCommitIntent,
        status: DocumentIntentStatus,
        *,
        updated_at: str,
        conflict_reason: str = "",
    ) -> DocumentCommitIntent:
        current = self.load_intent(intent.tenant_id, intent.owner_user_id, intent.intent_id)
        if current is None or current.identity_digest != intent.identity_digest:
            raise DocumentControlIntegrityError("document intent update is detached from its immutable identity")
        if current.status in {DocumentIntentStatus.COMPLETED, DocumentIntentStatus.CONFLICTED}:
            if current.status != status:
                return current
            return current
        if status != DocumentIntentStatus.CONFLICTED and _status_rank(status) < _status_rank(current.status):
            return current
        updated = replace(current, status=status, updated_at=updated_at, conflict_reason=conflict_reason[:500])
        atomic_write_json(
            self._intent_path(updated.tenant_id, updated.owner_user_id, updated.intent_id),
            updated.to_dict(),
            artifact_root=self._artifact_root(updated.tenant_id),
        )
        if status == DocumentIntentStatus.CONFLICTED:
            atomic_write_json(
                self._conflict_path(updated.tenant_id, updated.owner_user_id, updated.intent_id),
                updated.to_dict(),
                artifact_root=self._artifact_root(updated.tenant_id),
            )
        return updated

    def intents(self, tenant_id: str, owner_user_id: str) -> tuple[DocumentCommitIntent, ...]:
        """枚举所有耐久意图，包括已经结束的引用。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        directory = self._owner_root(tenant, owner) / "intents"
        names = self._json_names(directory, tenant)
        intents: list[DocumentCommitIntent] = []
        for name in names:
            intent = self.load_intent(tenant, owner, name.removesuffix(".json"))
            if intent is not None:
                intents.append(intent)
        return tuple(sorted(intents, key=lambda item: (item.created_at, item.intent_id)))

    def incomplete_intents(self, tenant_id: str, owner_user_id: str) -> tuple[DocumentCommitIntent, ...]:
        return tuple(
            intent
            for intent in self.intents(tenant_id, owner_user_id)
            if intent.status != DocumentIntentStatus.COMPLETED
        )

    def append_event(self, intent: DocumentCommitIntent, event: DocumentChangeEvent) -> None:
        if not _event_matches_intent(event, intent):
            raise ValueError("document event is detached from its prepared intent")
        payload = {
            "schema": _EVENT_SCHEMA,
            "intent_id": intent.intent_id,
            "intent_identity_digest": intent.identity_digest,
            **event.to_dict(),
        }
        atomic_create_json(
            self._event_path(intent, event.event_id),
            payload,
            artifact_root=self._artifact_root(intent.tenant_id),
        )

    def load_event(self, intent: DocumentCommitIntent) -> DocumentChangeEvent | None:
        payload = self._read_json(self._event_path(intent, intent.event_id), intent.tenant_id)
        if payload is None:
            return None
        if (
            payload.get("schema") != _EVENT_SCHEMA
            or payload.get("intent_id") != intent.intent_id
            or payload.get("intent_identity_digest") != intent.identity_digest
        ):
            raise DocumentControlIntegrityError("document event is detached from its intent")
        try:
            event = DocumentChangeEvent(
                event_id=str(payload["event_id"]),
                tenant_id=str(payload["tenant_id"]),
                owner_user_id=str(payload["owner_user_id"]),
                document_id=str(payload["document_id"]),
                edit_kind=DocumentEditKind(str(payload["edit_kind"])),
                old_relative_path=str(payload.get("old_relative_path") or ""),
                new_relative_path=str(payload.get("new_relative_path") or ""),
                before_raw_digest=str(payload.get("before_raw_digest") or ""),
                after_raw_digest=str(payload.get("after_raw_digest") or ""),
                logical_revision=int(payload["logical_revision"]),
                projection_generation=int(payload["projection_generation"]),
                occurred_at=str(payload["occurred_at"]),
                actor_binding=str(payload["actor_binding"]),
                evidence_reference=str(payload["evidence_reference"]),
                evidence_digest=str(payload["evidence_digest"]),
                edit_summary=str(payload["edit_summary"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentControlIntegrityError("document event is malformed") from exc
        if not _event_matches_intent(event, intent):
            raise DocumentControlIntegrityError("document event identity does not match its path")
        return event

    def load_event_binding(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        event_id: str,
    ) -> tuple[DocumentCommitIntent, DocumentChangeEvent] | None:
        """解析一个不可变变更事件 ID，不在其他位置重复保存意图 ID。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        _validate_prefixed_digest(event_id, "memchg_", "event_id")
        directory = self._owner_root(tenant, owner) / "events" / identifier
        if not directory.exists():
            return None
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant))
        try:
            names = sorted(os.listdir(descriptor))
        finally:
            os.close(descriptor)
        if len(names) > _MAX_LINEAGE_EVENTS:
            raise DocumentControlIntegrityError("document event count exceeds its bounded limit")
        suffix = f"-{event_id}.json"
        matches = [name for name in names if name.endswith(suffix)]
        if len(matches) > 1:
            raise DocumentControlIntegrityError("document event ID is duplicated")
        if not matches:
            return None
        name = matches[0]
        prefix = name.removesuffix(suffix)
        if len(prefix) != 20 or not prefix.isdigit() or "/" in name:
            raise DocumentControlIntegrityError("document event path is malformed")
        payload = self._read_json(directory / name, tenant)
        if payload is None:
            return None
        intent_id = str(payload.get("intent_id") or "")
        intent = self.load_intent(tenant, owner, intent_id)
        if intent is None or intent.document_id != identifier or intent.event_id != event_id:
            raise DocumentControlIntegrityError("document event is detached from its intent")
        event = self.load_event(intent)
        if event is None or event.event_id != event_id:
            raise DocumentControlIntegrityError("document event binding disappeared")
        return intent, event

    def write_control(self, record: DocumentControlRecord) -> None:
        atomic_write_json(
            self._control_path(record.tenant_id, record.owner_user_id, record.document_id),
            record.to_dict(),
            artifact_root=self._artifact_root(record.tenant_id),
        )

    def load_control(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> DocumentControlRecord | None:
        path = self._control_path(tenant_id, owner_user_id, document_id)
        payload = self._read_json(path, tenant_id)
        if payload is None:
            return None
        record = DocumentControlRecord.from_dict(payload)
        if (record.tenant_id, record.owner_user_id, record.document_id) != (
            tenant_id,
            owner_user_id,
            document_id,
        ):
            raise DocumentControlIntegrityError("document control path identity does not match its payload")
        return record

    def controls(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[DocumentControlRecord, ...]:
        """安全枚举一个所有者的精确耐久控制快照。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        directory = self._owner_root(tenant, owner) / "documents"
        descriptor = _open_control_parent(directory / ".scan", self._artifact_root(tenant))
        try:
            names = tuple(sorted(os.listdir(descriptor)))
        finally:
            os.close(descriptor)
        if len(names) > _MAX_DOCUMENT_CONTROLS:
            raise DocumentControlIntegrityError("document control count exceeds its bound")
        records: list[DocumentControlRecord] = []
        present_paths: set[str] = set()
        for name in names:
            if not name.endswith(".json") or "/" in name:
                raise DocumentControlIntegrityError(
                    "document control directory contains an unexpected artifact"
                )
            try:
                document_id = validate_document_id(name.removesuffix(".json"))
            except ValueError as exc:
                raise DocumentControlIntegrityError("document control filename is invalid") from exc
            record = self.load_control(tenant, owner, document_id)
            if record is None:  # pragma: no cover - stable directory snapshot cannot lose a cooperative file.
                raise DocumentControlIntegrityError("document control disappeared during enumeration")
            if record.status == "present":
                if record.relative_path in present_paths:
                    raise DocumentControlIntegrityError(
                        "multiple present document controls claim one relative path"
                    )
                present_paths.add(record.relative_path)
            records.append(record)
        return tuple(records)


def _status_rank(status: DocumentIntentStatus) -> int:
    return {
        DocumentIntentStatus.PREPARED: 0,
        DocumentIntentStatus.INSTALLED: 1,
        DocumentIntentStatus.EVENT_APPENDED: 2,
        DocumentIntentStatus.PROJECTION_ENQUEUED: 3,
        DocumentIntentStatus.COMPLETED: 4,
        DocumentIntentStatus.CONFLICTED: 5,
    }[status]


def _event_matches_intent(event: DocumentChangeEvent, intent: DocumentCommitIntent) -> bool:
    before_digest = next(
        (state.raw_sha256 for state in (effect.before for effect in intent.effects) if isinstance(state, PresentPath)),
        "",
    )
    after_digest = next(
        (
            state.raw_sha256
            for state in (effect.after for effect in reversed(intent.effects))
            if isinstance(state, PresentPath)
        ),
        "",
    )
    return (
        event.event_id == intent.event_id
        and event.tenant_id == intent.tenant_id
        and event.owner_user_id == intent.owner_user_id
        and event.document_id == intent.document_id
        and event.edit_kind == intent.edit_kind
        and event.old_relative_path == intent.old_relative_path
        and event.new_relative_path == intent.new_relative_path
        and event.before_raw_digest == before_digest
        and event.after_raw_digest == after_digest
        and event.logical_revision == intent.logical_revision
        and event.projection_generation == intent.projection_generation
        and event.occurred_at == intent.created_at
        and event.actor_binding == intent.actor_binding
        and event.evidence_reference == intent.evidence_reference
        and event.evidence_digest == intent.evidence_digest
        and event.edit_summary == intent.edit_summary
    )
