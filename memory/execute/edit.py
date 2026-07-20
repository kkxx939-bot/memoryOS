"""记忆文档的收养、正文编辑和重命名操作。"""

from __future__ import annotations

import hashlib

from foundation.identity import LocalUserContext
from foundation.integrity import canonical_json
from infrastructure.store.memory.control_store import (
    adoption_document_id,
    adoption_request_digest,
    document_intent_id,
)
from infrastructure.store.memory.scanner import ExternalChangeKind, ExternalDocumentChange
from memory.commit.document_commit import DocumentCommitResult
from memory.core.model import (
    ABSENT,
    DocumentEditKind,
    DocumentEditPlan,
    ManagedDocument,
    UnmanagedDocument,
)
from memory.core.structure.frontmatter import matches_adopted_source, parse_front_matter
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.execute.base import MemoryCommandBase, _require_sha256
from memory.execute.contracts import AdoptResult, DocumentEditResult
from memory.ports.document_store import DocumentConflictError


class EditOperation(MemoryCommandBase):
    """执行单文档身份收养、正文替换和稳定身份重命名。"""

    def adopt_memory_document(
        self,
        relative_path: str,
        expected_raw_sha256: str,
        *,
        caller: LocalUserContext,
    ) -> AdoptResult:
        """显式收养一个安全、属于调用者且尚未受管的 Markdown 文件。"""

        self._require_ready()
        _require_sha256(expected_raw_sha256, "expected_raw_sha256")
        relative = MemoryDocumentPathPolicy.normalize_relative_path(relative_path)
        request_digest = adoption_request_digest(
            caller.tenant_id,
            caller.user_id,
            relative,
            expected_raw_sha256,
        )
        receipt_id = f"mdadopt_{request_digest}"
        assigned_document_id = adoption_document_id(request_digest)
        receipt = self.control_store.load_adoption_receipt(
            caller.tenant_id,
            caller.user_id,
            receipt_id,
        )
        if receipt is not None:
            # 已持久化的回执证明首次根目录预检已经完成；重试不能根据当前文件重新推导授权。
            self.committer.verify_adoption_root(
                caller.tenant_id,
                caller.user_id,
                assigned_document_id,
            )
            # 修复回执发布成功、文档 ID 索引尚未安装时发生的进程中断。
            receipt = self.control_store.prepare_adoption_receipt(
                caller.tenant_id,
                caller.user_id,
                relative,
                expected_raw_sha256,
                actor_binding=receipt.actor_binding,
            )
            self.erase_store.assert_mutation_allowed(
                caller.tenant_id,
                caller.user_id,
                receipt.document_id,
            )
            replay_key = receipt.idempotency_key
            replay_intent_id = document_intent_id(
                caller.tenant_id,
                caller.user_id,
                receipt.document_id,
                hashlib.sha256(replay_key.encode()).hexdigest(),
            )
            existing_intent = self.control_store.load_intent(
                caller.tenant_id,
                caller.user_id,
                replay_intent_id,
            )
            if existing_intent is not None:
                result = self.committer.recover_intent(
                    caller.tenant_id,
                    caller.user_id,
                    replay_intent_id,
                )
                return self._adopt_result(receipt.relative_path, receipt.document_id, result)

        before = self.document_store.full_scan(caller.tenant_id, caller.user_id)
        if not before.complete or before.errors:
            raise DocumentConflictError("memory document adoption requires a complete registration scan")
        if receipt is None:
            matches = [
                item
                for item in before.registrations
                if item.relative_path == relative
                and isinstance(item, UnmanagedDocument)
                and item.raw_sha256 == expected_raw_sha256
            ]
            if len(matches) != 1:
                raise DocumentConflictError(
                    "adopt target must be one safe UNMANAGED Markdown file matching expected_raw_sha256"
                )
            self.committer.preflight_adoption_create(
                caller.tenant_id,
                caller.user_id,
                assigned_document_id,
            )
            receipt = self.control_store.prepare_adoption_receipt(
                caller.tenant_id,
                caller.user_id,
                relative,
                expected_raw_sha256,
                actor_binding=self._actor_binding(caller),
            )

        # 内容无关的回执同时是防重用身份映射，已经硬擦除的 ID 不能再次写回 live 文件。
        self.erase_store.assert_mutation_allowed(
            caller.tenant_id,
            caller.user_id,
            receipt.document_id,
        )
        idempotency_key = receipt.idempotency_key
        idempotency_digest = hashlib.sha256(idempotency_key.encode()).hexdigest()
        intent_id = document_intent_id(
            caller.tenant_id,
            caller.user_id,
            receipt.document_id,
            idempotency_digest,
        )
        existing_intent = self.control_store.load_intent(
            caller.tenant_id,
            caller.user_id,
            intent_id,
        )
        if existing_intent is not None:
            result = self.committer.recover_intent(
                caller.tenant_id,
                caller.user_id,
                intent_id,
            )
            return self._adopt_result(receipt.relative_path, receipt.document_id, result)

        managed = [
            item
            for item in before.registrations
            if isinstance(item, ManagedDocument)
            and item.document_id == receipt.document_id
            and item.relative_path == relative
        ]
        unmanaged = [
            item
            for item in before.registrations
            if isinstance(item, UnmanagedDocument)
            and item.relative_path == relative
            and item.raw_sha256 == expected_raw_sha256
        ]
        if len(managed) == 1:
            self.committer.verify_adoption_root(
                caller.tenant_id,
                caller.user_id,
                receipt.document_id,
            )
            raw = self.document_store.read_raw(
                caller.tenant_id,
                caller.user_id,
                document_id=receipt.document_id,
            )
            if not matches_adopted_source(raw, receipt.document_id, expected_raw_sha256):
                raise DocumentConflictError("managed adopt retry does not match the durable source digest")
        elif len(unmanaged) == 1:
            self.committer.verify_adoption_root(
                caller.tenant_id,
                caller.user_id,
                receipt.document_id,
            )
            try:
                self.document_store.adopt(
                    caller.tenant_id,
                    caller.user_id,
                    relative,
                    expected_raw_sha256=expected_raw_sha256,
                    assigned_document_id=receipt.document_id,
                    operation_id=receipt.receipt_id,
                )
            except DocumentConflictError:
                # 并发的同一请求可能已经完成 front matter CAS，下面的完整扫描负责证明最终状态。
                pass
        else:
            raise DocumentConflictError("adopt retry target is detached from its durable receipt")

        after = self.document_store.full_scan(caller.tenant_id, caller.user_id)
        if not after.complete or after.errors:
            raise DocumentConflictError("adopted memory document could not be confirmed by a complete scan")
        registrations = [
            item
            for item in after.registrations
            if isinstance(item, ManagedDocument)
            and item.document_id == receipt.document_id
            and item.relative_path == relative
        ]
        if len(registrations) != 1:
            raise DocumentConflictError("adopted memory document changed before durable registration")
        adopted = registrations[0]
        raw = self.document_store.read_raw(
            caller.tenant_id,
            caller.user_id,
            document_id=receipt.document_id,
        )
        if not matches_adopted_source(raw, receipt.document_id, expected_raw_sha256):
            raise DocumentConflictError("adopted memory document is not the exact authorized rewrite")

        commit_result = self.committer.record_external_change(
            ExternalDocumentChange(
                change_kind=ExternalChangeKind.CREATE,
                tenant_id=caller.tenant_id,
                owner_user_id=caller.user_id,
                document_id=receipt.document_id,
                old_relative_path="",
                new_relative_path=relative,
                before_raw_digest="",
                after_raw_digest=adopted.raw_sha256,
                scan_generation_id=after.generation_id,
            ),
            actor_binding=receipt.actor_binding,
            evidence_reference=receipt.evidence_reference,
            evidence_digest=receipt.evidence_digest,
            idempotency_key=idempotency_key,
            edit_summary=receipt.edit_summary,
        )
        if commit_result is None:
            # 并发请求可能在本次 intent 查询与 committer no-op 之间完成，必须由同一 intent 证明。
            concurrent_intent = self.control_store.load_intent(
                caller.tenant_id,
                caller.user_id,
                intent_id,
            )
            if concurrent_intent is None:
                raise DocumentConflictError("adopted memory document has no durable commit intent")
            commit_result = self.committer.recover_intent(
                caller.tenant_id,
                caller.user_id,
                intent_id,
            )
        return self._adopt_result(receipt.relative_path, receipt.document_id, commit_result)

    def _adopt_result(
        self,
        relative_path: str,
        document_id: str,
        result: DocumentCommitResult,
    ) -> AdoptResult:
        event = result.event
        control = result.control
        if (
            control is None
            or event is None
            or event.edit_kind is not DocumentEditKind.CREATE
            or event.document_id != document_id
            or event.new_relative_path != relative_path
        ):
            raise DocumentConflictError("adopted memory document was not durably registered")
        if self.bootstrapper is not None:
            self.bootstrapper.ensure_adopted_user(
                event.tenant_id,
                event.owner_user_id,
                event.new_relative_path,
                document_id=event.document_id,
                adopted_raw_sha256=event.after_raw_digest,
            )
        return AdoptResult(
            document_uri=MemoryDocumentPathPolicy.document_uri(control.owner_user_id, document_id),
            document_id=document_id,
            document_kind=MemoryDocumentPathPolicy.kind_for(relative_path).value,
            relative_path=event.new_relative_path,
            document_revision=event.logical_revision,
            source_digest=event.after_raw_digest,
            changed=not result.no_op,
            edit_summary="adopt unmanaged Markdown document",
            projection_status="ENQUEUED",
        )

    def edit_memory_document(
        self,
        document_uri: str,
        edit: str,
        expected_digest: str,
        *,
        caller: LocalUserContext,
    ) -> DocumentEditResult:
        """在精确摘要保护下替换一个记忆文档的完整正文。"""

        self._require_ready()
        live = self._load_live(document_uri, caller)
        _require_sha256(expected_digest, "expected_digest")
        if live.state.raw_sha256 != expected_digest:
            raise DocumentConflictError("document edit expected digest does not match live Markdown")
        replacement_body = str(edit)
        if not replacement_body.strip():
            raise ValueError("document edit body is empty; use forget for deletion")
        parsed = parse_front_matter(
            live.raw_bytes,
            max_header_bytes=self.planner.max_front_matter_bytes,
            max_depth=self.planner.max_front_matter_depth,
        )
        after = _replace_document_body(parsed.header_bytes, replacement_body)
        evidence_digest = hashlib.sha256(after).hexdigest()
        plan = DocumentEditPlan(
            idempotency_key="edit:"
            + hashlib.sha256(canonical_json([document_uri, expected_digest, evidence_digest]).encode()).hexdigest(),
            tenant_id=live.tenant_id,
            owner_user_id=live.owner_user_id,
            edit_kind=DocumentEditKind.UPDATE,
            expected_state=live.state,
            evidence_digest=evidence_digest,
            edit_summary="explicit full-document edit",
            document_id=live.document_id,
            relative_path=live.relative_path,
            after_bytes=after,
            expected_registration_document_id=live.document_id,
        )
        result = self._commit_or_replay(
            plan,
            caller=caller,
            evidence_reference=f"explicit-edit:sha256:{evidence_digest}",
        )
        return DocumentEditResult(**self._result_fields(plan, result))

    def rename_memory_document(
        self,
        document_uri: str,
        new_relative_path: str,
        expected_digest: str,
        edit: str | None = None,
        *,
        caller: LocalUserContext,
    ) -> DocumentEditResult:
        """在一个效果内重命名稳定文档，并可选择同时替换正文。"""

        self._require_ready()
        live = self._load_live(document_uri, caller)
        _require_sha256(expected_digest, "expected_digest")
        target = MemoryDocumentPathPolicy.normalize_relative_path(new_relative_path)
        MemoryDocumentPathPolicy.kind_for(target)
        replacement_body = None if edit is None else str(edit)
        if replacement_body is not None and not replacement_body.strip():
            raise ValueError("rename edit body is empty; omit edit for a pure rename")
        after: bytes | None = None
        if replacement_body is not None:
            parsed = parse_front_matter(
                live.raw_bytes,
                max_header_bytes=self.planner.max_front_matter_bytes,
                max_depth=self.planner.max_front_matter_depth,
            )
            after = _replace_document_body(parsed.header_bytes, replacement_body)
        after_digest = hashlib.sha256(after).hexdigest() if after is not None else expected_digest
        request_digest = hashlib.sha256(
            canonical_json(
                [
                    "RENAME_REQUEST_V2",
                    document_uri,
                    target,
                    expected_digest,
                    hashlib.sha256(replacement_body.encode()).hexdigest() if replacement_body is not None else "",
                ]
            ).encode()
        ).hexdigest()
        idempotency_key = f"rename:{request_digest}"
        edit_summary = "rename and edit memory document" if replacement_body is not None else "rename memory document"
        evidence_digest = hashlib.sha256(
            canonical_json(
                [
                    "RENAME_EFFECT_V2",
                    document_uri,
                    target,
                    expected_digest,
                    after_digest,
                ]
            ).encode()
        ).hexdigest()
        evidence_reference = f"explicit-rename:sha256:{evidence_digest}"
        if live.state.raw_sha256 != expected_digest:
            if after is None or live.relative_path != target or live.raw_bytes != after:
                raise DocumentConflictError("document rename expected digest does not match live Markdown")
            # 重命名并编辑会改变摘要，重试只能由完全相同的持久 intent 授权继续向前恢复。
            intent_id = document_intent_id(
                live.tenant_id,
                live.owner_user_id,
                live.document_id,
                hashlib.sha256(idempotency_key.encode()).hexdigest(),
            )
            if self.control_store.load_intent(live.tenant_id, live.owner_user_id, intent_id) is None:
                raise DocumentConflictError("rename and edit target matches requested bytes without its durable intent")
            replay_plan = DocumentEditPlan(
                idempotency_key=idempotency_key,
                tenant_id=live.tenant_id,
                owner_user_id=live.owner_user_id,
                edit_kind=DocumentEditKind.RENAME,
                expected_state=live.state,
                expected_new_state=ABSENT,
                evidence_digest=evidence_digest,
                edit_summary=edit_summary,
                document_id=live.document_id,
                relative_path=live.relative_path,
                new_relative_path=target,
                after_bytes=after,
                expected_registration_document_id=live.document_id,
            )
            result = self._commit_or_replay(
                replay_plan,
                caller=caller,
                evidence_reference=evidence_reference,
            )
            return DocumentEditResult(**self._result_fields(replay_plan, result))
        if target == live.relative_path:
            if replacement_body is not None:
                raise ValueError("rename and edit requires a distinct new_relative_path")
            control = self.control_store.load_control(live.tenant_id, live.owner_user_id, live.document_id)
            if control is None:
                raise DocumentConflictError("document rename is detached from its durable registration")
            return DocumentEditResult(
                document_uri=live.document_uri,
                document_id=live.document_id,
                document_kind=live.document_kind,
                relative_path=live.relative_path,
                document_revision=control.logical_revision,
                source_digest=live.state.raw_sha256,
                changed=False,
                edit_summary=edit_summary,
                projection_status="UNCHANGED",
            )
        target_state = self.document_store.read_state(live.tenant_id, live.owner_user_id, target)
        if target_state != ABSENT:
            raise DocumentConflictError("document rename target must be ABSENT")
        plan = DocumentEditPlan(
            idempotency_key=idempotency_key,
            tenant_id=live.tenant_id,
            owner_user_id=live.owner_user_id,
            edit_kind=DocumentEditKind.RENAME,
            expected_state=live.state,
            expected_new_state=ABSENT,
            evidence_digest=evidence_digest,
            edit_summary=edit_summary,
            document_id=live.document_id,
            relative_path=live.relative_path,
            new_relative_path=target,
            after_bytes=after,
            expected_registration_document_id=live.document_id,
        )
        result = self._commit_or_replay(
            plan,
            caller=caller,
            evidence_reference=evidence_reference,
        )
        return DocumentEditResult(**self._result_fields(plan, result))


def _replace_document_body(header_bytes: bytes, replacement_body: str) -> bytes:
    body_bytes = replacement_body.encode("utf-8")
    if body_bytes and not body_bytes.startswith(b"\n"):
        body_bytes = b"\n" + body_bytes
    if body_bytes and not body_bytes.endswith(b"\n"):
        body_bytes += b"\n"
    return header_bytes + body_bytes


__all__ = ["EditOperation"]
