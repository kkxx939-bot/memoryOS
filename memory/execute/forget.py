"""记忆文档的软遗忘、局部遗忘和硬擦除操作。"""

from __future__ import annotations

import hashlib
import re

from foundation.identity import LocalUserContext
from foundation.integrity import canonical_json
from memory.commit.erase import DocumentEraseResult
from memory.core.model import DocumentEditKind, DocumentEditPlan, ManagedDocument
from memory.core.structure.frontmatter import parse_front_matter
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.execute.base import MemoryCommandBase, _assert_optional_live_digest, _LiveDocument
from memory.execute.contracts import ForgetMode, ForgetResult
from memory.ports.document_store import DocumentConflictError

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")


class ForgetOperation(MemoryCommandBase):
    """区分可恢复的软遗忘与不可恢复的全量硬擦除。"""

    def forget(
        self,
        document_uri: str,
        section_anchor: str | None = None,
        mode: ForgetMode = "SOFT_FORGET",
        expected_digest: str | None = None,
        *,
        caller: LocalUserContext,
    ) -> ForgetResult:
        self._require_ready()
        owner, document_id = self._bind_document_uri(document_uri, caller)
        normalized_mode = str(mode or "").strip().upper()
        if normalized_mode not in {"SOFT_FORGET", "HARD_ERASE"}:
            raise ValueError("forget mode must be SOFT_FORGET or HARD_ERASE")
        if normalized_mode == "HARD_ERASE":
            if section_anchor is not None:
                raise ValueError("HARD_ERASE only accepts a whole-document target")
            return self._hard_erase(
                document_uri,
                owner,
                document_id,
                expected_digest=expected_digest,
                caller=caller,
            )

        live = self._load_live(document_uri, caller)
        _assert_optional_live_digest(live.state, expected_digest)
        evidence_digest = hashlib.sha256(
            canonical_json(["SOFT_FORGET", document_uri, section_anchor or "", live.state.raw_sha256]).encode()
        ).hexdigest()
        if section_anchor is None:
            edit_kind = DocumentEditKind.DELETE
            after = None
            summary = "soft forget whole document (recoverable)"
        else:
            edit_kind = DocumentEditKind.UPDATE
            after = _remove_markdown_section(
                live.raw_bytes,
                section_anchor,
                max_header_bytes=self.planner.max_front_matter_bytes,
                max_depth=self.planner.max_front_matter_depth,
            )
            summary = f"soft forget section: {_normalized_anchor(section_anchor)[:180]}"
        plan = DocumentEditPlan(
            idempotency_key="soft-forget:"
            + hashlib.sha256(
                canonical_json([document_uri, live.state.raw_sha256, section_anchor or ""]).encode()
            ).hexdigest(),
            tenant_id=live.tenant_id,
            owner_user_id=live.owner_user_id,
            edit_kind=edit_kind,
            expected_state=live.state,
            evidence_digest=evidence_digest,
            edit_summary=summary,
            document_id=live.document_id,
            relative_path=live.relative_path,
            after_bytes=after,
            expected_registration_document_id=live.document_id,
        )
        result = self._commit_or_replay(
            plan,
            caller=caller,
            evidence_reference=f"soft-forget:sha256:{evidence_digest}",
        )
        return ForgetResult(
            **self._result_fields(plan, result),
            mode="SOFT_FORGET",
            recoverable=True,
        )

    def _hard_erase(
        self,
        document_uri: str,
        owner: str,
        document_id: str,
        *,
        expected_digest: str | None,
        caller: LocalUserContext,
    ) -> ForgetResult:
        existing = self.erase_store.load(caller.tenant_id, owner, document_id)
        live: _LiveDocument | None = None
        if existing is None:
            control = self.control_store.load_control(caller.tenant_id, owner, document_id)
            if control is not None and control.status == "deleted":
                scan = self.document_store.full_scan(caller.tenant_id, owner)
                if not scan.complete or scan.errors:
                    raise DocumentConflictError(
                        "hard erase of soft-forgotten memory requires a complete registration scan"
                    )
                if any(
                    isinstance(item, ManagedDocument) and item.document_id == document_id
                    for item in scan.registrations
                ):
                    raise DocumentConflictError("soft-forgotten memory document identity unexpectedly remains live")
                revisions = self.revision_store.list_revisions(caller.tenant_id, owner, document_id)
                latest = revisions[-1] if revisions else None
                if (
                    latest is None
                    or latest.state != "ABSENT"
                    or latest.edit_kind is not DocumentEditKind.DELETE
                    or latest.content_blob_role != "before_delete"
                    or not latest.content_blob_digest
                    or latest.relative_path != control.relative_path
                ):
                    raise DocumentConflictError("soft-forgotten memory lacks one exact retained deletion revision")
                source_digest = latest.content_blob_digest
                if expected_digest is not None and expected_digest != source_digest:
                    raise DocumentConflictError("hard erase expected digest does not match the soft-forgotten source")
                relative_path = latest.relative_path
                document_kind = MemoryDocumentPathPolicy.kind_for(relative_path).value
            else:
                live = self._load_live(document_uri, caller, allow_erasure=False)
                _assert_optional_live_digest(live.state, expected_digest)
                source_digest = live.state.raw_sha256
                relative_path = live.relative_path
                document_kind = live.document_kind
            retained = tuple(
                self.independent_evidence_locator(
                    caller.tenant_id,
                    owner,
                    document_id,
                    source_digest,
                )
            )
        else:
            if expected_digest is not None and expected_digest != existing.source_digest:
                raise DocumentConflictError("hard erase retry changed its exact expected digest")
            source_digest = existing.source_digest
            relative_path = existing.relative_path
            document_kind = existing.document_kind
            retained = existing.independent_evidence_retained
        erased = self.eraser.hard_erase(
            tenant_id=caller.tenant_id,
            owner_user_id=owner,
            document_id=document_id,
            expected_source_digest=source_digest,
            relative_path=relative_path,
            independent_evidence_retained=retained,
        )
        if live is None:
            document_kind = document_kind or ""
        return _hard_erase_result(
            document_uri=document_uri,
            document_id=document_id,
            document_kind=document_kind,
            relative_path=relative_path,
            erased=erased,
        )


def _remove_markdown_section(
    raw: bytes,
    anchor: str,
    *,
    max_header_bytes: int,
    max_depth: int,
) -> bytes:
    parsed = parse_front_matter(raw, max_header_bytes=max_header_bytes, max_depth=max_depth)
    target = _normalized_anchor(anchor)
    lines = parsed.body.splitlines(keepends=True)
    matches: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        match = _HEADING.fullmatch(line.rstrip("\r\n"))
        if match and " ".join(match.group(2).split()) == target:
            matches.append((index, len(match.group(1))))
    if len(matches) != 1:
        raise ValueError("section_anchor must match exactly one Markdown heading")
    start, level = matches[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = _HEADING.fullmatch(lines[index].rstrip("\r\n"))
        if match and len(match.group(1)) <= level:
            end = index
            break
    body = "".join(lines[:start] + lines[end:]).encode()
    return parsed.header_bytes + body


def _normalized_anchor(anchor: str) -> str:
    value = re.sub(r"^#{1,6}[ \t]+", "", str(anchor or "").strip())
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError("section_anchor is required")
    return normalized


def _hard_erase_result(
    *,
    document_uri: str,
    document_id: str,
    document_kind: str,
    relative_path: str,
    erased: DocumentEraseResult,
) -> ForgetResult:
    record = erased.record
    return ForgetResult(
        document_uri=document_uri,
        document_id=document_id,
        document_kind=document_kind,
        relative_path=relative_path,
        document_revision=record.document_revision_floor,
        source_digest="",
        changed=True,
        edit_summary="hard erase whole memory document",
        projection_status=record.status.value,
        mode="HARD_ERASE",
        recoverable=False,
        erasure_status=record.status.value,
        erasure_epoch=record.erasure_epoch,
        pending_backends=record.pending_backends,
        independent_evidence_retained=erased.independent_evidence_retained,
        media_disclaimer=erased.media_disclaimer,
    )


__all__ = ["ForgetOperation"]
