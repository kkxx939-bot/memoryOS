"""多个记忆文档的合并、提案和断点恢复操作。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import unified_diff

from foundation.identity import LocalUserContext
from foundation.integrity import canonical_json
from infrastructure.store.memory.review import (
    MemoryEditReviewWorkflow,
    ReviewConsolidationSource,
)
from memory.commit.consolidation import ConsolidationResult, ConsolidationSource
from memory.core.model import DocumentEditKind, DocumentEditPlan
from memory.core.structure.frontmatter import parse_front_matter
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.execute.base import MemoryCommandBase, _LiveDocument, _require_sha256
from memory.execute.contracts import MemoryConsolidationProposalResult
from memory.ports.document_store import DocumentConflictError


@dataclass(frozen=True)
class _PreparedConsolidation:
    """已经绑定精确源摘要、可安全提交的合并计划。"""

    target: _LiveDocument
    plan: DocumentEditPlan
    sources: tuple[ConsolidationSource, ...]
    request_digest: str


class ConsolidateOperation(MemoryCommandBase):
    """负责记忆合并的校验、预览、提交和恢复。"""

    def consolidate_memory_documents(
        self,
        target_plan: DocumentEditPlan,
        sources: Sequence[ConsolidationSource],
        *,
        idempotency_key: str,
        caller: LocalUserContext,
    ) -> ConsolidationResult:
        """执行已经过校验的只向前合并计划。"""

        self._require_ready()
        caller.assert_identity(
            tenant_id=target_plan.tenant_id,
            user_id=target_plan.owner_user_id,
        )
        if self.consolidator is None:
            raise RuntimeError("memory document consolidation is not configured")
        return self.consolidator.consolidate(
            target_plan,
            sources,
            idempotency_key=idempotency_key,
            actor_binding=self._actor_binding(caller),
        )

    def merge_memory_documents(
        self,
        target_document_uri: str,
        merged_edit: str,
        expected_target_digest: str,
        source_documents: Sequence[Mapping[str, str]],
        *,
        caller: LocalUserContext,
    ) -> ConsolidationResult:
        """从调用者提供的文档 URI 和精确摘要构造并提交合并。"""

        self._require_ready()
        if self.consolidator is None:
            raise RuntimeError("memory document consolidation is not configured")
        prepared = self._prepare_memory_consolidation(
            target_document_uri,
            merged_edit,
            expected_target_digest,
            source_documents,
            caller=caller,
        )
        return self.consolidator.consolidate(
            prepared.plan,
            prepared.sources,
            idempotency_key=f"merge:{prepared.request_digest}",
            actor_binding=self._actor_binding(caller),
        )

    def propose_memory_consolidation(
        self,
        target_document_uri: str,
        merged_edit: str,
        expected_target_digest: str,
        source_documents: Sequence[Mapping[str, str]],
        *,
        caller: LocalUserContext,
    ) -> MemoryConsolidationProposalResult:
        """生成写时复制的合并预览，不修改当前文档。"""

        self._require_ready()
        if self.review_store is None:
            raise RuntimeError("memory consolidation review is not configured")
        prepared = self._prepare_memory_consolidation(
            target_document_uri,
            merged_edit,
            expected_target_digest,
            source_documents,
            caller=caller,
        )
        assert prepared.plan.after_bytes is not None
        proposed_diff = _document_diff(prepared.target.raw_bytes, prepared.plan.after_bytes)
        if not proposed_diff:
            raise ValueError("consolidation proposal must change the target Markdown")
        review_sources = tuple(
            ReviewConsolidationSource(
                document_id=source.document_id,
                relative_path=source.relative_path,
                raw_sha256=source.raw_sha256,
                size=source.size,
            )
            for source in prepared.sources
        )
        record = self.review_store.seal(
            prepared.plan,
            proposed_diff=proposed_diff,
            workflow_kind=MemoryEditReviewWorkflow.CONSOLIDATION,
            consolidation_sources=review_sources,
        )
        return MemoryConsolidationProposalResult(
            proposal_id=record.proposal_id,
            status=record.status.value,
            document_uri=prepared.target.document_uri,
            document_id=prepared.target.document_id,
            document_kind=prepared.target.document_kind,
            relative_path=prepared.target.relative_path,
            source_digest=prepared.target.state.raw_sha256,
            proposed_source_digest=record.after_blob_digest,
            proposed_diff_digest=record.proposed_diff_blob_digest,
            proposed_diff=proposed_diff.decode("utf-8", errors="strict"),
            edit_summary=record.edit_summary,
            workflow_kind=record.workflow_kind.value,
            consolidation_sources=tuple(
                _review_consolidation_source_payload(prepared.target.owner_user_id, source)
                for source in review_sources
            ),
        )

    def resume_memory_consolidation(
        self,
        saga_id: str,
        *,
        caller: LocalUserContext,
    ) -> ConsolidationResult:
        """从持久状态继续执行已封存的只向前合并。"""

        self._require_ready()
        if self.consolidator is None:
            raise RuntimeError("memory document consolidation is not configured")
        return self.consolidator.resume(
            tenant_id=caller.tenant_id,
            owner_user_id=caller.user_id,
            saga_id=str(saga_id),
            actor_binding=self._actor_binding(caller),
        )

    def _prepare_memory_consolidation(
        self,
        target_document_uri: str,
        merged_edit: str,
        expected_target_digest: str,
        source_documents: Sequence[Mapping[str, str]],
        *,
        caller: LocalUserContext,
    ) -> _PreparedConsolidation:
        target = self._load_live(target_document_uri, caller)
        _require_sha256(expected_target_digest, "expected_target_digest")
        if target.state.raw_sha256 != expected_target_digest:
            raise DocumentConflictError("merge target expected digest does not match live Markdown")
        replacement_body = str(merged_edit or "")
        if not replacement_body.strip():
            raise ValueError("merged_edit must contain the complete target Markdown body")
        bounded_sources = tuple(source_documents)
        if not bounded_sources or len(bounded_sources) > 100:
            raise ValueError("merge requires between 1 and 100 source documents")
        sources: list[ConsolidationSource] = []
        seen_ids: set[str] = set()
        source_identity: list[tuple[str, str]] = []
        for item in bounded_sources:
            if set(item) != {"document_uri", "expected_digest"}:
                raise ValueError("merge source must contain only document_uri and expected_digest")
            source_uri = str(item["document_uri"])
            expected_digest = str(item["expected_digest"])
            _require_sha256(expected_digest, "source expected_digest")
            source = self._load_live(source_uri, caller)
            if source.document_id == target.document_id:
                raise ValueError("merge target cannot also be a redundant source")
            if source.document_id in seen_ids:
                raise ValueError("merge source documents must be unique")
            seen_ids.add(source.document_id)
            if source.state.raw_sha256 != expected_digest:
                raise DocumentConflictError("merge source expected digest does not match live Markdown")
            sources.append(
                ConsolidationSource(
                    document_id=source.document_id,
                    relative_path=source.relative_path,
                    raw_sha256=source.state.raw_sha256,
                    size=source.state.size,
                )
            )
            source_identity.append((source.document_uri, source.state.raw_sha256))
        sources.sort(key=lambda source: (source.document_id, source.relative_path, source.raw_sha256, source.size))
        parsed = parse_front_matter(
            target.raw_bytes,
            max_header_bytes=self.planner.max_front_matter_bytes,
            max_depth=self.planner.max_front_matter_depth,
        )
        body_bytes = replacement_body.encode("utf-8")
        if not body_bytes.startswith(b"\n"):
            body_bytes = b"\n" + body_bytes
        if not body_bytes.endswith(b"\n"):
            body_bytes += b"\n"
        after = parsed.header_bytes + body_bytes
        after_digest = hashlib.sha256(after).hexdigest()
        request_identity = [
            target.document_uri,
            target.state.raw_sha256,
            after_digest,
            sorted(source_identity),
        ]
        request_digest = hashlib.sha256(canonical_json(request_identity).encode()).hexdigest()
        plan = DocumentEditPlan(
            idempotency_key=f"merge-target:{request_digest}",
            tenant_id=target.tenant_id,
            owner_user_id=target.owner_user_id,
            edit_kind=DocumentEditKind.UPDATE,
            expected_state=target.state,
            evidence_digest=request_digest,
            edit_summary="merge memory documents into target",
            document_id=target.document_id,
            relative_path=target.relative_path,
            after_bytes=after,
            expected_registration_document_id=target.document_id,
        )
        return _PreparedConsolidation(
            target=target,
            plan=plan,
            sources=tuple(sources),
            request_digest=request_digest,
        )


def _document_diff(before: bytes, after: bytes) -> bytes:
    try:
        before_lines = before.decode("utf-8", errors="strict").splitlines(keepends=True)
        after_lines = after.decode("utf-8", errors="strict").splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        raise ValueError("consolidation proposal Markdown must be UTF-8") from exc
    return "".join(
        unified_diff(
            before_lines,
            after_lines,
            fromfile="live-target-markdown",
            tofile="proposed-consolidated-markdown",
        )
    ).encode("utf-8")


def _review_consolidation_source_payload(
    owner_user_id: str,
    source: ReviewConsolidationSource,
) -> dict[str, object]:
    return {
        "document_uri": MemoryDocumentPathPolicy.document_uri(owner_user_id, source.document_id),
        "document_id": source.document_id,
        "relative_path": source.relative_path,
        "source_digest": source.raw_sha256,
        "size": source.size,
    }


__all__ = ["ConsolidateOperation"]
