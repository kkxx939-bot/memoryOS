from __future__ import annotations

from typing import Any

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.ids import stable_hash
from memoryos.memory.admission import MemoryAdmissionGate
from memoryos.memory.extraction import MemoryExtractorBackend, RuleFallbackExtractor
from memoryos.memory.model.memory import Memory, MemoryCandidate, MemoryKind
from memoryos.memory.schema import (
    MEMORY_SCHEMA_VERSION,
    AdmissionDecision,
    MemoryCandidateDraft,
    MemoryOperationGroup,
    MemoryOperationGroupItem,
    MemoryType,
    MemoryTypeRegistry,
)
from memoryos.memory.service.memory_updater import MemoryUpdater
from memoryos.memory.view import adapter_id_from_archive, project_id_from_archive
from memoryos.operations.model.context_operation import ContextOperation


class RuleMemoryCommitPlanner:
    """Schema-driven memory planner with deterministic fallback extraction."""

    def __init__(
        self,
        extractor: MemoryExtractorBackend | None = None,
        registry: MemoryTypeRegistry | None = None,
        admission_gate: MemoryAdmissionGate | None = None,
    ) -> None:
        self.registry = registry or MemoryTypeRegistry()
        self.extractor = extractor or RuleFallbackExtractor()
        self.admission_gate = admission_gate or MemoryAdmissionGate(self.registry)
        self.updater = MemoryUpdater()
        self.last_group = MemoryOperationGroup()

    def plan(self, archive: SessionArchive) -> list[ContextOperation]:
        operations: list[ContextOperation] = []
        group = MemoryOperationGroup()
        schemas = self.registry.list()
        project_id = project_id_from_archive(archive)
        adapter_id = adapter_id_from_archive(archive)
        for candidate in self.extractor.extract(archive, schemas):
            admission = self.admission_gate.evaluate(
                candidate,
                user_id=archive.user_id,
                project_id=project_id,
                adapter_id=adapter_id,
            )
            group.add(candidate, admission)
        for item in [*group.accepted, *group.pending]:
            operations.append(self._operation(archive, item))
        self.last_group = group
        return operations

    def _operation(self, archive: SessionArchive, item: MemoryOperationGroupItem) -> ContextOperation:
        candidate = item.candidate
        admission = item.admission
        memory = self._memory(archive, candidate, item)
        operation = self.updater.add_memory(memory, evidence=self._evidence(candidate))
        operation.source_session_id = candidate.source_session_id or archive.session_id
        operation.payload = {
            **operation.payload,
            "memory_type": candidate.memory_type.value,
            "admission": admission.to_metadata(),
            "retrieval_views": admission.retrieval_views,
            "source_adapter_id": candidate.source_adapter_id,
            "source_session_id": candidate.source_session_id or archive.session_id,
            "source_roles": [candidate.source_role],
            "merge_key": admission.merge_key or candidate.merge_key,
            "schema_version": MEMORY_SCHEMA_VERSION,
        }
        context_object = operation.payload.get("context_object")
        if isinstance(context_object, dict):
            metadata = dict(context_object.get("metadata", {}) or {})
            metadata.update(
                {
                    "memory_type": candidate.memory_type.value,
                    "admission": admission.to_metadata(),
                    "retrieval_views": admission.retrieval_views,
                    "source": memory.source,
                    "merge_key": admission.merge_key or candidate.merge_key,
                    "schema_version": MEMORY_SCHEMA_VERSION,
                }
            )
            context_object["metadata"] = metadata
        return operation

    def _memory(self, archive: SessionArchive, candidate: MemoryCandidateDraft, item: MemoryOperationGroupItem) -> Memory:
        admission = item.admission
        source: dict[str, Any] = {
            "adapter_id": candidate.source_adapter_id,
            "session_id": candidate.source_session_id or archive.session_id,
            "message_ids": candidate.source_message_ids,
            "roles": [candidate.source_role],
        }
        uri = self._uri(archive.user_id, candidate, admission.decision)
        title = candidate.title[:96] or candidate.memory_type.value
        kind = self._kind(candidate, admission.decision)
        confidence = min(candidate.confidence, admission.confidence)
        admission_metadata = admission.to_metadata()
        merge_key = admission.merge_key or candidate.merge_key
        if admission.decision == AdmissionDecision.PENDING:
            return MemoryCandidate(
                uri=uri,
                user_id=archive.user_id,
                title=title,
                content=candidate.content,
                kind=kind,
                confidence=confidence,
                memory_type=candidate.memory_type.value,
                retrieval_views=admission.retrieval_views,
                admission=admission_metadata,
                merge_key=merge_key,
                source=source,
                memory_schema_version=MEMORY_SCHEMA_VERSION,
            )
        return Memory(
            uri=uri,
            user_id=archive.user_id,
            title=title,
            content=candidate.content,
            kind=kind,
            confidence=confidence,
            memory_type=candidate.memory_type.value,
            retrieval_views=admission.retrieval_views,
            admission=admission_metadata,
            merge_key=merge_key,
            source=source,
            memory_schema_version=MEMORY_SCHEMA_VERSION,
        )

    def _uri(self, user_id: str, candidate: MemoryCandidateDraft, decision: AdmissionDecision) -> str:
        bucket = "candidates" if decision == AdmissionDecision.PENDING else self._bucket(candidate.memory_type)
        digest = stable_hash([user_id, candidate.memory_type.value, candidate.merge_key, candidate.content], length=20)
        return f"memoryos://user/{user_id}/memories/{bucket}/{digest}"

    def _bucket(self, memory_type: MemoryType) -> str:
        return {
            MemoryType.PROFILE: "profile",
            MemoryType.PREFERENCE: "preferences",
            MemoryType.ENTITY: "entities",
            MemoryType.EVENT: "events",
            MemoryType.PROJECT_RULE: "rules",
            MemoryType.PROJECT_DECISION: "decisions",
            MemoryType.AGENT_EXPERIENCE: "agent_experience",
        }[memory_type]

    def _kind(self, candidate: MemoryCandidateDraft, decision: AdmissionDecision) -> MemoryKind:
        if decision == AdmissionDecision.PENDING:
            return MemoryKind.CANDIDATE
        if candidate.memory_type == MemoryType.PROJECT_RULE:
            return MemoryKind.POLICY
        return MemoryKind.EXPLICIT

    def _evidence(self, candidate: MemoryCandidateDraft) -> list[dict]:
        return [
            *candidate.evidence,
            {
                "source_adapter_id": candidate.source_adapter_id,
                "source_session_id": candidate.source_session_id,
                "source_message_ids": candidate.source_message_ids,
                "reason": candidate.reason,
            },
        ]


MemoryCommitPlanner = RuleMemoryCommitPlanner
