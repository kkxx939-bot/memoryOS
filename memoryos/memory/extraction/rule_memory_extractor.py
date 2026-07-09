from __future__ import annotations

from memoryos.contextdb.model.context_type import ContextType
from memoryos.core.ids import stable_hash
from memoryos.memory.extraction.memory_extractor import ExtractionResult, MemoryExtractor
from memoryos.memory.model.memory import Memory, MemoryKind
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class RuleMemoryExtractor(MemoryExtractor):
    """Legacy operation-emitting extractor.

    Kept for compatibility tests and old callers. MemoryCommitPlanner uses
    RuleFallbackExtractor, which emits MemoryCandidateDraft objects instead.
    """

    markers = ("记住：", "记住:", "remember:", "Remember:")

    def extract(self, session_archive) -> ExtractionResult:
        result = ExtractionResult(extractor_version="rule_context_memory_extractor_v1")
        for message in getattr(session_archive, "messages", []):
            text = str(message.get("content", message.get("text", "")))
            for marker in self.markers:
                if marker not in text:
                    continue
                memory_text = text.split(marker, 1)[1].strip()
                if not memory_text:
                    continue
                memory = self._memory(getattr(session_archive, "user_id", ""), memory_text)
                result.accepted.append(
                    ContextOperation(
                        user_id=memory.user_id,
                        context_type=ContextType.MEMORY,
                        action=OperationAction.ADD,
                        target_uri=memory.uri,
                        payload={"context_object": memory.to_context_object().to_dict(), "content": memory.content},
                        evidence=[{"source_session_id": getattr(session_archive, "session_id", "")}],
                        confidence=memory.confidence,
                        source_session_id=getattr(session_archive, "session_id", None),
                    )
                )
                break
        return result

    def _memory(self, user_id: str, text: str) -> Memory:
        kind = MemoryKind.POLICY if any(word in text for word in ("不要", "不能", "允许", "必须")) else MemoryKind.EXPLICIT
        key = stable_hash([user_id, text], length=16)
        return Memory(
            uri=f"memoryos://user/{user_id}/memories/{'policies' if kind == MemoryKind.POLICY else 'explicit'}/{key}",
            user_id=user_id,
            title=text[:32],
            content=text,
            kind=kind,
            confidence=1.0,
        )
