"""相关旧记忆检索使用的严格配置和结果模型。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from infrastructure.editor.snapshot import SnapshotBatch
from memory.editor.reader import MemorySnapshotBatch
from memory.uri import MemoryURI, MemoryURINodeType


class MemoryRetrievalError(RuntimeError):
    """相关旧记忆检索失败，不能安全地继续生成编辑操作。"""


@dataclass(frozen=True)
class MemoryRetrievalConfig:
    """限制会话查询、语义命中和确定性工具 URI 的资源使用。"""

    max_query_chars: int = 5_000
    max_prompt_chars: int = 1_000
    max_completion_chars: int = 500
    max_tool_message_chars: int = 500
    search_limit: int = 5
    max_tool_uris: int = 32

    def __post_init__(self) -> None:
        for name, value in {
            "max_query_chars": self.max_query_chars,
            "max_prompt_chars": self.max_prompt_chars,
            "max_completion_chars": self.max_completion_chars,
            "max_tool_message_chars": self.max_tool_message_chars,
            "search_limit": self.search_limit,
            "max_tool_uris": self.max_tool_uris,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class MemorySearchHit:
    """语义搜索返回的一个 L2 URI，以及可审计的阶段分数。"""

    uri: MemoryURI
    score: float
    vector_score: float | None = None
    rerank_score: float | None = None

    def __post_init__(self) -> None:
        parsed = MemoryURI.parse(self.uri)
        if parsed.node_type is not MemoryURINodeType.DOCUMENT:
            raise ValueError("memory search hit must identify an L2 document")
        object.__setattr__(self, "uri", parsed)
        if isinstance(self.score, bool) or not isinstance(self.score, int | float):
            raise TypeError("memory search hit score must be numeric")
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("memory search hit score must be finite")
        object.__setattr__(self, "score", score)
        vector_score = score if self.vector_score is None else _finite_score(
            self.vector_score,
            "memory search vector_score",
        )
        if not -1.0 <= vector_score <= 1.0:
            raise ValueError("memory search vector_score must be between -1 and 1")
        object.__setattr__(self, "vector_score", vector_score)
        if self.rerank_score is not None:
            rerank_score = _finite_score(
                self.rerank_score,
                "memory search rerank_score",
            )
            if score != rerank_score:
                raise ValueError("memory search final score must equal its rerank score")
            object.__setattr__(self, "rerank_score", rerank_score)


@dataclass(frozen=True)
class MemoryRelatedContext:
    """绑定一个 ConversationSegment 的相关旧记忆完整读取结果。"""

    conversation_id: str
    segment_id: str
    source_segment_digest: str
    query: str
    search_roots: tuple[MemoryURI, ...]
    search_hits: tuple[MemorySearchHit, ...]
    snapshots: MemorySnapshotBatch

    def __post_init__(self) -> None:
        for name, value in {
            "conversation_id": self.conversation_id,
            "segment_id": self.segment_id,
        }.items():
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
        if not _is_sha256(self.source_segment_digest):
            raise ValueError("source_segment_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("related memory query must be non-empty")
        if not isinstance(self.snapshots, SnapshotBatch):
            raise TypeError("snapshots must be a MemorySnapshotBatch")

        roots = tuple(MemoryURI.parse(uri) for uri in self.search_roots)
        if any(uri.node_type is not MemoryURINodeType.DIRECTORY for uri in roots):
            raise ValueError("related memory search roots must identify directories")
        if roots != tuple(sorted(roots, key=str)) or len(roots) != len(set(roots)):
            raise ValueError("related memory search roots must be unique and sorted")
        object.__setattr__(self, "search_roots", roots)

        hits = tuple(self.search_hits)
        if any(not isinstance(hit, MemorySearchHit) for hit in hits):
            raise TypeError("search_hits must contain MemorySearchHit values")
        expected_hits = tuple(sorted(hits, key=lambda hit: (-hit.score, str(hit.uri))))
        if hits != expected_hits or len({hit.uri for hit in hits}) != len(hits):
            raise ValueError("related memory search hits must be unique and relevance-sorted")
        for hit in hits:
            if not any(hit.uri.matches_prefix(root) for root in roots):
                raise ValueError("related memory search hit is outside its allowed roots")
            if self.snapshots.get(str(hit.uri)) is None:
                raise ValueError("related memory search hit has no corresponding read snapshot")
        object.__setattr__(self, "search_hits", hits)

    @property
    def selected_uris(self) -> tuple[MemoryURI, ...]:
        """返回已经尝试完整读取的全部规范 L2 URI。"""

        return tuple(MemoryURI(snapshot.identity) for snapshot in self.snapshots.snapshots)


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_score(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{label} must be finite")
    return score


__all__ = [
    "MemoryRelatedContext",
    "MemoryRetrievalConfig",
    "MemoryRetrievalError",
    "MemorySearchHit",
]
