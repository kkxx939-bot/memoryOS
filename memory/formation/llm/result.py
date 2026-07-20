"""LLM 记忆提取的批次结果与候选拒绝原因。"""

from __future__ import annotations

from dataclasses import dataclass

from memory.core.model import MemoryEditProposal
from memory.formation.egress import EgressDecision


@dataclass(frozen=True)
class RejectedMemoryCandidate:
    """记录单个模型候选被确定性校验拒绝的原因。"""

    index: int
    reason: str


@dataclass(frozen=True)
class MemoryExtractionBatchResult:
    """区分可进入规划的候选与被拒绝候选，不隐藏部分失败。"""

    accepted: tuple[MemoryEditProposal, ...]
    rejected: tuple[RejectedMemoryCandidate, ...]
    outbound_digest: str = ""
    egress_decision: str = EgressDecision.LOCAL_ONLY.value


__all__ = ["MemoryExtractionBatchResult", "RejectedMemoryCandidate"]
