from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from memoryos.memory.canonical.episode import EvidenceEpisode


@dataclass(frozen=True)
class SalienceDecision:
    salient: bool
    reasons: tuple[str, ...]


class EpisodeSalienceGate:
    """A deliberately small episode gate; it never creates memory content."""

    IMPORTANT_EVENT_TYPES = {
        "DECISION",
        "PREFERENCE",
        "RULE",
        "TASK_RESULT",
        "TOOL_FAILURE",
        "TOOL_RECOVERY",
        "CONFIG_CHANGED",
        "STATE_CHANGED",
        "ENTITY_CHANGED",
        "FEEDBACK",
        "TOOL_RESULT",
    }

    def evaluate(self, episode: EvidenceEpisode) -> SalienceDecision:
        reasons: list[str] = []
        repeated = Counter(event.text().strip().casefold() for event in episode.events if event.text().strip())
        if any(count > 1 for count in repeated.values()):
            reasons.append("repeated_pattern")
        for event in episode.events:
            if event.event_type in self.IMPORTANT_EVENT_TYPES:
                reasons.append(f"event_type:{event.event_type.lower()}")
            if bool(event.metadata.get("salient")):
                reasons.append("adapter_marked")
            text = event.text().casefold()
            if any(marker in text for marker in ("remember:", "remember this", "记住", "请记住")):
                reasons.append("explicit_remember")
            if event.actor.kind == "user":
                reasons.append("user_episode_boundary")
            if event.actor.kind == "assistant" and any(
                marker in text
                for marker in ("implemented", "completed", "outcome", "reusable", "已实现", "完成", "结果")
            ):
                reasons.append("assistant_task_result")
        if not reasons and len(episode.events) > 1:
            reasons.append("episode_batch")
        return SalienceDecision(bool(reasons), tuple(dict.fromkeys(reasons)))
