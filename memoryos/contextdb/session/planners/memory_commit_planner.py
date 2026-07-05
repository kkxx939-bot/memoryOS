from __future__ import annotations

from memoryos.behavior.update.behavior_window import BehaviorWindowEvaluator
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.ids import stable_hash
from memoryos.memory.model.memory import Memory, MemoryAnchor, MemoryCandidate, MemoryKind
from memoryos.memory.service.memory_updater import MemoryUpdater
from memoryos.operations.model.context_operation import ContextOperation


class RuleMemoryCommitPlanner:
    """Fallback rule planner.

    Production LLM memory extraction must emit accepted, pending, and rejected
    operation groups before operations enter the commit plane. This planner is a
    deterministic fallback for local/session signals and tests.
    """

    def __init__(self) -> None:
        self.updater = MemoryUpdater()
        self.window_evaluator = BehaviorWindowEvaluator()

    def plan(self, archive: SessionArchive) -> list[ContextOperation]:
        operations: list[ContextOperation] = []
        for message in archive.messages:
            text = str(message.get("content", message.get("text", ""))).strip()
            if not text:
                continue
            lowered = text.lower()
            if any(token in lowered for token in ("记住", "我喜欢", "我不喜欢", "prefer", "remember")):
                operations.append(self.updater.add_memory(self._memory(archive.user_id, text, MemoryKind.EXPLICIT), evidence=[{"source": "session_message"}]))
            if any(token in lowered for token in ("以后别", "不要自动", "禁止自动", "do not automatically", "no auto")):
                operations.append(self.updater.policy_rule(self._memory(archive.user_id, text, MemoryKind.POLICY), evidence=[{"source": "explicit_rule"}]))

        scene_groups = self._scene_groups(archive)
        for (scene_key, _similarity_key), observations in scene_groups.items():
            if len(observations) >= 2:
                operations.append(
                    self.updater.add_memory(
                        self._anchor(archive.user_id, scene_key),
                        evidence=[{"source": "behavior_cluster", "count": len(observations)}],
                    )
                )
            positive_feedback = [item for item in archive.feedback if float(item.get("reward", item.get("reward_value", 0.0)) or 0.0) > 0]
            if len(observations) >= 3 and positive_feedback:
                operations.append(
                    self.updater.add_memory(
                        self._candidate(archive.user_id, scene_key),
                        evidence=[{"source": "behavior_pattern", "count": len(observations)}],
                    )
                )
        return operations

    def _scene_groups(self, archive: SessionArchive) -> dict[tuple[str, tuple[str, ...]], list[dict]]:
        groups: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
        for observation in archive.observations:
            scene_key = str(observation.get("scene_key", observation.get("scene", "default")))
            similarity_key = self.window_evaluator._similarity_key(observation)
            groups.setdefault((scene_key, similarity_key), []).append(observation)
        return groups

    def _memory(self, user_id: str, content: str, kind: MemoryKind) -> Memory:
        digest = stable_hash([user_id, kind.value, content], length=16)
        return Memory(
            uri=f"memoryos://user/{user_id}/memories/{kind.value}/{digest}",
            user_id=user_id,
            title=content[:48] or kind.value,
            content=content,
            kind=kind,
            confidence=1.0 if kind in {MemoryKind.EXPLICIT, MemoryKind.POLICY} else 0.65,
        )

    def _anchor(self, user_id: str, scene_key: str) -> MemoryAnchor:
        anchor_key = f"{scene_key}_anchor"
        return MemoryAnchor(
            uri=f"memoryos://user/{user_id}/memories/anchors/{anchor_key}",
            user_id=user_id,
            title=f"{scene_key} behavior anchor",
            content=f"User has a recurring behavior theme around {scene_key}; related preferences, resources, and policies require continued observation.",
            anchor_key=anchor_key,
        )

    def _candidate(self, user_id: str, scene_key: str) -> MemoryCandidate:
        return MemoryCandidate(
            uri=f"memoryos://user/{user_id}/memories/candidates/{scene_key}",
            user_id=user_id,
            title=f"{scene_key} inferred preference candidate",
            content=f"Repeated positive behavior suggests a candidate preference for {scene_key}.",
            kind=MemoryKind.CANDIDATE,
            confidence=0.6,
        )


MemoryCommitPlanner = RuleMemoryCommitPlanner
