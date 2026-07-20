"""从观察和案例形成稳定行为结构的领域规则。"""

from behavior.core.formation.behavior_extractor import BehaviorExtractor
from behavior.core.formation.lifecycle import (
    BehaviorLifecycleResult,
    BehaviorLifecycleService,
)
from behavior.core.formation.scene_key_builder import SceneKeyBuilder

__all__ = [
    "BehaviorExtractor",
    "BehaviorLifecycleResult",
    "BehaviorLifecycleService",
    "SceneKeyBuilder",
]
