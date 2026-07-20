"""行为领域核心：行为模型、形成规则和机会评估。"""

from behavior.core.evaluation import (
    BehaviorCoolingService,
    BehaviorWindowDecision,
    BehaviorWindowEvaluator,
    OpportunityAwareDecay,
    OpportunityWindow,
    build_opportunity_window,
)
from behavior.core.formation import (
    BehaviorExtractor,
    BehaviorLifecycleResult,
    BehaviorLifecycleService,
    SceneKeyBuilder,
)
from behavior.core.model import (
    BehaviorCase,
    BehaviorCluster,
    BehaviorPattern,
    FeedbackSignal,
    Observation,
    OpportunityDecayResult,
    OpportunityStats,
)
from behavior.core.support import BehaviorSupportAnchor

__all__ = [
    "BehaviorCase",
    "BehaviorCluster",
    "BehaviorCoolingService",
    "BehaviorExtractor",
    "BehaviorLifecycleResult",
    "BehaviorLifecycleService",
    "BehaviorPattern",
    "BehaviorSupportAnchor",
    "BehaviorWindowDecision",
    "BehaviorWindowEvaluator",
    "FeedbackSignal",
    "Observation",
    "OpportunityAwareDecay",
    "OpportunityDecayResult",
    "OpportunityStats",
    "OpportunityWindow",
    "SceneKeyBuilder",
    "build_opportunity_window",
]
