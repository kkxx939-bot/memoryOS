from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.contextdb.store.local_stores import InMemoryIndexStore
from memoryos.prediction.model.prediction_ledger import PredictionLedger
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.model.prediction_result import PredictionResult
from memoryos.prediction.pipeline.prediction_engine import PredictionEngine


class MemoryOSClient:
    def __init__(self, root: str, index_store: InMemoryIndexStore | None = None) -> None:
        self.root = root
        self.index_store = index_store or InMemoryIndexStore()
        self.engine = PredictionEngine(self.index_store, PredictionLedger(root))

    def predict(self, request: PredictionRequest, policies: list[ActionPolicy]) -> PredictionResult:
        return self.engine.process(request, policies=policies)
