from __future__ import annotations

from pathlib import Path

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.contextdb.store import FileSystemSourceStore, IndexStore, RelationStore, SourceStore
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.sqlite_relation_store import SQLiteRelationStore
from memoryos.prediction.model.prediction_ledger import PredictionLedger
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.model.prediction_result import PredictionResult
from memoryos.prediction.pipeline.prediction_engine import PredictionEngine


class MemoryOSClient:
    def __init__(
        self,
        root: str,
        index_store: IndexStore | None = None,
        source_store: SourceStore | None = None,
        relation_store: RelationStore | None = None,
    ) -> None:
        self.root = root
        root_path = Path(root)
        self.source_store = source_store or FileSystemSourceStore(root_path)
        self.index_store = index_store or SQLiteIndexStore(root_path / "indexes" / "context.sqlite3")
        self.relation_store = relation_store or SQLiteRelationStore(root_path / "indexes" / "relations.sqlite3")
        self.engine = PredictionEngine(
            self.index_store,
            PredictionLedger(root),
            source_store=self.source_store,
            relation_store=self.relation_store,
        )

    def predict(self, request: PredictionRequest, policies: list[ActionPolicy]) -> PredictionResult:
        return self.engine.process(request, policies=policies)
