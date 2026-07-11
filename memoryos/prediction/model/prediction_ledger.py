"""预测模块里的预测台账。"""

from __future__ import annotations

import json
from pathlib import Path

from memoryos.prediction.model.prediction_result import PredictionResult


class PredictionLedger:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def record(self, result: PredictionResult) -> Path:
        path = self.root / "tenants" / "default" / "users" / result.observation.user_id / "predictions" / "ledger.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        return path
