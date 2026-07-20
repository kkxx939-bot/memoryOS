"""ActionPolicy 在线决策的最小化、不可变审计存储。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from infrastructure.store.filesystem.durable_io import atomic_create_json
from foundation.ids import require_safe_path_segment, stable_hash
from policy.action_policy.decision.result import PredictionResult


class ActionPolicyDecisionLedger:
    """按租户保存不包含观察原文和上下文正文的决策审计记录。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def record(self, result: PredictionResult, *, tenant_id: str) -> Path:
        tenant = require_safe_path_segment(str(tenant_id or ""), "tenant_id")
        user_id = require_safe_path_segment(result.observation.user_id, "user_id")
        artifact_root = self.root if tenant == "default" else self.root / "tenants" / tenant
        payload = self._payload(result, tenant_id=tenant)
        record_id = stable_hash(payload, length=32)
        path = artifact_root / "action-policy" / "decision-ledger" / user_id / f"{record_id}.json"
        atomic_create_json(path, payload, artifact_root=artifact_root)
        return path

    @staticmethod
    def _payload(result: PredictionResult, *, tenant_id: str) -> dict[str, Any]:
        candidates = []
        for candidate in result.candidates:
            numeric_features = {
                str(key): float(value)
                for key, value in candidate.features.items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
            candidates.append(
                {
                    "action": candidate.action,
                    "score": candidate.score,
                    "policy_uri": candidate.policy_uri,
                    "features": numeric_features,
                }
            )
        source_uri_digests = [
            hashlib.sha256(uri.encode("utf-8")).hexdigest()
            for uri in result.action_context.source_uris
        ]
        return {
            "schema_version": "action_policy_decision_v1",
            "tenant_id": tenant_id,
            "user_id": result.observation.user_id,
            "request_id": result.request_id,
            "episode_id": result.episode_id,
            "scene_key": result.observation.scene_key,
            "observed_at": result.observation.observed_at,
            "candidates": candidates,
            "decision": result.decision.to_dict(),
            "source_uri_digests": source_uri_digests,
        }


__all__ = ["ActionPolicyDecisionLedger"]
