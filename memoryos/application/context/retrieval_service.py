"""Application retrieval service with durable recall traces."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from memoryos.application.context.assembler import ContextAssembler
from memoryos.core.clock import utc_now
from memoryos.core.durable_io import atomic_write_json
from memoryos.security.context_projection import ContextProjectionSanitizer


class RetrievalService:
    def __init__(self, assembler: ContextAssembler, trace_root: str | Path) -> None:
        self.assembler = assembler
        self.trace_root = Path(trace_root).expanduser().resolve()
        self.trace_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.trace_root.chmod(0o700)
        except OSError as exc:
            raise PermissionError("recall trace directory permissions could not be secured") from exc
        self.sanitizer = ContextProjectionSanitizer()

    def search(self, query: str, **kwargs: Any) -> tuple[list[dict[str, Any]], str]:
        self._require_ready()
        try:
            results = self.assembler.search(query, **kwargs)
        except Exception:
            # Canonical visibility marks the bound runtime NOT_READY before
            # raising its typed integrity error.  Re-check at this public
            # service boundary so callers receive the stable readiness
            # contract while the original integrity error remains the cause.
            self._require_ready()
            raise
        self._require_ready()
        trace_id = self._record(query, kwargs, results, [])
        return results, trace_id

    def assemble(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self._require_ready()
        try:
            result = self.assembler.assemble(query, **kwargs)
        except Exception:
            self._require_ready()
            raise
        self._require_ready()
        trace_id = self._record(
            query, kwargs, list(result.get("contexts", [])), list(result.get("dropped_contexts", []))
        )
        return {**result, "trace_id": trace_id}

    def _require_ready(self) -> None:
        context_db = getattr(self.assembler, "context_db", None)
        readiness = getattr(context_db, "readiness", None)
        require_ready = getattr(readiness, "require_ready", None)
        if callable(require_ready):
            require_ready()

    def read_trace(self, trace_id: str) -> dict[str, Any]:
        try:
            parsed = uuid.UUID(str(trace_id))
        except (AttributeError, TypeError, ValueError):
            raise ValueError("trace_id must be a canonical UUID") from None
        canonical_id = str(parsed)
        if canonical_id != str(trace_id):
            raise ValueError("trace_id must be a canonical UUID")
        path = (self.trace_root / f"{canonical_id}.json").resolve()
        try:
            path.relative_to(self.trace_root)
        except ValueError:
            raise ValueError("trace path escapes its tenant root") from None
        if not path.is_file():
            raise FileNotFoundError(canonical_id)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("trace_id") != canonical_id:
            raise ValueError("recall trace is invalid")
        self.sanitizer.assert_safe(value)
        return value

    def _record(
        self, query: str, kwargs: dict[str, Any], selected: list[dict[str, Any]], dropped: list[dict[str, Any]]
    ) -> str:
        trace_id = str(uuid.uuid4())
        trace = {
            "trace_id": trace_id,
            "created_at": utc_now(),
            "query": query,
            "scope": {
                key: kwargs.get(key) for key in ("tenant_id", "user_id", "project_id", "adapter_id", "search_scope")
            },
            "retrieval_views": kwargs.get("retrieval_views") or [],
            "metadata_filters": kwargs.get("connect_filters") or {},
            "candidate_count": len(selected) + len(dropped),
            "lexical_candidates": [
                item.get("uri")
                for item in selected
                if item.get("retrieval_source") in {None, "", "index", "lexical", "hybrid"}
            ],
            "vector_candidates": [
                item.get("uri") for item in selected if item.get("retrieval_source") in {"vector", "hybrid"}
            ],
            "selected": [
                {"uri": item.get("uri"), "score": item.get("score"), "layer": item.get("layer")} for item in selected
            ],
            "dropped": dropped,
            "token_budget": kwargs.get("token_budget"),
            "rerank_enabled": getattr(self.assembler, "reranker", None) is not None,
        }
        safe_trace = self.sanitizer.sanitize_trace(trace)
        if not isinstance(safe_trace, dict) or safe_trace.get("trace_id") != trace_id:
            raise ValueError("recall trace sanitization produced an invalid payload")
        atomic_write_json(
            self.trace_root / f"{trace_id}.json",
            safe_trace,
            artifact_root=self.trace_root,
        )
        return trace_id
