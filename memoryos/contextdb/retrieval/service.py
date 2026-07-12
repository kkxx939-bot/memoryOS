"""上下文数据库里的服务。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from memoryos.adapters.agent_hooks.sanitizer import sanitize_text
from memoryos.contextdb.retrieval.context_assembler import ContextAssembler
from memoryos.core.time import utc_now
from memoryos.operations.commit.effect_marker import atomic_write_json


class RetrievalService:
    def __init__(self, assembler: ContextAssembler, trace_root: str | Path) -> None:
        self.assembler = assembler
        self.trace_root = Path(trace_root).expanduser().resolve()
        self.trace_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.trace_root.chmod(0o700)
        except OSError:
            pass

    def search(self, query: str, **kwargs: Any) -> tuple[list[dict[str, Any]], str]:
        results = self.assembler.search(query, **kwargs)
        trace_id = self._record(query, kwargs, results, [])
        return results, trace_id

    def assemble(self, query: str, **kwargs: Any) -> dict[str, Any]:
        result = self.assembler.assemble(query, **kwargs)
        trace_id = self._record(query, kwargs, list(result.get("contexts", [])), list(result.get("dropped_contexts", [])))
        return {**result, "trace_id": trace_id}

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
        return value

    def _record(self, query: str, kwargs: dict[str, Any], selected: list[dict[str, Any]], dropped: list[dict[str, Any]]) -> str:
        trace_id = str(uuid.uuid4())
        trace = {
            "trace_id": trace_id,
            "created_at": utc_now(),
            "query": sanitize_text(query, max_text=1000),
            "scope": {
                key: kwargs.get(key)
                for key in ("tenant_id", "user_id", "project_id", "adapter_id", "search_scope")
            },
            "retrieval_views": kwargs.get("retrieval_views") or [],
            "metadata_filters": kwargs.get("connect_filters") or {},
            "candidate_count": len(selected) + len(dropped),
            "lexical_candidates": [item.get("uri") for item in selected if item.get("retrieval_source") in {None, "", "index", "lexical", "hybrid"}],
            "vector_candidates": [item.get("uri") for item in selected if item.get("retrieval_source") in {"vector", "hybrid"}],
            "selected": [{"uri": item.get("uri"), "score": item.get("score"), "layer": item.get("layer")} for item in selected],
            "dropped": dropped,
            "token_budget": kwargs.get("token_budget"),
            "rerank_enabled": getattr(self.assembler, "reranker", None) is not None,
        }
        atomic_write_json(self.trace_root / f"{trace_id}.json", trace)
        return trace_id
