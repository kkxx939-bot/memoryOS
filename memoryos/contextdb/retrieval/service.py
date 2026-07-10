from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from memoryos.contextdb.retrieval.context_assembler import ContextAssembler
from memoryos.core.time import utc_now


class RetrievalService:
    def __init__(self, assembler: ContextAssembler, trace_root: str | Path) -> None:
        self.assembler = assembler
        self.trace_root = Path(trace_root)
        self.trace_root.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, **kwargs: Any) -> tuple[list[dict[str, Any]], str]:
        results = self.assembler.search(query, **kwargs)
        trace_id = self._record(query, kwargs, results, [])
        return results, trace_id

    def assemble(self, query: str, **kwargs: Any) -> dict[str, Any]:
        result = self.assembler.assemble(query, **kwargs)
        trace_id = self._record(query, kwargs, list(result.get("contexts", [])), list(result.get("dropped_contexts", [])))
        return {**result, "trace_id": trace_id}

    def read_trace(self, trace_id: str) -> dict[str, Any]:
        path = self.trace_root / f"{trace_id}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("recall trace is invalid")
        return value

    def _record(self, query: str, kwargs: dict[str, Any], selected: list[dict[str, Any]], dropped: list[dict[str, Any]]) -> str:
        trace_id = str(uuid.uuid4())
        trace = {
            "trace_id": trace_id,
            "created_at": utc_now(),
            "query": query[:1000],
            "scope": {key: kwargs.get(key) for key in ("user_id", "project_id", "adapter_id", "search_scope")},
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
        (self.trace_root / f"{trace_id}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        return trace_id
