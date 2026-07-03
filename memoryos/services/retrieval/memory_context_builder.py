from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from memoryos.domain.memory.memory_item import TYPE_DIR
from memoryos.ports.repositories.memory_repository import MemoryRepository


@dataclass
class MemoryRoute:
    memory_type: str
    strategy: str
    reason: str
    selected_count: int = 0
    directory_abstract: str = ""
    target_uri: str = ""
    score: float = 0.0
    match_reason: str = ""
    level: str = "memory_type"

    def to_dict(self) -> dict:
        return {
            "memory_type": self.memory_type,
            "strategy": self.strategy,
            "reason": self.reason,
            "selected_count": self.selected_count,
            "directory_abstract": self.directory_abstract,
            "target_uri": self.target_uri,
            "score": self.score,
            "match_reason": self.match_reason,
            "level": self.level,
        }


@dataclass
class MemoryContext:
    stable_context: list[dict] = field(default_factory=list)
    recent_context: list[dict] = field(default_factory=list)
    relevant_memories: list[dict] = field(default_factory=list)
    route_trace: list[MemoryRoute] = field(default_factory=list)
    digest: str = ""

    def memories_for_prediction(self) -> list[dict]:
        return _dedupe_memories([*self.stable_context, *self.recent_context, *self.relevant_memories])

    def source_summary(self) -> dict[str, dict]:
        summary: dict[str, dict] = {}
        for memory in self.memories_for_prediction():
            memory_type = str(memory.get("type") or "unknown")
            item = summary.setdefault(memory_type, {"count": 0, "paths": []})
            item["count"] += 1
            item["paths"].append(memory.get("path"))
        return summary

    def to_dict(self) -> dict:
        return {
            "stable_context": [_summary(memory) for memory in self.stable_context],
            "recent_context": [_summary(memory) for memory in self.recent_context],
            "relevant_memories": [_summary(memory) for memory in self.relevant_memories],
            "route_trace": [route.to_dict() for route in self.route_trace],
            "source_summary": self.source_summary(),
            "digest": self.digest,
        }


class MemoryContextBuilder:
    STABLE_ROUTES = {
        "profile": "stable user model; always useful for personalization",
        "policy": "permissions and safety boundaries; always needed before action",
        "preference": "explicit user choices; useful even when query terms differ",
    }
    RECENT_ROUTES = {
        "feedback": "recent user feedback can change ranking and intervention style",
        "intervention": "recent agent actions prevent repeated or annoying behavior",
        "event": "recent auditable facts may explain current state",
    }
    RELEVANT_ROUTES = {
        "habit": "behavior patterns related to the current observation",
        "trigger": "scene signals that usually precede a need or behavior",
        "case": "similar reusable episodes",
        "event": "auditable facts directly related to the query",
    }

    def __init__(self, store: MemoryRepository) -> None:
        self.store = store

    def build(
        self,
        user_id: str,
        query: str,
        stable_limit_per_type: int = 3,
        recent_limit_per_type: int = 2,
        relevant_limit_per_type: int = 3,
        route_node_limit: int = 4,
        route_score_threshold: float = 0.12,
        digest_limit: int = 12,
        digest_char_budget: int = 3600,
    ) -> MemoryContext:
        route_trace: list[MemoryRoute] = []
        stable_context = self._stable_context(user_id, stable_limit_per_type, route_trace)
        recent_context = self._recent_context(user_id, recent_limit_per_type, route_trace)
        relevant_memories = self._relevant_memories(
            user_id,
            query,
            relevant_limit_per_type,
            route_node_limit,
            route_score_threshold,
            route_trace,
        )
        digest = self._digest(stable_context, recent_context, relevant_memories, route_trace, digest_limit, digest_char_budget)
        return MemoryContext(
            stable_context=stable_context,
            recent_context=recent_context,
            relevant_memories=relevant_memories,
            route_trace=route_trace,
            digest=digest,
        )

    def _stable_context(self, user_id: str, limit_per_type: int, route_trace: list[MemoryRoute]) -> list[dict]:
        memories = []
        for memory_type, reason in self.STABLE_ROUTES.items():
            selected = self.store.list_by_type(user_id, memory_type, limit=limit_per_type)
            memories.extend(selected)
            route_trace.append(
                self._route(
                    user_id=user_id,
                    memory_type=memory_type,
                    strategy="fixed_stable_context",
                    reason=reason,
                    selected_count=len(selected),
                )
            )
        return _dedupe_memories(memories)

    def _recent_context(self, user_id: str, limit_per_type: int, route_trace: list[MemoryRoute]) -> list[dict]:
        memories = []
        for memory_type, reason in self.RECENT_ROUTES.items():
            selected = self.store.list_by_type(user_id, memory_type, limit=limit_per_type)
            memories.extend(selected)
            route_trace.append(
                self._route(
                    user_id=user_id,
                    memory_type=memory_type,
                    strategy="recent_state_context",
                    reason=reason,
                    selected_count=len(selected),
                )
            )
        return _dedupe_memories(memories)

    def _relevant_memories(
        self,
        user_id: str,
        query: str,
        limit_per_type: int,
        route_node_limit: int,
        route_score_threshold: float,
        route_trace: list[MemoryRoute],
    ) -> list[dict]:
        memories = []
        directory_routes = self._rank_directory_routes(
            user_id=user_id,
            query=query,
            route_node_limit=route_node_limit,
            route_score_threshold=route_score_threshold,
        )
        for route in directory_routes:
            memory_type = route.memory_type
            selected = self.store.hybrid_search(
                query=query,
                user_id=user_id,
                memory_type=memory_type,
                limit=limit_per_type,
            )
            memories.extend(selected)
            route.selected_count = len(selected)
            route_trace.append(route)
        return _dedupe_memories(memories)

    def _rank_directory_routes(
        self,
        user_id: str,
        query: str,
        route_node_limit: int,
        route_score_threshold: float,
    ) -> list[MemoryRoute]:
        routes = []
        directory_hits = self.store.rank_directory_layers(
            query=query,
            user_id=user_id,
            memory_types=set(self.RELEVANT_ROUTES),
            limit=route_node_limit * 2,
        )
        for hit in directory_hits:
            memory_type = str(hit["type"])
            reason = self.RELEVANT_ROUTES.get(memory_type, "directory route selected by L0/L1 index")
            score = float(hit.get("score", 0.0))
            if score < route_score_threshold:
                continue
            keyword_score = float(hit.get("keyword_score", 0.0))
            semantic_score = float(hit.get("semantic_score", 0.0))
            level = str(hit.get("level") or "L1")
            if keyword_score > 0:
                match_reason = f"query matched {level} directory node"
            elif semantic_score > 0:
                match_reason = f"embedding matched {level} directory node"
            else:
                match_reason = f"{level} directory route fallback"
            routes.append(
                self._route(
                    user_id=user_id,
                    memory_type=memory_type,
                    strategy="directory_first_relevant_memory",
                    reason=reason,
                    selected_count=0,
                    score=score,
                    match_reason=(
                        f"{match_reason}; keyword={keyword_score:.3f}; semantic={semantic_score:.3f}"
                    ),
                    level=level,
                    target_uri=f"memory://{hit.get('directory_path')}",
                    directory_abstract=str(hit.get("abstract", "")),
                )
            )
        routes.sort(key=lambda route: route.score, reverse=True)
        return routes[:route_node_limit]

    def _digest(
        self,
        stable_context: list[dict],
        recent_context: list[dict],
        relevant_memories: list[dict],
        route_trace: list[MemoryRoute],
        limit: int,
        char_budget: int,
    ) -> str:
        lines = ['<personal-memory source="memoryos" format="context-digest">']
        self._append_route_summary(lines, route_trace)
        self._append_section(lines, "Stable context", stable_context, limit)
        self._append_section(lines, "Recent context", recent_context, limit)
        self._append_section(lines, "Relevant memories", relevant_memories, limit)
        if len(lines) == 1:
            lines.append("No personal memory context found.")
        lines.append("</personal-memory>")
        return _clip_digest("\n".join(lines), char_budget)

    def _append_route_summary(self, lines: list[str], route_trace: list[MemoryRoute]) -> None:
        selected = [route for route in route_trace if route.selected_count > 0]
        if not selected:
            return
        lines.append("Memory route trace:")
        for route in selected:
            lines.append(f"- {route.memory_type}: {route.strategy}, selected={route.selected_count}")

    def _append_section(self, lines: list[str], title: str, memories: list[dict], limit: int) -> None:
        if not memories:
            return
        lines.append(f"{title}:")
        for memory in memories[:limit]:
            abstract = memory.get("abstract") or memory.get("content", "")[:160]
            lines.append(f"- [{memory.get('type')}] {memory.get('title')}: {abstract} ({memory.get('path')})")

    def _route(
        self,
        user_id: str,
        memory_type: str,
        strategy: str,
        reason: str,
        selected_count: int,
        score: float = 0.0,
        match_reason: str = "",
        level: str = "memory_type",
        target_uri: str = "",
        directory_abstract: str = "",
    ) -> MemoryRoute:
        directory_name = TYPE_DIR.get(memory_type, memory_type)
        return MemoryRoute(
            memory_type=memory_type,
            strategy=strategy,
            reason=reason,
            selected_count=selected_count,
            directory_abstract=directory_abstract or self._directory_abstract(user_id, memory_type),
            target_uri=target_uri or f"memory://user/{user_id}/{directory_name}",
            score=round(max(0.0, min(1.0, score)), 6),
            match_reason=match_reason,
            level=level,
        )

    def _directory_abstract(self, user_id: str, memory_type: str) -> str:
        directory_name = TYPE_DIR.get(memory_type)
        if not directory_name:
            return ""
        directory = self.store.root / "user" / user_id / directory_name
        return _read_layer_text(directory / ".abstract.md") or _read_layer_text(directory / ".overview.md")


def _dedupe_memories(memories: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for memory in memories:
        key = memory.get("id") or memory.get("path")
        if not key or key in seen:
            continue
        deduped.append(memory)
        seen.add(key)
    return deduped


def _summary(memory: dict) -> dict:
    return {
        "id": memory.get("id"),
        "path": memory.get("path"),
        "type": memory.get("type"),
        "title": memory.get("title"),
        "score": memory.get("score"),
        "effective_weight": memory.get("effective_weight"),
        "hotness": memory.get("hotness"),
    }


def _read_layer_text(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    return text[:320]


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    tokens = set()
    current = []
    for ch in lowered:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.add("".join(current))
                current = []
            if "\u4e00" <= ch <= "\u9fff":
                tokens.add(ch)
    if current:
        tokens.add("".join(current))
    return {token for token in tokens if token}


def _overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return min(1.0, overlap / max(3, len(left)))


def _clip_digest(digest: str, char_budget: int) -> str:
    if len(digest) <= char_budget:
        return digest
    closing = "\n</personal-memory>"
    clipped = digest[: max(0, char_budget - len(closing) - 32)].rstrip()
    return clipped + "\n... clipped ..." + closing
