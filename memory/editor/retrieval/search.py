"""组合向量召回、目录层级召回和可选重排的记忆搜索。"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from LLMClient import Embedder, EmbeddingVector, Reranker
from memory.editor.retrieval.index import MemoryVectorIndex, MemoryVectorMatch
from memory.editor.retrieval.model import MemorySearchHit
from memory.model import MemoryLevel
from memory.uri import MemoryURI, MemoryURINodeType


class MemorySearchMode(str, Enum):
    """向量主召回是否同时使用目录 L0/L1 扩展候选。"""

    VECTOR = "vector"
    HIERARCHICAL = "hierarchical"


@dataclass(frozen=True)
class MemorySemanticSearchConfig:
    """限制分阶段召回和重排的候选规模。"""

    mode: MemorySearchMode = MemorySearchMode.HIERARCHICAL
    candidate_multiplier: int = 4
    min_vector_candidates: int = 20
    directory_candidates: int = 20
    child_candidates: int = 32
    max_directory_expansions: int = 128
    max_rerank_candidates: int = 50
    max_rerank_document_chars: int = 12_000
    vector_score_threshold: float = -1.0
    rerank_score_threshold: float = 0.0
    score_propagation_alpha: float = 0.8
    rerank_hierarchy: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", MemorySearchMode(self.mode))
        for name, value, maximum in (
            ("candidate_multiplier", self.candidate_multiplier, 10),
            ("min_vector_candidates", self.min_vector_candidates, 10_000),
            ("directory_candidates", self.directory_candidates, 10_000),
            ("child_candidates", self.child_candidates, 10_000),
            ("max_directory_expansions", self.max_directory_expansions, 100_000),
            ("max_rerank_candidates", self.max_rerank_candidates, 10_000),
            ("max_rerank_document_chars", self.max_rerank_document_chars, 1_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be between one and {maximum}")
        if not _finite_between(self.vector_score_threshold, -1.0, 1.0):
            raise ValueError("vector_score_threshold must be between -1 and 1")
        if isinstance(self.rerank_score_threshold, bool) or not isinstance(
            self.rerank_score_threshold,
            int | float,
        ):
            raise TypeError("rerank_score_threshold must be numeric")
        if not math.isfinite(float(self.rerank_score_threshold)):
            raise ValueError("rerank_score_threshold must be finite")
        if not _finite_between(self.score_propagation_alpha, 0.0, 1.0):
            raise ValueError("score_propagation_alpha must be between zero and one")
        if not isinstance(self.rerank_hierarchy, bool):
            raise TypeError("rerank_hierarchy must be boolean")


@dataclass(frozen=True)
class _Candidate:
    uri: MemoryURI
    content: str
    vector_score: float
    score: float


class MemorySemanticSearchEngine:
    """保证 query embedding 只生成一次的完整记忆语义搜索入口。"""

    def __init__(
        self,
        *,
        embedder: Embedder,
        index: MemoryVectorIndex,
        reranker: Reranker | None = None,
        config: MemorySemanticSearchConfig | None = None,
    ) -> None:
        if not callable(getattr(embedder, "embed_query", None)):
            raise TypeError("embedder must implement embed_query")
        if not callable(getattr(index, "search", None)) or not callable(
            getattr(index, "search_children", None)
        ):
            raise TypeError("index must implement vector search and child search")
        if reranker is not None and not callable(getattr(reranker, "rerank", None)):
            raise TypeError("reranker must implement rerank")
        self.embedder = embedder
        self.index = index
        self.reranker = reranker
        self.config = config or MemorySemanticSearchConfig()

    async def search(
        self,
        query: str,
        *,
        roots: tuple[MemoryURI, ...],
        limit: int,
    ) -> tuple[MemorySearchHit, ...]:
        """完成向量主召回、可选层级补召回及独立重排。"""

        normalized_query = self._query(query)
        normalized_roots = self._roots(roots)
        maximum = self._limit(limit)

        query_vector = await self.embedder.embed_query(normalized_query)
        if not isinstance(query_vector, EmbeddingVector):
            raise TypeError("embedder must return an EmbeddingVector for a query")

        vector_limit = max(
            self.config.min_vector_candidates,
            maximum * self.config.candidate_multiplier,
        )
        direct_matches = await self.index.search(
            query_vector,
            roots=normalized_roots,
            levels=(MemoryLevel.DETAIL,),
            limit=vector_limit,
        )
        candidates: dict[MemoryURI, _Candidate] = {}
        self._merge_matches(candidates, self._matches(direct_matches), parent_score=None)

        if self.config.mode is MemorySearchMode.HIERARCHICAL:
            hierarchical = await self._hierarchical(
                normalized_query,
                query_vector,
                normalized_roots,
            )
            for candidate in hierarchical:
                self._merge(candidates, candidate)

        ranked = sorted(candidates.values(), key=lambda item: (-item.score, str(item.uri)))
        if self.reranker is None:
            return tuple(
                MemorySearchHit(
                    uri=candidate.uri,
                    score=candidate.score,
                    vector_score=candidate.vector_score,
                )
                for candidate in ranked
                if candidate.score >= self.config.vector_score_threshold
            )[:maximum]
        return await self._rerank_final(normalized_query, ranked, maximum)

    async def _hierarchical(
        self,
        query: str,
        query_vector: EmbeddingVector,
        roots: tuple[MemoryURI, ...],
    ) -> tuple[_Candidate, ...]:
        directory_matches = self._matches(
            await self.index.search(
                query_vector,
                roots=roots,
                levels=(MemoryLevel.ABSTRACT, MemoryLevel.OVERVIEW),
                limit=self.config.directory_candidates,
            )
        )
        queue: list[tuple[float, str, MemoryURI]] = []
        best_directory_score: dict[MemoryURI, float] = {}
        for root in roots:
            self._queue_directory(queue, best_directory_score, root, 0.0)
        for match in directory_matches:
            directory_uri = MemoryURI.from_directory(match.directory)
            self._queue_directory(queue, best_directory_score, directory_uri, match.score)

        visited: set[MemoryURI] = set()
        collected: dict[MemoryURI, _Candidate] = {}
        while queue and len(visited) < self.config.max_directory_expansions:
            negative_score, _, directory_uri = heapq.heappop(queue)
            if directory_uri in visited:
                continue
            visited.add(directory_uri)
            parent_score = -negative_score
            children = self._matches(
                await self.index.search_children(
                    query_vector,
                    parent=directory_uri,
                    limit=self.config.child_candidates,
                )
            )
            scored_children = await self._rerank_children(query, children)
            for match, local_score in scored_children:
                final_score = self._propagated(local_score, parent_score)
                if match.level is MemoryLevel.DETAIL:
                    self._merge(
                        collected,
                        _Candidate(
                            uri=match.uri,
                            content=match.content,
                            vector_score=match.score,
                            score=final_score,
                        ),
                    )
                    continue
                directory = MemoryURI.from_directory(match.directory)
                self._queue_directory(queue, best_directory_score, directory, final_score)
        return tuple(sorted(collected.values(), key=lambda item: (-item.score, str(item.uri))))

    async def _rerank_children(
        self,
        query: str,
        matches: tuple[MemoryVectorMatch, ...],
    ) -> tuple[tuple[MemoryVectorMatch, float], ...]:
        if not matches:
            return ()
        if self.reranker is None or not self.config.rerank_hierarchy:
            return tuple((match, match.score) for match in matches)
        scores = await self.reranker.rerank(
            query,
            tuple(self._rerank_text(match.content) for match in matches),
        )
        if not isinstance(scores, tuple) or len(scores) != len(matches):
            raise ValueError("reranker returned an unexpected hierarchy score count")
        return tuple((match, self._score(score, "hierarchy rerank")) for match, score in zip(matches, scores, strict=True))

    async def _rerank_final(
        self,
        query: str,
        candidates: list[_Candidate],
        limit: int,
    ) -> tuple[MemorySearchHit, ...]:
        selected = candidates[: self.config.max_rerank_candidates]
        if not selected:
            return ()
        assert self.reranker is not None
        scores = await self.reranker.rerank(
            query,
            tuple(self._rerank_text(candidate.content) for candidate in selected),
        )
        if not isinstance(scores, tuple) or len(scores) != len(selected):
            raise ValueError("reranker returned an unexpected final score count")
        hits: list[MemorySearchHit] = []
        for candidate, raw_score in zip(selected, scores, strict=True):
            score = self._score(raw_score, "final rerank")
            if score < self.config.rerank_score_threshold:
                continue
            hits.append(
                MemorySearchHit(
                    uri=candidate.uri,
                    score=score,
                    vector_score=candidate.vector_score,
                    rerank_score=score,
                )
            )
        hits.sort(key=lambda item: (-item.score, str(item.uri)))
        return tuple(hits[:limit])

    def _merge_matches(
        self,
        candidates: dict[MemoryURI, _Candidate],
        matches: tuple[MemoryVectorMatch, ...],
        *,
        parent_score: float | None,
    ) -> None:
        for match in matches:
            if match.level is not MemoryLevel.DETAIL:
                continue
            score = match.score if parent_score is None else self._propagated(match.score, parent_score)
            self._merge(
                candidates,
                _Candidate(match.uri, match.content, match.score, score),
            )

    @staticmethod
    def _merge(candidates: dict[MemoryURI, _Candidate], candidate: _Candidate) -> None:
        previous = candidates.get(candidate.uri)
        if previous is None or candidate.score > previous.score:
            candidates[candidate.uri] = candidate

    @staticmethod
    def _queue_directory(
        queue: list[tuple[float, str, MemoryURI]],
        scores: dict[MemoryURI, float],
        uri: MemoryURI,
        score: float,
    ) -> None:
        current = scores.get(uri)
        if current is not None and current >= score:
            return
        scores[uri] = score
        heapq.heappush(queue, (-score, str(uri), uri))

    def _propagated(self, local_score: float, parent_score: float) -> float:
        if parent_score == 0:
            return local_score
        alpha = self.config.score_propagation_alpha
        return alpha * local_score + (1.0 - alpha) * parent_score

    def _rerank_text(self, text: str) -> str:
        maximum = self.config.max_rerank_document_chars
        if len(text) <= maximum:
            return text
        if maximum <= 3:
            return text[:maximum]
        return text[: maximum - 3].rstrip() + "..."

    @staticmethod
    def _matches(value: Sequence[MemoryVectorMatch]) -> tuple[MemoryVectorMatch, ...]:
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise TypeError("memory vector index must return a sequence")
        matches = tuple(value)
        if any(not isinstance(match, MemoryVectorMatch) for match in matches):
            raise TypeError("memory vector index returned an invalid match")
        return matches

    @staticmethod
    def _query(query: object) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("memory semantic query must be non-empty text")
        return query.strip()

    @staticmethod
    def _roots(roots: tuple[MemoryURI, ...]) -> tuple[MemoryURI, ...]:
        if not isinstance(roots, tuple) or not roots:
            raise ValueError("memory semantic search requires at least one root")
        normalized = tuple(MemoryURI.parse(root) for root in roots)
        if any(root.node_type is not MemoryURINodeType.DIRECTORY for root in normalized):
            raise ValueError("memory semantic search roots must identify directories")
        if len(normalized) != len(set(normalized)):
            raise ValueError("memory semantic search roots must be unique")
        return normalized

    @staticmethod
    def _limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1_000:
            raise ValueError("memory semantic search limit must be between one and 1000")
        return limit

    @staticmethod
    def _score(value: object, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{label} score must be numeric")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"{label} score must be finite")
        return score


def _finite_between(value: object, minimum: float, maximum: float) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and minimum <= float(value) <= maximum
    )


__all__ = [
    "MemorySearchMode",
    "MemorySemanticSearchConfig",
    "MemorySemanticSearchEngine",
]
