"""Deterministic lexical tokenization shared by ContextDB index implementations."""

from __future__ import annotations

import re


def lexical_terms(text: object) -> tuple[str, ...]:
    """Return deterministic complete Latin tokens and CJK character bigrams."""

    normalized = str(text).casefold()
    terms = re.findall(r"[a-z0-9_]+", normalized)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(sequence) == 1:
            terms.append(sequence)
        else:
            terms.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tuple(dict.fromkeys(term for term in terms if term))


def lexical_match_count(query: object, haystack: object) -> int:
    query_terms = lexical_terms(query)
    if not query_terms:
        return 0
    haystack_terms = set(lexical_terms(haystack))
    return sum(1 for term in query_terms if term in haystack_terms)


def lexical_relevance(query: object, haystack: object) -> float:
    query_terms = lexical_terms(query)
    if not query_terms:
        return 0.0
    return lexical_match_count(query, haystack) / len(query_terms)


__all__ = ["lexical_match_count", "lexical_relevance", "lexical_terms"]
