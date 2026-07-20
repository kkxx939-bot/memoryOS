"""存储查询实现共享的有界异常与确定性词法原语。"""

from __future__ import annotations

import re


class CatalogCandidateBoundExceeded(RuntimeError):
    """存储查询无法在硬性执行上限内完成。"""


def lexical_terms(text: object) -> tuple[str, ...]:
    """返回完整拉丁词元以及中日韩字符二元组。"""

    normalized = str(text).casefold()
    terms = re.findall(r"[a-z0-9_]+", normalized)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(sequence) == 1:
            terms.append(sequence)
        else:
            terms.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tuple(dict.fromkeys(term for term in terms if term))


def lexical_match_count(query: object, haystack: object) -> int:
    """计算查询词元在候选文本中的命中数量。"""

    query_terms = lexical_terms(query)
    if not query_terms:
        return 0
    haystack_terms = set(lexical_terms(haystack))
    return sum(1 for term in query_terms if term in haystack_terms)


def lexical_relevance(query: object, haystack: object) -> float:
    """返回用于存储候选粗排的词元命中比例。"""

    query_terms = lexical_terms(query)
    if not query_terms:
        return 0.0
    return lexical_match_count(query, haystack) / len(query_terms)


__all__ = [
    "CatalogCandidateBoundExceeded",
    "lexical_match_count",
    "lexical_relevance",
    "lexical_terms",
]
