"""从文档血缘引用中筛选独立 Session 证据。"""

from __future__ import annotations


def independent_session_archives(references: tuple[str, ...]) -> tuple[str, ...]:
    """返回与文档正文独立存在的 SessionArchive URI。"""

    archives: set[str] = set()
    for reference in references:
        logical = str(reference).split("#manifest=", 1)[0]
        if logical.startswith("memoryos://user/") and "/sessions/history/" in logical:
            archives.add(logical)
    return tuple(sorted(archives))


__all__ = ["independent_session_archives"]
