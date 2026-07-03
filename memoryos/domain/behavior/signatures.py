from __future__ import annotations

import hashlib

MATCH_LEVEL_WEIGHTS = {
    "exact": 1.0,
    "semantic": 0.7,
    "coarse": 0.45,
}


def context_tokens(retrieval_query: str, context_tags: list[str]) -> list[str]:
    tokens = [_normalize_token(tag) for tag in context_tags if _normalize_token(tag)]
    if not tokens:
        tokens = text_tokens(retrieval_query)
    return tokens


def combined_context_tokens(retrieval_query: str, context_tags: list[str]) -> list[str]:
    tokens = [_normalize_token(tag) for tag in context_tags if _normalize_token(tag)]
    tokens.extend(text_tokens(retrieval_query))
    return list(dict.fromkeys(tokens))


def scene_signatures(retrieval_query: str, context_tags: list[str]) -> dict[str, str]:
    tokens = context_tokens(retrieval_query, context_tags)
    return {
        "exact": hash_tokens(tokens),
        "semantic": hash_tokens([token for token in tokens if not is_exact_only_token(token)]),
        "coarse": hash_tokens([token for token in tokens if is_coarse_token(token)]),
    }


def layered_token_sets(retrieval_query: str, context_tags: list[str]) -> dict[str, set[str]]:
    tokens = context_tokens(retrieval_query, context_tags)
    return {
        "exact": set(tokens),
        "semantic": {token for token in tokens if not is_exact_only_token(token)},
        "coarse": {token for token in tokens if is_coarse_token(token)},
    }


def pattern_scene_signatures(retrieval_query: str, context_tags: list[str]) -> dict[str, str]:
    tokens = combined_context_tokens(retrieval_query, context_tags)
    return {
        "exact": hash_tokens(tokens),
        "semantic": hash_tokens([token for token in tokens if not is_exact_only_token(token)]),
        "coarse": hash_tokens([token for token in tokens if is_coarse_token(token)]),
    }


def pattern_layered_token_sets(retrieval_query: str, context_tags: list[str]) -> dict[str, set[str]]:
    tokens = combined_context_tokens(retrieval_query, context_tags)
    return {
        "exact": set(tokens),
        "semantic": {token for token in tokens if not is_exact_only_token(token)},
        "coarse": {token for token in tokens if is_coarse_token(token)},
    }


def hash_tokens(tokens: list[str]) -> str:
    key = "|".join(sorted(set(tokens))) or "global"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def is_exact_only_token(token: str) -> bool:
    if token.startswith("temperature_") or token.startswith("humidity_"):
        return True
    if token.startswith("duration_") and token.endswith("_minutes"):
        return True
    return False


def is_coarse_token(token: str) -> bool:
    if is_exact_only_token(token):
        return False
    if token.startswith("duration_"):
        return False
    if token in {"hot_environment", "cold_environment", "humid_environment"}:
        return False
    return True


def text_tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []
    current: list[str] = []
    for ch in lowered:
        if "\u4e00" <= ch <= "\u9fff":
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(ch)
            continue
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _normalize_token(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")
