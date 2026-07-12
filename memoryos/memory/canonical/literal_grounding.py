"""Language-neutral literal grounding for evidence-bound proposal fields."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


def normalize_literal(value: object) -> str:
    """Normalize representation without inventing aliases or translations."""

    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def literal_token_units(value: object) -> tuple[str, ...]:
    """Return deterministic script-aware units without doing word segmentation."""

    normalized = normalize_literal(value)
    units: list[str] = []
    buffered: list[str] = []
    bucket = ""

    def flush() -> None:
        nonlocal bucket
        if buffered:
            units.append("".join(buffered))
            buffered.clear()
        bucket = ""

    for character in normalized:
        if character.isspace():
            flush()
            continue
        if _is_cjk_character(character):
            flush()
            units.append(character)
            continue
        if character.isalnum() or character == "_":
            current = "ascii" if character.isascii() else _script_bucket(character)
            if buffered and current != bucket:
                flush()
            bucket = current
            buffered.append(character)
            continue
        flush()
        units.append(character)
    flush()
    return tuple(units)


def _script_bucket(character: str) -> str:
    name = unicodedata.name(character, "")
    return name.split(" ", 1)[0] if name else "unicode"


def _is_cjk_character(character: str) -> bool:
    name = unicodedata.name(character, "")
    return name.startswith(("CJK ", "HIRAGANA ", "KATAKANA ", "HANGUL "))


def _contains_cjk(value: str) -> bool:
    """Detect scripts whose word boundaries cannot be inferred without a segmenter."""

    return any(_is_cjk_character(character) for character in value)


def _scalar_supported(value: object, evidence_texts: Sequence[str]) -> bool:
    expected = normalize_literal(value)
    if not expected:
        return False
    normalized_texts = tuple(normalize_literal(text) for text in evidence_texts if str(text))
    if expected in normalized_texts:
        return True
    # CJK word segmentation is not deterministic here. New V3 proposals must
    # bind an exact child span rather than falling back to sentence substring.
    if _contains_cjk(expected):
        return False
    expected_units = literal_token_units(expected)
    if not expected_units:
        return False
    width = len(expected_units)
    for text in normalized_texts:
        units = literal_token_units(text)
        if any(units[index : index + width] == expected_units for index in range(len(units) - width + 1)):
            return True
    return False


def literal_value_supported(value: Any, evidence_texts: Sequence[str]) -> bool:
    """Require every structured scalar leaf to have one literal evidence match."""

    if value is None or value == "":
        return False
    if isinstance(value, Mapping):
        return bool(value) and all(literal_value_supported(item, evidence_texts) for item in value.values())
    if isinstance(value, list | tuple | set | frozenset):
        return bool(value) and all(literal_value_supported(item, evidence_texts) for item in value)
    return _scalar_supported(value, evidence_texts)


__all__ = ["literal_token_units", "literal_value_supported", "normalize_literal"]
