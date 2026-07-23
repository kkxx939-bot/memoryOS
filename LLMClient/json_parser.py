"""分层提取和修复 JSON 语法，不臆造语义字段。"""

from __future__ import annotations

import ast
import importlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

JSONParseMode = Literal[
    "strict",
    "code_fence",
    "extracted",
    "trailing_comma_repair",
    "json_repair",
    "python_literal_repair",
]

_CODE_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


@dataclass(frozen=True)
class ParsedJSON:
    """解析后的值及可审计的语法修复说明。"""

    value: object
    mode: JSONParseMode

    @property
    def repaired(self) -> bool:
        return self.mode != "strict"


def parse_json_response(source: str, *, allow_repair: bool = True) -> ParsedJSON:
    """解析模型返回的 JSON，只修复语法并拒绝非 JSON 的 Python 值。"""

    if not isinstance(source, str) or not source.strip():
        raise ValueError("structured model response must be non-empty text")
    text = source.strip()
    parsed = _loads(text)
    if parsed is not _MISSING:
        return ParsedJSON(parsed, "strict")

    candidates: list[tuple[str, JSONParseMode]] = []
    fence = _CODE_FENCE.search(text)
    if fence:
        candidates.append((fence.group(1).strip(), "code_fence"))
    extracted = _extract_balanced_json(text)
    if extracted and all(extracted != candidate for candidate, _mode in candidates):
        candidates.append((extracted, "extracted"))
    for candidate, mode in candidates:
        parsed = _loads(candidate)
        if parsed is not _MISSING:
            return ParsedJSON(parsed, mode)

    if not allow_repair:
        raise ValueError("model response is not valid JSON")

    repair_sources = [candidate for candidate, _mode in candidates] or [text]
    for candidate in repair_sources:
        without_trailing_commas = _TRAILING_COMMA.sub(r"\1", candidate)
        if without_trailing_commas != candidate:
            parsed = _loads(without_trailing_commas)
            if parsed is not _MISSING:
                return ParsedJSON(parsed, "trailing_comma_repair")

    repaired = _repair_with_optional_dependency(text)
    if repaired is not _MISSING:
        return ParsedJSON(repaired, "json_repair")

    for candidate in repair_sources:
        try:
            value = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if _is_json_value(value):
            return ParsedJSON(value, "python_literal_repair")
    raise ValueError("model response could not be repaired as JSON")


class _Missing:
    pass


_MISSING = _Missing()


def _loads(source: str) -> object | _Missing:
    try:
        value = json.loads(source, parse_constant=_reject_non_finite)
    except (json.JSONDecodeError, ValueError):
        return _MISSING
    return value if _is_json_value(value) else _MISSING


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _repair_with_optional_dependency(source: str) -> object | _Missing:
    try:
        module = importlib.import_module("json_repair")
    except ImportError:
        return _MISSING
    try:
        value = module.loads(source)
    except (TypeError, ValueError, json.JSONDecodeError):
        return _MISSING
    return value if _is_json_value(value) else _MISSING


def _extract_balanced_json(source: str) -> str | None:
    start = next((index for index, character in enumerate(source) if character in "[{"), None)
    if start is None:
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(source)):
        character = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character in "[{":
            stack.append(character)
            continue
        if character in "]}":
            if not stack:
                return None
            opening = stack.pop()
            if (opening, character) not in {("[", "]"), ("{", "}")}:
                return None
            if not stack:
                return source[start : index + 1]
    return None


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, str | bool | int | float):
        return not isinstance(value, float) or math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


__all__ = ["JSONParseMode", "ParsedJSON", "parse_json_response"]
