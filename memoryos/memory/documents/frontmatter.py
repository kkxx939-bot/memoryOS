"""Strict, non-round-tripping front matter parsing for Markdown memory."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml

from memoryos.core.ids import new_id

_DOCUMENT_ID = re.compile(r"^memdoc_[A-Za-z0-9]{16,64}$")


class FrontMatterError(ValueError):
    """A Markdown header is missing, unsafe or violates the system contract."""


class MissingFrontMatter(FrontMatterError):
    """The file has no YAML front matter block."""


class MissingDocumentId(FrontMatterError):
    """The header is syntactically safe but is not managed by MemoryOS."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeySafeLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> Any:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise FrontMatterError("front matter mapping keys must be scalar") from exc
        if duplicate:
            raise FrontMatterError(f"duplicate front matter key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class ParsedFrontMatter:
    values: Mapping[str, Any]
    body: str
    header_bytes: bytes
    body_bytes: bytes

    @property
    def document_id(self) -> str:
        value = self.values.get("document_id")
        if not isinstance(value, str) or not _DOCUMENT_ID.fullmatch(value):
            raise MissingDocumentId("front matter document_id is missing or invalid")
        return value


def new_document_id() -> str:
    return new_id("memdoc")


def validate_document_id(value: object) -> str:
    document_id = str(value or "")
    if not _DOCUMENT_ID.fullmatch(document_id):
        raise FrontMatterError("document_id is invalid")
    return document_id


def parse_front_matter(
    raw: bytes,
    *,
    max_header_bytes: int,
    max_depth: int = 12,
    require_document_id: bool = True,
) -> ParsedFrontMatter:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise FrontMatterError("UTF-8 BOM is not allowed")
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise FrontMatterError("memory document must be strict UTF-8") from exc
    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].rstrip(b"\r\n") != b"---":
        raise MissingFrontMatter("memory document has no front matter")
    consumed = len(lines[0])
    closing_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        consumed += len(line)
        if consumed > max_header_bytes:
            raise FrontMatterError("front matter exceeds the configured byte limit")
        if line.rstrip(b"\r\n") == b"---":
            closing_index = index
            break
    if closing_index is None:
        raise FrontMatterError("front matter is not terminated")
    header_length = sum(len(line) for line in lines[: closing_index + 1])
    header_bytes = raw[:header_length]
    yaml_bytes = b"".join(lines[1:closing_index])
    try:
        yaml_text = yaml_bytes.decode("utf-8", errors="strict")
        for token in yaml.scan(yaml_text, Loader=_UniqueKeySafeLoader):
            if isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken, yaml.tokens.TagToken)):
                raise FrontMatterError("front matter anchors, aliases and explicit tags are forbidden")
        documents = list(yaml.load_all(yaml_text, Loader=_UniqueKeySafeLoader))
    except FrontMatterError:
        raise
    except yaml.YAMLError as exc:
        raise FrontMatterError("front matter is not safe YAML") from exc
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise FrontMatterError("front matter must contain exactly one mapping document")
    values = documents[0]
    if any(not isinstance(key, str) for key in values):
        raise FrontMatterError("front matter keys must be strings")
    _validate_value(values, depth=0, max_depth=max_depth)
    schema = values.get("memoryos_schema")
    if type(schema) is not int or schema != 1:
        raise FrontMatterError("memoryos_schema must equal 1")
    parsed = ParsedFrontMatter(
        values=values,
        body=raw[header_length:].decode("utf-8"),
        header_bytes=header_bytes,
        body_bytes=raw[header_length:],
    )
    if require_document_id:
        _ = parsed.document_id
    return parsed


def render_new_document(document_id: str, body: str = "") -> bytes:
    identifier = validate_document_id(document_id)
    normalized_body = str(body)
    if normalized_body and not normalized_body.startswith("\n"):
        normalized_body = f"\n{normalized_body}"
    return (
        f"---\nmemoryos_schema: 1\ndocument_id: {identifier}\n---\n{normalized_body}".encode()
    )


def adopt_raw_document(raw: bytes, document_id: str, *, max_header_bytes: int, max_depth: int = 12) -> bytes:
    """Add only system fields while preserving all existing body bytes."""

    identifier = validate_document_id(document_id)
    if raw.startswith(b"\xef\xbb\xbf"):
        raise FrontMatterError("UTF-8 BOM is not allowed")
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise FrontMatterError("memory document must be strict UTF-8") from exc
    try:
        parsed = parse_front_matter(
            raw,
            max_header_bytes=max_header_bytes,
            max_depth=max_depth,
            require_document_id=False,
        )
    except MissingFrontMatter:
        return render_new_document(identifier) + raw
    if "document_id" in parsed.values:
        raise FrontMatterError("existing document_id cannot be replaced during adopt")
    insertion = f"document_id: {identifier}\n".encode()
    first_line_end = len(raw.splitlines(keepends=True)[0])
    return raw[:first_line_end] + insertion + raw[first_line_end:]


def matches_adopted_source(raw: bytes, document_id: str, expected_raw_sha256: str) -> bool:
    """Verify that managed bytes are exactly one adoption of a known source digest.

    This reverses only the two byte-exact insertions performed by
    :func:`adopt_raw_document`; no source bytes need to be retained in the
    durable adoption receipt.
    """

    identifier = validate_document_id(document_id)
    if len(expected_raw_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_raw_sha256
    ):
        raise ValueError("expected_raw_sha256 must be a lowercase SHA-256 digest")
    unmanaged_prefix = render_new_document(identifier)
    if raw.startswith(unmanaged_prefix):
        original = raw[len(unmanaged_prefix) :]
        if hashlib.sha256(original).hexdigest() == expected_raw_sha256:
            return True
    first_line_end = len(raw.splitlines(keepends=True)[0]) if raw else 0
    insertion = f"document_id: {identifier}\n".encode()
    if first_line_end and raw[first_line_end:].startswith(insertion):
        original = raw[:first_line_end] + raw[first_line_end + len(insertion) :]
        if hashlib.sha256(original).hexdigest() == expected_raw_sha256:
            return True
    return False


def _validate_value(value: Any, *, depth: int, max_depth: int) -> None:
    if depth > max_depth:
        raise FrontMatterError("front matter nesting is too deep")
    if value is None or isinstance(value, str | bool | int | float):
        return
    if isinstance(value, list):
        for item in value:
            _validate_value(item, depth=depth + 1, max_depth=max_depth)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FrontMatterError("front matter mapping keys must be strings")
            _validate_value(item, depth=depth + 1, max_depth=max_depth)
        return
    raise FrontMatterError("front matter contains a non-portable YAML value")


__all__ = [
    "FrontMatterError",
    "MissingDocumentId",
    "MissingFrontMatter",
    "ParsedFrontMatter",
    "adopt_raw_document",
    "matches_adopted_source",
    "new_document_id",
    "parse_front_matter",
    "render_new_document",
    "validate_document_id",
]
