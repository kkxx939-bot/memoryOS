from __future__ import annotations

import pytest

from memoryos.memory.documents.frontmatter import (
    FrontMatterError,
    MissingDocumentId,
    MissingFrontMatter,
    adopt_raw_document,
    parse_front_matter,
    render_new_document,
)
from memoryos.memory.documents.model import (
    ABSENT,
    ManagedDocument,
    PresentPath,
    QuarantinedDocument,
    RegistrationStatus,
    UnmanagedDocument,
    UnsafePath,
    raw_state_from_dict,
    raw_state_to_dict,
)

DOCUMENT_ID = "memdoc_0123456789ABCDEF"


def test_raw_path_states_round_trip_without_collapsing_empty_present() -> None:
    present = PresentPath("profile.md", "0" * 64, 0)
    unsafe = UnsafePath("profile.md", "permission denied")

    assert raw_state_from_dict(raw_state_to_dict(ABSENT)) == ABSENT
    assert raw_state_from_dict(raw_state_to_dict(present)) == present
    assert raw_state_from_dict(raw_state_to_dict(unsafe)) == unsafe
    assert present != ABSENT


def test_raw_path_state_validation_rejects_incomplete_values() -> None:
    with pytest.raises(ValueError, match="invalid PRESENT"):
        PresentPath("profile.md", "short", 1)
    with pytest.raises(ValueError, match="invalid PRESENT"):
        PresentPath("", "0" * 64, 1)
    with pytest.raises(ValueError, match="invalid UNSAFE"):
        UnsafePath("profile.md", "")
    with pytest.raises(ValueError, match="unknown raw path state"):
        raw_state_from_dict({"state": "DELETED"})


def test_registration_states_remain_distinct() -> None:
    managed = ManagedDocument("profile.md", DOCUMENT_ID, "1" * 64, 12)
    unmanaged = UnmanagedDocument("profile.md", "2" * 64, 0, "missing document_id")
    quarantined = QuarantinedDocument("profile.md", "invalid UTF-8", "3" * 64, 4)

    assert managed.status is RegistrationStatus.MANAGED
    assert unmanaged.status is RegistrationStatus.UNMANAGED
    assert quarantined.status is RegistrationStatus.QUARANTINED


def test_front_matter_parser_preserves_exact_header_and_body_bytes() -> None:
    raw = (
        b"---\r\n"
        b"memoryos_schema: 1\r\n"
        b"document_id: memdoc_0123456789ABCDEF\r\n"
        b"note: keep formatting\r\n"
        b"---\r\n"
        b"# Heading\r\n\r\nBody\r\n"
    )

    parsed = parse_front_matter(raw, max_header_bytes=1024)

    assert parsed.document_id == DOCUMENT_ID
    assert parsed.header_bytes + parsed.body_bytes == raw
    assert parsed.body_bytes == b"# Heading\r\n\r\nBody\r\n"
    assert parsed.body == "# Heading\r\n\r\nBody\r\n"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"# no header\n", "no front matter"),
        (b"\xef\xbb\xbf---\nmemoryos_schema: 1\n---\n", "BOM"),
        (b"---\nmemoryos_schema: 1\n---\n\xff", "strict UTF-8"),
        (b"---\nmemoryos_schema: 1\n", "not terminated"),
        (
            b"---\nmemoryos_schema: 1\nmemoryos_schema: 1\ndocument_id: memdoc_0123456789ABCDEF\n---\n",
            "duplicate front matter key",
        ),
        (
            b"---\nmemoryos_schema: 1\ndocument_id: memdoc_0123456789ABCDEF\nvalue: !unsafe x\n---\n",
            "anchors, aliases and explicit tags",
        ),
        (
            b"---\nmemoryos_schema: 1\ndocument_id: memdoc_0123456789ABCDEF\nvalue: &x [1]\ncopy: *x\n---\n",
            "anchors, aliases and explicit tags",
        ),
        (
            b"---\nmemoryos_schema: 1\ndocument_id: memdoc_0123456789ABCDEF\n"
            b"--- # second YAML document\nmemoryos_schema: 1\n---\n",
            "exactly one mapping document",
        ),
        (b"---\nmemoryos_schema: true\ndocument_id: memdoc_0123456789ABCDEF\n---\n", "must equal 1"),
    ],
)
def test_front_matter_parser_rejects_unsafe_yaml_and_encoding(raw: bytes, message: str) -> None:
    exception = MissingFrontMatter if message == "no front matter" else FrontMatterError
    with pytest.raises(exception, match=message):
        parse_front_matter(raw, max_header_bytes=1024)


def test_front_matter_parser_rejects_oversized_and_overdeep_headers() -> None:
    oversized = (
        b"---\nmemoryos_schema: 1\ndocument_id: memdoc_0123456789ABCDEF\nnote: "
        + (b"x" * 200)
        + b"\n---\n"
    )
    overdeep = (
        b"---\nmemoryos_schema: 1\ndocument_id: memdoc_0123456789ABCDEF\nvalue: [[[[[1]]]]]\n---\n"
    )

    with pytest.raises(FrontMatterError, match="byte limit"):
        parse_front_matter(oversized, max_header_bytes=64)
    with pytest.raises(FrontMatterError, match="nesting is too deep"):
        parse_front_matter(overdeep, max_header_bytes=1024, max_depth=2)


def test_document_id_is_required_and_must_be_valid() -> None:
    missing = b"---\nmemoryos_schema: 1\n---\nBody\n"
    invalid = b"---\nmemoryos_schema: 1\ndocument_id: ../escape\n---\nBody\n"

    with pytest.raises(MissingDocumentId):
        parse_front_matter(missing, max_header_bytes=1024)
    with pytest.raises(MissingDocumentId):
        parse_front_matter(invalid, max_header_bytes=1024)


def test_rendered_document_has_only_minimal_system_fields() -> None:
    rendered = render_new_document(DOCUMENT_ID, "# Profile\n")
    parsed = parse_front_matter(rendered, max_header_bytes=1024)

    assert dict(parsed.values) == {"memoryos_schema": 1, "document_id": DOCUMENT_ID}
    assert parsed.body.endswith("# Profile\n")


def test_adopt_without_front_matter_preserves_all_original_bytes() -> None:
    original = b"# User note\r\n\r\nKeep this exact.\r\n"

    adopted = adopt_raw_document(original, DOCUMENT_ID, max_header_bytes=1024)

    parsed = parse_front_matter(adopted, max_header_bytes=1024)
    assert parsed.document_id == DOCUMENT_ID
    assert adopted.endswith(original)


def test_adopt_existing_safe_front_matter_inserts_id_without_round_trip() -> None:
    original = b"---\r\nmemoryos_schema: 1\r\nnote: keep\r\n---\r\n# Body\r\n"

    adopted = adopt_raw_document(original, DOCUMENT_ID, max_header_bytes=1024)

    parsed = parse_front_matter(adopted, max_header_bytes=1024)
    assert parsed.document_id == DOCUMENT_ID
    assert b"memoryos_schema: 1\r\nnote: keep\r\n---\r\n# Body\r\n" in adopted


def test_adopt_refuses_existing_id_or_unsafe_bytes() -> None:
    managed = render_new_document(DOCUMENT_ID, "body")

    with pytest.raises(FrontMatterError, match="cannot be replaced"):
        adopt_raw_document(managed, "memdoc_FEDCBA9876543210", max_header_bytes=1024)
    with pytest.raises(FrontMatterError, match="BOM"):
        adopt_raw_document(b"\xef\xbb\xbfbody", DOCUMENT_ID, max_header_bytes=1024)

