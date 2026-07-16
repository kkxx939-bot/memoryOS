from __future__ import annotations

import pytest

from memoryos.contextdb.catalog import (
    CatalogRecord,
    CatalogRecordKind,
    ServingTier,
    normalize_tree_path,
    validate_tree_paths,
)
from memoryos.security.context_projection import ContextProjectionSanitizer


def test_projection_sanitizer_allows_logical_uri_but_removes_credentials_and_absolute_path() -> None:
    safe = ContextProjectionSanitizer().sanitize(
        title="tool result",
        l1_text=(
            "Authorization: Bearer abcdefghijklmnop\n"
            "Authorization: Token opaque-authorization-value\n"
            "Proxy-Authorization: Digest proxy-credential\n"
            "Cookie: session=raw-cookie\n"
            "password=hunter2\n"
            "read /Users/gulf/Desktop/report.csv"
        ),
        metadata={
            "source_uri": "memoryos://user/u1/sessions/history/s1",
            "resource_uri": "file:///Users/gulf/Desktop/report.csv",
            "cookie": "session=secret",
            "environment": {"HOME": "/Users/gulf", "TOKEN": "secret"},
        },
        source_kind="tool_result",
    )

    encoded = repr(safe)
    assert "memoryos://user/u1/sessions/history/s1" in encoded
    assert "hunter2" not in encoded
    assert "abcdefghijklmnop" not in encoded
    assert "opaque-authorization-value" not in encoded
    assert "proxy-credential" not in encoded
    assert "raw-cookie" not in encoded
    assert "session=secret" not in encoded
    assert "/Users/gulf" not in encoded
    assert safe.resource_name == "report.csv"
    assert safe.resource_location == "desktop"


def test_catalog_record_validates_taxonomy_enum_and_finite_hotness() -> None:
    record = CatalogRecord(
        record_key="slot:s1:current",
        uri="memoryos://user/u1/memories/canonical/slots/s1",
        tenant_id="t1",
        record_kind=CatalogRecordKind.CURRENT_SLOT,
        serving_tier=ServingTier.WARM,
        tree_paths=("memories/preferences/food/ice_cream_flavors", "timeline/2026/07/14"),
        event_time="2026-07-14T00:00:00+08:00",
    )

    assert record.record_kind == "current_slot"
    assert record.serving_tier == "WARM"
    assert record.event_time == "2026-07-13T16:00:00+00:00"
    assert record.path_depth == 4

    with pytest.raises(ValueError, match="controlled taxonomy"):
        validate_tree_paths(("invented/unbounded/path",))
    with pytest.raises(ValueError, match="resource path kind"):
        validate_tree_paths(("resources/llm-invented-folder",))
    with pytest.raises(ValueError, match="memory path kind"):
        validate_tree_paths(("memories/random/new/folder",))
    with pytest.raises(ValueError, match="calendar date"):
        validate_tree_paths(("timeline/2026/02/30",))
    with pytest.raises(ValueError, match="one controlled identifier"):
        validate_tree_paths(("agents/codex/private",))
    with pytest.raises(ValueError, match="too many secondary"):
        validate_tree_paths(tuple(f"sessions/s{i}" for i in range(9)))


def test_catalog_tree_paths_pseudonymize_sensitive_dynamic_segments_and_metadata_mirrors() -> None:
    raw_primary = "memories/preferences/Users_gulf_Desktop_private.txt/ice_cream"
    raw_secondary = "projects/sk-abcdefghijk123456"
    record = CatalogRecord(
        record_key="slot:sensitive:current",
        uri="memoryos://user/u1/memories/canonical/slots/sensitive",
        tenant_id="t1",
        record_kind=CatalogRecordKind.CURRENT_SLOT,
        primary_tree_path=raw_primary,
        tree_paths=(raw_primary, raw_secondary, "timeline/2026/07/14"),
        metadata={
            "primary_tree_path": raw_primary,
            "tree_paths": [raw_primary, raw_secondary],
            "nested": {"secondary_tree_paths": [raw_secondary]},
        },
    )

    safe = record.with_sanitized_projection()
    encoded = repr(safe.to_dict())

    assert safe.primary_tree_path.startswith("memories/preferences/id-")
    assert safe.tree_paths[1].startswith("projects/id-")
    assert safe.tree_paths[2] == "timeline/2026/07/14"
    assert safe.metadata["primary_tree_path"] == safe.primary_tree_path
    assert safe.metadata["tree_paths"] == list(safe.tree_paths)
    assert "sk-abcdefghijk123456" not in encoded
    assert "Users_gulf" not in encoded
    assert "gulf_Desktop" not in encoded
    assert normalize_tree_path(raw_primary) == safe.primary_tree_path
    assert normalize_tree_path(safe.primary_tree_path) == safe.primary_tree_path
    assert normalize_tree_path("projects/project-a") == "projects/project-a"


def test_catalog_tree_path_sanitization_is_fail_closed_for_invalid_dynamic_segments() -> None:
    with pytest.raises(ValueError, match="unsafe segment"):
        normalize_tree_path("projects/bad\x00segment")
