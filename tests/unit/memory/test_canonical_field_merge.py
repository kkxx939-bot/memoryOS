from __future__ import annotations

import pytest

from memoryos.memory.canonical.field_merge import FieldMergeError, FieldMerger
from memoryos.memory.schema import FieldMergeMode, MemoryType, MemoryTypeSchema


def _schema() -> MemoryTypeSchema:
    return MemoryTypeSchema(
        memory_type=MemoryType.ENTITY,
        description="field merge test",
        required_fields=(),
        field_merge_rules={
            "identity": FieldMergeMode.IMMUTABLE,
            "status": FieldMergeMode.REPLACE,
            "aliases": FieldMergeMode.APPEND_UNIQUE,
            "summary": FieldMergeMode.PATCH_TEXT,
            "untouched": FieldMergeMode.REPLACE,
        },
    )


def _ref(name: str) -> dict[str, str]:
    return {"event_id": name, "content_hash": f"hash-{name}"}


def test_partial_merge_materializes_complete_state_and_preserves_provenance() -> None:
    merger = FieldMerger()
    current = {
        "identity": "database",
        "status": "candidate",
        "aliases": ["db"],
        "summary": "Existing summary",
        "untouched": "keep me",
    }
    old_status = (_ref("old-status"),)
    old_untouched = (_ref("old-untouched"),)
    result = merger.merge(
        _schema(),
        current,
        {
            "value.status": old_status,
            "value.untouched": old_untouched,
        },
        {"status": "adopted"},
        {"value.status": (_ref("new-status"),), "transition": (_ref("transition"),)},
        relation="SUPPLEMENTS",
        review_authority=False,
    )

    assert result.value_fields == {**current, "status": "adopted"}
    assert result.changed_fields == ("status",)
    assert set(result.unchanged_fields) == {"aliases", "identity", "summary", "untouched"}
    assert result.field_evidence_refs["value.status"] == (_ref("new-status"),)
    assert result.field_evidence_refs["value.untouched"] == old_untouched


def test_immutable_field_rejects_overwrite() -> None:
    with pytest.raises(FieldMergeError, match="identity"):
        FieldMerger().merge(
            _schema(),
            {"identity": "database"},
            {"value.identity": (_ref("old"),)},
            {"identity": "cache"},
            {"value.identity": (_ref("new"),)},
            relation="SUPPLEMENTS",
            review_authority=True,
        )


def test_append_unique_is_stable_and_retains_old_and_new_evidence() -> None:
    args = (
        _schema(),
        {"aliases": ["db", "database"]},
        {"value.aliases": (_ref("old"),)},
        {"aliases": ["database", "store", "db", "engine"]},
        {"value.aliases": (_ref("new"),)},
    )
    first = FieldMerger().merge(
        *args,
        relation="SUPPLEMENTS",
        review_authority=False,
    )
    second = FieldMerger().merge(
        *args,
        relation="SUPPLEMENTS",
        review_authority=False,
    )

    assert first.value_fields["aliases"] == ["db", "database", "store", "engine"]
    assert first.field_evidence_refs["value.aliases"] == (_ref("old"), _ref("new"))
    assert first.merge_digest == second.merge_digest
    assert first.decisions == second.decisions


def test_initial_append_unique_is_materialized_as_a_stable_complete_list() -> None:
    result = FieldMerger().merge(
        _schema(),
        {},
        {},
        {"aliases": ["db", "db", "database"]},
        {"value.aliases": (_ref("new"),), "transition": (_ref("transition"),)},
        relation="UNRELATED",
        review_authority=False,
    )

    assert result.value_fields == {"aliases": ["db", "database"]}
    assert result.field_evidence_refs["value.aliases"] == (_ref("new"),)
    assert result.changed_fields == ("aliases",)


def test_field_merger_rejects_undeclared_current_or_incoming_state() -> None:
    for current, incoming in (
        ({"forged_semantic_field": "already persisted"}, {}),
        ({}, {"forged_semantic_field": "new value"}),
    ):
        with pytest.raises(FieldMergeError, match="outside the memory schema"):
            FieldMerger().merge(
                _schema(),
                current,
                {},
                incoming,
                {"value.forged_semantic_field": (_ref("forged"),)},
                relation="SUPPLEMENTS",
                review_authority=True,
            )


def test_patch_text_requires_structured_deterministic_expression() -> None:
    merger = FieldMerger()
    with pytest.raises(FieldMergeError, match="summary"):
        merger.merge(
            _schema(),
            {"summary": "base"},
            {"value.summary": (_ref("old"),)},
            {"summary": "silently replace"},
            {"value.summary": (_ref("new"),)},
            relation="SUPPLEMENTS",
            review_authority=False,
        )

    result = merger.merge(
        _schema(),
        {"summary": "base"},
        {"value.summary": (_ref("old"),)},
        {"summary": {"op": "append", "text": "detail", "separator": ": "}},
        {"value.summary": (_ref("new"),)},
        relation="SUPPLEMENTS",
        review_authority=False,
    )
    assert result.value_fields["summary"] == "base: detail"


def test_destructive_replace_requires_review_authority() -> None:
    with pytest.raises(FieldMergeError, match="status"):
        FieldMerger().merge(
            _schema(),
            {"status": "adopted"},
            {"value.status": (_ref("old"),)},
            {"status": ""},
            {"value.status": (_ref("new"),)},
            relation="CORRECTS",
            review_authority=False,
        )

    reviewed = FieldMerger().merge(
        _schema(),
        {"status": "adopted"},
        {"value.status": (_ref("old"),)},
        {"status": ""},
        {"value.status": (_ref("new"),)},
        relation="CORRECTS",
        review_authority=True,
    )
    assert reviewed.value_fields["status"] == ""
