from __future__ import annotations

from inspect import Parameter, Signature, signature

from memoryos.contextdb.context_db import ContextDB


def test_contextdb_constructor_signature_remains_exactly_compatible() -> None:
    public = signature(ContextDB.__init__)
    expected_names = (
        "self",
        "source_store",
        "index_store",
        "relation_store",
        "queue_store",
        "session_commit_service",
        "committer",
        "projection_store",
        "canonical_projector",
        "current_slot_projector",
        "tombstone_service",
        "retention_manager",
        "migration_gate",
        "unified_context_migration",
        "readiness",
    )
    assert tuple(public.parameters) == expected_names
    assert all(parameter.kind is Parameter.POSITIONAL_OR_KEYWORD for parameter in public.parameters.values())
    assert {
        name: parameter.annotation
        for name, parameter in public.parameters.items()
        if parameter.annotation is not Signature.empty
    } == {
        "source_store": "SourceStore",
        "index_store": "IndexStore",
        "relation_store": "RelationStore",
        "queue_store": "QueueStore | None",
        "committer": "OperationCommitter | None",
        "projection_store": "ProjectionRecordStore | None",
    }
    assert all(
        public.parameters[name].default is Signature.empty
        for name in ("self", "source_store", "index_store", "relation_store")
    )
    assert all(public.parameters[name].default is None for name in expected_names[4:])
    assert public.return_annotation == "None"


def test_contextdb_commit_signatures_remain_exactly_compatible() -> None:
    commit_one = signature(ContextDB.commit_operation)
    assert tuple(commit_one.parameters) == ("self", "operation")
    assert commit_one.parameters["operation"].annotation == "ContextOperation"
    assert commit_one.return_annotation == "CommitResult"

    commit_many = signature(ContextDB.commit_operations)
    assert tuple(commit_many.parameters) == ("self", "operations")
    assert commit_many.parameters["operations"].annotation == "list[ContextOperation]"
    assert commit_many.return_annotation == "list[CommitResult]"
