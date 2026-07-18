from __future__ import annotations

from inspect import Parameter, Signature, signature

from memoryos.contextdb.context_db import ContextDB


def test_contextdb_constructor_exposes_only_greenfield_composition_boundaries() -> None:
    public = signature(ContextDB.__init__)
    expected_names = (
        "self",
        "source_store",
        "index_store",
        "relation_store",
        "queue_store",
        "session_commit_service",
        "committer",
        "document_overlay",
        "tombstone_service",
        "retention_manager",
        "readiness",
        "tenant_id",
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
        "document_overlay": "Any | None",
        "tenant_id": "str",
    }
    assert all(
        public.parameters[name].default is Signature.empty
        for name in ("self", "source_store", "index_store", "relation_store")
    )
    assert all(public.parameters[name].default is None for name in expected_names[4:-1])
    assert public.parameters["tenant_id"].default == ""
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
