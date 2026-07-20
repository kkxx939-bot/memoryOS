from __future__ import annotations

from inspect import Parameter, Signature, signature

from infrastructure.context.facade import ContextDB


def test_contextdb_constructor_exposes_only_greenfield_composition_boundaries() -> None:
    public = signature(ContextDB.__init__)
    expected_names = (
        "self",
        "source_store",
        "index_store",
        "relation_store",
        "relation_committer",
        "readiness",
        "tenant_id",
        "serving_lock",
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
        "relation_committer": "OrdinaryRelationCommitter | None",
        "tenant_id": "str",
        "serving_lock": "RLock | None",
    }
    assert all(
        public.parameters[name].default is Signature.empty
        for name in ("self", "source_store", "index_store", "relation_store")
    )
    assert public.parameters["relation_committer"].default is None
    assert public.parameters["readiness"].default is None
    assert public.parameters["tenant_id"].default == ""
    assert public.parameters["serving_lock"].default is None
    assert public.return_annotation == "None"


def test_contextdb_does_not_proxy_unrelated_application_services() -> None:
    removed = {
        "commit_operation",
        "commit_operations",
        "commit_session",
        "write_object",
        "seed_object",
        "import_object",
        "rebuild_index",
        "verify_consistency",
        "run_retention_cycle",
    }
    assert not removed.intersection(vars(ContextDB))
