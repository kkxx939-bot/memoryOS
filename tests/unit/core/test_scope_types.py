from __future__ import annotations

import memoryos.contextdb.scope as legacy_scope
import memoryos.core.types as scope_types
from memoryos.core.types import (
    AuthorityPolicy,
    ContextScope,
    ScopeRef,
    ScopeSelector,
    VisibilityPolicy,
)


def test_scope_types_round_trip_at_core_owner_without_identity_change() -> None:
    subject = ScopeRef(namespace="memoryos", kind="principal", id="user-1")
    value = ContextScope(
        applicability=ScopeSelector((subject,)),
        visibility=VisibilityPolicy(tenant_id="tenant-1", allowed_principal_ids=("user-1",)),
        canonical_subject=subject,
        authority=AuthorityPolicy(principal_ids=("user-1",)),
    )

    assert ContextScope.from_dict(value.to_dict()) == value


def test_historical_contextdb_scope_exports_preserve_object_identity() -> None:
    assert legacy_scope.__all__ == scope_types.__all__
    for name in legacy_scope.__all__:
        assert getattr(legacy_scope, name) is getattr(scope_types, name)
