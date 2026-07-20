from __future__ import annotations

import pytest

from foundation.scope import (
    ScopeRef,
    ScopeSelector,
    scope_keys_from_payloads,
)
from pre.evidence import ScopeRef as EvidenceScopeRef


def test_scope_model_is_shared_without_type_conversion() -> None:
    """Evidence 使用 Foundation 的同一个类型，不能复制出第二套作用域模型。"""

    assert EvidenceScopeRef is ScopeRef


def test_scope_selector_round_trip_preserves_business_hierarchy() -> None:
    workspace = ScopeRef(namespace="memoryos", kind="workspace", id="workspace-1")
    camera = ScopeRef(
        namespace="memoryos",
        kind="asset",
        id="camera-1",
        parent_path=("factory-1",),
    )
    value = ScopeSelector((workspace, camera))

    restored = ScopeSelector.from_dict(value.to_dict())

    assert restored == value
    assert {scope.key for scope in restored.all_of} == {
        "memoryos:workspace:workspace-1",
        "memoryos:asset:path:factory-1/camera-1",
    }


def test_scope_array_rejects_malformed_members_instead_of_dropping_them() -> None:
    with pytest.raises(ValueError, match="scope objects"):
        scope_keys_from_payloads(
            [
                ScopeRef(namespace="memoryos", kind="workspace", id="workspace-1").to_dict(),
                "invalid",
            ]
        )
