from __future__ import annotations

from pathlib import Path

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.trusted_context import READ_CONTEXT, TrustedRequestContext
from memoryos.contextdb.retrieval.query_plan import RetrievalOptions
from memoryos.contextdb.retrieval.query_planner import RetrievalScopeViolation


@pytest.mark.parametrize(
    "forged_scope_key",
    [
        "memoryos:principal:victim",
        "memoryos:team:administrators",
        "memoryos:workspace:project-b",
    ],
)
def test_public_search_rejects_user_declared_scope_escalation(
    tmp_path: Path,
    forged_scope_key: str,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    caller = TrustedRequestContext(
        tenant_id="default",
        user_id="u1",
        actor_kind="agent",
        actor_id="codex",
        capabilities=frozenset({READ_CONTEXT}),
        allowed_workspace_ids=frozenset({"project-a"}),
    )

    with pytest.raises(RetrievalScopeViolation, match="exceed trusted caller scope"):
        client.search_context(
            "private context",
            options=RetrievalOptions(
                tenant_id="default",
                owner_user_id="u1",
                workspace_ids=("project-a",),
                adapter_id="codex",
                metadata_filters={"applicability_scope_keys": [forged_scope_key]},
            ),
            caller=caller,
        )


def test_public_search_keeps_authorized_principal_workspace_and_adapter_scope(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    caller = TrustedRequestContext(
        tenant_id="default",
        user_id="u1",
        actor_kind="agent",
        actor_id="codex",
        capabilities=frozenset({READ_CONTEXT}),
        allowed_workspace_ids=frozenset({"project-a"}),
    )

    assert (
        client.search_context(
            "no matching context",
            options=RetrievalOptions(
                tenant_id="default",
                owner_user_id="u1",
                workspace_ids=("project-a",),
                adapter_id="codex",
                target_paths=("agents/codex",),
                metadata_filters={
                    "applicability_scope_keys": [
                        "memoryos:principal:u1",
                        "memoryos:workspace:project-a",
                    ]
                },
            ),
            caller=caller,
        )
        == []
    )
