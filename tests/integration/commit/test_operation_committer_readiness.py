from __future__ import annotations

from pathlib import Path

import pytest

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.runtime import RuntimeConfig, build_runtime_container
from memoryos.runtime.readiness import RuntimeNotReadyError, RuntimeReadinessState


class _CountingExtractor:
    semantic_proposal_backend = True
    llm_semantic_backend = True
    is_remote = False

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, archive, schemas):  # noqa: ANN001, ANN201
        del archive, schemas
        self.calls += 1
        return ()


def _resource_operation() -> ContextOperation:
    obj = ContextObject(
        uri="memoryos://user/u1/resources/readiness-proof",
        context_type=ContextType.RESOURCE,
        title="readiness proof",
        owner_user_id="u1",
        tenant_id="t1",
    )
    return ContextOperation(
        context_type=ContextType.RESOURCE,
        action=OperationAction.ADD,
        target_uri=obj.uri,
        user_id="u1",
        payload={"tenant_id": "t1", "context_object": obj.to_dict(), "content": "proof"},
    )


def _assert_no_operation_artifacts(root: Path) -> None:
    system = root / "tenants" / "t1" / "system"
    for name in ("redo", "operations", "diffs"):
        assert not list((system / name).glob("*.json"))


@pytest.mark.parametrize(
    "state",
    [RuntimeReadinessState.NOT_READY, RuntimeReadinessState.RECOVERING],
)
def test_ordinary_committer_and_contextdb_reject_before_artifacts(
    tmp_path: Path,
    state: RuntimeReadinessState,
) -> None:
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    runtime.readiness.transition(state, reasons=("startup proof incomplete",))

    with pytest.raises(RuntimeNotReadyError, match=f"runtime is {state.value}"):
        runtime.committer.commit("u1", [_resource_operation()])
    with pytest.raises(RuntimeNotReadyError, match=f"runtime is {state.value}"):
        runtime.context_db.commit_operations([_resource_operation()])

    _assert_no_operation_artifacts(tmp_path)
    with pytest.raises(FileNotFoundError):
        runtime.source_store.read_object("memoryos://user/u1/resources/readiness-proof")


@pytest.mark.parametrize(
    "state",
    [RuntimeReadinessState.NOT_READY, RuntimeReadinessState.RECOVERING],
)
def test_session_service_rejects_before_archive_extraction_or_group_mutation(
    tmp_path: Path,
    state: RuntimeReadinessState,
) -> None:
    extractor = _CountingExtractor()
    runtime = build_runtime_container(
        RuntimeConfig(root=str(tmp_path), tenant_id="t1", memory_extractor=extractor)
    )
    archive = SessionArchive(
        user_id="u1",
        session_id=f"blocked-{state.value.lower()}",
        archive_uri=f"memoryos://user/u1/sessions/history/blocked-{state.value.lower()}",
        messages=[{"id": "m1", "role": "user", "content": "Remember this durable preference."}],
        metadata={"tenant_id": "t1", "project_id": "memoryos"},
        task_id=f"blocked-{state.value.lower()}-task",
    )
    runtime.readiness.transition(state, reasons=("startup proof incomplete",))

    with pytest.raises(RuntimeNotReadyError, match=f"runtime is {state.value}"):
        runtime.session_commit_service.sync_archive(archive)
    with pytest.raises(RuntimeNotReadyError, match=f"runtime is {state.value}"):
        runtime.session_commit_service.async_commit(archive)

    assert extractor.calls == 0
    assert not runtime.session_archive_store.archive_exists(archive.archive_uri, tenant_id="t1")
    system = tmp_path / "tenants" / "t1" / "system"
    assert not list((system / "commit_groups").glob("*.json"))
    assert runtime.queue_store.stats().get("pending", 0) == 0
    _assert_no_operation_artifacts(tmp_path)


def test_ordinary_committer_rejects_document_uri_before_redo_publication(tmp_path: Path) -> None:
    runtime = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    document_uri = "memoryos://user/u1/memory/documents/memdoc_01J00000000000000000000000"
    operation = ContextOperation(
        context_type=ContextType.RESOURCE,
        action=OperationAction.ADD,
        target_uri=document_uri,
        user_id="u1",
        payload={"tenant_id": "t1"},
    )

    with pytest.raises(PermissionError, match="Markdown memory documents cannot pass"):
        runtime.committer.commit("u1", [operation])

    _assert_no_operation_artifacts(tmp_path)
