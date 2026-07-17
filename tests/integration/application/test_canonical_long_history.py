from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.memory.canonical import (
    CanonicalMemoryQuery,
    CanonicalMemoryRepository,
    CanonicalQueryIntent,
    OfflineCanonicalMemoryRetriever,
)
from memoryos.memory.canonical.current_head import load_current_head, publish_current_head_sets
from memoryos.memory.canonical.history import validate_canonical_receipt_history
from memoryos.operations.commit.receipt import load_transaction_receipt, validate_transaction_receipt
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.runtime import RuntimeConfig, build_runtime_container
from memoryos.runtime.readiness import RuntimeReadinessState
from tests.support.canonical_transactions import (
    _artifact_root,
    _persisted_episode,
    _plan,
    _proposal,
    _replacement_proposal,
    _reviewed_resolution_plan,
    _setup,
)


def test_ten_revision_history_keeps_every_receipt_immutable_and_only_latest_head_current(
    tmp_path: Path,
) -> None:
    source, _index, _queue, _relations, committer, first_episode, scope = _setup(tmp_path)
    first = _proposal(first_episode, "history-1", "Backend-1", "confirmation", "confirmed")
    identity, _transition, plan = _plan(source, first_episode, scope, first)
    operations = plan.to_context_operations(
        user_id="u1",
        tenant_id="t1",
        episode_id=first_episode.episode_id,
    )
    first_diff = committer.commit("u1", operations)
    first_replay_operations = [ContextOperation.from_dict(operation.to_dict()) for operation in first_diff.operations]
    receipt_paths = [committer._transaction_marker(str(operations[0].payload["idempotency_key"]))]
    receipt_bytes = [receipt_paths[0].read_bytes()]
    diff_ids = [f"diff_{operations[0].payload['transaction_id']}"]

    for revision in range(2, 11):
        _slot, claims = CanonicalMemoryRepository(source).load(identity)
        active = next(claim for claim in claims if claim.current.state == "ACTIVE")
        value = f"Backend-{revision}"
        archive = SessionArchive(
            user_id="u1",
            session_id=f"history-{revision}",
            archive_uri=f"memoryos://user/u1/sessions/history/history-{revision}",
            messages=[
                {
                    "id": f"m{revision}",
                    "role": "user",
                    "content": (
                        f"I formally change the primary storage backend from the previous value to {value} now."
                    ),
                }
            ],
            metadata={"tenant_id": "t1", "project_id": "memoryos"},
            task_id=f"history-task-{revision}",
            # Keep every replacement later than the previously committed
            # revision while also making it effective at retrieval time.  A
            # fixed future timestamp would correctly create a not-yet-current
            # Claim and turn this history/receipt test into a clock-dependent
            # retrieval failure.
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        episode = _persisted_episode(tmp_path, archive)
        replacement = _replacement_proposal(
            episode,
            f"history-{revision}",
            value,
            active,
        )
        replacement_plan = _reviewed_resolution_plan(
            source,
            committer,
            episode,
            replacement,
            command_suffix=f"long-history-{revision}",
        )
        replacement_operations = list(replacement_plan.operations)
        committer.commit("u1", replacement_operations)
        canonical_operation = next(
            operation
            for operation in replacement_operations
            if operation.payload.get("canonical_pending_resolution") is not True
        )
        path = committer._transaction_marker(str(canonical_operation.payload["idempotency_key"]))
        receipt_paths.append(path)
        receipt_bytes.append(path.read_bytes())
        diff_ids.append(f"diff_{canonical_operation.payload['transaction_id']}")

    artifact_root = _artifact_root(tmp_path)
    history = validate_canonical_receipt_history(artifact_root, tenant_id="t1")
    assert history["transaction_receipts"] == 10
    assert all(validate_transaction_receipt(load_transaction_receipt(path)) for path in receipt_paths)
    assert [path.read_bytes() for path in receipt_paths] == receipt_bytes
    assert all((artifact_root / "system" / "diffs" / f"{diff_id}.json").exists() for diff_id in diff_ids)

    slot, claims = CanonicalMemoryRepository(source).load(identity)
    assert slot is not None and slot.revision == 10
    assert len(claims) == 10
    assert sum(claim.current.state == "ACTIVE" for claim in claims) == 1
    active = next(claim for claim in claims if claim.current.state == "ACTIVE")
    assert active.latest_revision.value_fields["canonical_value"] == "Backend-10"
    head_before = load_current_head(artifact_root, identity.slot_uri)[0]
    source_before_replay = {
        identity.slot_uri: source.read_object(identity.slot_uri).to_dict(),
        **{claim.uri: source.read_object(claim.uri).to_dict() for claim in claims},
    }
    replayed = committer.commit("u1", first_replay_operations)
    assert replayed.diff_id == first_diff.diff_id
    assert load_current_head(artifact_root, identity.slot_uri)[0] == head_before
    assert {uri: source.read_object(uri).to_dict() for uri in source_before_replay} == source_before_replay
    assert [path.read_bytes() for path in receipt_paths] == receipt_bytes
    publish_current_head_sets(
        artifact_root,
        receipt_paths[0],
        load_transaction_receipt(receipt_paths[0]),
    )
    assert load_current_head(artifact_root, identity.slot_uri)[0] == head_before

    restarted = build_runtime_container(RuntimeConfig(root=str(tmp_path), tenant_id="t1"))
    assert restarted.readiness.state == RuntimeReadinessState.READY, restarted.readiness.reasons
    assert not list((artifact_root / "system" / "quarantine").glob("**/*.original"))
    assert not list((artifact_root / "system" / "redo").glob("*.json"))
    assert validate_canonical_receipt_history(artifact_root, tenant_id="t1")["transaction_receipts"] == 10

    results = OfflineCanonicalMemoryRetriever(
        restarted.source_store,
        restarted.index_store,
        restarted.relation_store,
        projection_store=restarted.memory_projection_worker.projector.record_store,
        offline_admin=True,
    ).search(
        CanonicalMemoryQuery(
            text="Backend-10",
            tenant_id="t1",
            principal_id="u1",
            applicability_scope_keys=("memoryos:workspace:memoryos",),
            intent=CanonicalQueryIntent.CURRENT,
            limit=20,
        )
    )
    assert len(results) == 1
    assert results[0]["metadata"]["canonical_value"] == "backend-10"
