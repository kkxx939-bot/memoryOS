from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.api.trusted_context import TrustedRequestContext
from memoryos.memory.canonical import CanonicalMemoryRepository
from memoryos.memory.canonical.event import canonical_digest
from memoryos.memory.canonical.review_command import (
    PendingReviewCommandIntegrityError,
    PendingReviewCommandStore,
    PendingReviewIdempotencyConflict,
)
from memoryos.operations.commit.effect_marker import atomic_write_json
from memoryos.runtime.readiness import RuntimeReadinessState


def _pending(client: MemoryOSClient) -> dict:
    identity = {"decision_topic": "primary storage backend"}
    first = client.remember(
        user_id="u1",
        content="PostgreSQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields=identity,
    )
    assert first["status"] == "COMMITTED"
    second = client.remember(
        user_id="u1",
        content="MySQL",
        memory_type="project_decision",
        project_id="memoryos",
        identity_fields=identity,
    )
    assert second["status"] == "PENDING"
    return client.list_pending(user_id="u1", lifecycle_states=["PENDING"])[0]


def _kwargs(record: dict, command_id: str) -> dict:
    return {
        "user_id": "u1",
        "pending_uri": record["uri"],
        "decision": "CONFIRM_AND_APPLY",
        "expected_lifecycle_revision": record["lifecycle_revision"],
        "expected_proposal_fingerprint": record["proposal_fingerprint"],
        "command_id": command_id,
        "reason": "structured human authority",
    }


def test_same_review_command_returns_exact_result_and_conflicting_decision_is_rejected(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    record = _pending(client)
    request = _kwargs(record, "same-review-command")

    first = client.review_pending(**request)
    repeated = client.review_pending(**request)

    assert first == repeated
    assert first["status"] == "resolved"
    assert len(first["resolved_claim_uris"]) == 1
    with pytest.raises(PendingReviewIdempotencyConflict, match="different decision or effect"):
        client.review_pending(**{**request, "decision": "REJECT"})


def test_unauthorized_review_creates_no_command_or_recovery_side_effect(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    record = _pending(client)
    request = _kwargs(record, "unauthorized-review-command")
    caller = TrustedRequestContext(
        tenant_id="default",
        user_id="u1",
        actor_kind="agent",
        actor_id="codex",
        capabilities=frozenset(),
        allowed_workspace_ids=frozenset({"memoryos"}),
    )

    with pytest.raises(PermissionError):
        client.review_pending(**request, caller=caller)

    commands = PendingReviewCommandStore(tmp_path, tenant_id="default")
    assert not commands.path(request["command_id"]).exists()


def test_committed_review_receipt_survives_command_record_deletion(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    record = _pending(client)
    request = _kwargs(record, "receipt-bound-review-command")
    commands = PendingReviewCommandStore(tmp_path, tenant_id="default")

    first = client.review_pending(**request)
    commands.path(request["command_id"]).unlink()

    recovered = client.review_pending(**request)
    assert recovered == first

    commands.path(request["command_id"]).unlink()
    with pytest.raises(PendingReviewIdempotencyConflict, match="bound by a receipt"):
        client.review_pending(**{**request, "decision": "REJECT"})


def test_two_concurrent_reviews_have_one_winner_and_one_terminal_cas_failure(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    record = _pending(client)
    barrier = Barrier(2)

    def review(command_id: str):  # noqa: ANN202
        barrier.wait(timeout=10)
        return client.review_pending(**_kwargs(record, command_id))

    outcomes: list[object] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(review, command_id) for command_id in ("review-a", "review-b")]
        for future in futures:
            try:
                outcomes.append(future.result(timeout=20))
            except Exception as exc:  # noqa: BLE001 - asserted by exact outcome below.
                outcomes.append(exc)

    successes = [item for item in outcomes if isinstance(item, dict)]
    failures = [item for item in outcomes if isinstance(item, Exception)]
    assert len(successes) == 1 and successes[0]["status"] == "resolved"
    assert len(failures) == 1
    assert "expected revision" in str(failures[0]) or "Lock already held" in str(failures[0])

    pending = CanonicalMemoryRepository(
        client.source_store,
        client.relation_store,
    ).load_pending(record["uri"], tenant_id="default", owner_user_id="u1")
    assert pending.lifecycle_state.value == "resolved"
    assert pending.lifecycle_revision == 3
    active = client.search_context(
        "storage backend",
        user_id="u1",
        project_id="memoryos",
        context_type="memory",
        memory_states=["ACTIVE"],
    )
    mysql = [item for item in active if item["metadata"].get("canonical_value") == "mysql"]
    assert len(mysql) == 1
    commands = PendingReviewCommandStore(tmp_path, tenant_id="default")
    existing_states = []
    for command_id in ("review-a", "review-b"):
        if commands.path(command_id).exists():
            existing_states.append(commands.load(command_id)["status"])
    assert existing_states.count("completed") == 1
    assert existing_states.count("failed") <= 1

    failed_command = "stale-review-command"
    with pytest.raises(ValueError, match="expected revision"):
        client.review_pending(**_kwargs(record, failed_command))
    assert commands.load(failed_command)["status"] == "failed"
    with pytest.raises(ValueError, match="previously failed"):
        client.review_pending(**_kwargs(record, failed_command))


def test_retryable_review_failure_keeps_command_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    record = _pending(client)
    request = _kwargs(record, "retryable-review-command")
    original = client._review_pending_locked
    calls = 0

    def fail_once(**kwargs):  # noqa: ANN003, ANN202
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected retryable review outage")
        return original(**kwargs)

    monkeypatch.setattr(client, "_review_pending_locked", fail_once)
    with pytest.raises(OSError, match="retryable review outage"):
        client.review_pending(**request)
    commands = PendingReviewCommandStore(tmp_path, tenant_id="default")
    assert commands.load("retryable-review-command")["status"] == "running"

    result = client.review_pending(**request)

    assert result["status"] == "resolved"
    assert commands.load("retryable-review-command")["status"] == "completed"


def test_crashed_confirm_and_apply_command_keeps_exclusive_resolution_ownership(
    tmp_path: Path,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    record = _pending(client)
    first_request = _kwargs(record, "crashed-confirm-and-apply")
    crashed = False

    def crash_after_confirmation_head(stage: str, _transaction_id: str) -> None:
        nonlocal crashed
        if stage == "after_current_head" and not crashed:
            crashed = True
            raise SystemExit("crash after committed confirmation head")

    client.committer.test_hook = crash_after_confirmation_head
    with pytest.raises(SystemExit, match="committed confirmation head"):
        client.review_pending(**first_request)
    client.committer.test_hook = None

    recovered = MemoryOSClient(str(tmp_path))
    assert recovered.readiness.state == RuntimeReadinessState.READY, recovered.readiness.reasons
    confirmed = recovered.list_pending(user_id="u1", lifecycle_states=["CONFIRMED"])[0]
    commands = PendingReviewCommandStore(tmp_path, tenant_id="default")
    assert commands.load(first_request["command_id"])["status"] == "running"

    competing = {
        **_kwargs(record, "competing-confirm-and-apply"),
        "expected_lifecycle_revision": confirmed["lifecycle_revision"],
    }
    with pytest.raises(PendingReviewIdempotencyConflict, match="owns the in-flight resolution"):
        recovered.review_pending(**competing)

    result = recovered.review_pending(**first_request)
    assert result["status"] == "resolved"
    assert commands.load(first_request["command_id"])["status"] == "completed"


def test_startup_completes_running_review_command_from_resolved_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MemoryOSClient(str(tmp_path))
    record = _pending(client)
    request = _kwargs(record, "resolved-before-command-complete")
    original_complete = PendingReviewCommandStore.complete

    def crash_before_command_completion(self, command_id, result):  # noqa: ANN001, ANN202
        del self, command_id, result
        raise SystemExit("crash before review command completion")

    monkeypatch.setattr(PendingReviewCommandStore, "complete", crash_before_command_completion)
    with pytest.raises(SystemExit, match="before review command completion"):
        client.review_pending(**request)
    monkeypatch.setattr(PendingReviewCommandStore, "complete", original_complete)

    recovered = MemoryOSClient(str(tmp_path))
    assert recovered.readiness.state == RuntimeReadinessState.READY, recovered.readiness.reasons
    command = PendingReviewCommandStore(tmp_path, tenant_id="default").load(request["command_id"])
    assert command["status"] == "completed"
    assert command["result"]["status"] == "resolved"
    assert command["result"]["resolved_claim_uris"]
    assert recovered.review_pending(**request) == command["result"]


def test_completed_review_result_must_match_receipt_backed_lifecycle_history(tmp_path: Path) -> None:
    client = MemoryOSClient(str(tmp_path))
    record = _pending(client)
    request = _kwargs(record, "tampered-review-result")
    client.review_pending(**request)
    commands = PendingReviewCommandStore(tmp_path, tenant_id="default")
    path = commands.path(request["command_id"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result"]["lifecycle_revision"] = 999
    core = {key: value for key, value in payload.items() if key != "record_digest"}
    payload["record_digest"] = canonical_digest(core)
    atomic_write_json(path, payload, artifact_root=commands.artifact_root)

    with pytest.raises(PendingReviewCommandIntegrityError, match="differs from committed history"):
        client.review_pending(**request)

    restarted = MemoryOSClient(str(tmp_path))
    assert restarted.readiness.state == RuntimeReadinessState.NOT_READY
    assert "pending review" in " ".join(restarted.readiness.reasons).casefold()
