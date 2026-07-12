from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from memoryos.api.limits import MAX_RETRIEVAL_LIMIT, MAX_TOKEN_BUDGET, bounded_int
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.service import RetrievalService
from memoryos.contextdb.session.commit_group import CommitGroupIntegrityError, CommitGroupStore
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.memory.canonical.evidence import EvidenceRef
from memoryos.memory.canonical.retrieval import CanonicalMemoryRetriever, CanonicalQueryIntent
from memoryos.memory.canonical.state import MemoryClaim, MemoryRevision, TransitionProfile


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["hotness", "semantic_hotness", "behavior_support_hotness"])
def test_context_hotness_rejects_nonfinite_values(field: str, value: float) -> None:
    kwargs: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match="finite"):
        ContextObject(
            uri="memoryos://user/u1/memories/nonfinite",
            context_type=ContextType.MEMORY,
            title="nonfinite",
            **kwargs,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_relation_and_vector_inputs_reject_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ContextRelation(
            source_uri="memoryos://user/u1/memories/a",
            relation_type="related",
            target_uri="memoryos://user/u1/memories/b",
            weight=value,
        )
    vectors = InMemoryVectorStore()
    with pytest.raises(ValueError, match="finite"):
        vectors.upsert_vector("memoryos://user/u1/memories/a", [1.0, value])
    vectors.upsert_vector("memoryos://user/u1/memories/a", [1.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        vectors.search_vector([value, 0.0], namespace="")


def test_api_limits_share_one_hard_bound() -> None:
    assert bounded_int(
        MAX_RETRIEVAL_LIMIT,
        default=10,
        minimum=0,
        maximum=MAX_RETRIEVAL_LIMIT,
        label="limit",
    ) == MAX_RETRIEVAL_LIMIT
    assert bounded_int(
        MAX_TOKEN_BUDGET,
        default=2000,
        minimum=0,
        maximum=MAX_TOKEN_BUDGET,
        label="token_budget",
    ) == MAX_TOKEN_BUDGET
    with pytest.raises(ValueError, match="between"):
        bounded_int(101, default=10, minimum=0, maximum=MAX_RETRIEVAL_LIMIT, label="limit")
    with pytest.raises(ValueError, match="between"):
        bounded_int(
            MAX_TOKEN_BUDGET + 1,
            default=2000,
            minimum=0,
            maximum=MAX_TOKEN_BUDGET,
            label="token_budget",
        )


def _revision(revision: int, *, valid_from: str, valid_to: str | None = None) -> MemoryRevision:
    return MemoryRevision(
        revision=revision,
        state="ACTIVE",
        value_fields={"canonical_value": f"value-{revision}"},
        evidence_refs=(EvidenceRef(f"e{revision}", None, f"hash-{revision}"),),
        proposal_id=f"p{revision}",
        relation="UNRELATED",
        epistemic_status="EXPLICIT",
        previous_revision=revision - 1 if revision > 1 else None,
        valid_from=valid_from,
        valid_to=valid_to,
        transaction_time=valid_from,
    )


def test_revision_intervals_require_timezone_and_positive_duration() -> None:
    with pytest.raises(ValueError, match="timezone"):
        _revision(1, valid_from="2026-01-01T00:00:00")
    with pytest.raises(ValueError, match="later"):
        _revision(
            1,
            valid_from="2026-01-01T00:00:00+00:00",
            valid_to="2026-01-01T00:00:00+00:00",
        )
    with pytest.raises(ValueError, match="later"):
        _revision(
            1,
            valid_from="2026-01-02T00:00:00+00:00",
            valid_to="2026-01-01T00:00:00+00:00",
        )


def test_equal_effective_time_transition_is_normalized_to_legal_interval() -> None:
    timestamp = "2026-01-01T00:00:00+00:00"
    first = _revision(1, valid_from=timestamp)
    claim = MemoryClaim(
        "claim",
        "memoryos://user/u1/memories/canonical/slots/slot/claims/claim",
        "slot",
        "value",
        TransitionProfile.AUTHORITATIVE_STATE,
        (first,),
    )
    updated = claim.with_revision(_revision(2, valid_from=timestamp))
    first_end = datetime.fromisoformat(str(updated.revisions[0].valid_to))
    first_start = datetime.fromisoformat(updated.revisions[0].valid_from)
    second_start = datetime.fromisoformat(updated.revisions[1].valid_from)
    assert first_start < first_end == second_start


def test_current_effective_time_and_negated_intent_are_fail_closed(tmp_path: Path) -> None:
    retriever = CanonicalMemoryRetriever(
        FileSystemSourceStore(tmp_path),
        InMemoryIndexStore(),
    )
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    expired_start = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    expired_end = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    assert retriever._revision_is_effective({"valid_from": future, "valid_to": None}) is False
    assert retriever._revision_is_effective(
        {"valid_from": expired_start, "valid_to": expired_end}
    ) is False
    for text in ("没有冲突", "不看历史", "do not show alternatives", "without history"):
        assert retriever.classify_intent(text) == CanonicalQueryIntent.CURRENT


def test_commit_group_retry_is_finite_and_corrupt_state_is_quarantined(tmp_path: Path) -> None:
    store = CommitGroupStore(tmp_path)
    status = store.create(
        "group-a",
        task_id="task-a",
        archive_uri="memoryos://user/u1/sessions/history/a",
        user_id="u1",
        tenant_id="default",
    )
    assert status.canonical_status == "pending"
    for attempt in range(1, store.MAX_ATTEMPTS + 1):
        attempt_id = f"attempt-{attempt}"
        assert store.claim_canonical("group-a", attempt_id=attempt_id)
        status = store.fail_canonical(
            "group-a",
            "OSError",
            retryable=True,
            attempt_id=attempt_id,
        )
    assert status.canonical_status == "dead_letter"
    assert status.canonical_attempt_count == store.MAX_ATTEMPTS
    assert status.canonical_last_error == "OSError"
    assert status.canonical_next_retry_at == ""
    assert store.claim_canonical("group-a", attempt_id="stale") is False
    assert store.pending() == []

    broken = store.path("broken-group")
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("{broken", encoding="utf-8")
    with pytest.raises(CommitGroupIntegrityError, match="quarantined"):
        store.load("broken-group")
    assert not broken.exists()
    quarantined = list((tmp_path / "system" / "quarantine" / "commit_group").glob("*.original"))
    assert len(quarantined) == 1


def test_recall_trace_redacts_secret_query_and_uses_private_file(tmp_path: Path) -> None:
    class EmptyAssembler:
        reranker = None

        def search(self, _query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    service = RetrievalService(EmptyAssembler(), tmp_path / "traces")  # type: ignore[arg-type]
    _results, trace_id = service.search(
        "OPENAI_API_KEY=sk-live-secret",
        user_id="u1",
        tenant_id="default",
    )
    trace = service.read_trace(trace_id)
    assert "sk-live-secret" not in trace["query"]
    assert "<redacted>" in trace["query"]
    path = service.trace_root / f"{trace_id}.json"
    assert path.stat().st_mode & 0o777 == 0o600
