"""明确记住内容的单一用例。"""

from __future__ import annotations

import hashlib
import re

from foundation.identity import LocalUserContext
from foundation.integrity import canonical_json
from memory.core.model import MemoryCandidateKind, MemoryEditProposal
from memory.execute.base import MemoryCommandBase, _assert_expected_digest
from memory.execute.contracts import RememberResult
from memory.execute.write_planner import explicit_evidence_digest


class RememberOperation(MemoryCommandBase):
    """把可信用户明确输入转换为确定性的记忆文档 CAS。"""

    def remember(
        self,
        content: str,
        occurred_at: str | None = None,
        target_hint: str | None = None,
        expected_document_digest: str | None = None,
        *,
        caller: LocalUserContext,
    ) -> RememberResult:
        self._require_ready()
        if self.bootstrapper is not None:
            self.bootstrapper.ensure_user(caller.tenant_id, caller.user_id)
        body = str(content or "").strip()
        if not body:
            raise ValueError("remember content is required")
        evidence_digest = explicit_evidence_digest(body)
        proposal = _explicit_proposal(
            body,
            occurred_at=occurred_at,
            target_hint=target_hint,
            evidence_reference=f"explicit-input:sha256:{evidence_digest}",
        )
        request_key = "remember:" + hashlib.sha256(
            canonical_json([caller.tenant_id, caller.user_id, body, occurred_at or "", target_hint or ""]).encode()
        ).hexdigest()
        plan = self.planner.plan(
            proposal,
            tenant_id=caller.tenant_id,
            owner_user_id=caller.user_id,
            idempotency_key=request_key,
            evidence_digest=evidence_digest,
        )
        _assert_expected_digest(plan.expected_state, expected_document_digest)
        self.erase_store.assert_mutation_allowed(caller.tenant_id, caller.user_id, plan.document_id)
        result = self._commit_or_replay(
            plan,
            caller=caller,
            evidence_reference=f"explicit-input:sha256:{evidence_digest}",
        )
        return RememberResult(**self._result_fields(plan, result))


def _explicit_proposal(
    content: str,
    *,
    occurred_at: str | None,
    target_hint: str | None,
    evidence_reference: str,
) -> MemoryEditProposal:
    raw_hint = str(target_hint or "").strip()
    normalized_hint = raw_hint.casefold().replace("-", "_")
    kind_aliases = {
        "profile": MemoryCandidateKind.PROFILE_FACT,
        "profile_fact": MemoryCandidateKind.PROFILE_FACT,
        "preference": MemoryCandidateKind.PREFERENCE,
        "preferences": MemoryCandidateKind.PREFERENCE,
        "entity": MemoryCandidateKind.ENTITY_NOTE,
        "entity_note": MemoryCandidateKind.ENTITY_NOTE,
        "topic": MemoryCandidateKind.TOPIC_NOTE,
        "topic_note": MemoryCandidateKind.TOPIC_NOTE,
        "episode": MemoryCandidateKind.EPISODE,
        "open_loop": MemoryCandidateKind.OPEN_LOOP,
        "experience": MemoryCandidateKind.EXPERIENCE,
    }
    subject_hint = ""
    prefix, separator, suffix = raw_hint.partition(":")
    if separator and prefix.casefold().replace("-", "_") in kind_aliases:
        kind = kind_aliases[prefix.casefold().replace("-", "_")]
        subject_hint = suffix.strip()
    else:
        kind = kind_aliases.get(normalized_hint, MemoryCandidateKind.TOPIC_NOTE)
        if raw_hint and normalized_hint not in kind_aliases:
            subject_hint = raw_hint
    title = subject_hint or _content_title(content)
    entity_hints = (title,) if kind == MemoryCandidateKind.ENTITY_NOTE else ()
    topic_hints = (title,) if kind == MemoryCandidateKind.TOPIC_NOTE else ()
    return MemoryEditProposal(
        candidate_kind=kind,
        title=title,
        body=content,
        evidence_refs=(evidence_reference,),
        subject=title,
        entity_hints=entity_hints,
        topic_hints=topic_hints,
        occurred_at=str(occurred_at or ""),
    )


def _content_title(content: str) -> str:
    first = next((line.strip() for line in content.splitlines() if line.strip()), "Memory")
    first = re.sub(r"^#{1,6}[ \t]+", "", first)
    collapsed = " ".join(first.split())
    return (collapsed[:120] or "Memory").rstrip()


__all__ = ["RememberOperation"]
