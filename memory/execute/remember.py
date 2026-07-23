"""明确记住内容的单一可信命令。"""

from __future__ import annotations

import hashlib
import re

from foundation.identity import LocalUserContext
from foundation.integrity import canonical_json
from memory.commit.remember_plan import RememberTarget, RememberTargetKind
from memory.execute.base import MemoryCommandBase, _assert_expected_digest
from memory.execute.contracts import RememberResult


class RememberOperation(MemoryCommandBase):
    """把可信用户显式输入转换为确定性的 Markdown CAS。"""

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
        request_material = canonical_json(
            [caller.tenant_id, caller.user_id, body, occurred_at or "", target_hint or ""]
        )
        command_digest = hashlib.sha256(request_material.encode()).hexdigest()
        target = _explicit_target(body, target_hint)
        request_key = f"remember:{command_digest}"
        plan = self.planner.plan(
            body,
            target,
            tenant_id=caller.tenant_id,
            owner_user_id=caller.user_id,
            idempotency_key=request_key,
            command_digest=command_digest,
        )
        _assert_expected_digest(plan.expected_state, expected_document_digest)
        self.erase_store.assert_mutation_allowed(caller.tenant_id, caller.user_id, plan.document_id)
        result = self._commit_or_replay(
            plan,
            caller=caller,
            evidence_reference=f"explicit-command:sha256:{command_digest}",
        )
        return RememberResult(**self._result_fields(plan, result))


def _explicit_target(content: str, target_hint: str | None) -> RememberTarget:
    raw_hint = str(target_hint or "").strip()
    aliases = {
        "profile": RememberTargetKind.PROFILE,
        "profile_fact": RememberTargetKind.PROFILE,
        "preference": RememberTargetKind.PREFERENCE,
        "preferences": RememberTargetKind.PREFERENCE,
        "entity": RememberTargetKind.ENTITY,
        "entity_note": RememberTargetKind.ENTITY,
        "topic": RememberTargetKind.TOPIC,
        "topic_note": RememberTargetKind.TOPIC,
        "episode": RememberTargetKind.EPISODE,
        "experience": RememberTargetKind.EPISODE,
        "open_loop": RememberTargetKind.OPEN_LOOP,
    }
    prefix, separator, suffix = raw_hint.partition(":")
    normalized_prefix = prefix.casefold().replace("-", "_")
    if separator and normalized_prefix in aliases:
        kind = aliases[normalized_prefix]
        subject = suffix.strip()
    else:
        normalized = raw_hint.casefold().replace("-", "_")
        kind = aliases.get(normalized, RememberTargetKind.TOPIC)
        subject = "" if normalized in aliases else raw_hint
    return RememberTarget(kind, subject or _content_title(content))


def _content_title(content: str) -> str:
    first = next((line.strip() for line in content.splitlines() if line.strip()), "Memory")
    first = re.sub(r"^#{1,6}[ \t]+", "", first)
    collapsed = " ".join(first.split())
    return (collapsed[:120] or "Memory").rstrip()


__all__ = ["RememberOperation"]
