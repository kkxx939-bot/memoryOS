from __future__ import annotations

import re
from dataclasses import dataclass, field

from memoryos.services.memory.extractor import MEMORY_ACTIONS, MemoryOperation


@dataclass(frozen=True)
class MemoryOperationValidation:
    accepted: bool
    errors: list[str] = field(default_factory=list)
    needs_user_confirmation: bool = False
    sensitive: bool = False
    policy_decision: str = "accepted"

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "errors": self.errors,
            "needs_user_confirmation": self.needs_user_confirmation,
            "sensitive": self.sensitive,
            "policy_decision": self.policy_decision,
        }


class MemoryOperationValidator:
    def validate(
        self,
        operation: MemoryOperation,
        source_message_ids: list[str] | None = None,
        explicit_user_intent: bool = False,
    ) -> MemoryOperationValidation:
        errors = []
        if operation.action not in MEMORY_ACTIONS:
            errors.append(f"unknown action: {operation.action}")
        if operation.action in {"add", "update"} and not operation.text.strip():
            errors.append("text is required")
        if operation.memory_type == "policy" and not (explicit_user_intent or self._has_explicit_intent(operation)):
            return MemoryOperationValidation(
                accepted=False,
                errors=["policy memory requires explicit user intent"],
                needs_user_confirmation=True,
                policy_decision="needs_confirmation",
            )
        sensitive = SensitiveMemoryClassifier().is_sensitive(operation.text, operation.tags)
        if sensitive and "user_confirmed" not in {str(tag) for tag in operation.tags}:
            return MemoryOperationValidation(
                accepted=False,
                errors=errors,
                needs_user_confirmation=True,
                sensitive=True,
                policy_decision="sensitive_needs_confirmation",
            )
        return MemoryOperationValidation(
            accepted=not errors,
            errors=errors,
            needs_user_confirmation=False,
            sensitive=sensitive,
            policy_decision="accepted" if not errors else "rejected",
        )

    def _has_explicit_intent(self, operation: MemoryOperation) -> bool:
        tags = {str(tag) for tag in operation.tags}
        return bool(tags & {"explicit_user_intent", "user_confirmed"})


class SensitiveMemoryClassifier:
    SENSITIVE_TERMS = {
        "身份证",
        "password",
        "密码",
        "token",
        "api_key",
        "secret",
        "银行卡",
        "病历",
        "medical",
    }
    SENSITIVE_PATTERNS = (
        re.compile(r"\b1[3-9]\d{9}\b"),
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        re.compile(r"\b\d{15}(\d{2}[0-9Xx])?\b"),
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        re.compile(r"\b(?:sk|pk|rk|api)[-_][A-Za-z0-9]{16,}\b"),
        re.compile(r"\b(?:eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"),
    )

    def is_sensitive(self, text: str, tags: list[str] | None = None) -> bool:
        lowered = str(text).lower()
        if any(term.lower() in lowered for term in self.SENSITIVE_TERMS):
            return True
        if any(pattern.search(str(text)) for pattern in self.SENSITIVE_PATTERNS):
            return True
        tag_set = {str(tag).lower() for tag in (tags or [])}
        return bool(tag_set & {"sensitive", "secret", "credential", "medical"})
