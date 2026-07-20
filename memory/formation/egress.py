"""记忆模型调用前的结构化、可替换出站数据策略。"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from pre.evidence.model import EvidenceEpisode
from pre.session import SessionArchive
from sanitization import (
    ENV_SECRET_RE,
    INLINE_SECRET_RE,
    PRIVATE_KEY_RE,
    SECRET_KEY_RE,
)


class EgressDecision(str, Enum):
    ALLOW = "ALLOW"
    ALLOW_REDACTED = "ALLOW_REDACTED"
    LOCAL_ONLY = "LOCAL_ONLY"
    DENY = "DENY"


class SensitivityCategory(str, Enum):
    SECRET = "secret"
    IDENTITY = "identity"
    CONTACT = "contact"
    ADDRESS = "address"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    PRIVATE_RELATIONSHIP = "private_relationship"
    PRIVATE_CONVERSATION = "private_conversation"
    ENTERPRISE_CODE_CONFIG = "enterprise_code_config"
    RESTRICTED_SCOPE = "restricted_scope"


@dataclass(frozen=True)
class EgressAssessment:
    decision: EgressDecision
    categories: tuple[SensitivityCategory, ...] = ()
    reasons: tuple[str, ...] = ()


class EgressClassifier(Protocol):
    def classify(self, archive: SessionArchive, episode: EvidenceEpisode) -> set[SensitivityCategory]: ...


class MetadataSensitivityClassifier:
    """优先服从接入层给出的结构化敏感标签。"""

    def classify(self, archive: SessionArchive, episode: EvidenceEpisode) -> set[SensitivityCategory]:
        result: set[SensitivityCategory] = set()
        metadata_rows: list[Mapping[str, Any]] = [dict(archive.metadata or {})]
        metadata_rows.extend(dict(event.metadata or {}) for event in episode.events)
        aliases = {item.value: item for item in SensitivityCategory}
        for metadata in metadata_rows:
            raw = metadata.get("sensitivity", metadata.get("sensitivity_categories", []))
            values = [raw] if isinstance(raw, str) else list(raw or []) if isinstance(raw, Sequence) else []
            for value in values:
                category = aliases.get(str(value).strip().casefold())
                if category is not None:
                    result.add(category)
            scope = dict(metadata.get("scope", {}) or {}) if isinstance(metadata.get("scope", {}), Mapping) else {}
            if bool(metadata.get("restricted") or metadata.get("private") or scope.get("restricted")):
                result.add(SensitivityCategory.RESTRICTED_SCOPE)
        return result

class StructuredTextSensitivityClassifier:
    """有限的默认文本分类器；部署方可以注入更强的实现。"""

    _CREDENTIAL_VALUE_RE = re.compile(
        r"(?i)\b(?:api[_ -]?key|access[_ -]?token|password|passcode|passwd|secret)\b"
        r"\s*(?:is|equals|was|[:=])\s*[^\s,;]+"
    )

    _PATTERNS: dict[SensitivityCategory, tuple[re.Pattern[str], ...]] = {
        SensitivityCategory.IDENTITY: (
            re.compile(r"(?i)\b(?:passport|national\s+id|social\s+security|ssn)\b"),
            re.compile(r"(?:身份证|护照号)"),
        ),
        SensitivityCategory.CONTACT: (
            re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b"),
            # 限定电话号码的合理位数并在非电话分隔符处停止，
            # 避免把 ISO 时间戳误判为联系方式。
            re.compile(r"(?<!\d)(?:\+?\d[\s()-]*){9,15}(?!\d)"),
        ),
        SensitivityCategory.ADDRESS: (
            re.compile(r"(?i)\b(?:home|residential|mailing)\s+address\b"),
            re.compile(r"(?:家庭住址|居住地址|收货地址)"),
        ),
        SensitivityCategory.MEDICAL: (
            re.compile(r"(?i)\b(?:diagnosis|diagnosed|hiv|cancer|therapy|medical\s+record|prescription)\b"),
            re.compile(r"(?:诊断|病历|处方|艾滋|癌症|心理治疗)"),
        ),
        SensitivityCategory.FINANCIAL: (
            re.compile(r"(?i)\b(?:bank\s+account|credit\s+card|routing\s+number|salary|tax\s+id)\b"),
            re.compile(r"(?:银行卡|银行账户|工资|税号)"),
        ),
        SensitivityCategory.PRIVATE_RELATIONSHIP: (
            re.compile(r"(?i)\b(?:affair|intimate\s+partner|private\s+relationship)\b"),
            re.compile(r"(?:婚外情|亲密关系|私人关系)"),
        ),
        SensitivityCategory.PRIVATE_CONVERSATION: (
            re.compile(r"(?i)\b(?:private|confidential|off\s+the\s+record)\s+(?:chat|conversation|message)\b"),
            re.compile(r"(?:私密对话|保密聊天|私人消息)"),
        ),
        SensitivityCategory.ENTERPRISE_CODE_CONFIG: (
            re.compile(r"(?i)\b(?:proprietary|internal|confidential)\s+(?:source\s+code|configuration|repository)\b"),
            re.compile(r"(?:企业敏感代码|内部源码|机密配置)"),
        ),
    }
    redaction_safe_categories = frozenset({SensitivityCategory.CONTACT})

    def classify(self, archive: SessionArchive, episode: EvidenceEpisode) -> set[SensitivityCategory]:
        result: set[SensitivityCategory] = set()
        # 自定义语义后端可能读取完整不可变归档，因此必须检查整个边界，
        # 不能只检查内置 Prompt 使用的规范化事件文本。
        archive_boundary = {
            "user_id": archive.user_id,
            "session_id": archive.session_id,
            "archive_uri": archive.archive_uri,
            "messages": archive.messages,
            "observations": archive.observations,
            "predictions": archive.predictions,
            "action_results": archive.action_results,
            "feedback": archive.feedback,
            "used_contexts": archive.used_contexts,
            "used_skills": archive.used_skills,
            "tool_results": archive.tool_results,
            "metadata": archive.metadata,
        }
        if _contains_secret_field(archive_boundary):
            result.add(SensitivityCategory.SECRET)
        for text in _iter_strings(archive_boundary):
            result.update(self.classify_text(text))
        for event in episode.events:
            outbound_event = {
                "text": event.text(),
                "actor": event.actor.to_dict(),
                "subjects": [subject.to_dict() for subject in event.subjects],
            }
            for text in _iter_strings(outbound_event):
                result.update(self.classify_text(text))
        for scope in episode.legal_scope_candidates():
            for text in _iter_strings(scope.to_dict()):
                result.update(self.classify_text(text))
        return result

    def classify_text(self, text: str) -> set[SensitivityCategory]:
        result: set[SensitivityCategory] = set()
        if (
            PRIVATE_KEY_RE.search(text)
            or ENV_SECRET_RE.search(text)
            or INLINE_SECRET_RE.search(text)
            or self._CREDENTIAL_VALUE_RE.search(text)
            or re.search(r"(?i)\b(?:password|passwd|authorization|cookie)\s*[:=]", text)
        ):
            result.add(SensitivityCategory.SECRET)
        for category, patterns in self._PATTERNS.items():
            if any(pattern.search(text) for pattern in patterns):
                result.add(category)
        return result

    def redact(self, text: str, categories: set[SensitivityCategory]) -> str:
        redacted = text
        if SensitivityCategory.SECRET in categories:
            redacted = PRIVATE_KEY_RE.sub("[REDACTED_SECRET]", redacted)
            redacted = ENV_SECRET_RE.sub("[REDACTED_SECRET]", redacted)
            redacted = INLINE_SECRET_RE.sub("[REDACTED_SECRET]", redacted)
        for category in categories:
            for pattern in self._PATTERNS.get(category, ()):
                redacted = pattern.sub(f"[REDACTED_{category.value.upper()}]", redacted)
        return redacted


class MemoryEgressPolicy:
    """远程 Provider 默认拒绝发送敏感数据。"""

    def __init__(
        self,
        classifiers: Sequence[EgressClassifier] | None = None,
        *,
        redact_categories: Sequence[SensitivityCategory] = (),
    ) -> None:
        self.classifiers = tuple(
            classifiers or (MetadataSensitivityClassifier(), StructuredTextSensitivityClassifier())
        )
        self.redact_categories = frozenset(redact_categories)

    def evaluate(
        self,
        archive: SessionArchive,
        episode: EvidenceEpisode,
        *,
        remote: bool,
    ) -> EgressAssessment:
        if not remote:
            return EgressAssessment(EgressDecision.LOCAL_ONLY, reasons=("local_provider",))
        categories: set[SensitivityCategory] = set()
        unsafe_redaction: set[SensitivityCategory] = set()
        for classifier in self.classifiers:
            detected = classifier.classify(archive, episode)
            categories.update(detected)
            safe_categories = set(getattr(classifier, "redaction_safe_categories", ()))
            unsafe_redaction.update(detected - safe_categories)
        ordered = tuple(sorted(categories, key=lambda item: item.value))
        if not ordered:
            return EgressAssessment(EgressDecision.ALLOW, reasons=("no_sensitive_category",))
        if SensitivityCategory.SECRET in categories or SensitivityCategory.RESTRICTED_SCOPE in categories:
            return EgressAssessment(EgressDecision.DENY, ordered, ("secret_or_restricted",))
        if categories.issubset(self.redact_categories) and not unsafe_redaction:
            return EgressAssessment(EgressDecision.ALLOW_REDACTED, ordered, ("configured_redaction",))
        return EgressAssessment(EgressDecision.LOCAL_ONLY, ordered, ("sensitive_remote_default_deny",))

    def redact(self, prompt: str, assessment: EgressAssessment) -> str:
        if assessment.decision != EgressDecision.ALLOW_REDACTED:
            return prompt
        categories = set(assessment.categories)
        for classifier in self.classifiers:
            redact = getattr(classifier, "redact", None)
            if callable(redact):
                redacted = redact(prompt, categories)
                if not isinstance(redacted, str):
                    raise TypeError("egress classifier redact() must return text")
                prompt = redacted
        return prompt


def _iter_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(text for item in value.values() for text in _iter_strings(item))
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return tuple(text for item in value for text in _iter_strings(item))
    return ()


def _contains_secret_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if SECRET_KEY_RE.search(str(key)) and item is not None and item != "" and item != "<redacted>":
                return True
            if _contains_secret_field(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_contains_secret_field(item) for item in value)
    return False


__all__ = [
    "EgressAssessment",
    "EgressClassifier",
    "EgressDecision",
    "MemoryEgressPolicy",
    "MetadataSensitivityClassifier",
    "SensitivityCategory",
    "StructuredTextSensitivityClassifier",
]
