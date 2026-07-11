"""记忆系统里的证据。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from memoryos.memory.canonical.episode import EvidenceEpisode
from memoryos.memory.canonical.event import EventEnvelope
from memoryos.memory.canonical.proposal import EpistemicStatus, MemorySemanticProposal


def evidence_hash(text: str) -> str:
    """计算证据文本的 SHA-256，用来发现内容被改动。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EvidenceSignalKind(str, Enum):
    """列出确认、约束、偏好、提议、评估、撤回和否定这些信号类型。"""

    CONFIRMATION = "confirmation"
    CONSTRAINT = "constraint"
    PREFERENCE = "preference"
    PROPOSAL = "proposal"
    EVALUATION = "evaluation"
    RETRACTION = "retraction"
    NEGATION = "negation"


EVIDENCE_SIGNAL_PHRASES: dict[EvidenceSignalKind, tuple[str, ...]] = {
    EvidenceSignalKind.CONFIRMATION: (
        "formally change",
        "continue using",
        "remains active",
        "confirmed",
        "confirm",
        "decide",
        "decided",
        "adopt",
        "adopted",
        "正式改成",
        "继续使用",
        "保持为当前",
        "确认",
        "决定",
        "采用",
    ),
    EvidenceSignalKind.CONSTRAINT: (
        "must not",
        "do not",
        "forbidden",
        "must",
        "不得",
        "禁止",
        "不允许",
        "不要",
        "必须",
    ),
    EvidenceSignalKind.PREFERENCE: (
        "preference",
        "do not like",
        "dislike",
        "prefer",
        "like",
        "不喜欢",
        "偏好",
        "喜欢",
    ),
    EvidenceSignalKind.PROPOSAL: (
        "recommended",
        "recommend",
        "can consider",
        "consider",
        "might",
        "possible",
        "may",
        "可以考虑",
        "以后考虑",
        "候选",
        "建议",
    ),
    EvidenceSignalKind.EVALUATION: (
        "can evaluate",
        "evaluate",
        "可以评估",
        "评估",
    ),
    EvidenceSignalKind.RETRACTION: (
        "no longer",
        "retract",
        "revoke",
        "不再",
        "取消",
        "撤回",
    ),
    EvidenceSignalKind.NEGATION: (
        "not confirmed",
        "unconfirmed",
        "did not approve",
        "没有同意",
        "还没有确认",
        "尚未确认",
        "未确认",
    ),
}


@dataclass(frozen=True)
class EvidenceSignalMatch:
    """保存一次词法命中的位置和上下文状态。"""

    kind: EvidenceSignalKind
    phrase: str
    start: int
    end: int
    negated: bool
    hypothetical: bool
    quoted: bool
    metalinguistic: bool
    confidence: float


class EvidenceSignalMatcher:
    """找出词法信号，同时标记否定、假设、引用和元语言。"""

    _HYPOTHETICAL_RE = re.compile(r"(?i)(?:\bif\b|\blater\b|如果|假如|以后).{0,32}$")
    _NEGATION_RE = re.compile(r"(?i)(?:\bnot\b|\bno\b|\bnever\b|\bdid\s+not\b|不|未|没有|尚未).{0,16}$")
    _POST_NEGATION_RE = re.compile(r"(?i)(?:did\s+not\s+approve|not\s+approved|没有同意|未同意|没有批准|未批准)")
    _META_RE = re.compile(r"(?i)(interpret|phrase|example|counterexample|理解成|词语|短语|文档|反例|示例)")
    _ATTRIBUTED_RE = re.compile(
        r"(?i)(?:codex|agent|assistant|模型|助手).{0,24}(?:said|says|recommended|recommend|说|建议)"
    )
    _QUOTE_PAIRS = (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"))

    def match(self, text: str) -> tuple[EvidenceSignalMatch, ...]:
        """处理 match 这一步。"""

        matches: list[EvidenceSignalMatch] = []
        for kind, phrases in EVIDENCE_SIGNAL_PHRASES.items():
            for phrase in phrases:
                pattern = self._pattern(phrase)
                for found in pattern.finditer(text):
                    start, end = found.span()
                    before = text[max(0, start - 48) : start]
                    sentence = self._sentence(text, start, end)
                    after = text[end : min(len(text), end + 64)]
                    explicitly_negative = kind == EvidenceSignalKind.NEGATION
                    matches.append(
                        EvidenceSignalMatch(
                            kind=kind,
                            phrase=found.group(0),
                            start=start,
                            end=end,
                            negated=(
                                explicitly_negative
                                or bool(self._NEGATION_RE.search(before))
                                or bool(self._POST_NEGATION_RE.search(after))
                            ),
                            hypothetical=bool(self._HYPOTHETICAL_RE.search(before)),
                            quoted=self._quoted(text, start, end),
                            metalinguistic=bool(
                                self._META_RE.search(sentence) or self._ATTRIBUTED_RE.search(sentence)
                            ),
                            confidence=0.95 if not explicitly_negative else 0.99,
                        )
                    )
        return tuple(sorted(matches, key=lambda item: (item.start, -(item.end - item.start), item.kind.value)))

    def _pattern(self, phrase: str) -> re.Pattern[str]:
        escaped = re.escape(phrase)
        if phrase.isascii():
            return re.compile(rf"(?i)(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])")
        return re.compile(escaped, re.IGNORECASE)

    def _sentence(self, text: str, start: int, end: int) -> str:
        left = max(text.rfind(mark, 0, start) for mark in ("。", "！", "？", ".", "!", "?", "\n")) + 1
        positions = [position for mark in ("。", "！", "？", ".", "!", "?", "\n") if (position := text.find(mark, end)) >= 0]
        right = min(positions) if positions else len(text)
        return text[left:right]

    def _quoted(self, text: str, start: int, end: int) -> bool:
        for opening, closing in self._QUOTE_PAIRS:
            left = text.rfind(opening, 0, start)
            if left < 0:
                continue
            right = text.find(closing, end)
            if right >= 0 and (opening != closing or text.count(opening, left, start + 1) % 2 == 1):
                return True
        return False


@dataclass(frozen=True)
class EvidenceRef:
    """指向原始事件或其中一段经过哈希校验的文本。"""

    event_id: str
    source_uri: str | None
    content_hash: str
    span_start: int | None = None
    span_end: int | None = None
    quoted_text_hash: str | None = None

    @classmethod
    def from_event(
        cls,
        event: EventEnvelope,
        *,
        source_uri: str | None = None,
        span_start: int | None = None,
        span_end: int | None = None,
    ) -> EvidenceRef:
        """从原始事件生成带内容哈希和可选文本片段的 EvidenceRef。"""

        text = event.text()
        quoted_hash = None
        if span_start is not None and span_end is not None:
            quoted_hash = evidence_hash(text[span_start:span_end])
        return cls(
            event_id=event.event_id,
            source_uri=source_uri,
            content_hash=evidence_hash(text),
            span_start=span_start,
            span_end=span_end,
            quoted_text_hash=quoted_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_uri": self.source_uri,
            "content_hash": self.content_hash,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "quoted_text_hash": self.quoted_text_hash,
        }


def bind_field_evidence(
    identity_fields: Mapping[str, Any],
    value_fields: Mapping[str, Any],
    evidence_refs: tuple[EvidenceRef, ...],
) -> dict[str, tuple[EvidenceRef, ...]]:
    """给原子提案里的身份、值和状态依据分别绑定证据。"""

    return {
        **{f"identity.{key}": evidence_refs for key in identity_fields},
        **{f"value.{key}": evidence_refs for key in value_fields},
        "semantic.speech_act": evidence_refs,
        "semantic.temporal_scope": evidence_refs,
        "transition": evidence_refs,
    }


@dataclass(frozen=True)
class ProposalValidationResult:
    """保存校验后的提案、错误原因和缺少证据的字段。"""

    valid: bool
    proposal: MemorySemanticProposal
    errors: tuple[str, ...] = ()
    unsupported_fields: tuple[str, ...] = ()


class ProposalEvidenceValidator:
    """拿原始事件核对提案字段、角色和语义是否站得住。"""

    def __init__(self, signal_matcher: EvidenceSignalMatcher | None = None) -> None:
        self.signal_matcher = signal_matcher or EvidenceSignalMatcher()

    def validate(self, proposal: MemorySemanticProposal, episode: EvidenceEpisode) -> ProposalValidationResult:
        """检查输入是否满足这里的约束。"""

        errors: list[str] = []
        evidence_texts: list[str] = []
        evidence_text_by_ref: dict[EvidenceRef, str] = {}
        evidence_actor_kinds: list[str] = []
        if not proposal.evidence_refs:
            errors.append("missing_evidence")
        for ref in proposal.evidence_refs:
            event = episode.event(ref.event_id)
            if event is None:
                errors.append(f"unknown_event:{ref.event_id}")
                continue
            text = event.text()
            evidence_actor_kinds.append(event.actor.kind)
            if evidence_hash(text) != ref.content_hash:
                errors.append(f"content_hash_mismatch:{ref.event_id}")
                continue
            if (ref.span_start is None) != (ref.span_end is None):
                errors.append(f"incomplete_span:{ref.event_id}")
                continue
            selected = text
            if ref.span_start is not None and ref.span_end is not None:
                if ref.span_start < 0 or ref.span_end <= ref.span_start or ref.span_end > len(text):
                    errors.append(f"invalid_span:{ref.event_id}")
                    continue
                selected = text[ref.span_start : ref.span_end]
                if ref.quoted_text_hash and evidence_hash(selected) != ref.quoted_text_hash:
                    errors.append(f"quoted_text_hash_mismatch:{ref.event_id}")
                    continue
            evidence_texts.append(selected)
            evidence_text_by_ref[ref] = selected

        errors.extend(self._field_evidence_errors(proposal, evidence_text_by_ref))
        unsupported = self._unsupported_fields(proposal, evidence_texts, evidence_text_by_ref)
        hardened = proposal
        if unsupported and proposal.epistemic_status in {EpistemicStatus.EXPLICIT, EpistemicStatus.OBSERVED}:
            hardened = replace(proposal, epistemic_status=EpistemicStatus.INFERRED)
            errors.append("unsupported_core_fields")
        semantic_errors = self._semantic_errors(proposal, evidence_actor_kinds, evidence_text_by_ref)
        if semantic_errors and hardened.epistemic_status == EpistemicStatus.EXPLICIT:
            hardened = replace(hardened, epistemic_status=EpistemicStatus.INFERRED)
        errors.extend(semantic_errors)
        return ProposalValidationResult(
            valid=not errors,
            proposal=hardened,
            errors=tuple(dict.fromkeys(errors)),
            unsupported_fields=tuple(unsupported),
        )

    def _semantic_errors(
        self,
        proposal: MemorySemanticProposal,
        actor_kinds: list[str],
        evidence_text_by_ref: Mapping[EvidenceRef, str],
    ) -> list[str]:
        errors = []
        if proposal.epistemic_status == EpistemicStatus.EXPLICIT and not any(
            kind in {"user", "system"} for kind in actor_kinds
        ):
            errors.append("explicit_status_requires_authoritative_evidence")
        semantic = proposal.semantic
        speech = str(getattr(semantic.speech_act, "value", semantic.speech_act)).casefold()
        commitment = str(getattr(semantic.commitment, "value", semantic.commitment)).casefold()
        temporal_scope = str(getattr(semantic.temporal_scope, "value", semantic.temporal_scope)).casefold()
        semantic_refs = tuple(proposal.field_evidence_refs.get("transition", ())) or tuple(
            proposal.field_evidence_refs.get("semantic.speech_act", ())
        )
        semantic_texts = [evidence_text_by_ref[ref] for ref in semantic_refs if ref in evidence_text_by_ref]
        matches = tuple(match for text in semantic_texts for match in self.signal_matcher.match(text))
        usable = tuple(
            match
            for match in matches
            if not (match.negated or match.hypothetical or match.quoted or match.metalinguistic)
        )
        if speech in {"confirmation", "correction"} or commitment in {"confirmed", "committed"}:
            compatible = {
                "preference": {EvidenceSignalKind.PREFERENCE, EvidenceSignalKind.CONFIRMATION},
                "project_rule": {EvidenceSignalKind.CONSTRAINT, EvidenceSignalKind.CONFIRMATION},
                "project_decision": {EvidenceSignalKind.CONFIRMATION},
                "profile": {EvidenceSignalKind.CONFIRMATION},
            }.get(proposal.memory_type, {EvidenceSignalKind.CONFIRMATION})
            if not any(match.kind in compatible for match in usable):
                errors.append("semantic_confirmation_unsupported")
        if speech == "retraction" and not any(match.kind == EvidenceSignalKind.RETRACTION for match in usable):
            errors.append("semantic_retraction_unsupported")
        if temporal_scope == "future":
            temporal_refs = tuple(proposal.field_evidence_refs.get("semantic.temporal_scope", ()))
            text = "\n".join(evidence_text_by_ref[ref] for ref in temporal_refs if ref in evidence_text_by_ref)
            if not re.search(r"(?i)(\bfuture\b|\blater\b|\bwill\b|以后|未来|稍后|届时)", text):
                errors.append("temporal_scope_unsupported")
        return errors

    def _unsupported_fields(
        self,
        proposal: MemorySemanticProposal,
        evidence_texts: list[str],
        evidence_text_by_ref: Mapping[EvidenceRef, str],
    ) -> list[str]:
        if not evidence_texts:
            return [
                *[f"identity.{key}" for key in proposal.identity_fields],
                *[f"value.{key}" for key in proposal.value_fields],
            ]
        unsupported = []
        system_identity_fields = {str(item) for item in proposal.metadata.get("system_identity_fields", []) or []}
        for prefix, fields in (("identity", proposal.identity_fields), ("value", proposal.value_fields)):
            for key, value in fields.items():
                if prefix == "identity" and key in system_identity_fields:
                    continue
                binding = f"{prefix}.{key}"
                field_texts = [
                    evidence_text_by_ref[ref]
                    for ref in proposal.field_evidence_refs.get(binding, ())
                    if ref in evidence_text_by_ref
                ]
                haystack = "\n".join(field_texts).casefold()
                if not self._supported(value, haystack) and not self._semantically_supported(key, value, field_texts):
                    unsupported.append(f"{prefix}.{key}")
        return unsupported

    def _field_evidence_errors(
        self,
        proposal: MemorySemanticProposal,
        evidence_text_by_ref: Mapping[EvidenceRef, str],
    ) -> list[str]:
        required = {
            *[f"identity.{key}" for key in proposal.identity_fields],
            *[f"value.{key}" for key in proposal.value_fields],
            "semantic.speech_act",
            "semantic.temporal_scope",
            "transition",
        }
        errors = []
        for field_name in sorted(required):
            refs = tuple(proposal.field_evidence_refs.get(field_name, ()))
            if not refs:
                errors.append(f"missing_field_evidence:{field_name}")
            elif any(ref not in evidence_text_by_ref for ref in refs):
                errors.append(f"invalid_field_evidence:{field_name}")
        unknown = set(proposal.field_evidence_refs) - required
        if unknown:
            errors.append(f"unknown_field_evidence:{','.join(sorted(unknown))}")
        return errors

    def _semantically_supported(self, key: str, value: Any, evidence_texts: list[str]) -> bool:
        if key != "canonical_value" or str(value).casefold() not in {"forbidden", "required"}:
            return False
        usable = tuple(
            match
            for text in evidence_texts
            for match in self.signal_matcher.match(text)
            if not (match.negated or match.hypothetical or match.quoted or match.metalinguistic)
        )
        return any(match.kind == EvidenceSignalKind.CONSTRAINT for match in usable)

    def _supported(self, value: Any, haystack: str) -> bool:
        if value is None or value == "":
            return False
        if isinstance(value, bool | int | float):
            return str(value).casefold() in haystack
        if isinstance(value, str):
            candidates = {value.casefold(), value.replace("_", " ").casefold(), value.replace("-", " ").casefold()}
            return any(candidate in haystack for candidate in candidates)
        if isinstance(value, Mapping):
            return all(self._supported(item, haystack) for item in value.values())
        if isinstance(value, list | tuple | set):
            return all(self._supported(item, haystack) for item in value)
        return str(value).casefold() in haystack
