"""记忆系统里的证据。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from memoryos.memory.canonical.episode import EvidenceEpisode
from memoryos.memory.canonical.event import EventEnvelope, canonical_json
from memoryos.memory.canonical.proposal import EpistemicStatus, MemorySemanticProposal

_V3_SEMANTIC_BINDINGS = (
    "semantic.utterance_mode",
    "semantic.attribution",
    "semantic.durability",
    "semantic.modal_force",
    "semantic.atomicity",
)
_V3_BASE_SEMANTIC_BINDINGS = (
    "semantic.speech_act",
    "semantic.commitment",
    "semantic.temporal_scope",
    "semantic.relation_to_existing",
)
_V3_ATOMIC_BINDINGS = ("transition", *_V3_BASE_SEMANTIC_BINDINGS, *_V3_SEMANTIC_BINDINGS)


def _semantic_contract_version(proposal: MemorySemanticProposal) -> str:
    return str(getattr(proposal, "semantic_contract_version", "v2") or "v2").strip().casefold()


def _semantic_value(value: object) -> str:
    return str(getattr(value, "value", value) or "").strip().upper()

_REPLACEMENT_CUE_RE = re.compile(
    r"(?i)(?:\bformally\s+change\b|\bswitch(?:ed|ing)?\s+from\b.{0,64}\bto\b|"
    r"\breplace(?:d|s|ment)?\b.{0,48}\b(?:with|by)\b|\bsupersed(?:e|es|ed|ing)\b|"
    r"\bno\s+longer\b.{0,48}\b(?:use|using)\b.{0,64}\b(?:use|using|adopt)\b|"
    r"正式.{0,12}(?:改成|改为|替换为|切换到)|改成|改为|改用|替换为|切换到|"
    r"从.{1,48}切换到.{1,48}|不再使用.{1,48}(?:改用|改为|使用)|"
    r"撤销之前|之前.{0,32}作废|取代)"
)
_REPLACEMENT_NEGATION_RE = re.compile(
    r"(?i)(?:不是|并非|没有|未曾|不打算|不准备|否认|\bnot\b|\bnever\b|\bdid\s+not\b|"
    r"\bdoes\s+not\b).{0,48}$"
)
_REPLACEMENT_DISCUSSION_RE = re.compile(
    r"(?i)(?:讨论|考虑|评估|设想|假设|提议|建议|可能|也许|是否|"
    r"\bdiscuss(?:ed|ing)?\b|\bconsider(?:ed|ing)?\b|\bevaluat(?:e|ed|ing)\b|"
    r"\bsuggest(?:ed|ing)?\b|\bpropos(?:e|ed|ing)\b|\bmight\b|\bmaybe\b|\bwhether\b)"
)
_REPLACEMENT_COMMITMENT_RE = re.compile(
    r"(?i)(?:最终|最后|正式|现在|当前|通过|批准|确认|"
    r"\bfinally\b|\bformally\b|\bnow\b|\bdecid(?:e|ed)\b|\bapproved?\b|\bpassed?\b)"
)
_REPLACEMENT_KEEP_OVERRIDE_RE = re.compile(
    r"(?i)(?:不过|但是|但|而是|最终|最后|\bhowever\b|\bbut\b|\byet\b).{0,96}"
    r"(?:继续使用|仍然(?:保持|使用)|保持不变|决定保留|不切换|"
    r"\bcontinue\s+using\b|\bkeep\b|\bremain(?:s|ed)?\b|\bdo\s+not\s+switch\b)"
)
_ATTRIBUTED_SPEECH_RE = re.compile(
    r"(?i)(?:他说|她说|他们说|有人(?:说|建议|要求|认为)|"
    r"我(?:曾经?)?听说|听说|据说|传闻|\bi\s+heard\b|\breportedly\b|\baccording\s+to\b|"
    r"(?:[A-Za-z0-9_\u4e00-\u9fff]{1,20})\s*(?:说|表示|声称|认为|建议|提到)|"
    r"he\s+(?:said|suggested|required)|she\s+(?:said|suggested|required)|"
    r"they\s+(?:said|suggested|required)|someone\s+(?:said|suggested|required|recommended)|"
    r"(?:[A-Za-z0-9_]{1,32})\s+(?:said|says|suggested|recommended|claimed))"
)
_USER_RESOLUTION_RE = re.compile(
    r"(?i)(?:但|但是|不过|而我|\bbut\b).{0,80}(?:我|\bwe\b)"
    r".{0,32}(?:最终(?:决定)?|最后(?:决定)?|决定(?:使用|采用)|正式(?:改成|改为)|"
    r"decid(?:e|ed)|adopt(?:ed)?)"
)


def has_explicit_replacement_cue(text: str) -> bool:
    """Return true only for an affirmed, final replacement proposition."""

    for match in _REPLACEMENT_CUE_RE.finditer(text):
        before = text[max(0, match.start() - 96) : match.start()]
        after = text[match.end() :]
        if _REPLACEMENT_NEGATION_RE.search(before):
            continue
        discussions = tuple(_REPLACEMENT_DISCUSSION_RE.finditer(before))
        commitments = tuple(_REPLACEMENT_COMMITMENT_RE.finditer(before))
        if discussions and (not commitments or discussions[-1].start() > commitments[-1].start()):
            continue
        if _REPLACEMENT_KEEP_OVERRIDE_RE.search(after):
            continue
        return True
    return False


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


class ConstraintPolarity(str, Enum):
    """Direction carried by one explicit constraint or preference signal."""

    REQUIRE = "REQUIRE"
    FORBID = "FORBID"
    ALLOW = "ALLOW"
    PREFER = "PREFER"
    DISCOURAGE = "DISCOURAGE"
    CONDITIONAL_REQUIRE = "CONDITIONAL_REQUIRE"
    CONDITIONAL_FORBID = "CONDITIONAL_FORBID"


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
        "正式改为",
        "替换为",
        "切换到",
        "改成",
        "改为",
        "继续使用",
        "保持为当前",
        "确认",
        "决定",
        "采用",
    ),
    EvidenceSignalKind.CONSTRAINT: (
        "must not",
        "should not",
        "do not",
        "forbidden",
        "prohibited",
        "not allowed",
        "must",
        "required",
        "require",
        "need to",
        "needs to",
        "may use",
        "can use",
        "allowed",
        "permitted",
        "avoid",
        "discouraged",
        "不得",
        "禁止",
        "不允许",
        "不要",
        "尽量不要",
        "不建议",
        "避免",
        "必须",
        "需要",
        "要求",
        "可以使用",
        "允许使用",
        "允许",
        "可使用",
    ),
    EvidenceSignalKind.PREFERENCE: (
        "preference",
        "preferably",
        "should use",
        "do not like",
        "dislike",
        "prefer",
        "like",
        "不喜欢",
        "偏好",
        "优先",
        "最好",
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
    polarity: ConstraintPolarity | None = None


class EvidenceSignalMatcher:
    """找出词法信号，同时标记否定、假设、引用和元语言。"""

    _HYPOTHETICAL_RE = re.compile(r"(?i)(?:\bif\b|\blater\b|如果|假如|以后).{0,32}$")
    _NEGATION_RE = re.compile(r"(?i)(?:\bnot\b|\bno\b|\bnever\b|\bdid\s+not\b|不|未|没有|尚未).{0,16}$")
    _POST_NEGATION_RE = re.compile(r"(?i)(?:did\s+not\s+approve|not\s+approved|没有同意|未同意|没有批准|未批准)")
    _META_RE = re.compile(r"(?i)(interpret|phrase|example|counterexample|理解成|词语|短语|文档|反例|示例)")
    _CONDITIONAL_BEFORE_RE = re.compile(
        r"(?i)(?:\bif\b|\bwhen\b|\bwhenever\b|\bprovided\s+that\b|\bonly\s+if\b|"
        r"\bunless\b|\bexcept(?:\s+when|\s+for)?\b|如果|若|当|仅当|只有|只要|除非).{0,64}$"
    )
    _EXCEPTION_AFTER_RE = re.compile(
        r"(?i)(?:\bunless\b|\bexcept(?:\s+when|\s+for)?\b|\bonly\s+if\b|除非|例外|仅限|只有)"
    )
    _CONTRAST_BEFORE_RE = re.compile(r"(?i)(?:\bbut\b|\brather\b|\binstead\b|而是|但是|但|却)\s*$")
    _QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))

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
                    negated = explicitly_negative or (
                        not self._CONTRAST_BEFORE_RE.search(before)
                        and (bool(self._NEGATION_RE.search(before)) or bool(self._POST_NEGATION_RE.search(after)))
                    )
                    matches.append(
                        EvidenceSignalMatch(
                            kind=kind,
                            phrase=found.group(0),
                            start=start,
                            end=end,
                            negated=negated,
                            hypothetical=bool(self._HYPOTHETICAL_RE.search(before)),
                            quoted=self._quoted(text, start, end),
                            metalinguistic=bool(
                                self._META_RE.search(sentence) or self._attributed_signal(text, start)
                            ),
                            confidence=0.95 if not explicitly_negative else 0.99,
                            polarity=self._polarity(kind, found.group(0), before, after),
                        )
                    )
        ordered = sorted(matches, key=lambda item: (item.start, -(item.end - item.start), item.kind.value))
        return tuple(
            match
            for match in ordered
            if not any(
                other is not match
                and match.kind == EvidenceSignalKind.CONSTRAINT
                and other.kind == match.kind
                and other.start <= match.start
                and other.end >= match.end
                and (other.end - other.start) > (match.end - match.start)
                for other in ordered
            )
        )

    def _polarity(
        self,
        kind: EvidenceSignalKind,
        phrase: str,
        before: str,
        after: str,
    ) -> ConstraintPolarity | None:
        normalized = re.sub(r"\s+", " ", phrase.strip().casefold())
        forbid = {
            "must not",
            "do not",
            "forbidden",
            "prohibited",
            "not allowed",
            "不得",
            "禁止",
            "不允许",
            "不要",
        }
        require = {
            "must",
            "required",
            "require",
            "need to",
            "needs to",
            "必须",
            "需要",
            "要求",
        }
        allow = {
            "may use",
            "can use",
            "allowed",
            "permitted",
            "可以使用",
            "允许使用",
            "允许",
            "可使用",
        }
        discourage = {
            "should not",
            "avoid",
            "discouraged",
            "尽量不要",
            "不建议",
            "避免",
            "do not like",
            "dislike",
            "不喜欢",
        }
        prefer = {
            "preference",
            "preferably",
            "should use",
            "prefer",
            "like",
            "偏好",
            "优先",
            "最好",
            "喜欢",
        }
        if normalized in forbid:
            conditional = bool(self._CONDITIONAL_BEFORE_RE.search(before) or self._EXCEPTION_AFTER_RE.search(after))
            return ConstraintPolarity.CONDITIONAL_FORBID if conditional else ConstraintPolarity.FORBID
        if normalized in require:
            conditional = bool(self._CONDITIONAL_BEFORE_RE.search(before) or self._EXCEPTION_AFTER_RE.search(after))
            return ConstraintPolarity.CONDITIONAL_REQUIRE if conditional else ConstraintPolarity.REQUIRE
        if normalized in allow:
            return ConstraintPolarity.ALLOW
        if normalized in discourage:
            return ConstraintPolarity.DISCOURAGE
        if kind == EvidenceSignalKind.PREFERENCE and normalized in prefer:
            return ConstraintPolarity.PREFER
        return None

    def _pattern(self, phrase: str) -> re.Pattern[str]:
        escaped = re.escape(phrase)
        if phrase.isascii():
            return re.compile(rf"(?i)(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])")
        return re.compile(escaped, re.IGNORECASE)

    def _sentence(self, text: str, start: int, end: int) -> str:
        left = max(text.rfind(mark, 0, start) for mark in ("。", "！", "？", ".", "!", "?", "\n")) + 1
        positions = [
            position for mark in ("。", "！", "？", ".", "!", "?", "\n") if (position := text.find(mark, end)) >= 0
        ]
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

    def _attributed_signal(self, text: str, signal_start: int) -> bool:
        for attributed in _ATTRIBUTED_SPEECH_RE.finditer(text):
            if attributed.start() > signal_start:
                break
            between = text[attributed.end() : signal_start]
            if re.search(r"[。！？.!?\n]", between):
                continue
            resolution = _USER_RESOLUTION_RE.search(text, attributed.end())
            if resolution is not None and resolution.start() <= signal_start:
                continue
            return True
        return False


@dataclass(frozen=True)
class EvidenceRef:
    """A verifiable reference to an immutable event or one exact field span."""

    event_id: str
    source_uri: str | None
    content_hash: str
    span_start: int | None = None
    span_end: int | None = None
    quoted_text_hash: str | None = None
    event_digest: str | None = None
    event_schema_version: str | None = None
    tenant_id: str | None = None
    episode_id: str | None = None
    actor_id: str | None = None
    actor_kind: str | None = None
    actor_role: str | None = None
    actor_id_inferred: bool | None = None
    actor_role_inferred: bool | None = None
    subject_refs: tuple[str, ...] = ()
    content_path: str | None = None
    quoted_text: str | None = None
    occurred_at: str | None = None
    ingested_at: str | None = None
    sequence: int | None = None
    evidence_strength: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_refs", tuple(str(item) for item in self.subject_refs))

    @classmethod
    def from_event(
        cls,
        event: EventEnvelope,
        *,
        source_uri: str | None = None,
        content_path: str | None = None,
        span_start: int | None = None,
        span_end: int | None = None,
    ) -> EvidenceRef:
        """Build a strong V2 reference exclusively from the immutable envelope."""

        path = content_path or event.content_path
        text = event.text(path)
        quoted_hash = None
        quoted_text = None
        if (span_start is None) != (span_end is None):
            raise ValueError("evidence span requires both start and end")
        if span_start is not None and span_end is not None:
            if span_start < 0 or span_end <= span_start or span_end > len(text):
                raise ValueError("evidence span is outside the selected content")
            quoted_text = text[span_start:span_end]
            quoted_hash = evidence_hash(quoted_text)
        inferred = (
            event.actor.inferred
            or any(subject.inferred for subject in event.subjects)
            or event.occurred_at_inferred
            or event.ingested_at_inferred
            or event.sequence_inferred
        )
        return cls(
            event_id=event.event_id,
            source_uri=source_uri,
            content_hash=evidence_hash(text),
            span_start=span_start,
            span_end=span_end,
            quoted_text_hash=quoted_hash,
            event_digest=event.digest,
            event_schema_version=event.schema_version,
            tenant_id=event.tenant_id,
            episode_id=event.episode_id,
            actor_id=event.actor.id,
            actor_kind=event.actor.kind,
            actor_role=event.actor.role,
            actor_id_inferred=event.actor.id_inferred,
            actor_role_inferred=event.actor.role_inferred,
            subject_refs=tuple(canonical_json(subject.to_dict()) for subject in event.subjects),
            content_path=path,
            quoted_text=quoted_text,
            occurred_at=event.occurred_at.isoformat(),
            ingested_at=(event.ingested_at or event.occurred_at).isoformat(),
            sequence=event.sequence,
            evidence_strength="INFERRED" if inferred else "EXPLICIT",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_uri": self.source_uri,
            "content_hash": self.content_hash,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "quoted_text_hash": self.quoted_text_hash,
            "event_digest": self.event_digest,
            "event_schema_version": self.event_schema_version,
            "tenant_id": self.tenant_id,
            "episode_id": self.episode_id,
            "actor_id": self.actor_id,
            "actor_kind": self.actor_kind,
            "actor_role": self.actor_role,
            "actor_id_inferred": self.actor_id_inferred,
            "actor_role_inferred": self.actor_role_inferred,
            "subject_refs": list(self.subject_refs),
            "content_path": self.content_path,
            "quoted_text": self.quoted_text,
            "occurred_at": self.occurred_at,
            "ingested_at": self.ingested_at,
            "sequence": self.sequence,
            "evidence_strength": self.evidence_strength,
        }


def bind_field_evidence(
    identity_fields: Mapping[str, Any],
    value_fields: Mapping[str, Any],
    evidence_refs: tuple[EvidenceRef, ...],
    *,
    bindings: Mapping[str, tuple[EvidenceRef, ...]] | None = None,
    semantic_contract_version: str = "v2",
) -> dict[str, tuple[EvidenceRef, ...]]:
    """Validate an explicit field-to-evidence map.

    Proposal-level evidence is deliberately not copied to every semantic field.
    Callers that cannot identify field evidence must leave the binding empty so
    admission fails closed.
    """

    defaults = {
        **{f"identity.{key}": evidence_refs for key in identity_fields},
        **{f"value.{key}": evidence_refs for key in value_fields},
        "semantic.speech_act": evidence_refs,
        "semantic.commitment": evidence_refs,
        "semantic.temporal_scope": evidence_refs,
        "semantic.relation_to_existing": evidence_refs,
        "transition": evidence_refs,
    }
    if str(semantic_contract_version).strip().casefold() == "v3":
        defaults.update({field_name: evidence_refs for field_name in _V3_SEMANTIC_BINDINGS})
    if bindings is None:
        raise ValueError("field evidence bindings require explicit field bindings")
    missing = set(defaults) - set(bindings)
    unknown = set(bindings) - set(defaults)
    if missing or unknown:
        details = [
            *(f"missing:{key}" for key in sorted(missing)),
            *(f"unknown:{key}" for key in sorted(unknown)),
        ]
        raise ValueError(f"field evidence bindings mismatch: {','.join(details)}")
    return {key: tuple(bindings[key]) for key in defaults}


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
        evidence_actor_roles: list[str] = []
        atomic_errors: list[str] = []
        if not proposal.evidence_refs:
            errors.append("missing_evidence")
        for ref in proposal.evidence_refs:
            event = episode.event(ref.event_id)
            if event is None:
                errors.append(f"unknown_event:{ref.event_id}")
                continue
            reference_errors = self._reference_errors(ref, event, episode)
            if reference_errors:
                errors.extend(reference_errors)
                continue
            try:
                text = event.text(ref.content_path)
            except ValueError:
                errors.append(f"content_path_mismatch:{ref.event_id}")
                continue
            evidence_actor_kinds.append(event.actor.kind)
            evidence_actor_roles.append(str(event.actor.role or event.actor.kind))
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
                if not ref.quoted_text_hash or evidence_hash(selected) != ref.quoted_text_hash:
                    errors.append(f"quoted_text_hash_mismatch:{ref.event_id}")
                    continue
                if ref.quoted_text is None or selected != ref.quoted_text:
                    errors.append(f"quoted_text_mismatch:{ref.event_id}")
                    continue
            elif ref.quoted_text_hash is not None or ref.quoted_text is not None:
                errors.append(f"unexpected_quote_without_span:{ref.event_id}")
                continue
            evidence_texts.append(selected)
            evidence_text_by_ref[ref] = selected

        if _semantic_contract_version(proposal) == "v3":
            atomic_errors = self._atomic_evidence_errors(proposal, evidence_text_by_ref)
            errors.extend(atomic_errors)
        field_errors = self._field_evidence_errors(proposal, evidence_text_by_ref)
        errors.extend(field_errors)
        unsupported = self._unsupported_fields(proposal, evidence_texts, evidence_text_by_ref)
        hardened = proposal
        if unsupported and proposal.epistemic_status in {EpistemicStatus.EXPLICIT, EpistemicStatus.OBSERVED}:
            hardened = replace(proposal, epistemic_status=EpistemicStatus.INFERRED)
            errors.append("unsupported_core_fields")
        semantic_errors = self._semantic_errors(proposal, evidence_actor_kinds, evidence_text_by_ref)
        declared_role = str(proposal.metadata.get("source_role") or "").strip().casefold()
        declared_role = "assistant" if declared_role == "agent" else declared_role
        if _semantic_contract_version(proposal) == "v3":
            transition_ref = self._v3_transition_ref(proposal)
            actual_role = str(getattr(transition_ref, "actor_role", "") or "").strip().casefold()
            if not actual_role:
                actual_role = str(getattr(transition_ref, "actor_kind", "") or "").strip().casefold()
            actual_roles = {"assistant" if actual_role == "agent" else actual_role} if actual_role else set()
        else:
            actual_roles = {"assistant" if role == "agent" else role for role in evidence_actor_roles}
        if declared_role and actual_roles and declared_role not in actual_roles:
            semantic_errors.append("source_role_evidence_mismatch")
        if semantic_errors and hardened.epistemic_status == EpistemicStatus.EXPLICIT:
            hardened = replace(hardened, epistemic_status=EpistemicStatus.INFERRED)
        errors.extend(semantic_errors)
        metadata = dict(hardened.metadata)
        metadata["transition_evidence_validated"] = bool(
            proposal.field_evidence_refs.get("transition")
            and not semantic_errors
            and not atomic_errors
        )
        relation_evidence_validated = self._relation_evidence_supported(
            proposal,
            evidence_text_by_ref,
        )
        metadata["semantic_relation_evidence_validated"] = relation_evidence_validated
        relation = str(
            getattr(proposal.semantic.relation_to_existing, "value", proposal.semantic.relation_to_existing)
        ).casefold()
        metadata["replacement_evidence_validated"] = bool(
            relation in {"corrects", "supersedes"} and relation_evidence_validated
        )
        if _semantic_contract_version(proposal) == "v3":
            metadata["atomic_evidence_validated"] = not atomic_errors
            metadata["semantic_contract_validated"] = not errors and not unsupported
        hardened = replace(hardened, metadata=metadata)
        return ProposalValidationResult(
            valid=not errors,
            proposal=hardened,
            errors=tuple(dict.fromkeys(errors)),
            unsupported_fields=tuple(unsupported),
        )

    def _atomic_evidence_errors(
        self,
        proposal: MemorySemanticProposal,
        evidence_text_by_ref: Mapping[EvidenceRef, str],
    ) -> list[str]:
        """Validate the single source-grounded proposition span required by V3."""

        atomic = getattr(proposal, "atomic_evidence_ref", None)
        if atomic is None:
            return ["missing_atomic_evidence_ref"]
        errors: list[str] = []
        matching = tuple(ref for ref in proposal.evidence_refs if ref == atomic)
        if not matching:
            errors.append("atomic_evidence_ref_not_in_evidence")
        elif len(matching) != 1:
            errors.append("atomic_evidence_ref_not_unique")
        if atomic.span_start is None or atomic.span_end is None:
            errors.append("atomic_evidence_requires_exact_span")
        if atomic not in evidence_text_by_ref and matching:
            errors.append("atomic_evidence_ref_not_validated")
        for field_name in _V3_ATOMIC_BINDINGS:
            if tuple(proposal.field_evidence_refs.get(field_name, ())) != (atomic,):
                errors.append(f"atomic_evidence_binding_mismatch:{field_name}")
        return errors

    def _v3_transition_ref(self, proposal: MemorySemanticProposal) -> EvidenceRef | None:
        refs = tuple(proposal.field_evidence_refs.get("transition", ()))
        return refs[0] if len(refs) == 1 and refs[0] == getattr(proposal, "atomic_evidence_ref", None) else None

    def _relation_evidence_supported(
        self,
        proposal: MemorySemanticProposal,
        evidence_text_by_ref: Mapping[EvidenceRef, str],
    ) -> bool:
        relation = (
            str(
                getattr(
                    proposal.semantic.relation_to_existing,
                    "value",
                    proposal.semantic.relation_to_existing,
                )
            )
            .strip()
            .casefold()
        )
        refs = tuple(proposal.field_evidence_refs.get("semantic.relation_to_existing", ()))
        texts = [evidence_text_by_ref[ref] for ref in refs if ref in evidence_text_by_ref]
        if not refs or len(texts) != len(refs):
            return False
        if _semantic_contract_version(proposal) == "v3":
            atomic = getattr(proposal, "atomic_evidence_ref", None)
            if atomic is None or atomic not in evidence_text_by_ref or atomic not in refs:
                return False
            related = bool(proposal.all_related_memory_ids)
            if relation == "unrelated":
                return not related
            if relation in {"duplicate", "contradicts", "corrects", "supersedes"}:
                return related
            if relation == "supplements":
                return related or proposal.memory_type == "agent_experience"
            return relation == "alternative"
        text = "\n".join(texts)
        related = bool(proposal.all_related_memory_ids)
        usable_signals = tuple(
            match
            for item in texts
            for match in self.signal_matcher.match(item)
            if not (match.negated or match.hypothetical or match.quoted or match.metalinguistic)
        )
        if relation in {"unrelated", "duplicate"}:
            return not related
        if relation == "alternative":
            return any(
                match.kind in {EvidenceSignalKind.PROPOSAL, EvidenceSignalKind.EVALUATION} for match in usable_signals
            ) or bool(re.search(r"(?i)(\balternative\b|\boption\b|候选|备选|备用|可选)", text))
        if relation == "contradicts":
            return related and bool(
                re.search(
                    r"(?i)(\bcontradict|\bconflict|\binstead\b|\brather\s+than\b|\bnot\b.{0,32}\bbut\b|冲突|矛盾|而不是|改为)",
                    text,
                )
            )
        if relation == "supplements":
            return (related or proposal.memory_type == "agent_experience") and bool(
                re.search(r"(?i)(\balso\b|\badditionally\b|\bsupplement|\bin\s+addition\b|同时|补充|此外)", text)
            )
        if relation == "corrects":
            return related and bool(
                re.search(
                    r"(?i)(\bcorrect(?:ion|ed|s)?\b|\bwas\s+wrong\b|\bactually\b|"
                    r"纠正|更正|修正|之前.{0,24}错|不是.{0,32}而是)",
                    text,
                )
            )
        if relation == "supersedes":
            return related and has_explicit_replacement_cue(text)
        return False

    def _reference_errors(
        self,
        ref: EvidenceRef,
        event: EventEnvelope,
        episode: EvidenceEpisode,
    ) -> list[str]:
        errors = []
        required = {
            "event_digest": ref.event_digest,
            "event_schema_version": ref.event_schema_version,
            "tenant_id": ref.tenant_id,
            "episode_id": ref.episode_id,
            "actor_id": ref.actor_id,
            "actor_kind": ref.actor_kind,
            "actor_role": ref.actor_role,
            "actor_id_inferred": ref.actor_id_inferred,
            "actor_role_inferred": ref.actor_role_inferred,
            "content_path": ref.content_path,
            "occurred_at": ref.occurred_at,
            "ingested_at": ref.ingested_at,
            "sequence": ref.sequence,
            "evidence_strength": ref.evidence_strength,
        }
        for name, value in required.items():
            if value is None or value == "":
                errors.append(f"missing_{name}:{ref.event_id}")
        if errors:
            return errors
        if ref.event_digest != event.digest:
            errors.append(f"event_digest_mismatch:{ref.event_id}")
        if ref.event_schema_version != event.schema_version:
            errors.append(f"event_schema_mismatch:{ref.event_id}")
        if ref.tenant_id != event.tenant_id or ref.tenant_id != episode.tenant_id:
            errors.append(f"tenant_mismatch:{ref.event_id}")
        if ref.episode_id != event.episode_id or ref.episode_id != episode.episode_id:
            errors.append(f"episode_mismatch:{ref.event_id}")
        if ref.actor_id != event.actor.id:
            errors.append(f"actor_id_mismatch:{ref.event_id}")
        if ref.actor_kind != event.actor.kind:
            errors.append(f"actor_kind_mismatch:{ref.event_id}")
        if ref.actor_role != event.actor.role:
            errors.append(f"actor_role_mismatch:{ref.event_id}")
        if ref.actor_id_inferred is not None and ref.actor_id_inferred != event.actor.id_inferred:
            errors.append(f"actor_id_inference_mismatch:{ref.event_id}")
        if ref.actor_role_inferred is not None and ref.actor_role_inferred != event.actor.role_inferred:
            errors.append(f"actor_role_inference_mismatch:{ref.event_id}")
        if ref.occurred_at != event.occurred_at.isoformat():
            errors.append(f"occurred_at_mismatch:{ref.event_id}")
        if ref.ingested_at != (event.ingested_at or event.occurred_at).isoformat():
            errors.append(f"ingested_at_mismatch:{ref.event_id}")
        if ref.sequence != event.sequence:
            errors.append(f"sequence_mismatch:{ref.event_id}")
        expected_strength = (
            "INFERRED"
            if (
                event.actor.inferred
                or any(subject.inferred for subject in event.subjects)
                or event.occurred_at_inferred
                or event.ingested_at_inferred
                or event.sequence_inferred
            )
            else "EXPLICIT"
        )
        if ref.evidence_strength != expected_strength:
            errors.append(f"evidence_strength_mismatch:{ref.event_id}")
        expected_subjects = tuple(canonical_json(subject.to_dict()) for subject in event.subjects)
        if not ref.subject_refs:
            errors.append(f"missing_subject_refs:{ref.event_id}")
        elif ref.subject_refs != expected_subjects:
            errors.append(f"subject_mismatch:{ref.event_id}")
        if ref.source_uri and (not episode.source_uris or ref.source_uri != episode.source_uris[0]):
            errors.append(f"source_uri_mismatch:{ref.event_id}")
        return errors

    def _semantic_errors(
        self,
        proposal: MemorySemanticProposal,
        actor_kinds: list[str],
        evidence_text_by_ref: Mapping[EvidenceRef, str],
    ) -> list[str]:
        if _semantic_contract_version(proposal) == "v3":
            return self._v3_semantic_errors(proposal, evidence_text_by_ref)
        errors = []
        if proposal.epistemic_status == EpistemicStatus.EXPLICIT and not any(
            kind in {"user", "system"} for kind in actor_kinds
        ):
            errors.append("explicit_status_requires_authoritative_evidence")
        semantic = proposal.semantic
        speech = str(getattr(semantic.speech_act, "value", semantic.speech_act)).casefold()
        commitment = str(getattr(semantic.commitment, "value", semantic.commitment)).casefold()
        temporal_scope = str(getattr(semantic.temporal_scope, "value", semantic.temporal_scope)).casefold()
        semantic_refs = tuple(
            dict.fromkeys(
                (
                    *proposal.field_evidence_refs.get("transition", ()),
                    *proposal.field_evidence_refs.get("semantic.speech_act", ()),
                    *proposal.field_evidence_refs.get("semantic.commitment", ()),
                    *proposal.field_evidence_refs.get("semantic.relation_to_existing", ()),
                )
            )
        )
        semantic_texts = [evidence_text_by_ref[ref] for ref in semantic_refs if ref in evidence_text_by_ref]
        matches = tuple(match for text in semantic_texts for match in self.signal_matcher.match(text))
        usable = tuple(
            match
            for match in matches
            if not (
                match.negated
                or match.quoted
                or match.metalinguistic
                or (
                    match.hypothetical
                    and match.polarity
                    not in {ConstraintPolarity.CONDITIONAL_REQUIRE, ConstraintPolarity.CONDITIONAL_FORBID}
                )
            )
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

    def _v3_semantic_errors(
        self,
        proposal: MemorySemanticProposal,
        evidence_text_by_ref: Mapping[EvidenceRef, str],
    ) -> list[str]:
        """Check V3 contract consistency and fail closed on explicit surface contradictions."""

        errors: list[str] = []
        transition_ref = self._v3_transition_ref(proposal)
        if transition_ref is None or transition_ref not in evidence_text_by_ref:
            return ["semantic_transition_atomic_evidence_missing"]
        if proposal.epistemic_status == EpistemicStatus.EXPLICIT and str(
            transition_ref.actor_kind or ""
        ).casefold() not in {"user", "system"}:
            errors.append("explicit_status_requires_authoritative_evidence")
        semantic = proposal.semantic
        speech = _semantic_value(semantic.speech_act)
        commitment = _semantic_value(semantic.commitment)
        relation = _semantic_value(semantic.relation_to_existing)
        modal_force = _semantic_value(getattr(semantic, "modal_force", "UNKNOWN"))
        utterance_mode = _semantic_value(getattr(semantic, "utterance_mode", "UNKNOWN"))
        transition_text = evidence_text_by_ref[transition_ref]
        if utterance_mode in {"ASSERTION", "DIRECTIVE"} and re.search(r"[?？]", transition_text):
            errors.append("semantic_utterance_mode_conflicts_with_question_surface")
        if speech in {"RETRACTION", "REJECTION"} and relation != "CORRECTS":
            errors.append("semantic_retraction_relation_inconsistent")
        if speech in {"PROPOSAL", "EVALUATION_REQUEST"} and commitment == "CONFIRMED":
            errors.append("semantic_proposal_commitment_inconsistent")
        if (
            proposal.memory_type == "project_rule"
            and speech not in {"RETRACTION", "REJECTION"}
            and modal_force == "NONE"
        ):
            errors.append("project_rule_modal_force_missing")
        if proposal.memory_type not in {"project_rule", "preference"} and modal_force not in {
            "NONE",
            "UNKNOWN",
            "SCHEMA_MISMATCH",
        }:
            errors.append("modal_force_not_allowed_for_memory_type")
        if not self._relation_evidence_supported(proposal, evidence_text_by_ref):
            errors.append("semantic_relation_structure_invalid")
        return errors

    def _unsupported_fields(
        self,
        proposal: MemorySemanticProposal,
        evidence_texts: list[str],
        evidence_text_by_ref: Mapping[EvidenceRef, str],
    ) -> list[str]:
        if _semantic_contract_version(proposal) == "v3":
            return self._unsupported_v3_fields(proposal, evidence_text_by_ref)
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
                polarity_field = key in {"canonical_value", "polarity", "constraint_polarity"} and str(
                    value
                ).casefold() in {
                    "required",
                    "require",
                    "forbidden",
                    "forbid",
                    "allowed",
                    "allow",
                    "prefer",
                    "preferred",
                    "discourage",
                    "discouraged",
                    "conditional_require",
                    "conditional_forbid",
                }
                supported = (
                    self._semantically_supported(
                        key,
                        value,
                        field_texts,
                        proposal.value_fields,
                        proposal.identity_fields,
                    )
                    if polarity_field
                    else self._supported(value, haystack)
                    or self._semantically_supported(
                        key,
                        value,
                        field_texts,
                        proposal.value_fields,
                        proposal.identity_fields,
                    )
                )
                if not supported:
                    unsupported.append(f"{prefix}.{key}")
        return unsupported

    def _unsupported_v3_fields(
        self,
        proposal: MemorySemanticProposal,
        evidence_text_by_ref: Mapping[EvidenceRef, str],
    ) -> list[str]:
        """Require model-authored fields to be traceable to their bound source span."""

        atomic = getattr(proposal, "atomic_evidence_ref", None)
        if atomic is None:
            return [
                *[f"identity.{key}" for key in proposal.identity_fields],
                *[f"value.{key}" for key in proposal.value_fields],
            ]
        unsupported: list[str] = []
        system_identity_fields = {str(item) for item in proposal.metadata.get("system_identity_fields", []) or []}
        system_value_fields = {str(item) for item in proposal.metadata.get("system_value_fields", []) or []}
        for prefix, fields in (("identity", proposal.identity_fields), ("value", proposal.value_fields)):
            for key, value in fields.items():
                field_name = f"{prefix}.{key}"
                refs = tuple(proposal.field_evidence_refs.get(field_name, ()))
                if value is None or value == "" or value == () or value == [] or value == {}:
                    unsupported.append(field_name)
                    continue
                if (prefix == "identity" and key in system_identity_fields) or (
                    prefix == "value" and key in system_value_fields
                ):
                    if not refs or any(ref not in evidence_text_by_ref for ref in refs):
                        unsupported.append(field_name)
                    continue
                if not refs or any(
                    ref not in evidence_text_by_ref or not self._within_atomic_span(ref, atomic) for ref in refs
                ):
                    unsupported.append(field_name)
                    continue
                field_texts = [evidence_text_by_ref[ref] for ref in refs]
                haystack = "\n".join(field_texts).casefold()
                source_value = dict(proposal.metadata.get("source_grounded_field_values", {}) or {}).get(
                    field_name
                )
                polarity_value = key in {"polarity", "constraint_polarity"} or (
                    key == "canonical_value"
                    and proposal.memory_type == "project_rule"
                    and str(value).strip().casefold()
                    in {
                        "required",
                        "require",
                        "forbidden",
                        "forbid",
                        "allowed",
                        "allow",
                        "prefer",
                        "preferred",
                        "discourage",
                        "discouraged",
                        "conditional_require",
                        "conditional_forbid",
                    }
                )
                if not self._supported(value, haystack) and not self._supported(source_value, haystack) and not (
                    polarity_value
                    and self._semantically_supported(
                        key,
                        value,
                        field_texts,
                        proposal.value_fields,
                        proposal.identity_fields,
                    )
                ):
                    unsupported.append(field_name)
        return unsupported

    def _within_atomic_span(self, ref: EvidenceRef, atomic: EvidenceRef) -> bool:
        if ref == atomic:
            return True
        return bool(
            atomic.span_start is not None
            and atomic.span_end is not None
            and ref.span_start is not None
            and ref.span_end is not None
            and ref.event_id == atomic.event_id
            and ref.content_path == atomic.content_path
            and ref.content_hash == atomic.content_hash
            and ref.span_start >= atomic.span_start
            and ref.span_end <= atomic.span_end
        )

    def _field_evidence_errors(
        self,
        proposal: MemorySemanticProposal,
        evidence_text_by_ref: Mapping[EvidenceRef, str],
    ) -> list[str]:
        required = {
            *[f"identity.{key}" for key in proposal.identity_fields],
            *[f"value.{key}" for key in proposal.value_fields],
            "semantic.speech_act",
            "semantic.commitment",
            "semantic.temporal_scope",
            "semantic.relation_to_existing",
            "transition",
        }
        if _semantic_contract_version(proposal) == "v3":
            required.update(_V3_SEMANTIC_BINDINGS)
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

    def _semantically_supported(
        self,
        key: str,
        value: Any,
        evidence_texts: list[str],
        value_fields: Mapping[str, Any],
        identity_fields: Mapping[str, Any],
    ) -> bool:
        if key not in {"canonical_value", "polarity", "constraint_polarity"}:
            return False
        expected = {
            "required": {ConstraintPolarity.REQUIRE, ConstraintPolarity.CONDITIONAL_REQUIRE},
            "require": {ConstraintPolarity.REQUIRE, ConstraintPolarity.CONDITIONAL_REQUIRE},
            "forbidden": {ConstraintPolarity.FORBID, ConstraintPolarity.CONDITIONAL_FORBID},
            "forbid": {ConstraintPolarity.FORBID, ConstraintPolarity.CONDITIONAL_FORBID},
            "allowed": {ConstraintPolarity.ALLOW},
            "allow": {ConstraintPolarity.ALLOW},
            "prefer": {ConstraintPolarity.PREFER},
            "preferred": {ConstraintPolarity.PREFER},
            "discourage": {ConstraintPolarity.DISCOURAGE},
            "discouraged": {ConstraintPolarity.DISCOURAGE},
            "conditional_require": {ConstraintPolarity.CONDITIONAL_REQUIRE},
            "conditional_forbid": {ConstraintPolarity.CONDITIONAL_FORBID},
        }.get(str(value).strip().casefold())
        if expected is None:
            return False
        usable = tuple(
            (match, text)
            for text in evidence_texts
            for match in self.signal_matcher.match(text)
            if not (
                match.negated
                or match.quoted
                or match.metalinguistic
                or (
                    match.hypothetical
                    and match.polarity
                    not in {ConstraintPolarity.CONDITIONAL_REQUIRE, ConstraintPolarity.CONDITIONAL_FORBID}
                )
            )
        )
        matching = tuple(
            (match, text)
            for match, text in usable
            if match.polarity in expected
            and self._constraint_subject_supported(match, text, identity_fields, value_fields)
        )
        if not matching:
            return False
        conditional = any(
            match.polarity in {ConstraintPolarity.CONDITIONAL_REQUIRE, ConstraintPolarity.CONDITIONAL_FORBID}
            for match, _text in matching
        )
        if not conditional:
            return True
        return any(
            value_fields.get(field_name) is not None
            and value_fields.get(field_name) != ""
            and value_fields.get(field_name) != ()
            and value_fields.get(field_name) != []
            and value_fields.get(field_name) != {}
            for field_name in (
                "condition",
                "conditions",
                "exception",
                "exceptions",
                "applicability_qualifier",
            )
        )

    def _constraint_subject_supported(
        self,
        match: EvidenceSignalMatch,
        text: str,
        identity_fields: Mapping[str, Any],
        value_fields: Mapping[str, Any],
    ) -> bool:
        topic = str(identity_fields.get("rule_topic") or value_fields.get("subject") or "").strip().casefold()
        if not topic:
            return True
        anchors = {
            token
            for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", topic.replace("_", "-").replace("-", " "))
            if token not in {"use", "usage", "rule", "policy", "constraint", "project", "使用", "规则", "约束"}
            and len(token) > 1
        }
        if not anchors or not any(anchor in text.casefold() for anchor in anchors):
            # Some identity topics are deterministic normalized labels rather
            # than source wording. Existing identity-field validation remains
            # authoritative when no lexical anchor is available.
            return True
        clause_pattern = re.compile(r"(?i)(?:[，,;；。.!?！？]|但是|不过|而是|但|\bbut\b|\bhowever\b)")
        left = max((item.end() for item in clause_pattern.finditer(text, 0, match.start)), default=0)
        right_match = clause_pattern.search(text, match.end)
        right = right_match.start() if right_match else len(text)
        clause = text[left:right].casefold()
        if any(anchor in clause for anchor in anchors):
            return True
        prefix = text[:left].casefold()
        return bool(
            any(anchor in prefix for anchor in anchors)
            and re.match(r"\s*(?:(?:otherwise)\s+)?(?:it|they|this|that)\b", clause)
        )

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
