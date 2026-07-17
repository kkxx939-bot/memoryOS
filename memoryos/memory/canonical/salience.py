"""Deterministic salience admission before semantic memory extraction."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from memoryos.core.integrity import canonical_digest
from memoryos.memory.canonical.episode import EvidenceEpisode
from memoryos.security.sanitization import ENV_SECRET_RE, INLINE_SECRET_RE, PRIVATE_KEY_RE

_REMEMBER = re.compile(r"(?i)(?:\bremember(?:\s+this|\s+that|:)|请记住|记住[：:]?)")
_PREFERENCE = re.compile(
    r"(?i)(?:\bi\s+(?:strongly\s+)?(?:prefer|like|dislike)\b|\bpreference\b|我(?:一直)?(?:喜欢|不喜欢)|偏好|以后请)"
)
_PROFILE = re.compile(
    r"(?i)(?:^i\s+am\b|^i\s+work\b|^je\s+travaille\s+comme\b|^我是|^我的职业是|^我在.+工作|长期从事|负责人)"
)
_RULE = re.compile(r"(?i)(?:\bmust(?:\s+not)?\b|\bnever\b|\bdo\s+not\b|项目规则|必须|不得|禁止|不允许|不要)")
_DECISION = re.compile(
    r"(?i)(?:\b(?:we|i)\s+(?:have\s+)?decided\b|\bconfirm(?:ed)?\b|\bformally\s+change\b|"
    r"\b(?:continue|keep)\s+(?:to\s+)?(?:use|using)\b|\badopt(?:ed)?\b|"
    r"正式决定|正式改成|确认采用|确认|决定|已采用|继续使用|保持使用|本项目.{0,24}(?:选型|采用|定为))"
)
_CORRECTION = re.compile(
    r"(?i)(?:\bcorrection\b|\bcorrect(?:ing|ed)?\b|\bactually\b|\bno\s+longer\b|纠正|更正|改为|不再|撤回)"
)
_COMMITMENT = re.compile(r"(?i)(?:\bwill\b|\bcommit(?:ted)?\b|\bpromise\b|承诺|以后会|将会)")
_DURABLE = re.compile(r"(?i)(?:\balways\b|\blong[- ]term\b|\bfrom\s+now\s+on\b|长期|一直|今后|以后都)")
_FUTURE_UTILITY = re.compile(r"(?i)(?:\breuse|reusable|next\s+time|future\s+work|以后(?:继续|都|还要)|可复用|下次)")
_FUTURE_OPTION = re.compile(
    r"(?i)(?:\bfuture\b.{0,48}\b(?:option|candidate|alternative|evaluate)\b|"
    r"\b(?:option|candidate|alternative)\b.{0,48}\bfuture\b|以后(?:可以|可)?(?:评估|考虑)|未来(?:候选|选项|方案)|"
    r"(?:可能|也许|或许).{0,24}(?:以后|未来).{0,24}(?:使用|采用)|"
    r"(?:他说|她说|有人说).{0,64}(?:不同意|不接受|拒绝))"
)
_OUTCOME = re.compile(r"(?i)(?:\b(?:completed|implemented|fixed|verified|resolved)\b|已完成|已实现|已修复|已验证|解决)")
_TRANSIENT = re.compile(
    r"(?i)(?:\bjust\s+for\s+(?:now|today)\b|\bone[- ]off\b|\btemporary\b|\bthis\s+time\s+only\b|临时|仅这一次|今天先|暂时)"
)
_ORDINARY_CHAT = re.compile(r"(?i)^(?:hi|hello|hey|thanks|thank\s+you|你好|您好|谢谢|在吗)[.!！。\s]*$")


@dataclass(frozen=True)
class SalienceFactor:
    name: str
    weight: int
    event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SalienceDecision:
    salient: bool
    reasons: tuple[str, ...]
    score: int = 0
    factors: tuple[SalienceFactor, ...] = ()
    episode_fingerprint: str = ""
    budget_cost: int = 0
    duplicate: bool = False
    privacy_risk: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class EpisodeSalienceGate:
    """Pure rules: no request/session state is retained on the gate instance."""

    MIN_SCORE = 35
    IMPORTANT_EVENT_TYPES = {
        "DECISION",
        "PREFERENCE",
        "RULE",
        "CORRECTION",
        "RETRACTION",
        "TASK_RESULT",
        "TOOL_FAILURE",
        "TOOL_RECOVERY",
        "CONFIG_CHANGED",
    }

    def evaluate(
        self,
        episode: EvidenceEpisode,
        *,
        existing_memories: Sequence[Any] = (),
        seen_episode_fingerprints: Collection[str] = (),
        prior_episode_counts: Mapping[str, int] | None = None,
        consumed_budget: int = 0,
        max_episode_budget: int = 8,
    ) -> SalienceDecision:
        fingerprint, budget_key, scope_keys = self.fingerprint(episode)
        if fingerprint in seen_episode_fingerprints:
            return SalienceDecision(
                False,
                ("duplicate_episode",),
                score=-100,
                factors=(SalienceFactor("duplicate_episode", -100, tuple(event.event_id for event in episode.events)),),
                episode_fingerprint=fingerprint,
                duplicate=True,
                metadata={"budget_key": budget_key, "scope_keys": scope_keys},
            )
        if consumed_budget >= max(0, int(max_episode_budget)):
            return SalienceDecision(
                False,
                ("episode_budget_exhausted",),
                score=-100,
                factors=(SalienceFactor("episode_budget_exhausted", -100),),
                episode_fingerprint=fingerprint,
                metadata={"budget_key": budget_key, "scope_keys": scope_keys},
            )

        factors: list[SalienceFactor] = []
        normalized_texts = [self._normalize(event.text()) for event in episode.events if event.text().strip()]
        repetitions = Counter(normalized_texts)
        prior_counts = {self._normalize(key): int(value) for key, value in dict(prior_episode_counts or {}).items()}
        existing_values = self._existing_values(existing_memories)
        episode_values = {value for value in normalized_texts if value}
        exact_duplicates = episode_values & existing_values
        privacy_events = tuple(event.event_id for event in episode.events if self._private(event.text()))
        if privacy_events:
            factors.append(SalienceFactor("privacy_or_sensitivity_risk", -100, privacy_events))

        for event in episode.events:
            text = event.text()
            lowered = text.casefold()
            authoritative_actor = event.actor.kind in {"user", "system"}
            event_factors: list[tuple[str, int]] = []
            if event.event_type in self.IMPORTANT_EVENT_TYPES:
                event_factors.append((f"event_type:{event.event_type.lower()}", 45))
            if event.event_type in {"STATE_CHANGED", "ENTITY_CHANGED"} and bool(
                event.metadata.get("durable") or event.metadata.get("user_confirmed")
            ):
                event_factors.append((f"durable_event_type:{event.event_type.lower()}", 40))
            if event.event_type == "FEEDBACK" and (event.actor.kind == "user" or bool(event.metadata.get("explicit"))):
                event_factors.append(("explicit_feedback", 40))
            if bool(event.metadata.get("salient")):
                event_factors.append(("adapter_marked", 45))
            if bool(event.metadata.get("user_confirmed") or event.metadata.get("explicitly_confirmed")):
                event_factors.append(("user_confirmed_result", 65))
            if authoritative_actor and _REMEMBER.search(text):
                event_factors.append(("explicit_remember", 100))
            if authoritative_actor and _CORRECTION.search(text):
                event_factors.append(("correction_or_contradiction", 75))
            if authoritative_actor and _RULE.search(text):
                event_factors.append(("durable_rule", 70))
            if authoritative_actor and _PREFERENCE.search(text):
                event_factors.append(("durable_preference", 65))
            if authoritative_actor and _PROFILE.search(text.strip()):
                event_factors.append(("durable_profile", 50))
            if authoritative_actor and _DECISION.search(text):
                event_factors.append(("confirmed_decision", 70))
            if authoritative_actor and _COMMITMENT.search(text):
                event_factors.append(("commitment", 45))
            if authoritative_actor and _DURABLE.search(text):
                event_factors.append(("durability", 25))
            if _FUTURE_UTILITY.search(text):
                event_factors.append(("future_utility", 25))
            if authoritative_actor and _FUTURE_OPTION.search(text):
                event_factors.append(("future_option", 45))
            if _OUTCOME.search(text) and (
                _FUTURE_UTILITY.search(text)
                or event.event_type == "TASK_RESULT"
                or any(token in lowered for token in ("lesson", "pattern", "approach", "经验", "做法"))
            ):
                event_factors.append(("reusable_task_outcome", 55))
            if _TRANSIENT.search(text):
                event_factors.append(("transient_or_one_off", -35))
            if event.event_type == "TOOL_RESULT":
                event_factors.append(("ordinary_tool_result", -45))
            if (
                event.actor.kind == "tool"
                and event.event_type not in {"TOOL_FAILURE", "TOOL_RECOVERY", "TASK_RESULT"}
                and not bool(event.metadata.get("user_confirmed") or event.metadata.get("explicitly_confirmed"))
            ):
                event_factors.append(("unconfirmed_tool_output", -25))
            if _ORDINARY_CHAT.fullmatch(text.strip()):
                event_factors.append(("ordinary_chat", -30))
            factors.extend(SalienceFactor(name, weight, (event.event_id,)) for name, weight in event_factors)

        repeated_ids = tuple(
            event.event_id for event in episode.events if repetitions[self._normalize(event.text())] > 1
        )
        if repeated_ids:
            factors.append(SalienceFactor("repetition_within_episode", 10, repeated_ids))
        cross_episode_ids = tuple(
            event.event_id for event in episode.events if prior_counts.get(self._normalize(event.text()), 0) >= 2
        )
        if cross_episode_ids:
            factors.append(SalienceFactor("repetition_across_episodes", 20, cross_episode_ids))
        if exact_duplicates:
            factors.append(
                SalienceFactor("duplicate_canonical_content", -80, tuple(event.event_id for event in episode.events))
            )
        elif existing_memories:
            factors.append(
                SalienceFactor("novel_against_canonical", 15, tuple(event.event_id for event in episode.events))
            )

        factors = self._dedupe_factors(factors)
        score = sum(factor.weight for factor in factors)
        privacy_risk = bool(privacy_events)
        duplicate = bool(exact_duplicates)
        hard_positive = any(
            factor.name
            in {
                "explicit_remember",
                "correction_or_contradiction",
                "durable_rule",
                "durable_preference",
                "durable_profile",
                "confirmed_decision",
            }
            for factor in factors
        )
        salient = not privacy_risk and not duplicate and (hard_positive or score >= self.MIN_SCORE)
        reasons = tuple(factor.name for factor in factors)
        return SalienceDecision(
            salient,
            reasons,
            score=score,
            factors=tuple(factors),
            episode_fingerprint=fingerprint,
            budget_cost=1 if salient else 0,
            duplicate=duplicate,
            privacy_risk=privacy_risk,
            metadata={
                "consumed_budget": int(consumed_budget),
                "max_episode_budget": int(max_episode_budget),
                "existing_memory_count": len(existing_memories),
                "budget_key": budget_key,
                "scope_keys": scope_keys,
            },
        )

    def fingerprint(self, episode: EvidenceEpisode) -> tuple[str, str, tuple[str, ...]]:
        """Return the stable episode and budget identities used by admission.

        ``task_id`` and ``episode_id`` are request-local and deliberately do
        not participate.  Durable reservation replay uses this same function
        to prove that a task still refers to the exact semantic episode that
        consumed (or skipped) its extraction budget.
        """

        scope_keys = tuple(sorted(scope.key for scope in episode.legal_scope_candidates() if scope.kind != "episode"))
        budget_key = canonical_digest([episode.tenant_id, scope_keys])
        episode_fingerprint = canonical_digest(
            [
                episode.tenant_id,
                scope_keys,
                [(event.event_type, event.actor.kind, self._normalize(event.text())) for event in episode.events],
            ]
        )
        return episode_fingerprint, budget_key, scope_keys

    def _existing_values(self, memories: Sequence[Any]) -> set[str]:
        values = set()
        for memory in memories:
            if isinstance(memory, Mapping):
                raw = memory.get("canonical_value") or memory.get("content") or memory.get("l2")
            else:
                raw = getattr(memory, "canonical_value", "") or getattr(memory, "l2", "")
            normalized = self._normalize(str(raw or ""))
            if normalized:
                values.add(normalized)
        return values

    def _private(self, text: str) -> bool:
        return bool(
            PRIVATE_KEY_RE.search(text)
            or ENV_SECRET_RE.search(text)
            or INLINE_SECRET_RE.search(text)
            or re.search(r"(?i)\b(?:password|passwd|authorization|cookie)\s*[:=]", text)
        )

    def _normalize(self, text: str) -> str:
        return " ".join(str(text).strip().casefold().split())

    def _dedupe_factors(self, factors: list[SalienceFactor]) -> list[SalienceFactor]:
        grouped: dict[tuple[str, int], list[str]] = {}
        for factor in factors:
            grouped.setdefault((factor.name, factor.weight), []).extend(factor.event_ids)
        return [
            SalienceFactor(name, weight, tuple(dict.fromkeys(event_ids)))
            for (name, weight), event_ids in grouped.items()
        ]
