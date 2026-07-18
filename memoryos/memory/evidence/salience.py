"""Deterministic, storage-neutral salience admission for Session evidence."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from memoryos.core.integrity import canonical_digest
from memoryos.memory.evidence.episode import EvidenceEpisode

_REMEMBER = re.compile(r"(?i)(?:记住|请记得|以后记得|remember|keep in mind)")
_PREFERENCE = re.compile(r"(?i)(?:我(?:不)?喜欢|我更喜欢|偏好|prefer|dislike)")
_PROFILE = re.compile(r"(?i)(?:我是|我叫|我住在|i am|my name is)")
_DURABLE = re.compile(r"(?i)(?:以后|长期|一直|将来|from now on|always|long[- ]term)")
_CORRECTION = re.compile(r"(?i)(?:更正|不是.+是|改成|改为|correction|actually)")
_OPEN_LOOP = re.compile(r"(?i)(?:待确认|以后再看|尚未解决|todo|open question|follow up)")
_TRANSIENT = re.compile(r"(?i)(?:这一次|临时|仅本次|just this once|temporary)")
_PRIVATE = re.compile(r"(?i)(?:password|api[_ -]?key|密码|密钥|身份证|银行卡)")


@dataclass(frozen=True)
class SalienceFactor:
    name: str
    weight: int
    event_ids: tuple[str, ...]


@dataclass(frozen=True)
class SalienceDecision:
    salient: bool
    reasons: tuple[str, ...]
    score: int
    factors: tuple[SalienceFactor, ...]
    episode_fingerprint: str
    budget_cost: int
    duplicate: bool = False
    privacy_risk: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EpisodeSalienceGate:
    """Select only durable, useful evidence without granting write authority."""

    MIN_SCORE = 45

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    def fingerprint(self, episode: EvidenceEpisode) -> tuple[str, str, tuple[str, ...]]:
        scopes = tuple(sorted(item.key for item in episode.legal_scope_candidates() if item.kind != "episode"))
        budget_key = canonical_digest([episode.tenant_id, scopes])
        fingerprint = canonical_digest(
            [episode.tenant_id, scopes, [(item.event_type, item.actor.kind, self._normalize(item.text())) for item in episode.events]]
        )
        return fingerprint, budget_key, scopes

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
        fingerprint, budget_key, scopes = self.fingerprint(episode)
        if fingerprint in set(seen_episode_fingerprints):
            return SalienceDecision(
                False,
                ("duplicate_episode",),
                -80,
                (SalienceFactor("duplicate_episode", -80, tuple(item.event_id for item in episode.events)),),
                fingerprint,
                0,
                duplicate=True,
                metadata={"budget_key": budget_key, "scope_keys": scopes},
            )
        if consumed_budget >= max_episode_budget:
            return SalienceDecision(
                False,
                ("budget_exhausted",),
                0,
                (),
                fingerprint,
                0,
                metadata={"budget_key": budget_key, "scope_keys": scopes},
            )
        factors: list[SalienceFactor] = []
        counts = Counter(self._normalize(item.text()) for item in episode.events)
        prior = dict(prior_episode_counts or {})
        for event in episode.events:
            text = event.text()
            if _PRIVATE.search(text):
                factors.append(SalienceFactor("privacy_or_sensitivity_risk", -100, (event.event_id,)))
            if event.actor.kind in {"user", "system"}:
                for name, pattern, weight in (
                    ("explicit_remember", _REMEMBER, 100),
                    ("durable_preference", _PREFERENCE, 65),
                    ("durable_profile", _PROFILE, 55),
                    ("correction", _CORRECTION, 75),
                    ("open_loop", _OPEN_LOOP, 50),
                    ("durability", _DURABLE, 30),
                ):
                    if pattern.search(text):
                        factors.append(SalienceFactor(name, weight, (event.event_id,)))
            if _TRANSIENT.search(text):
                factors.append(SalienceFactor("transient", -40, (event.event_id,)))
            normalized = self._normalize(text)
            if counts[normalized] > 1 or prior.get(normalized, 0) >= 2:
                factors.append(SalienceFactor("repetition", 15, (event.event_id,)))
        unique = {(item.name, item.event_ids): item for item in factors}
        factors = list(unique.values())
        score = sum(item.weight for item in factors)
        privacy = any(item.name == "privacy_or_sensitivity_risk" for item in factors)
        salient = not privacy and score >= self.MIN_SCORE
        return SalienceDecision(
            salient,
            tuple(item.name for item in factors),
            score,
            tuple(factors),
            fingerprint,
            1 if salient else 0,
            privacy_risk=privacy,
            metadata={
                "budget_key": budget_key,
                "scope_keys": scopes,
                "consumed_budget": int(consumed_budget),
                "max_episode_budget": int(max_episode_budget),
                "existing_memory_count": len(existing_memories),
            },
        )


__all__ = ["EpisodeSalienceGate", "SalienceDecision", "SalienceFactor"]
