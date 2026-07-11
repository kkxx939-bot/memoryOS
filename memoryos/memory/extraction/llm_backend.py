"""记忆系统里的大模型后端。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.episode import EvidenceEpisode, SessionArchiveEpisodeAdapter
from memoryos.memory.canonical.evidence import EvidenceRef
from memoryos.memory.canonical.prefetch import PrefetchedMemory
from memoryos.memory.canonical.proposal import (
    Commitment,
    EpistemicStatus,
    MemorySemanticProposal,
    SemanticAssessment,
    SemanticRelation,
    SpeechAct,
    TemporalScope,
)
from memoryos.memory.canonical.scope import ScopeRef
from memoryos.memory.schema import MemoryTypeSchema
from memoryos.memory.view import adapter_id_from_archive


class MemoryModelProvider(Protocol):
    """约定 MemoryModelProvider 需要提供的接口。"""

    def complete(self, prompt: str) -> str: ...


class MemoryExtractionPromptBuilder:
    """负责 MemoryExtractionPromptBuilder 这部分逻辑。"""

    def build(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
        *,
        existing_memories: Sequence[PrefetchedMemory] = (),
        episode: EvidenceEpisode | None = None,
    ) -> str:
        """根据输入组装结果对象。"""

        schema_names = ", ".join(schema.memory_type.value for schema in schemas)
        schema_payload = [
            {
                "memory_type": schema.memory_type.value,
                "required_fields": list(schema.required_fields),
                "optional_fields": list(schema.optional_fields),
            }
            for schema in schemas
        ]
        messages = "\n".join(json.dumps(message, ensure_ascii=False, sort_keys=True) for message in archive.messages)
        existing = json.dumps(
            [
                {
                    "uri": item.uri,
                    "memory_type": item.memory_type,
                    "state": item.state,
                    "revision": item.revision,
                    "slot_id": item.slot_id,
                    "claim_id": item.claim_id,
                    "canonical_value": item.canonical_value,
                    "identity_fields": item.identity_fields,
                    "scope": item.scope,
                    "l0": item.l0,
                    "l1": item.l1,
                    "l2": item.l2,
                    "relations": list(item.relations),
                }
                for item in existing_memories
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        events = []
        legal_scopes = []
        if episode is not None:
            events = [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "actor": event.actor.to_dict(),
                    "subjects": [subject.to_dict() for subject in event.subjects],
                    "event_digest": event.digest,
                    "content_hash": EvidenceRef.from_event(event).content_hash,
                    "content_path": event.content_path,
                    "occurred_at": event.occurred_at.isoformat(),
                    "ingested_at": (event.ingested_at or event.occurred_at).isoformat(),
                    "sequence": event.sequence,
                    "text": event.text(),
                }
                for event in episode.events
            ]
            legal_scopes = [scope.to_dict() for scope in episode.legal_scope_candidates()]
        return (
            "Extract durable memory semantic proposals as JSON. "
            "Return an object with a candidates array. Each proposal may describe semantics but is not a database operation. "
            f"Allowed memory_type values: {schema_names}. "
            f"Allowed speech_act values: {', '.join(item.value for item in SpeechAct)}. "
            f"Allowed commitment values: {', '.join(item.value for item in Commitment)}. "
            f"Allowed temporal_scope values: {', '.join(item.value for item in TemporalScope)}. "
            f"Allowed relation_to_existing values: {', '.join(item.value for item in SemanticRelation)}. "
            f"Allowed epistemic_status values: {', '.join(item.value for item in EpistemicStatus)}. "
            "Use identity_fields and value_fields separately. Include speech_act, commitment, temporal_scope, "
            "relation_to_existing, epistemic_status, evidence_refs, and field_evidence_refs. "
            "Bind every identity field, value field, semantic.speech_act, semantic.commitment, "
            "semantic.temporal_scope, semantic.relation_to_existing, and transition "
            "to evidence in field_evidence_refs. Use only suggested_scope_refs selected from legal scopes. "
            "Use related_slot_ids and related_claim_ids only for identities present in EXISTING_MEMORIES. "
            "Do not output operations or target URIs. Do not output tenant IDs, visibility policy, revisions, DELETE, or scope moves.\n"
            f"LEGAL_SCOPES={json.dumps(legal_scopes, ensure_ascii=False, sort_keys=True)}\n"
            f"MEMORY_SCHEMAS={json.dumps(schema_payload, ensure_ascii=False, sort_keys=True)}\n"
            f"EXISTING_MEMORIES={existing}\n"
            f"EPISODE_EVENTS={json.dumps(events, ensure_ascii=False, sort_keys=True)}\n"
            f"MESSAGES={messages}"
        )


class _MemoryExtractionJsonParser:
    """只负责拆出并解析严格的 JSON 响应信封。"""

    def _load_json(self, response: str) -> dict | list:
        text = response.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM memory response is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict | list):
            raise ValueError("LLM memory response must be an object or list")
        return payload


class LLMMemoryExtractorBackend:
    """让模型只产出语义提案，不能直接决定增删改。"""

    candidate_backend = True
    _PROPOSAL_FIELDS = {
        "proposal_id",
        "memory_type",
        "identity_fields",
        "value_fields",
        "semantic",
        "epistemic_status",
        "suggested_scope_refs",
        "related_memory_ids",
        "related_slot_ids",
        "related_claim_ids",
        "evidence_refs",
        "field_evidence_refs",
        "confidence",
        "source_role",
    }
    _SEMANTIC_FIELDS = {"speech_act", "commitment", "temporal_scope", "relation_to_existing"}
    _SPEECH_VALUES = {item.value.casefold() for item in SpeechAct} | {
        "recommendation",
        "future_option",
        "possible_alternative",
        "exploratory_alternative",
        "under_consideration",
    }
    _COMMITMENT_VALUES = {item.value.casefold() for item in Commitment} | {
        "possible",
        "exploratory_alternative",
        "future_option",
        "recommendation",
        "plan",
        "committed",
    }
    _TEMPORAL_VALUES = {item.value.casefold() for item in TemporalScope}
    _RELATION_VALUES = {item.value.casefold() for item in SemanticRelation} | {
        "possible_alternative",
        "exploratory_alternative",
    }

    def __init__(
        self,
        provider: MemoryModelProvider,
        prompt_builder: MemoryExtractionPromptBuilder | None = None,
        parser: _MemoryExtractionJsonParser | None = None,
        model_id: str | None = None,
        extractor_version: str = "llm_semantic_extractor_v2",
    ) -> None:
        self.provider = provider
        self.prompt_builder = prompt_builder or MemoryExtractionPromptBuilder()
        self.parser = parser or _MemoryExtractionJsonParser()
        self.model_id = model_id
        self.extractor_version = extractor_version

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> list[MemorySemanticProposal]:
        """处理 extract 这一步。"""

        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        return list(self.extract_with_context(archive, schemas, existing_memories=(), episode=episode))

    def extract_with_context(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
        *,
        existing_memories: Sequence[PrefetchedMemory],
        episode: EvidenceEpisode,
    ) -> list[MemorySemanticProposal]:
        """处理 extract with context 这一步。"""

        prompt = self.prompt_builder.build(
            archive,
            schemas,
            existing_memories=existing_memories,
            episode=episode,
        )
        response = self.provider.complete(prompt)
        payload = self.parser._load_json(response)
        if not isinstance(payload, dict):
            raise ValueError("semantic memory extraction response must be an object with candidates")
        if set(payload) != {"candidates"}:
            unknown = set(payload) - {"candidates"}
            raise ValueError(f"memory extraction response contains unknown fields: {','.join(sorted(unknown))}")
        raw_candidates = payload.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raise ValueError("memory extraction candidates must be a list")
        allowed = {schema.memory_type.value for schema in schemas}
        proposals: list[MemorySemanticProposal] = []
        proposal_ids: set[str] = set()
        legal_scopes = {scope.key for scope in episode.legal_scope_candidates()}
        legal_related_ids = {
            identifier
            for item in existing_memories
            for identifier in (item.uri, item.slot_id, item.claim_id)
            if identifier
        }
        for index, raw in enumerate(raw_candidates):
            if not isinstance(raw, dict):
                raise ValueError(f"candidate[{index}] must be an object")
            unknown = set(raw) - self._PROPOSAL_FIELDS
            if unknown:
                raise ValueError(f"candidate[{index}] contains unknown fields: {','.join(sorted(unknown))}")
            memory_type = str(raw.get("memory_type", ""))
            if memory_type not in allowed:
                raise ValueError(f"candidate[{index}] memory_type is not allowed: {memory_type}")
            identity_fields = raw.get("identity_fields", {})
            value_fields = raw.get("value_fields", {})
            semantic = raw.get("semantic", {})
            if (
                not isinstance(identity_fields, dict)
                or not isinstance(value_fields, dict)
                or not isinstance(semantic, dict)
            ):
                raise ValueError(f"candidate[{index}] semantic fields must be objects")
            if (
                not identity_fields
                or not value_fields
                or not any(value is not None and value != "" for value in value_fields.values())
            ):
                raise ValueError(f"candidate[{index}] requires non-empty identity_fields and value_fields")
            unknown_semantic = set(semantic) - self._SEMANTIC_FIELDS
            if unknown_semantic:
                raise ValueError(
                    f"candidate[{index}] semantic contains unknown fields: {','.join(sorted(unknown_semantic))}"
                )
            self._validate_semantic_enums(index, semantic)
            evidence_refs = self._evidence_refs(raw.get("evidence_refs", []), episode)
            field_evidence_refs = self._field_evidence_refs(
                index,
                raw.get("field_evidence_refs"),
                episode,
                identity_fields=identity_fields,
                value_fields=value_fields,
                evidence_refs=evidence_refs,
            )
            raw_scopes = raw.get("suggested_scope_refs", []) or []
            if not isinstance(raw_scopes, list) or any(not isinstance(item, dict) for item in raw_scopes):
                raise ValueError(f"candidate[{index}] suggested_scope_refs must contain only objects")
            scopes = tuple(ScopeRef.from_dict(item) for item in raw_scopes)
            if any(scope.key not in legal_scopes for scope in scopes):
                raise ValueError(f"candidate[{index}] suggested_scope_refs contains an illegal scope")
            proposal_id = str(
                raw.get("proposal_id") or f"proposal_{stable_hash([episode.episode_id, index, raw], length=32)}"
            )
            if proposal_id in proposal_ids:
                raise ValueError(f"duplicate proposal_id: {proposal_id}")
            proposal_ids.add(proposal_id)
            related_memory_ids = self._string_list(index, "related_memory_ids", raw.get("related_memory_ids", []))
            related_slot_ids = self._string_list(index, "related_slot_ids", raw.get("related_slot_ids", []))
            related_claim_ids = self._string_list(index, "related_claim_ids", raw.get("related_claim_ids", []))
            legal_slot_ids = {item.slot_id for item in existing_memories if item.slot_id}
            legal_claim_ids = {item.claim_id for item in existing_memories if item.claim_id}
            if (
                any(identifier not in legal_related_ids for identifier in related_memory_ids)
                or any(identifier not in legal_slot_ids for identifier in related_slot_ids)
                or any(identifier not in legal_claim_ids for identifier in related_claim_ids)
            ):
                raise ValueError(f"candidate[{index}] related_memory_ids contains an illegal reference")
            actual_source_role = self._source_role(evidence_refs, episode)
            reported_source_role = str(raw.get("source_role") or "").casefold()
            normalized_reported = "assistant" if reported_source_role == "agent" else reported_source_role
            if normalized_reported and normalized_reported != actual_source_role:
                raise ValueError(f"candidate[{index}] source_role does not match referenced evidence")
            speech = str(semantic.get("speech_act", "")).casefold()
            commitment = str(semantic.get("commitment", "")).casefold()
            if (
                speech in {"confirmation", "correction"} or commitment in {"confirmed", "committed"}
            ) and not evidence_refs:
                raise ValueError(f"candidate[{index}] authoritative semantics require evidence_refs")
            proposals.append(
                MemorySemanticProposal(
                    proposal_id=proposal_id,
                    memory_type=memory_type,
                    identity_fields=identity_fields,
                    value_fields=value_fields,
                    semantic=SemanticAssessment(
                        str(semantic.get("speech_act", "observation")),
                        str(semantic.get("commitment", "weak")),
                        str(semantic.get("temporal_scope", "unspecified")),
                        str(semantic.get("relation_to_existing", "unrelated")),
                    ),
                    epistemic_status=EpistemicStatus(str(raw.get("epistemic_status", "INFERRED")).upper()),
                    suggested_scope_refs=scopes,
                    related_memory_ids=related_memory_ids,
                    related_slot_ids=related_slot_ids,
                    related_claim_ids=related_claim_ids,
                    evidence_refs=evidence_refs,
                    field_evidence_refs=field_evidence_refs,
                    confidence=float(raw.get("confidence", 0.5)),
                    extractor_version=self.extractor_version,
                    model_id=self.model_id,
                    metadata={
                        "source_role": actual_source_role,
                        "source_adapter_id": adapter_id_from_archive(archive),
                        "source_session_id": archive.session_id,
                        "source_connect": dict(archive.metadata.get("connect", {}) or {}),
                    },
                )
            )
        return proposals

    def _validate_semantic_enums(self, index: int, semantic: dict) -> None:
        values = {
            "speech_act": (semantic.get("speech_act", ""), self._SPEECH_VALUES),
            "commitment": (semantic.get("commitment", ""), self._COMMITMENT_VALUES),
            "temporal_scope": (semantic.get("temporal_scope", ""), self._TEMPORAL_VALUES),
            "relation_to_existing": (semantic.get("relation_to_existing", "unrelated"), self._RELATION_VALUES),
        }
        for field_name, (raw_value, allowed) in values.items():
            normalized = str(raw_value).strip().casefold().replace("-", "_").replace(" ", "_")
            if normalized not in allowed:
                raise ValueError(f"candidate[{index}] {field_name} is not allowed: {raw_value}")

    def _string_list(self, index: int, field_name: str, payload: object) -> tuple[str, ...]:
        if payload is None:
            return ()
        if not isinstance(payload, list) or any(not isinstance(item, str) or not item for item in payload):
            raise ValueError(f"candidate[{index}] {field_name} must contain only non-empty strings")
        return tuple(payload)

    def _source_role(self, evidence_refs: tuple[EvidenceRef, ...], episode: EvidenceEpisode) -> str:
        roles = {event.actor.kind for ref in evidence_refs if (event := episode.event(ref.event_id)) is not None}
        if roles == {"user"}:
            return "user"
        if "tool" in roles or "sensor" in roles or "robot" in roles:
            return "tool"
        if "assistant" in roles:
            return "assistant"
        return "unknown"

    def _evidence_refs(self, payload: object, episode: EvidenceEpisode) -> tuple[EvidenceRef, ...]:
        if not isinstance(payload, list):
            raise ValueError("evidence_refs must be a list")
        refs = []
        for item in payload:
            if not isinstance(item, dict) or not item.get("event_id"):
                raise ValueError("each evidence_ref requires event_id")
            unknown = set(item) - {
                "event_id",
                "event_digest",
                "content_hash",
                "content_path",
                "span_start",
                "span_end",
                "quoted_text",
                "quoted_text_hash",
            }
            if unknown:
                raise ValueError(f"evidence_ref contains unknown fields: {','.join(sorted(unknown))}")
            event = episode.event(str(item["event_id"]))
            if event is None:
                raise ValueError(f"evidence_ref event does not exist in episode: {item['event_id']}")
            span_start = int(item["span_start"]) if item.get("span_start") is not None else None
            span_end = int(item["span_end"]) if item.get("span_end") is not None else None
            text = event.text()
            if (span_start is None) != (span_end is None):
                raise ValueError(f"evidence_ref has an incomplete span: {item['event_id']}")
            if (
                span_start is not None
                and span_end is not None
                and (span_start < 0 or span_end <= span_start or span_end > len(text))
            ):
                raise ValueError(f"evidence_ref span is invalid: {item['event_id']}")
            derived = EvidenceRef.from_event(
                event,
                source_uri=episode.source_uris[0] if episode.source_uris else None,
                content_path=str(item.get("content_path") or event.content_path),
                span_start=span_start,
                span_end=span_end,
            )
            if item.get("event_digest") and str(item["event_digest"]) != derived.event_digest:
                raise ValueError(f"evidence_ref event_digest mismatch: {item['event_id']}")
            if item.get("content_hash") and str(item["content_hash"]) != derived.content_hash:
                raise ValueError(f"evidence_ref content_hash mismatch: {item['event_id']}")
            if item.get("quoted_text") is not None and str(item["quoted_text"]) != derived.quoted_text:
                raise ValueError(f"evidence_ref quoted_text mismatch: {item['event_id']}")
            if item.get("quoted_text_hash") and str(item["quoted_text_hash"]) != derived.quoted_text_hash:
                raise ValueError(f"evidence_ref quoted_text_hash mismatch: {item['event_id']}")
            refs.append(derived)
        return tuple(refs)

    def _field_evidence_refs(
        self,
        index: int,
        payload: object,
        episode: EvidenceEpisode,
        *,
        identity_fields: dict,
        value_fields: dict,
        evidence_refs: tuple[EvidenceRef, ...],
    ) -> dict[str, tuple[EvidenceRef, ...]]:
        if not isinstance(payload, dict):
            raise ValueError(f"candidate[{index}] field_evidence_refs must be an object")
        required = {
            *[f"identity.{key}" for key in identity_fields],
            *[f"value.{key}" for key in value_fields],
            "semantic.speech_act",
            "semantic.commitment",
            "semantic.temporal_scope",
            "semantic.relation_to_existing",
            "transition",
        }
        if set(payload) != required:
            missing = required - set(payload)
            unknown = set(payload) - required
            details = [
                *(f"missing:{key}" for key in sorted(missing)),
                *(f"unknown:{key}" for key in sorted(unknown)),
            ]
            raise ValueError(f"candidate[{index}] field_evidence_refs mismatch: {','.join(details)}")
        allowed = set(evidence_refs)
        results = {}
        for field_name in sorted(required):
            refs = self._evidence_refs(payload[field_name], episode)
            if not refs or any(ref not in allowed for ref in refs):
                raise ValueError(f"candidate[{index}] field_evidence_refs invalid for {field_name}")
            results[field_name] = refs
        return results


@dataclass
class FakeMemoryModelProvider:
    """约定 FakeMemoryModelProvider 需要提供的接口。"""

    response: str
    prompts: list[str] | None = None

    def complete(self, prompt: str) -> str:
        if self.prompts is not None:
            self.prompts.append(prompt)
        return self.response
