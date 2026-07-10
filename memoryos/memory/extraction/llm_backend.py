from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.episode import EvidenceEpisode
from memoryos.memory.canonical.evidence import EvidenceRef
from memoryos.memory.canonical.prefetch import PrefetchedMemory
from memoryos.memory.canonical.proposal import (
    EpistemicStatus,
    MemorySemanticProposal,
    SemanticAssessment,
)
from memoryos.memory.canonical.scope import ScopeRef
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType, MemoryTypeSchema
from memoryos.memory.view import adapter_id_from_archive


class MemoryModelProvider(Protocol):
    def complete(self, prompt: str) -> str: ...


class MemoryExtractionPromptBuilder:
    def build(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
        *,
        existing_memories: Sequence[PrefetchedMemory] = (),
        episode: EvidenceEpisode | None = None,
    ) -> str:
        schema_names = ", ".join(schema.memory_type.value for schema in schemas)
        messages = "\n".join(json.dumps(message, ensure_ascii=False, sort_keys=True) for message in archive.messages)
        existing = json.dumps(
            [
                {
                    "uri": item.uri,
                    "memory_type": item.memory_type,
                    "state": item.state,
                    "revision": item.revision,
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
                    "content_hash": EvidenceRef.from_event(event).content_hash,
                    "text": event.text(),
                }
                for event in episode.events
            ]
            legal_scopes = [scope.to_dict() for scope in episode.legal_scope_candidates()]
        return (
            "Extract durable memory semantic proposals as JSON. "
            "Return an object with a candidates array. Each proposal may describe semantics but is not a database operation. "
            f"Allowed memory_type values: {schema_names}. "
            "Use identity_fields and value_fields separately. Include speech_act, commitment, temporal_scope, "
            "relation_to_existing, epistemic_status, evidence_refs, and only suggested_scope_refs selected from legal scopes. "
            "Do not output operations or target URIs. Do not output tenant IDs, visibility policy, revisions, DELETE, or scope moves.\n"
            f"LEGAL_SCOPES={json.dumps(legal_scopes, ensure_ascii=False, sort_keys=True)}\n"
            f"EXISTING_MEMORIES={existing}\n"
            f"EPISODE_EVENTS={json.dumps(events, ensure_ascii=False, sort_keys=True)}\n"
            f"MESSAGES={messages}"
        )


class MemoryExtractionJsonParser:
    VALID_SOURCE_ROLES = {"user", "assistant", "agent", "tool", "unknown"}

    def parse(
        self,
        response: str,
        *,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> list[MemoryCandidateDraft]:
        payload = self._load_json(response)
        if isinstance(payload, dict):
            raw_candidates = payload.get("candidates", [])
        else:
            raw_candidates = payload
        if not isinstance(raw_candidates, list):
            raise ValueError("memory extraction candidates must be a list")
        allowed = {schema.memory_type for schema in schemas}
        candidates = []
        adapter_id = adapter_id_from_archive(archive)
        for index, raw in enumerate(raw_candidates):
            if not isinstance(raw, dict):
                raise ValueError(f"candidate[{index}] must be an object")
            memory_type = MemoryType(str(raw.get("memory_type", "")))
            if memory_type not in allowed:
                raise ValueError(f"candidate[{index}] memory_type is not allowed: {memory_type.value}")
            role = str(raw.get("source_role", "unknown") or "unknown").lower()
            if role not in self.VALID_SOURCE_ROLES:
                raise ValueError(f"candidate[{index}] source_role is not allowed: {role}")
            fields = raw.get("fields", {})
            if not isinstance(fields, dict):
                raise ValueError(f"candidate[{index}] fields must be an object")
            content = str(raw.get("content", raw.get("text", ""))).strip()
            title = str(raw.get("title", content[:64] or memory_type.value)).strip()
            if not content:
                raise ValueError(f"candidate[{index}] content is required")
            candidates.append(
                MemoryCandidateDraft(
                    memory_type=memory_type,
                    title=title,
                    content=content,
                    fields=fields,
                    confidence=float(raw.get("confidence", 0.5)),
                    source_role=role,
                    source_adapter_id=str(raw.get("source_adapter_id") or adapter_id),
                    source_session_id=str(raw.get("source_session_id") or archive.session_id),
                    source_message_ids=[str(item) for item in raw.get("source_message_ids", []) if item],
                    evidence=[item for item in raw.get("evidence", []) if isinstance(item, dict)],
                    suggested_retrieval_views=[],
                    merge_key=str(raw.get("merge_key", "")),
                    reason=str(raw.get("reason", "llm_candidate")),
                )
            )
        return candidates

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
    def __init__(
        self,
        provider: MemoryModelProvider,
        prompt_builder: MemoryExtractionPromptBuilder | None = None,
        parser: MemoryExtractionJsonParser | None = None,
        model_id: str | None = None,
        extractor_version: str = "llm_semantic_extractor_v1",
    ) -> None:
        self.provider = provider
        self.prompt_builder = prompt_builder or MemoryExtractionPromptBuilder()
        self.parser = parser or MemoryExtractionJsonParser()
        self.model_id = model_id
        self.extractor_version = extractor_version

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> list[MemoryCandidateDraft]:
        prompt = self.prompt_builder.build(archive, schemas)
        response = self.provider.complete(prompt)
        return self.parser.parse(response, archive=archive, schemas=schemas)

    def extract_with_context(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
        *,
        existing_memories: Sequence[PrefetchedMemory],
        episode: EvidenceEpisode,
    ) -> list[MemoryCandidateDraft | MemorySemanticProposal]:
        prompt = self.prompt_builder.build(
            archive,
            schemas,
            existing_memories=existing_memories,
            episode=episode,
        )
        response = self.provider.complete(prompt)
        payload = self.parser._load_json(response)
        raw_candidates = payload.get("candidates", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_candidates, list):
            raise ValueError("memory extraction candidates must be a list")
        if not any(isinstance(item, dict) and "identity_fields" in item for item in raw_candidates):
            legacy: list[MemoryCandidateDraft | MemorySemanticProposal] = list(
                self.parser.parse(response, archive=archive, schemas=schemas)
            )
            return legacy
        allowed = {schema.memory_type.value for schema in schemas}
        proposals: list[MemoryCandidateDraft | MemorySemanticProposal] = []
        for index, raw in enumerate(raw_candidates):
            if not isinstance(raw, dict):
                raise ValueError(f"candidate[{index}] must be an object")
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
            evidence_refs = self._evidence_refs(raw.get("evidence_refs", []), episode)
            scopes = tuple(
                ScopeRef.from_dict(item) for item in raw.get("suggested_scope_refs", []) or [] if isinstance(item, dict)
            )
            proposal_id = str(
                raw.get("proposal_id") or f"proposal_{stable_hash([episode.episode_id, index, raw], length=32)}"
            )
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
                    related_memory_ids=tuple(str(item) for item in raw.get("related_memory_ids", []) or []),
                    evidence_refs=evidence_refs,
                    confidence=float(raw.get("confidence", 0.5)),
                    extractor_version=self.extractor_version,
                    model_id=self.model_id,
                    metadata={
                        "source_role": self._source_role(evidence_refs, episode),
                        "source_adapter_id": adapter_id_from_archive(archive),
                        "source_session_id": archive.session_id,
                    },
                )
            )
        return proposals

    def _source_role(
        self, evidence_refs: tuple[EvidenceRef, ...], episode: EvidenceEpisode
    ) -> str:
        roles = {
            event.actor.kind
            for ref in evidence_refs
            if (event := episode.event(ref.event_id)) is not None
        }
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
                continue
            event = episode.event(str(item["event_id"]))
            if event is None:
                refs.append(
                    EvidenceRef(
                        event_id=str(item["event_id"]),
                        source_uri=None,
                        content_hash=str(item.get("content_hash", "")),
                    )
                )
                continue
            derived = EvidenceRef.from_event(
                event,
                source_uri=episode.source_uris[0] if episode.source_uris else None,
                span_start=int(item["span_start"]) if item.get("span_start") is not None else None,
                span_end=int(item["span_end"]) if item.get("span_end") is not None else None,
            )
            refs.append(
                EvidenceRef(
                    event_id=derived.event_id,
                    source_uri=derived.source_uri,
                    content_hash=str(item.get("content_hash") or derived.content_hash),
                    span_start=derived.span_start,
                    span_end=derived.span_end,
                    quoted_text_hash=str(item.get("quoted_text_hash") or derived.quoted_text_hash or "") or None,
                )
            )
        return tuple(refs)


@dataclass
class FakeMemoryModelProvider:
    response: str
    prompts: list[str] | None = None

    def complete(self, prompt: str) -> str:
        if self.prompts is not None:
            self.prompts.append(prompt)
        return self.response
