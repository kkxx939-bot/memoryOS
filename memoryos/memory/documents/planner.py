"""Deterministic routing and read-before-write planning for memory proposals."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from memoryos.core.ids import stable_hash
from memoryos.memory.documents.frontmatter import parse_front_matter, render_new_document
from memoryos.memory.documents.model import (
    ABSENT,
    DocumentEditKind,
    DocumentEditPlan,
    MemoryCandidateKind,
    MemoryEditProposal,
    PresentPath,
    UnsafePath,
)
from memoryos.memory.documents.path_policy import MemoryDocumentPathPolicy
from memoryos.memory.documents.store import DocumentUnsafeError, MemoryDocumentStore

_SLUG_TOKEN = re.compile(r"[^a-z0-9]+")
_SECTION_HEADING = re.compile(r"(?m)^#{1,2}[ \t]+(.+?)[ \t]*(?:\r?\n|$)")
_SEMANTIC_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_GENERIC_SEMANTIC_TERMS = frozenset(
    {"and", "for", "from", "memory", "note", "the", "this", "topic", "user", "with"}
)


@dataclass(frozen=True)
class RelatedDocumentCandidate:
    """Content-free Catalog hint that must be revalidated against live Markdown."""

    tenant_id: str
    owner_user_id: str
    document_id: str
    relative_path: str
    source_digest: str
    relevance: float = 0.0


RelatedDocumentFinder = Callable[
    [str, str, MemoryEditProposal, int],
    Sequence[RelatedDocumentCandidate],
]


@dataclass(frozen=True)
class MemoryDocumentRouter:
    """Map semantic candidates to the finite user-visible directory taxonomy."""

    def route(self, proposal: MemoryEditProposal) -> str:
        kind = proposal.candidate_kind
        if kind is MemoryCandidateKind.PROFILE_FACT:
            return "profile.md"
        if kind is MemoryCandidateKind.PREFERENCE:
            return "preferences.md"
        if kind is MemoryCandidateKind.OPEN_LOOP:
            return "knowledge/open-loops.md"
        subject = proposal.subject or proposal.title
        if kind is MemoryCandidateKind.ENTITY_NOTE:
            hint = proposal.entity_hints[0] if proposal.entity_hints else subject
            return f"knowledge/entities/{self.safe_slug(hint)}.md"
        if kind is MemoryCandidateKind.TOPIC_NOTE:
            hint = proposal.topic_hints[0] if proposal.topic_hints else subject
            return f"knowledge/topics/{self.safe_slug(hint)}.md"
        if kind is MemoryCandidateKind.EPISODE:
            return f"knowledge/episodes/{self._date(proposal.occurred_at)}-{self.safe_slug(subject)}.md"
        if kind is MemoryCandidateKind.EXPERIENCE:
            return f"experiences/{self._date(proposal.occurred_at)}-{self.safe_slug(subject)}.md"
        raise ValueError(f"unsupported memory candidate kind: {kind}")

    @staticmethod
    def safe_slug(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
        slug = _SLUG_TOKEN.sub("-", normalized).strip("-")[:120]
        if not slug:
            slug = f"note-{stable_hash(str(value), 16)}"
        return slug

    @staticmethod
    def _date(value: str) -> str:
        if not value:
            raise ValueError("episode and experience candidates require occurred_at")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("occurred_at must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("occurred_at must include a timezone")
        return parsed.date().isoformat()


class MemoryDocumentPlanner:
    """Produce exact CAS plans without allowing the model to select storage controls."""

    def __init__(
        self,
        store: MemoryDocumentStore,
        *,
        router: MemoryDocumentRouter | None = None,
        max_front_matter_bytes: int = 32 * 1024,
        max_front_matter_depth: int = 12,
        max_edit_bytes: int = 256 * 1024,
        related_document_finder: RelatedDocumentFinder | None = None,
        max_related_documents: int = 8,
    ) -> None:
        if max_related_documents <= 0:
            raise ValueError("max_related_documents must be positive")
        self.store = store
        self.router = router or MemoryDocumentRouter()
        self.max_front_matter_bytes = max_front_matter_bytes
        self.max_front_matter_depth = max_front_matter_depth
        self.max_edit_bytes = max_edit_bytes
        self.related_document_finder = related_document_finder
        self.max_related_documents = max_related_documents

    def plan(
        self,
        proposal: MemoryEditProposal,
        *,
        tenant_id: str,
        owner_user_id: str,
        idempotency_key: str,
        evidence_digest: str,
    ) -> DocumentEditPlan:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        if len(evidence_digest) != 64:
            raise ValueError("evidence_digest must be a full SHA-256 digest")
        relative = self.router.route(proposal)
        section = self._render_section(proposal)
        related_plan = self._related_update_plan(
            proposal,
            section=section,
            tenant_id=tenant,
            owner_user_id=owner,
            routed_relative_path=relative,
            idempotency_key=idempotency_key,
            evidence_digest=evidence_digest,
        )
        if related_plan is not None:
            return related_plan
        state = self.store.read_state(tenant, owner, relative)
        if state == ABSENT:
            # A replay before the first CAS install must render the same file
            # identity and bytes. The idempotency key is already bound to the
            # trusted commit group and sealed proposal set.
            document_id = f"memdoc_{stable_hash([tenant, owner, relative, idempotency_key], 32)}"
            after = render_new_document(document_id, section)
            edit_kind = DocumentEditKind.CREATE
        elif isinstance(state, PresentPath):
            raw = self.store.read_raw(tenant, owner, relative_path=relative)
            parsed = parse_front_matter(
                raw,
                max_header_bytes=self.max_front_matter_bytes,
                max_depth=self.max_front_matter_depth,
            )
            document_id = parsed.document_id
            after = self._merge_exact(raw, parsed.body, section)
            edit_kind = DocumentEditKind.UPDATE
        elif isinstance(state, UnsafePath):
            raise DocumentUnsafeError(state.reason)
        else:  # pragma: no cover - closed union guard.
            raise DocumentUnsafeError("unknown raw path state")
        if len(after) > self.max_edit_bytes:
            raise ValueError("planned document edit exceeds the bounded edit size")
        return DocumentEditPlan(
            idempotency_key=str(idempotency_key),
            tenant_id=tenant,
            owner_user_id=owner,
            edit_kind=edit_kind,
            expected_state=state,
            evidence_digest=evidence_digest,
            edit_summary=self._summary(proposal),
            document_id=document_id,
            relative_path=relative,
            after_bytes=after,
            expected_registration_document_id=document_id if edit_kind is DocumentEditKind.UPDATE else "",
        )

    def replan(
        self,
        sealed_proposal: MemoryEditProposal,
        *,
        tenant_id: str,
        owner_user_id: str,
        idempotency_key: str,
        evidence_digest: str,
    ) -> DocumentEditPlan:
        """Deterministic CAS retry over the same sealed semantic proposal."""

        return self.plan(
            sealed_proposal,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            idempotency_key=idempotency_key,
            evidence_digest=evidence_digest,
        )

    def _related_update_plan(
        self,
        proposal: MemoryEditProposal,
        *,
        section: str,
        tenant_id: str,
        owner_user_id: str,
        routed_relative_path: str,
        idempotency_key: str,
        evidence_digest: str,
    ) -> DocumentEditPlan | None:
        """Plan against the best valid related document after an exact source CAS read.

        Catalog is deliberately a bounded path/identity hint, never write
        authority.  Its source digest may be stale: the planner always rereads
        and binds the latest live state.  A missing, cross-owner, unsafe,
        identity-mismatched, or semantically incompatible candidate is ignored.
        A verified candidate
        receives one of three deterministic outcomes: an exact duplicate is a
        no-op, a matching level-two section is replaced as a correction, and an
        otherwise new section is appended as a supplement.
        """

        if self.related_document_finder is None:
            return None
        candidates = tuple(
            self.related_document_finder(
                tenant_id,
                owner_user_id,
                proposal,
                self.max_related_documents,
            )
        )
        if len(candidates) > self.max_related_documents:
            raise ValueError("related document lookup exceeded its bound")
        ordered = sorted(
            candidates,
            key=lambda item: (-float(item.relevance), item.relative_path, item.document_id),
        )
        seen: set[tuple[str, str]] = set()
        for candidate in ordered:
            if not isinstance(candidate, RelatedDocumentCandidate):
                raise TypeError("related document lookup returned an invalid candidate")
            if candidate.tenant_id != tenant_id or candidate.owner_user_id != owner_user_id:
                raise PermissionError("related document candidate crosses trusted scope")
            relative = MemoryDocumentPathPolicy.normalize_relative_path(candidate.relative_path)
            if relative == routed_relative_path:
                continue
            identity = (candidate.document_id, relative)
            if identity in seen:
                continue
            seen.add(identity)
            if len(candidate.source_digest) != 64:
                continue
            state = self.store.read_state(tenant_id, owner_user_id, relative)
            if not isinstance(state, PresentPath):
                continue
            raw = self.store.read_raw(tenant_id, owner_user_id, relative_path=relative)
            if hashlib.sha256(raw).hexdigest() != state.raw_sha256:
                continue
            parsed = parse_front_matter(
                raw,
                max_header_bytes=self.max_front_matter_bytes,
                max_depth=self.max_front_matter_depth,
            )
            if parsed.document_id != candidate.document_id:
                continue
            if not self._semantically_compatible_related_document(
                proposal,
                live_body=parsed.body,
                section=section,
                candidate_relative_path=relative,
                routed_relative_path=routed_relative_path,
            ):
                continue
            after, disposition = self._merge_related(
                raw,
                body=parsed.body,
                header_bytes=parsed.header_bytes,
                section=section,
                title=proposal.title,
            )
            if len(after) > self.max_edit_bytes:
                raise ValueError("planned related document edit exceeds the bounded edit size")
            return DocumentEditPlan(
                idempotency_key=str(idempotency_key),
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                edit_kind=DocumentEditKind.UPDATE,
                expected_state=state,
                evidence_digest=evidence_digest,
                edit_summary=f"{disposition}: {self._summary(proposal)}"[:240],
                document_id=parsed.document_id,
                relative_path=relative,
                after_bytes=after,
                expected_registration_document_id=parsed.document_id,
            )
        return None

    @staticmethod
    def _render_section(proposal: MemoryEditProposal) -> str:
        title = " ".join(proposal.title.split()).strip()
        body = proposal.body.strip()
        if not title or not body:
            raise ValueError("proposal title and body cannot be empty")
        return f"## {title}\n\n{body}\n"

    @staticmethod
    def _merge_exact(raw: bytes, body: str, section: str) -> bytes:
        normalized_existing = "\n".join(line.rstrip() for line in body.splitlines()).strip()
        normalized_section = "\n".join(line.rstrip() for line in section.splitlines()).strip()
        if normalized_section and normalized_section in normalized_existing:
            return raw
        separator = b"" if not raw or raw.endswith(b"\n\n") else (b"\n" if raw.endswith(b"\n") else b"\n\n")
        return raw + separator + section.encode()

    @staticmethod
    def _contains_section(body: str, section: str) -> bool:
        normalized_existing = "\n".join(line.rstrip() for line in body.splitlines()).strip()
        normalized_section = "\n".join(line.rstrip() for line in section.splitlines()).strip()
        return bool(normalized_section and normalized_section in normalized_existing)

    @classmethod
    def _merge_related(
        cls,
        raw: bytes,
        *,
        body: str,
        header_bytes: bytes,
        section: str,
        title: str,
    ) -> tuple[bytes, str]:
        """Return exact related-document bytes and a content-free disposition."""

        if cls._contains_section(body, section):
            return raw, "deduplicated"

        wanted = cls._normalized_heading(title)
        headings = tuple(_SECTION_HEADING.finditer(body))
        matching_index = next(
            (
                index
                for index, match in enumerate(headings)
                if match.group(0).startswith("##")
                and not match.group(0).startswith("###")
                and cls._normalized_heading(match.group(1)) == wanted
            ),
            None,
        )
        if matching_index is None:
            return cls._merge_exact(raw, body, section), "supplemented"

        match = headings[matching_index]
        end = headings[matching_index + 1].start() if matching_index + 1 < len(headings) else len(body)
        replacement = section.rstrip("\r\n") + "\n"
        suffix = body[end:]
        if suffix:
            replacement += "\n"
        after_body = body[: match.start()] + replacement + suffix
        return header_bytes + after_body.encode("utf-8"), "corrected"

    @staticmethod
    def _normalized_heading(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())

    @classmethod
    def _semantically_compatible_related_document(
        cls,
        proposal: MemoryEditProposal,
        *,
        live_body: str,
        section: str,
        candidate_relative_path: str,
        routed_relative_path: str,
    ) -> bool:
        """Revalidate semantic compatibility from live bytes, never Catalog score."""

        if cls._contains_section(live_body, section):
            return True
        normalized_live = cls._normalized_heading(live_body)
        live_tokens = tuple(_SEMANTIC_TOKEN.findall(normalized_live))
        live_headings = {
            cls._normalized_heading(match.group(1))
            for match in _SECTION_HEADING.finditer(live_body)
        }
        semantic_values = tuple(
            value
            for value in (
                proposal.title,
                proposal.subject,
                *proposal.entity_hints,
                *proposal.topic_hints,
            )
            if cls._normalized_heading(value)
        )
        normalized_values = tuple(cls._normalized_heading(value) for value in semantic_values)
        for value in normalized_values:
            value_tokens = tuple(_SEMANTIC_TOKEN.findall(value))
            meaningful = tuple(
                token
                for token in value_tokens
                if len(token) >= 3 and token not in _GENERIC_SEMANTIC_TERMS
            )
            if not meaningful:
                continue
            if value in live_headings or _contains_token_sequence(live_tokens, value_tokens):
                return True

        proposal_tokens = {
            token
            for value in normalized_values
            for token in _SEMANTIC_TOKEN.findall(value)
            if len(token) >= 3 and token not in _GENERIC_SEMANTIC_TERMS
        }
        live_token_set = {
            token
            for token in live_tokens
            if len(token) >= 3 and token not in _GENERIC_SEMANTIC_TERMS
        }
        if not proposal_tokens.intersection(live_token_set):
            return False
        return cls._route_family(candidate_relative_path) == cls._route_family(
            routed_relative_path
        )

    @staticmethod
    def _route_family(relative_path: str) -> str:
        relative = MemoryDocumentPathPolicy.normalize_relative_path(relative_path)
        if relative.startswith("knowledge/"):
            return "knowledge"
        if relative.startswith("experiences/"):
            return "experiences"
        return relative

    @staticmethod
    def _summary(proposal: MemoryEditProposal) -> str:
        value = f"{proposal.candidate_kind.value}: {' '.join(proposal.title.split())}"
        return value[:240]


def _contains_token_sequence(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(haystack[index : index + width] == needle for index in range(len(haystack) - width + 1))


def explicit_evidence_digest(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


__all__ = [
    "MemoryDocumentPlanner",
    "MemoryDocumentRouter",
    "RelatedDocumentCandidate",
    "RelatedDocumentFinder",
    "explicit_evidence_digest",
]
