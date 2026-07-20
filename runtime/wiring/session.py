"""SessionArchive、提交规划和消费流程的装配。"""

from __future__ import annotations

from typing import cast

from behavior.execute.session_commit_planner import BehaviorCommitPlanner
from infrastructure.context.projection_journal import SessionProjectionJournal
from infrastructure.context.session_projector import CatalogProjectionStore, SessionContextProjector
from infrastructure.store.filesystem.session_archive import SessionArchiveStore
from infrastructure.store.memory.evidence import DurableSalienceLedger
from infrastructure.store.session.commit_group import CommitGroupStore
from memory.commit.evidence.archive_encoder import SessionEvidenceArchiveEncoder
from memory.commit.planner import MemoryCommitPlanner
from memory.commit.session_commit import SessionCommitService
from memory.formation.llm import LLMMemoryExtractorBackend
from policy.action_policy.planning.session_commit_planner import ActionPolicyCommitPlanner
from runtime.config import RuntimeConfig
from runtime.container import MemoryRuntime, SessionRuntime, StoreRuntime, TransactionRuntime
from runtime.dependencies import RuntimeDependencies


def wire_session(
    stores: StoreRuntime,
    transaction: TransactionRuntime,
    memory: MemoryRuntime,
    config: RuntimeConfig,
    dependencies: RuntimeDependencies,
    *,
    tenant_root,  # noqa: ANN001
) -> SessionRuntime:
    """创建会话证据 Store，并把领域编码器作为显式依赖传入。"""

    extractor = dependencies.memory_extractor
    if extractor is None and stores.model_client is not None:
        extractor = LLMMemoryExtractorBackend(stores.model_client)
    archive_store = SessionArchiveStore(
        config.root_path,
        tenant_id=config.tenant_id,
        evidence_encoder=SessionEvidenceArchiveEncoder(),
    )
    session_projector = SessionContextProjector(
        cast(CatalogProjectionStore, stores.index),
        vector_store=stores.vector,
        embedding_provider=stores.embedding,
        vectorize_important_events=config.retrieval.vectorize_important_session_events,
    )
    memory_planner = MemoryCommitPlanner(
        memory.planner,
        extractor=extractor,
        archive_store=archive_store,
        salience_ledger=DurableSalienceLedger(config.root_path, tenant_id=config.tenant_id),
        bootstrapper=memory.bootstrapper,
        proposal_store=memory.proposal_store,
        review_store=memory.review_store,
        tenant_id=config.tenant_id,
    )
    commit_service = SessionCommitService(
        archive_store,
        stores.queue,
        committer=transaction.committer,
        memory_planner=memory_planner,
        behavior_planner=BehaviorCommitPlanner(index_store=stores.index, source_store=stores.source),
        action_policy_planner=ActionPolicyCommitPlanner(index_store=stores.index, source_store=stores.source),
        session_projector=session_projector,
        commit_group_store=CommitGroupStore(tenant_root),
        memory_committer=memory.committer,
        document_planner=memory.planner,
        projection_journal=SessionProjectionJournal(stores.index),
    )
    return SessionRuntime(
        archive_store=archive_store,
        memory_planner=memory_planner,
        commit_service=commit_service,
    )


__all__ = ["wire_session"]
