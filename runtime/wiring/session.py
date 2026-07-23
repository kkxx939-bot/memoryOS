"""SessionArchive、提交规划和消费流程的装配。"""

from __future__ import annotations

from typing import cast

from behavior.execute.session_commit_planner import BehaviorCommitPlanner
from infrastructure.context.projection_journal import SessionProjectionJournal
from infrastructure.context.session_projector import CatalogProjectionStore, SessionContextProjector
from infrastructure.store.filesystem.session_archive import SessionArchiveStore
from infrastructure.store.session.archive_event_encoder import CanonicalSessionArchiveEventEncoder
from infrastructure.store.session.commit_group import CommitGroupStore
from policy.action_policy.planning.session_commit_planner import ActionPolicyCommitPlanner
from runtime.config import RuntimeConfig
from runtime.container import SessionRuntime, StoreRuntime, TransactionRuntime
from runtime.session.commit_service import SessionCommitService


def wire_session(
    stores: StoreRuntime,
    transaction: TransactionRuntime,
    config: RuntimeConfig,
    *,
    tenant_root,  # noqa: ANN001
) -> SessionRuntime:
    """创建 SessionArchive Store，并把事件编码器作为显式依赖传入。"""

    archive_store = SessionArchiveStore(
        config.root_path,
        tenant_id=config.tenant_id,
        event_encoder=CanonicalSessionArchiveEventEncoder(),
    )
    session_projector = SessionContextProjector(
        cast(CatalogProjectionStore, stores.index),
        vector_store=stores.vector,
        embedding_provider=stores.embedding,
        vectorize_important_events=config.retrieval.vectorize_important_session_events,
    )
    commit_service = SessionCommitService(
        archive_store,
        stores.queue,
        committer=transaction.committer,
        behavior_planner=BehaviorCommitPlanner(index_store=stores.index, source_store=stores.source),
        action_policy_planner=ActionPolicyCommitPlanner(index_store=stores.index, source_store=stores.source),
        session_projector=session_projector,
        commit_group_store=CommitGroupStore(tenant_root),
        projection_journal=SessionProjectionJournal(stores.index),
    )
    return SessionRuntime(
        archive_store=archive_store,
        commit_service=commit_service,
    )


__all__ = ["wire_session"]
