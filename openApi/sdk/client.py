"""Context、Session 与 ActionPolicy 在线决策的进程内公开 SDK。

``MemoryOSClient`` 是本地部署的对外门面和运行时组合根：它构造应用服务、绑定
当前本地用户，然后委托给根级 Runtime 和各领域服务。协议解析属于 HTTP/MCP/CLI，
领域实现不应反向进入本模块。
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any

from foundation.identity import (
    LOCAL_STORAGE_NAMESPACE,
    LocalUserContext,
    workspace_ids_from_metadata,
)
from foundation.identity.local import PRINCIPAL_ONLY_WORKSPACE
from infrastructure.context.orchestrator import UnifiedRetrievalOrchestrator
from infrastructure.context.query_service import ContextQueryService
from infrastructure.context.query_support import (
    _scope_keys as _scope_keys,
)
from infrastructure.context.reranking import Reranker
from infrastructure.context.retrieval.embedding import EmbeddingProvider
from infrastructure.context.retrieval.hybrid_search import HybridSearch
from infrastructure.context.retrieval.query_plan import RetrievalOptions
from infrastructure.store.contracts.index import IndexStore
from infrastructure.store.contracts.lock import LockStore
from infrastructure.store.contracts.queue import QueueStore
from infrastructure.store.contracts.relation import RelationStore
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.contracts.vector import VectorStore
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.model.context.context_uri import ContextURI
from infrastructure.store.model.context.lifecycle import LifecycleState
from LLMClient import LLMClient
from LLMClient.config import ModelConfig
from openApi.retrieval_contract import parse_retrieval_options
from openApi.session_service import SessionApplicationService
from policy.action_policy.decision.request import PredictionRequest
from policy.action_policy.decision.result import PredictionResult
from policy.action_policy.execution.tool_registry import ToolRegistry
from policy.action_policy.model.action_policy import ActionPolicy
from policy.action_policy.workflow.result import ProcessObservationResult
from policy.action_policy.workflow.service import ActionPolicyWorkflowService
from pre.connect import ConnectMetadata, ConnectType, PipelineMode


class MemoryOSClient:
    """MemoryOS 进程内应用的稳定公开门面。"""

    def __init__(
        self,
        root: str,
        index_store: IndexStore | None = None,
        source_store: SourceStore | None = None,
        relation_store: RelationStore | None = None,
        queue_store: QueueStore | None = None,
        lock_store: LockStore | None = None,
        tool_registry: ToolRegistry | None = None,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        hybrid_search: HybridSearch | None = None,
        reranker: Reranker | None = None,
        model_config: ModelConfig | None = None,
        model_client: LLMClient | None = None,
        mode: str = "local",
        user_id: str = "local-user",
        adapter_id: str = "local_sdk",
    ) -> None:
        # SDK 只请求根级 RuntimeBuilder 创建进程运行时，不再复制内部依赖图。
        from runtime import RuntimeBuilder, RuntimeConfig, RuntimeDependencies

        self.root = root
        self.mode = mode
        self.tenant_id = LOCAL_STORAGE_NAMESPACE
        self.local_context = LocalUserContext(user_id=user_id, adapter_id=adapter_id)
        runtime = RuntimeBuilder(
            RuntimeConfig(
                root=root,
                mode=mode,
                tenant_id=LOCAL_STORAGE_NAMESPACE,
                model=model_config or ModelConfig(),
            ),
            RuntimeDependencies(
                index_store=index_store,
                source_store=source_store,
                relation_store=relation_store,
                queue_store=queue_store,
                lock_store=lock_store,
                tool_registry=tool_registry,
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                hybrid_search=hybrid_search,
                reranker=reranker,
                model_client=model_client,
            ),
        ).build()
        runtime.start()
        self.runtime = runtime
        self._last_recall_trace_id: ContextVar[str] = ContextVar(
            f"memoryos_last_recall_trace_id_{id(self)}",
            default="",
        )
        # 应用服务按需创建，每个服务只接收自己实际使用的依赖。
        self._context_queries: ContextQueryService | None = None
        self._action_policy_workflow_service: ActionPolicyWorkflowService | None = None
        self._session_application: SessionApplicationService | None = None

    def _build_context_queries(self) -> ContextQueryService:
        """装配上下文查询服务所需的最小依赖集合。"""

        return ContextQueryService(
            root=str(getattr(self, "root", "/tmp/memoryos-test")),
            tenant_id=str(getattr(self, "tenant_id", "default")),
            source_store=self.runtime.stores.source,
            context_reader=self.runtime.context.facade,
            readiness=self.runtime.readiness,
            effective_tenant=self._effective_tenant,
            connect_filters_from_metadata=self._connect_filters_from_metadata,
            project_id_from_metadata=self._project_id_from_metadata,
            parse_connect_metadata=self._parse_connect_metadata,
            retrieval_orchestrator=lambda: self._retrieval_orchestrator(),
            require_exact_workspace=self._require_exact_workspace,
            require_exact_read_scope=self._require_exact_read_scope,
            set_last_recall_trace_id=self._set_last_recall_trace_id,
        )

    def _build_action_policy_workflow_service(self) -> ActionPolicyWorkflowService:
        """装配 ActionPolicy 工作流，不把完整 SDK 运行时传入内部逻辑。"""

        return ActionPolicyWorkflowService(
            tenant_id=str(getattr(self, "tenant_id", "default")),
            engine=self.runtime.policy.engine,
            executor=self.runtime.policy.executor,
            session_commit_service=self.runtime.session.commit_service,
            readiness=self.runtime.readiness,
            require_predict_metadata=self._require_predict_metadata,
            require_process_observation_metadata=self._require_process_observation_metadata,
        )

    def _build_session_application(self) -> SessionApplicationService:
        """装配会话服务所需的归档、提交、检索和健康检查依赖。"""

        return SessionApplicationService(
            root=str(getattr(self, "root", "/tmp/memoryos-test")),
            mode=str(getattr(self, "mode", "local")),
            tenant_id=str(getattr(self, "tenant_id", "default")),
            search_context=self.search_context,
            session_archive_store=self.runtime.session.archive_store,
            queue_store=self.runtime.stores.queue,
            readiness=self.runtime.readiness,
            session_commit_service=self.runtime.session.commit_service,
            embedding_provider=self.runtime.stores.embedding,
            vector_store=self.runtime.stores.vector,
            reranker=self.runtime.stores.reranker,
            model_client=self.runtime.stores.model_client,
            effective_tenant=self._effective_tenant,
            require_exact_workspace=self._require_exact_workspace,
            parse_connect_metadata=self._parse_connect_metadata,
            project_id_from_metadata=self._project_id_from_metadata,
        )

    def _set_last_recall_trace_id(self, trace_id: str) -> None:
        """在当前请求上下文中记录轨迹，避免并发请求相互覆盖。"""

        self._trace_id_context().set(trace_id)

    @property
    def last_recall_trace_id(self) -> str:
        """返回当前线程或异步任务最近一次查询产生的轨迹 ID。"""

        return self._trace_id_context().get()

    def _trace_id_context(self) -> ContextVar[str]:
        """兼容最小测试装配，并始终返回请求隔离的轨迹容器。"""

        context = getattr(self, "_last_recall_trace_id", None)
        if isinstance(context, ContextVar):
            return context
        context = ContextVar(f"memoryos_last_recall_trace_id_{id(self)}", default="")
        self._last_recall_trace_id = context
        return context

    def _get_context_queries(self) -> ContextQueryService:
        service = getattr(self, "_context_queries", None)
        if service is None:
            service = self._build_context_queries()
            self._context_queries = service
        return service

    def _get_action_policy_workflow_service(self) -> ActionPolicyWorkflowService:
        service = getattr(self, "_action_policy_workflow_service", None)
        if service is None:
            service = self._build_action_policy_workflow_service()
            self._action_policy_workflow_service = service
        return service

    def _get_session_application(self) -> SessionApplicationService:
        service = getattr(self, "_session_application", None)
        if service is None:
            service = self._build_session_application()
            self._session_application = service
        return service

    def predict(self, request: PredictionRequest, policies: list[ActionPolicy] | None = None) -> PredictionResult:
        """调用行为预测应用服务，不在 SDK 内实现预测规则。"""

        """处理具身场景的在线动作预测请求。"""

        return self._get_action_policy_workflow_service().predict(request, policies=policies)

    def process_observation(
        self,
        request: PredictionRequest,
        policies: list[ActionPolicy] | None = None,
        *,
        archive_session: bool = True,
        async_commit: bool = True,
    ) -> ProcessObservationResult:
        """处理观察、动作执行以及可选的会话归档。"""

        return self._get_action_policy_workflow_service().process_observation(
            request,
            policies=policies,
            archive_session=archive_session,
            async_commit=async_commit,
        )

    def search_context(
        self,
        query: str,
        *,
        options: RetrievalOptions | Mapping[str, Any] | None = None,
        user_id: str | None = None,
        context_type: object | None = None,
        limit: int = 10,
        connect_metadata: dict[str, Any] | None = None,
        search_scope: str | None = None,
        retrieval_views: list[str] | None = None,
        project_id: str = "",
        applicability_scopes: list[dict[str, Any]] | None = None,
        record_kinds: list[str] | None = None,
        query_intent: str | None = None,
        caller: LocalUserContext | None = None,
    ) -> list[dict[str, Any]]:
        """执行受本地用户和工作区边界约束的统一上下文搜索。"""

        structured = parse_retrieval_options(options)
        effective_tenant = self._effective_tenant(caller, structured.tenant_id if structured else None)
        return self._get_context_queries().search_context(
            query,
            options=structured,
            user_id=user_id,
            context_type=context_type,
            limit=limit,
            connect_metadata=connect_metadata,
            search_scope=search_scope,
            retrieval_views=retrieval_views,
            project_id=project_id,
            tenant_id=effective_tenant,
            applicability_scopes=applicability_scopes,
            record_kinds=record_kinds,
            query_intent=query_intent,
            caller=caller,
        )

    def assemble_context(
        self,
        query: str,
        *,
        options: RetrievalOptions | Mapping[str, Any] | None = None,
        user_id: str | None = None,
        context_types: list[object] | None = None,
        limit: int = 20,
        connect_metadata: dict[str, Any] | None = None,
        search_scope: str | None = None,
        retrieval_views: list[str] | None = None,
        project_id: str = "",
        applicability_scopes: list[dict[str, Any]] | None = None,
        record_kinds: list[str] | None = None,
        query_intent: str | None = None,
        caller: LocalUserContext | None = None,
    ) -> dict[str, Any]:
        structured = parse_retrieval_options(options)
        effective_tenant = self._effective_tenant(caller, structured.tenant_id if structured else None)
        return self._get_context_queries().assemble_context(
            query,
            options=structured,
            user_id=user_id,
            context_types=context_types,
            limit=limit,
            connect_metadata=connect_metadata,
            search_scope=search_scope,
            retrieval_views=retrieval_views,
            project_id=project_id,
            tenant_id=effective_tenant,
            applicability_scopes=applicability_scopes,
            record_kinds=record_kinds,
            query_intent=query_intent,
            caller=caller,
        )

    def recall_trace(
        self,
        trace_id: str,
        *,
        caller: LocalUserContext | None = None,
    ) -> dict[str, Any]:
        return self._get_context_queries().recall_trace(trace_id, caller=caller)

    def read(
        self,
        uri: str,
        *,
        layer: str = "L2",
        caller: LocalUserContext | None = None,
    ) -> dict[str, Any]:
        effective_tenant = self._effective_tenant(caller, None)
        return self._get_context_queries().read(uri, layer=layer, tenant_id=effective_tenant, caller=caller)

    def archive_read(
        self,
        archive_uri: str,
        *,
        caller: LocalUserContext | None = None,
    ) -> dict[str, Any]:
        effective_tenant = self._effective_tenant(caller, None)
        return self._get_session_application().archive_read(
            archive_uri,
            tenant_id=effective_tenant,
            caller=caller,
        )

    def archive_search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 20,
        caller: LocalUserContext | None = None,
        project_id: str = "",
        timezone_name: str = "UTC",
    ) -> list[dict[str, Any]]:
        effective_tenant = self._effective_tenant(caller, None)
        return self._get_session_application().archive_search(
            query,
            user_id=user_id,
            limit=limit,
            tenant_id=effective_tenant,
            caller=caller,
            project_id=project_id,
            timezone_name=timezone_name,
            search_context=self.search_context,
            archive_read=self.archive_read,
        )

    def health(self) -> dict[str, Any]:
        """返回外部通道可消费的运行时健康状态。"""

        return self._get_session_application().health()

    def commit_agent_session(
        self,
        *,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]] | None = None,
        used_contexts: list[dict[str, Any]] | None = None,
        used_skills: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        connect_metadata: dict[str, Any] | None = None,
        async_commit: bool = True,
        project_id: str = "",
        session_key: str = "",
        scope: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        caller: LocalUserContext | None = None,
    ) -> Any:
        """按当前本地用户提交一次已规范化的 Agent 会话。"""

        effective_tenant = self._effective_tenant(caller, None)
        return self._get_session_application().commit_agent_session(
            user_id=user_id,
            session_id=session_id,
            messages=messages,
            used_contexts=used_contexts,
            used_skills=used_skills,
            tool_results=tool_results,
            connect_metadata=connect_metadata,
            async_commit=async_commit,
            project_id=project_id,
            session_key=session_key,
            scope=scope,
            provenance=provenance,
            tenant_id=effective_tenant,
            caller=caller,
        )

    def _require_exact_read_scope(
        self,
        uri: str,
        obj: Any,
        caller: LocalUserContext,
    ) -> None:
        if obj.uri != uri or str(obj.tenant_id or "default") != caller.tenant_id:
            raise FileNotFoundError(uri)
        if obj.owner_user_id != caller.user_id:
            raise FileNotFoundError(uri)
        parsed = ContextURI.parse(uri)
        if parsed.authority == "user" and parsed.user_id != caller.user_id:
            raise FileNotFoundError(uri)
        if obj.lifecycle_state != LifecycleState.ACTIVE:
            raise FileNotFoundError(uri)
        metadata = dict(obj.metadata or {})
        self._require_exact_workspace(metadata, caller, uri)

    def _effective_tenant(
        self,
        caller: LocalUserContext | None,
        explicit_tenant_id: str | None,
    ) -> str:
        if caller is not None:
            caller.assert_identity(tenant_id=explicit_tenant_id)
        if explicit_tenant_id not in {None, LOCAL_STORAGE_NAMESPACE}:
            raise ValueError("tenant selection is unavailable in local single-user mode")
        return LOCAL_STORAGE_NAMESPACE

    def _require_ready(self) -> None:
        self.runtime.readiness.require_ready()

    def _require_exact_workspace(
        self,
        metadata: dict[str, Any],
        caller: LocalUserContext,
        target: str,
    ) -> None:
        try:
            workspace_ids = workspace_ids_from_metadata(metadata)
        except (TypeError, ValueError):
            raise FileNotFoundError(target) from None
        if caller.workspace_id and workspace_ids and workspace_ids != {caller.workspace_id}:
            raise FileNotFoundError(target)

    def _workspace_matches(
        self,
        metadata: dict[str, Any],
        project_id: str,
        caller: LocalUserContext | None,
    ) -> bool:
        try:
            workspace_ids = workspace_ids_from_metadata(metadata)
        except (TypeError, ValueError):
            return False
        if not project_id:
            return True
        if project_id == PRINCIPAL_ONLY_WORKSPACE:
            return not workspace_ids
        return not workspace_ids or workspace_ids == {project_id}

    def _parse_connect_metadata(self, payload: dict[str, Any] | None) -> ConnectMetadata:
        return ConnectMetadata.from_dict(payload)

    def _require_predict_metadata(self, payload: dict[str, Any] | None) -> ConnectMetadata:
        if not payload:
            raise PermissionError(
                "predict() requires explicit embodied/action_capable connect metadata with can_predict_behavior=True."
            )
        metadata = self._parse_connect_metadata(payload)
        if (
            metadata.connect_type != ConnectType.EMBODIED
            or metadata.run_mode != PipelineMode.ACTION_CAPABLE
            or not metadata.capabilities.can_predict_behavior
        ):
            raise PermissionError(
                "predict() requires embodied/action_capable connect metadata "
                "with can_predict_behavior=True; use assemble_context() for context_reduction agents."
            )
        return metadata

    def _require_process_observation_metadata(self, payload: dict[str, Any] | None) -> ConnectMetadata:
        metadata = self._require_predict_metadata(payload)
        if not metadata.capabilities.can_execute_action:
            raise PermissionError(
                "process_observation() requires embodied/action_capable connect metadata "
                "with can_predict_behavior=True and can_execute_action=True."
            )
        return metadata

    def _connect_filters_from_metadata(self, connect_metadata: dict[str, Any] | None) -> dict[str, str]:
        if not connect_metadata:
            return {}
        allowed = {"connect_type", "adapter_id", "run_mode", "world_domain", "source_kind"}
        metadata = self._parse_connect_metadata(connect_metadata)
        metadata.validate()
        metadata_dict = metadata.to_dict()
        return {
            key: str(metadata_dict[key])
            for key in allowed
            if key in connect_metadata and metadata_dict.get(key) not in {None, ""}
        }

    def _parse_context_type(self, context_type: object) -> ContextType:
        if isinstance(context_type, ContextType):
            return context_type
        return ContextType(str(context_type))

    def _retrieval_orchestrator(self) -> UnifiedRetrievalOrchestrator:
        return UnifiedRetrievalOrchestrator(
            self.runtime.stores.index,
            source_store=self.runtime.stores.source,
            relation_store=self.runtime.stores.relation,
            session_archive_store=self.runtime.session.archive_store,
            readiness=self.runtime.readiness,
            serving_lock=self.runtime.context.facade.serving_lock,
            serving_generation_token=self.runtime.context.facade.serving_generation_token,
            vector_store=self.runtime.stores.vector,
            embedding_provider=self.runtime.stores.embedding,
            reranker=self.runtime.stores.reranker,
        )

    def _project_id_from_metadata(self, connect_metadata: dict[str, Any] | None) -> str:
        metadata = dict(connect_metadata or {})
        for key in ("project_id", "project"):
            if metadata.get(key):
                return str(metadata[key])
        extra = metadata.get("extra")
        if isinstance(extra, dict):
            for key in ("project_id", "project", "repo"):
                if extra.get(key):
                    return str(extra[key])
        return ""

    def _uri_items(self, uris: list[str]) -> list[dict[str, str]]:
        return [{"uri": uri} for uri in dict.fromkeys(str(uri) for uri in uris if uri)]

    def _merge_uri_items(self, *groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                uri = str(item.get("uri", ""))
                if not uri or uri in seen:
                    continue
                seen.add(uri)
                merged.append(dict(item))
        return merged


class LocalMemoryOSClient(MemoryOSClient):
    """进程内客户端的本地兼容名称。"""
