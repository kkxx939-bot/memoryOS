from __future__ import annotations

from memoryos.behavior.model.behavior_pattern import BehaviorCluster
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class BehaviorClusterUpdater:
    def add_cluster(self, cluster: BehaviorCluster) -> ContextOperation:
        uri = f"memoryos://user/{cluster.user_id}/behavior/clusters/{cluster.scene_key}/{cluster.cluster_id}"
        obj = ContextObject(
            uri=uri,
            context_type=ContextType.BEHAVIOR_CLUSTER,
            title=f"BehaviorCluster {cluster.scene_key}",
            owner_user_id=cluster.user_id,
            metadata=cluster.__dict__,
        )
        return ContextOperation(
            user_id=cluster.user_id,
            context_type=ContextType.BEHAVIOR_CLUSTER,
            action=OperationAction.ADD,
            target_uri=uri,
            payload={"context_object": obj.to_dict(), "content": cluster.__dict__},
            evidence=[{"case_refs": cluster.case_refs}],
            confidence=cluster.confidence,
        )
