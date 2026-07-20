"""构造由 Runtime 注入统一事务内核的 ActionPolicy 扩展。"""

from __future__ import annotations

from policy.action_policy.integration.commit_handler import ActionPolicyCommitHandler
from policy.action_policy.integration.conflict_policy import ActionPolicyConflictPolicy
from transaction.commit.domain_protocols import TransactionDomainExtensions


def build_action_policy_transaction_extensions() -> TransactionDomainExtensions:
    return TransactionDomainExtensions(
        conflict_policy=ActionPolicyConflictPolicy(),
        handlers=(ActionPolicyCommitHandler(),),
    )


__all__ = [
    "build_action_policy_transaction_extensions",
]
