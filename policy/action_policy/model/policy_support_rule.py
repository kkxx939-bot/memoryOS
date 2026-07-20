"""由本地用户明确反馈形成的 ActionPolicy 约束规则。"""

from __future__ import annotations

from dataclasses import dataclass, field

from foundation.clock import utc_now


@dataclass(frozen=True)
class PolicySupportRule:
    """描述一条策略为什么被约束，不包含 Context 或 Store 细节。"""

    uri: str
    user_id: str
    title: str
    content: str
    rule_key: str
    constrains_policy_uris: list[str] = field(default_factory=list)
    policy_rule_type: str = ""
    policy_rule_value: str = ""
    related_action: str = ""
    confidence: float = 1.0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.uri or not self.user_id or not self.rule_key:
            raise ValueError("policy support rule requires URI, user and rule key")
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(
            self,
            "constrains_policy_uris",
            list(dict.fromkeys(str(uri) for uri in self.constrains_policy_uris if str(uri))),
        )


__all__ = ["PolicySupportRule"]
