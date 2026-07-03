from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ContextLayerName(str, Enum):
    L0 = "l0"
    L1 = "l1"
    L2 = "l2"


@dataclass(frozen=True)
class ContextLayer:
    name: ContextLayerName
    uri: str
    token_estimate: int = 0
    content_type: str = "text/markdown"

    def to_dict(self) -> dict:
        return {
            "name": self.name.value,
            "uri": self.uri,
            "token_estimate": self.token_estimate,
            "content_type": self.content_type,
        }


@dataclass(frozen=True)
class ContextLayers:
    l0_uri: str | None = None
    l1_uri: str | None = None
    l2_uri: str | None = None

    def select(self, layer: ContextLayerName) -> str | None:
        if layer == ContextLayerName.L0:
            return self.l0_uri
        if layer == ContextLayerName.L1:
            return self.l1_uri
        return self.l2_uri

    def to_dict(self) -> dict:
        return {"l0_uri": self.l0_uri, "l1_uri": self.l1_uri, "l2_uri": self.l2_uri}
