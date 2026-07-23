"""内置供应商协议适配器。"""

from LLMClient.providers.litellm import LiteLLMProvider
from LLMClient.providers.openai_compatible import OpenAICompatibleProvider

__all__ = ["LiteLLMProvider", "OpenAICompatibleProvider"]
