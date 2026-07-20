"""MemoryOS 对外交付层。

这里统一承载 Python SDK、HTTP、MCP 和 CLI 四种外部访问方式。各通道只负责
协议解析、输入校验、可信身份绑定、错误转换和调用编排；ContextDB、Memory、
Session 逻辑由对应领域与 Runtime 负责，在线决策归 :mod:`policy.action_policy`。
"""

from openApi.version import __version__

__all__ = ["__version__"]
