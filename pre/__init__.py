"""在 Memory、Context、Behavior 和 ActionPolicy 分流前准备公共业务事实。"""

# 根包保持轻量；调用者必须从 pre.connect、pre.session 或 pre.evidence
# 显式导入，避免只需要 SessionArchive 时提前加载整套证据模型。
__all__: list[str] = []
