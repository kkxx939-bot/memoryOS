"""在 Memory、Context、Behavior 和 ActionPolicy 分流前准备公共业务事实。"""

# 根包保持轻量；调用者必须从 pre.connect、pre.conversation 或 pre.session
# 显式导入，避免公共输入模型之间产生隐式耦合。
__all__: list[str] = []
