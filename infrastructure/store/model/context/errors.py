"""上下文存储标识的校验异常。"""


class InvalidContextURI(ValueError):
    """上下文 URI 不符合受控命名空间规则。"""


__all__ = ["InvalidContextURI"]
