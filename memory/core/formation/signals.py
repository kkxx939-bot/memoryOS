"""记忆显著性判断与规则提取共享的确定性语义信号。"""

from __future__ import annotations

import re
from enum import Enum


class MemorySignal(str, Enum):
    EXPLICIT_REMEMBER = "explicit_remember"
    PREFERENCE = "durable_preference"
    PROFILE = "durable_profile"
    DURABILITY = "durability"
    CORRECTION = "correction"
    OPEN_LOOP = "open_loop"
    TRANSIENT = "transient"
    SENSITIVE = "privacy_or_sensitivity_risk"
    EXPERIENCE = "experience"
    ENTITY = "entity"


_REMEMBER_PREFIX = re.compile(
    r"(?i)^(?:(?:请|please)\s*)?(?:记住|记得|remember(?:\s+(?:that|this))?)\s*[:：,，]?\s*(.+)$"
)
_PATTERNS: dict[MemorySignal, re.Pattern[str]] = {
    MemorySignal.EXPLICIT_REMEMBER: re.compile(r"(?i)(?:记住|请记得|以后记得|remember|keep in mind)"),
    MemorySignal.PREFERENCE: re.compile(r"(?i)(?:我(?:不)?喜欢|我更喜欢|我的偏好是|偏好|i (?:prefer|dislike|like))"),
    MemorySignal.PROFILE: re.compile(r"(?i)(?:^|[。.!?]\s*)(?:我是|我叫|我住在|我的.+是|i am|my name is)"),
    MemorySignal.DURABILITY: re.compile(r"(?i)(?:以后|长期|一直|将来|from now on|always|long[- ]term)"),
    MemorySignal.CORRECTION: re.compile(r"(?i)(?:更正|不是.+是|改成|改为|correction|actually)"),
    MemorySignal.OPEN_LOOP: re.compile(r"(?i)(?:待确认|以后再看|尚未解决|需要跟进|todo|open question|follow up)"),
    MemorySignal.TRANSIENT: re.compile(r"(?i)(?:这一次|临时|仅本次|just this once|temporary)"),
    MemorySignal.SENSITIVE: re.compile(r"(?i)(?:password|api[_ -]?key|密码|密钥|身份证|银行卡)"),
    MemorySignal.EXPERIENCE: re.compile(r"(?i)(?:经验|教训|有效做法|复用|lesson|worked well|reusable)"),
    MemorySignal.ENTITY: re.compile(r"(?i)(?:项目|系统|产品|公司|组织|project|system|product|organization)"),
}


def detect_memory_signals(text: str) -> frozenset[MemorySignal]:
    """一次扫描返回全部记忆信号，确保显著性与规则提取口径一致。"""

    return frozenset(signal for signal, pattern in _PATTERNS.items() if pattern.search(text))


def strip_remember_prefix(text: str) -> tuple[str, bool]:
    """去掉明确记住指令，只返回需要持久化的语义正文。"""

    matched = _REMEMBER_PREFIX.match(text)
    return ((matched.group(1) if matched else text).strip(), matched is not None)


__all__ = ["MemorySignal", "detect_memory_signals", "strip_remember_prefix"]
