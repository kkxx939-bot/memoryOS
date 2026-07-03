from __future__ import annotations

from dataclasses import dataclass


ACTION_SCHEMA_VERSION = "action_schema_v1"


@dataclass(frozen=True)
class ActionSpec:
    action: str
    need: str
    risk_level: str
    predictable: bool
    intervenable: bool
    executable: bool
    requires_confirmation: bool
    aliases: tuple[str, ...] = ()


ACTION_SPECS = {
    "turn_on_ac": ActionSpec(
        action="turn_on_ac",
        need="cool_down",
        risk_level="low",
        predictable=True,
        intervenable=True,
        executable=True,
        requires_confirmation=True,
        aliases=("open_ac", "seek_cooling", "ac_on", "打开空调", "开启空调"),
    ),
    "turn_on_fan": ActionSpec(
        action="turn_on_fan",
        need="cool_down",
        risk_level="low",
        predictable=True,
        intervenable=True,
        executable=True,
        requires_confirmation=False,
        aliases=("open_fan", "fan_on", "打开电扇", "打开风扇"),
    ),
    "drink_water": ActionSpec(
        action="drink_water",
        need="hydrate",
        risk_level="none",
        predictable=True,
        intervenable=True,
        executable=False,
        requires_confirmation=False,
        aliases=("喝水", "补水"),
    ),
    "take_shower": ActionSpec(
        action="take_shower",
        need="comfort",
        risk_level="private",
        predictable=True,
        intervenable=False,
        executable=False,
        requires_confirmation=False,
        aliases=("洗澡", "冲澡"),
    ),
    "smoke": ActionSpec(
        action="smoke",
        need="habit_trigger",
        risk_level="medium",
        predictable=True,
        intervenable=True,
        executable=False,
        requires_confirmation=False,
        aliases=("抽烟", "吸烟"),
    ),
    "organize_desk": ActionSpec(
        action="organize_desk",
        need="organize",
        risk_level="none",
        predictable=True,
        intervenable=False,
        executable=False,
        requires_confirmation=False,
        aliases=("整理桌面",),
    ),
    "continue_current_activity": ActionSpec(
        action="continue_current_activity",
        need="none",
        risk_level="none",
        predictable=True,
        intervenable=False,
        executable=False,
        requires_confirmation=False,
        aliases=("do_nothing",),
    ),
}


def canonical_action(action: str) -> str:
    value = str(action or "").strip()
    if not value:
        return ""
    if value in ACTION_SPECS:
        return value
    lowered = value.lower()
    for spec in ACTION_SPECS.values():
        if lowered == spec.action.lower() or lowered in {alias.lower() for alias in spec.aliases}:
            return spec.action
    return value


def action_spec(action: str) -> ActionSpec:
    canonical = canonical_action(action)
    return ACTION_SPECS.get(
        canonical,
        ActionSpec(
            action=canonical or "unknown",
            need="unknown",
            risk_level="unknown",
            predictable=True,
            intervenable=True,
            executable=False,
            requires_confirmation=True,
        ),
    )


def action_need(action: str) -> str:
    return action_spec(action).need
