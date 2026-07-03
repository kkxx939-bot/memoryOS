from __future__ import annotations


def l0_abstract(text: str, max_chars: int = 220) -> str:
    compact = " ".join(str(text).split())
    return compact[:max_chars]


def l1_overview(title: str, bullets: list[str], max_bullets: int = 12) -> str:
    lines = [f"# {title}", ""]
    lines.extend(f"- {bullet}" for bullet in bullets[:max_bullets] if str(bullet).strip())
    return "\n".join(lines).strip() + "\n"
