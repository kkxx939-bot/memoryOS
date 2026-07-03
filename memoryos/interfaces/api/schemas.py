from __future__ import annotations


def require_fields(payload: dict, fields: list[str]) -> None:
    missing = [field for field in fields if field not in payload or payload.get(field) in {None, ""}]
    if missing:
        raise ValueError(f"Missing required API payload field(s): {', '.join(missing)}")


def require_user_payload(payload: dict) -> None:
    require_fields(payload, ["user_id"])


def optional_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected a list")
    return [str(item) for item in value]
