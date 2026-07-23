"""严格校验 JSON Schema，并可选使用完整实现标准的依赖。"""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Mapping


class JSONSchemaValidationError(ValueError):
    """解析后的模型返回值不符合请求的 JSON Schema。"""


def validate_json_schema(value: object, schema: Mapping[str, object]) -> object:
    """校验时不进行类型强制转换；已安装 jsonschema 时使用它，否则使用严格核心实现。"""

    if not isinstance(schema, Mapping) or not schema:
        raise ValueError("JSON Schema must be a non-empty object")
    try:
        module = importlib.import_module("jsonschema")
    except ImportError:
        _validate(value, schema, root=schema, path="$")
        return value
    try:
        validator_class = module.validators.validator_for(schema)
        validator_class.check_schema(schema)
        validator_class(schema).validate(value)
    except Exception as exc:
        exceptions = getattr(module, "exceptions", None)
        validation_error = getattr(exceptions, "ValidationError", ())
        schema_error = getattr(exceptions, "SchemaError", ())
        if validation_error and isinstance(exc, validation_error):
            raise JSONSchemaValidationError(str(exc)) from exc
        if schema_error and isinstance(exc, schema_error):
            raise ValueError(f"invalid JSON Schema: {exc}") from exc
        raise
    return value


def _validate(
    value: object,
    schema: Mapping[str, object],
    *,
    root: Mapping[str, object],
    path: str,
) -> None:
    if "$ref" in schema:
        reference = schema["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise ValueError("fallback JSON Schema validator only supports local $ref values")
        target: object = root
        for raw_part in reference[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, Mapping) or part not in target:
                raise ValueError(f"unresolved JSON Schema reference: {reference}")
            target = target[part]
        if not isinstance(target, Mapping):
            raise ValueError(f"JSON Schema reference is not an object: {reference}")
        _validate(value, target, root=root, path=path)
        return

    if "allOf" in schema:
        for index, branch in enumerate(_schema_array(schema["allOf"], "allOf")):
            _validate(value, branch, root=root, path=f"{path}.allOf[{index}]")
    if "anyOf" in schema:
        branches = _schema_array(schema["anyOf"], "anyOf")
        if not any(_is_valid(value, branch, root) for branch in branches):
            raise JSONSchemaValidationError(f"{path} does not satisfy anyOf")
    if "oneOf" in schema:
        branches = _schema_array(schema["oneOf"], "oneOf")
        matches = sum(_is_valid(value, branch, root) for branch in branches)
        if matches != 1:
            raise JSONSchemaValidationError(f"{path} must satisfy exactly one oneOf branch")
    if "not" in schema:
        not_schema = schema["not"]
        if not isinstance(not_schema, Mapping):
            raise ValueError("JSON Schema not must be an object")
        if _is_valid(value, not_schema, root):
            raise JSONSchemaValidationError(f"{path} satisfies a forbidden schema")

    if "const" in schema and value != schema["const"]:
        raise JSONSchemaValidationError(f"{path} does not equal const")
    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or value not in enum:
            raise JSONSchemaValidationError(f"{path} is not an allowed enum value")

    expected_type = schema.get("type")
    if expected_type is not None:
        allowed_types = [expected_type] if isinstance(expected_type, str) else expected_type
        if not isinstance(allowed_types, list) or not all(isinstance(item, str) for item in allowed_types):
            raise ValueError("JSON Schema type must be a string or array of strings")
        if not any(_matches_type(value, item) for item in allowed_types):
            raise JSONSchemaValidationError(f"{path} must have type {' or '.join(allowed_types)}")

    if isinstance(value, dict):
        _validate_object(value, schema, root=root, path=path)
    elif isinstance(value, list):
        _validate_array(value, schema, root=root, path=path)
    elif isinstance(value, str):
        _validate_string(value, schema, path=path)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        _validate_number(value, schema, path=path)


def _validate_object(
    value: dict[object, object],
    schema: Mapping[str, object],
    *,
    root: Mapping[str, object],
    path: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise JSONSchemaValidationError(f"{path} object keys must be strings")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ValueError("JSON Schema properties must be an object")
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ValueError("JSON Schema required must be an array of strings")
    missing = [name for name in required if name not in value]
    if missing:
        raise JSONSchemaValidationError(f"{path} is missing required fields: {missing}")
    for key, item in value.items():
        child_schema = properties.get(key)
        if child_schema is not None:
            if not isinstance(child_schema, Mapping):
                raise ValueError(f"JSON Schema property {key} must be an object")
            _validate(item, child_schema, root=root, path=f"{path}.{key}")
            continue
        additional = schema.get("additionalProperties", True)
        if additional is False:
            raise JSONSchemaValidationError(f"{path} contains unknown field: {key}")
        if isinstance(additional, Mapping):
            _validate(item, additional, root=root, path=f"{path}.{key}")
    _require_size(value, schema, path=path, minimum_key="minProperties", maximum_key="maxProperties")


def _validate_array(
    value: list[object],
    schema: Mapping[str, object],
    *,
    root: Mapping[str, object],
    path: str,
) -> None:
    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise ValueError("JSON Schema items must be an object")
        for index, item in enumerate(value):
            _validate(item, items, root=root, path=f"{path}[{index}]")
    _require_size(value, schema, path=path, minimum_key="minItems", maximum_key="maxItems")
    if schema.get("uniqueItems") is True:
        for index, item in enumerate(value):
            if item in value[:index]:
                raise JSONSchemaValidationError(f"{path} array items must be unique")


def _validate_string(value: str, schema: Mapping[str, object], *, path: str) -> None:
    _require_size(value, schema, path=path, minimum_key="minLength", maximum_key="maxLength")
    pattern = schema.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise ValueError("JSON Schema pattern must be a string")
        if re.search(pattern, value) is None:
            raise JSONSchemaValidationError(f"{path} does not match required pattern")


def _validate_number(value: int | float, schema: Mapping[str, object], *, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise JSONSchemaValidationError(f"{path} must be finite")
    bounds = (
        ("minimum", lambda actual, bound: actual >= bound),
        ("maximum", lambda actual, bound: actual <= bound),
        ("exclusiveMinimum", lambda actual, bound: actual > bound),
        ("exclusiveMaximum", lambda actual, bound: actual < bound),
    )
    for key, predicate in bounds:
        bound = schema.get(key)
        if bound is None:
            continue
        if isinstance(bound, bool) or not isinstance(bound, int | float):
            raise ValueError(f"JSON Schema {key} must be numeric")
        if not predicate(value, bound):
            raise JSONSchemaValidationError(f"{path} violates {key}")


def _require_size(
    value: object,
    schema: Mapping[str, object],
    *,
    path: str,
    minimum_key: str,
    maximum_key: str,
) -> None:
    size = len(value)  # type: ignore[arg-type]
    for key, predicate in (
        (minimum_key, lambda actual, bound: actual >= bound),
        (maximum_key, lambda actual, bound: actual <= bound),
    ):
        bound = schema.get(key)
        if bound is None:
            continue
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
            raise ValueError(f"JSON Schema {key} must be a non-negative integer")
        if not predicate(size, bound):
            raise JSONSchemaValidationError(f"{path} violates {key}")


def _schema_array(value: object, keyword: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not value or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"JSON Schema {keyword} must be a non-empty array of objects")
    return value


def _is_valid(value: object, schema: Mapping[str, object], root: Mapping[str, object]) -> bool:
    try:
        _validate(value, schema, root=root, path="$")
    except JSONSchemaValidationError:
        return False
    return True


def _matches_type(value: object, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    raise ValueError(f"unsupported JSON Schema type: {expected}")


__all__ = ["JSONSchemaValidationError", "validate_json_schema"]
