from __future__ import annotations

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


def json_object(value: JsonValue) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object.")
    return value


def json_string(value: JsonValue, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def optional_json_string(value: JsonValue, *, name: str) -> str | None:
    if value is None:
        return None
    return json_string(value, name=name)


def optional_json_int(value: JsonValue, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value
