"""A small JSON-schema check for tool arguments (the subset the tool specifications use).

Every tool call is validated before it has any effect, so a malformed request never moves the robot or changes a
memory. ``problems`` returns human-readable messages the model can act on; an empty list means the value is valid.
"""
from __future__ import annotations

import math

_PYTHON_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}


def problems(value, schema: dict, path: str = "$") -> list[str]:
    if "anyOf" in schema:
        if any(not problems(value, option, path) for option in schema["anyOf"]):
            return []
        return [f"{path}: matches none of the allowed forms"]
    kind = schema.get("type")
    if kind == "null":
        return [] if value is None else [f"{path}: must be null"]
    found = []
    if kind in ("integer", "number"):
        allowed = int if kind == "integer" else (int, float)
        if isinstance(value, bool) or not isinstance(value, allowed) or not math.isfinite(value):
            return [f"{path}: must be a finite {kind}"]
        if "minimum" in schema and value < schema["minimum"]:
            found.append(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            found.append(f"{path}: must be <= {schema['maximum']}")
    elif kind in _PYTHON_TYPES and (not isinstance(value, _PYTHON_TYPES[kind])
                                    or (kind != "boolean" and isinstance(value, bool))):
        return [f"{path}: must be of type {kind}"]
    if "enum" in schema and value not in schema["enum"]:
        found.append(f"{path}: must be one of {schema['enum']}")
    if kind == "string" and "maxLength" in schema and len(value) > schema["maxLength"]:
        found.append(f"{path}: at most {schema['maxLength']} characters")
    if kind == "array":
        if len(value) < schema.get("minItems", 0):
            found.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            found.append(f"{path}: at most {schema['maxItems']} items")
        if schema.get("uniqueItems") and len({repr(item) for item in value}) != len(value):
            found.append(f"{path}: items must be unique")
        if "items" in schema:
            for index, item in enumerate(value):
                found += problems(item, schema["items"], f"{path}[{index}]")
    if kind == "object":
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                found.append(f"{path}: missing '{name}'")
        if schema.get("additionalProperties") is False:
            found += [f"{path}: unknown field '{name}'" for name in value if name not in properties]
        for name, sub in properties.items():
            if name in value:
                found += problems(value[name], sub, f"{path}.{name}")
    return found


def tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    """One tool specification: name, description shown to the model, and a closed argument schema."""
    return {"type": "function", "name": name, "description": description,
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": properties,
                            "required": list(properties) if required is None else required}}
