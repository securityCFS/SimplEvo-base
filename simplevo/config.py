"""Validated YAML configuration shared by the CLI and Python API."""
from __future__ import annotations

import math
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

import yaml

DEFAULTS = yaml.safe_load(files("simplevo").joinpath("defaults.yaml").read_text(encoding="utf-8"))


def resolve_config(overrides: dict | None = None) -> dict:
    """Merge partial overrides into fresh defaults; reject typos and invalid budgets."""
    result = deepcopy(DEFAULTS)
    if overrides is None:
        return result
    if not isinstance(overrides, dict):
        raise ValueError("configuration must be a mapping of sections")
    for section, values in overrides.items():
        if section not in result or not isinstance(values, dict):
            raise ValueError(f"unknown section or invalid mapping: {section}")
        for key, value in values.items():
            label = f"{section}.{key}"
            if key not in result[section]:
                raise ValueError(f"unknown configuration key: {label}")
            default = DEFAULTS[section][key]
            if label == "model.turn_timeout_s":
                valid = value is None or (type(value) in (int, float) and math.isfinite(value) and value > 0)
            elif type(default) is bool:
                valid = type(value) is bool
            elif type(default) is int:
                minimum = 0 if label in ("episode.retries", "segmentation.gap") else 1
                valid = type(value) is int and value >= minimum
            elif type(default) is float:
                valid = type(value) in (int, float) and math.isfinite(value) and value > 0
            else:
                valid = isinstance(value, str) and bool(value.strip())
            if not valid:
                raise ValueError(f"invalid value for {label}: {value!r}")
            result[section][key] = value
    if result["memory"]["carry"] not in ("none", "lessons", "lessons+outcomes", "lessons+outcomes+frames"):
        raise ValueError("invalid memory.carry")
    if result["demonstration"]["max_action_rows"] < 2:
        raise ValueError("demonstration.max_action_rows must be at least 2")
    if not 3 <= result["annotation"]["max_keyframes"] <= 20:
        raise ValueError("annotation.max_keyframes must be between 3 and 20 (task-memory capacity)")
    return result


def load_config(path: Path | None = None) -> dict:
    if path is None:
        return resolve_config()
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {path}: {error}") from error
    return resolve_config(raw)


def save_config(path: Path, config: dict) -> None:
    """Save the effective settings alongside the episode or annotation artifacts."""
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
