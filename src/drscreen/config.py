"""Typed configuration loading.

All tunable values live in YAML files under ``configs/``. This module turns a
YAML file into a validated, immutable Pydantic object and applies command-line
overrides of the form ``section.key=value`` before validation, so an override
is checked exactly as strictly as the file itself.

Schemas for each pipeline stage subclass ``StrictModel``. Unknown keys are
rejected, which means a typo or an orphaned key fails loudly instead of being
silently ignored.

Relative paths in config files are interpreted relative to the current working
directory; all documented commands run from the repository root.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from drscreen.exceptions import ConfigError

OVERRIDE_SEPARATOR = "="
KEY_PATH_SEPARATOR = "."


class StrictModel(BaseModel):
    """Base for all config schemas: immutable and rejects unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def load_config[SchemaT: StrictModel](
    path: Path, schema: type[SchemaT], overrides: Sequence[str] = ()
) -> SchemaT:
    """Load a YAML config file, apply overrides and validate it.

    Args:
        path: Path to the YAML file.
        schema: The ``StrictModel`` subclass describing the expected structure.
        overrides: Strings of the form ``"a.b.c=value"``. The value is parsed
            as YAML, so ``"train.epochs=5"`` yields an int and
            ``"train.lr=1e-4"`` a float.

    Returns:
        A validated, immutable instance of ``schema``.

    Raises:
        ConfigError: If the file is missing or unparsable, an override is
            malformed or targets a key absent from the file, or the result
            fails schema validation.
    """
    raw = _read_yaml_mapping(path)
    for override in overrides:
        _apply_override(raw, override)
    try:
        return schema.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {path}:\n{exc}") from exc


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file whose top level must be a mapping.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping.

    Raises:
        ConfigError: If the file is missing, unparsable or not a mapping.
    """
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path.resolve()}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file {path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file {path} must contain a mapping at the top level, "
            f"got {type(data).__name__}."
        )
    return data


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply one ``a.b.c=value`` override to a nested mapping in place.

    Only keys already present in the file can be overridden. This keeps the
    YAML file the single complete record of what a run can be configured with.

    Args:
        raw: The nested mapping loaded from YAML. Modified in place.
        override: The override string.

    Raises:
        ConfigError: If the override is malformed or names a missing key.
    """
    key_path, separator, value_text = override.partition(OVERRIDE_SEPARATOR)
    if not separator or not key_path:
        raise ConfigError(
            f"Malformed override {override!r}; expected the form 'section.key=value'."
        )
    keys = key_path.split(KEY_PATH_SEPARATOR)
    node: Any = raw
    for depth, key in enumerate(keys):
        if not isinstance(node, dict) or key not in node:
            known = ", ".join(sorted(node)) if isinstance(node, dict) else "none"
            location = KEY_PATH_SEPARATOR.join(keys[:depth]) or "<root>"
            raise ConfigError(
                f"Override {override!r} targets unknown key {key!r} under {location}. "
                f"Known keys: {known}."
            )
        if depth == len(keys) - 1:
            node[key] = yaml.safe_load(value_text)
        else:
            node = node[key]
