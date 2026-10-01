from pathlib import Path

import pytest

from drscreen.config import StrictModel, load_config
from drscreen.exceptions import ConfigError


class _Train(StrictModel):
    epochs: int
    lr: float


class _Schema(StrictModel):
    name: str
    output_dir: Path
    train: _Train


VALID_YAML = """
name: demo
output_dir: runs/demo
train:
  epochs: 10
  lr: 0.001
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")
    return path


def test_loads_valid_file_into_typed_object(config_file: Path) -> None:
    config = load_config(config_file, _Schema)

    assert config.name == "demo"
    assert config.output_dir == Path("runs/demo")
    assert config.train.epochs == 10
    assert config.train.lr == pytest.approx(0.001)


def test_config_is_immutable(config_file: Path) -> None:
    config = load_config(config_file, _Schema)

    with pytest.raises(ValueError, match="frozen"):
        config.name = "changed"  # type: ignore[misc]


def test_overrides_are_parsed_as_yaml_scalars(config_file: Path) -> None:
    config = load_config(config_file, _Schema, ["train.epochs=3", "train.lr=1e-4", "name=x"])

    assert config.train.epochs == 3
    assert config.train.lr == pytest.approx(1e-4)
    assert config.name == "x"


def test_overrides_are_validated(config_file: Path) -> None:
    with pytest.raises(ConfigError, match="epochs"):
        load_config(config_file, _Schema, ["train.epochs=many"])


@pytest.mark.parametrize("override", ["train.epochs", "=5", ""])
def test_malformed_override_is_rejected(config_file: Path, override: str) -> None:
    with pytest.raises(ConfigError, match="Malformed override"):
        load_config(config_file, _Schema, [override])


@pytest.mark.parametrize("override", ["train.momentum=0.9", "missing.key=1", "name.inner=1"])
def test_override_of_unknown_key_is_rejected(config_file: Path, override: str) -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(config_file, _Schema, [override])


def test_unknown_key_in_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(VALID_YAML + "stray_key: 1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="stray_key"):
        load_config(path, _Schema)


def test_missing_required_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("name: demo\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="output_dir"):
        load_config(path, _Schema)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml", _Schema)


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path, _Schema)


def test_non_mapping_top_level_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="mapping"):
        load_config(path, _Schema)
