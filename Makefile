PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
DATA_CONFIG ?= configs/data.yaml

.PHONY: install lock lint format test fetch-data prepare-data inspect

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: $(BIN)/python
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install --no-deps -e .
	$(BIN)/pre-commit install

lock: $(BIN)/python
	$(BIN)/pip install pip-tools
	$(BIN)/pip-compile --quiet --strip-extras --output-file requirements.txt pyproject.toml
	$(BIN)/pip-compile --quiet --strip-extras --extra dev --constraint requirements.txt \
		--output-file requirements-dev.txt pyproject.toml

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	$(BIN)/mypy
	$(BIN)/lint-imports

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:
	$(BIN)/pytest

fetch-data:
	$(BIN)/python scripts/fetch_data.py --config $(DATA_CONFIG)

prepare-data:
	$(BIN)/python scripts/prepare_data.py --config $(DATA_CONFIG)

inspect:
	$(BIN)/python scripts/inspect_data.py --config $(DATA_CONFIG)
