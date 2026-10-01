PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: install lock lint format test

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

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:
	$(BIN)/pytest
