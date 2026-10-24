PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
DATA_CONFIG ?= configs/data.yaml
GRADING_CONFIG ?= configs/train_grading.yaml
QUALITY_CONFIG ?= configs/train_quality.yaml
EVAL_CONFIG ?= configs/evaluate.yaml
APP_CONFIG ?= configs/app.yaml
SPACE_DIR := build/hf_space

.PHONY: install lock lint format test fetch-data prepare-data inspect train train-grading train-quality evaluate tune-referral serve space

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

train: train-grading train-quality

train-grading:
	$(BIN)/python scripts/train.py --config $(GRADING_CONFIG)

train-quality:
	$(BIN)/python scripts/train.py --config $(QUALITY_CONFIG)

evaluate:
	@test -n "$(CHECKPOINT)" || { echo "Usage: make evaluate CHECKPOINT=runs/<run>/best.pt"; exit 1; }
	$(BIN)/python scripts/evaluate.py --config $(EVAL_CONFIG) --checkpoint $(CHECKPOINT) $(EVAL_ARGS)

tune-referral:
	@test -n "$(CHECKPOINT)" || { echo "Usage: make tune-referral CHECKPOINT=runs/<run>/best.pt"; exit 1; }
	$(BIN)/python scripts/tune_referral.py --config $(EVAL_CONFIG) --checkpoint $(CHECKPOINT)

serve:
	$(BIN)/python scripts/serve.py --config $(APP_CONFIG)

# Assemble a self-contained Hugging Face Space in $(SPACE_DIR); push it with
# `hf upload <user>/<space> $(SPACE_DIR) . --repo-type space`.
space:
	rm -rf $(SPACE_DIR)
	mkdir -p $(SPACE_DIR)
	cp -R deploy/hf_space/. $(SPACE_DIR)/
	cp -R src/drscreen $(SPACE_DIR)/
	cp requirements.txt $(SPACE_DIR)/
	cp configs/risk.yaml $(SPACE_DIR)/configs/
	find $(SPACE_DIR) -name __pycache__ -prune -exec rm -rf {} +
