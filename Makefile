# Developer quality-gate shortcuts. Every target maps to a check that must pass before a
# change is considered mergeable (see SPEC.md §10, §13).

.DEFAULT_GOAL := check
PY := python

.PHONY: help install lint format format-fix typecheck test test-fast cov check clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev + RL extras (editable).
	$(PY) -m pip install -e ".[dev,rl]"

lint: ## Run ruff lint checks.
	ruff check src tests

format: ## Check formatting with black (no writes).
	black --check src tests

format-fix: ## Apply black + ruff autofixes.
	black src tests
	ruff check --fix src tests

typecheck: ## Run mypy in strict mode.
	mypy src

test: ## Run the full test suite (including slow validation/learning proofs).
	$(PY) -m pytest

test-fast: ## Run only the fast test suite (excludes slow proofs).
	$(PY) -m pytest -m "not slow"

cov: ## Run tests with a coverage report.
	$(PY) -m pytest --cov --cov-report=term-missing

check: lint format typecheck test-fast ## Run all quality gates (fast tests).

clean: ## Remove caches and build artifacts.
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
