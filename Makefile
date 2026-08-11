# Thin wrappers around uv, so the commands used locally and in CI are the same.
#
#   make test                          # run the test suite
#   make test ARGS="tests/test_metrics.py -k d_prime"
#   make check                         # everything CI checks
#
# Pass FROZEN=1 to fail on a stale uv.lock 

FROZEN_FLAG := $(if $(FROZEN),--frozen,)
UV_RUN := uv run $(FROZEN_FLAG)
ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help install test test-cov types check build clean lock

help:  ## Show this help
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) \
		| sed -e 's/:.*##/:/' \
		| awk -F':' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Sync the virtualenv with pyproject.toml and uv.lock
	uv sync $(FROZEN_FLAG)

test:  ## Run the test suite
	$(UV_RUN) pytest $(ARGS)

test-cov:  ## Run the test suite with coverage (what CI runs)
	$(UV_RUN) pytest --cov $(ARGS)

types:  ## Type check with mypy
	$(UV_RUN) mypy $(ARGS)

check: test-cov types  ## Run the full test suite with coverage, then type check

build:  ## Build the sdist and wheel into dist/
	uv build

lock:  ## Re-resolve and update uv.lock
	uv lock

clean:  ## Remove build artefacts and tool caches
	rm -rf dist .coverage .coverage.* .pytest_cache .mypy_cache
