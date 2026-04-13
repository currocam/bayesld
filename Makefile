.PHONY: build test test-all format

build:
	uvx maturin develop --release

test:
	uv run pytest python/tests -m "not slow"

test-all:
	uv run pytest python/tests -m ""

format:
	uvx ruff format

