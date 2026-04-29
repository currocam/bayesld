.PHONY: build test test-all format

build:
	uvx maturin develop --release

test:
	uv run --with pytest,joblib pytest python/tests -m "not slow"

test-all:
	uv run --with pytest,joblib pytest python/tests -m ""

format:
	uvx ruff format

