.PHONY: build test test-all test-inference format

build:
	uvx maturin develop --release

test:
	uv run --with pytest,joblib pytest python/tests -m "not slow"

test-all:
	uv run --with pytest,joblib pytest python/tests -m ""

test-inference:
	uv run --with pytest,joblib,cmdstanpy,arviz pytest python/tests/models -m "slow"

format:
	uvx ruff format

