.PHONY: build test test-all test-inference-api format

build:
	RUSTUP_TOOLCHAIN=nightly uvx maturin develop --release

test:
	uv run --with pytest,joblib pytest python/tests -m "not slow"

test-all:
	uv run --with pytest,joblib pytest python/tests -m ""

test-inference-api:
	uv run --with pytest,joblib,msprime,cmdstanpy,arviz,xarray pytest python/tests/inference -m ""

format:
	uvx ruff format

