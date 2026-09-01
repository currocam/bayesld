.PHONY: build test format

build:
	RUSTUP_TOOLCHAIN=nightly uvx maturin develop --release

test:
	RUSTUP_TOOLCHAIN=nightly uv run cargo test
	uv run --with pytest,joblib,msprime,cmdstanpy,arviz,xarray,scipy pytest python/tests -m ""

format:
	uvx ruff format
