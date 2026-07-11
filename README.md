# bayesld

`bayesld` is a Python package for Bayesian inference of very recent population size history \(N_e(t)\) from linkage disequilibrium (LD) and genetic diversity.

## Installation

Requires **Python ≥ 3.11**. All commands install from GitHub: [`https://github.com/currocam/bayesld`](https://github.com/currocam/bayesld).

### conda

```bash
conda create -n bayesld python=3.12 -y
conda activate bayesld
conda install -c conda-forge cmdstanpy pip -y
pip install "git+https://github.com/currocam/bayesld.git"
```

### pip

```bash
pip install "git+https://github.com/currocam/bayesld.git"
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

### uv

```bash
uv add "git+https://github.com/currocam/bayesld.git"
uv run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

## Getting started

```python
from bayesld.inference import PiecewiseConstant
model = (
    PiecewiseConstant(num_epochs=2)
    .with_data(
        mean_diversity=pi,
        mean_ld=ld,
        left_bins=left_bins,
        right_bins=right_bins,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        num_samples=100,
        sequence_length=2e7,
    )
)
for _ in range(3):
    model = model.active_learning_round(num_points=5, rtol=0.1, min_replicates=30)
idata = model.sample(draws=1000, tune=1000, chains=4)
```

See [`examples/inference_two_epoch_bottleneck.ipynb`](examples/inference_two_epoch_bottleneck.ipynb) for a full analysis of a very recent bottleneck using simulated data.

For preparing the input data from a VCF, see [`examples/data_from_vcf.ipynb`](examples/data_from_vcf.ipynb).

## Development

The package includes a compiled Rust extension (requires Rust nightly for `portable_simd`). Build it before running any code:

```bash
make build    # RUSTUP_TOOLCHAIN=nightly uvx maturin develop --release
make test     # full test suite (Stan/msprime/arviz included)
make format   # ruff format
```