"""
Benchmark: surrogate likelihood evaluation with parallel Monte Carlo.

Usage:
    uv run python benches/mc_parallel.py --workers 8 --replicates 16

Workflow
--------
1. Simulate "observed" data at known true parameters (treated as windows of data).
2. Build a jax.jit-compiled likelihood function that captures the fixed
   hyperparameters and observed data in a closure; only the parameter under
   inference and random_seed are JAX-traced.
3. Time first call (JIT compile + run) and second call (cached trace).

Parallelism is provided by shard_map inside montecarlo.*: replicates are
distributed across jax_num_cpu_devices virtual CPU threads.  The Rust
smc_prime extension releases the GIL (py.detach), so all threads run
concurrently.
"""

import argparse
import functools
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workers",
    type=int,
    default=4,
    help="Number of virtual CPU devices (parallelism level)",
)
parser.add_argument(
    "--replicates",
    type=int,
    default=8,
    help="MC replicates used inside each likelihood evaluation",
)
parser.add_argument(
    "--data-replicates",
    type=int,
    default=20,
    help="Simulated data windows (replicates treated as observations)",
)
args = parser.parse_args()

# Must be set before JAX initialises.
import jax

jax.config.update("jax_num_cpu_devices", args.workers)
jax.config.update("jax_enable_x64", True)

import functools
import pathlib
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "python"))
import bayesld
from bayesld import montecarlo, surrogate_likelihoods

# ── Shared simulation parameters ──────────────────────────────────────────────

left_bins, right_bins = bayesld.linear_bins()
MUTATION_RATE = 1.25e-8
RECOMBINATION_RATE = 1e-8
SEQUENCE_LENGTH = right_bins[-1] * 2 / RECOMBINATION_RATE
SAMPLE_SIZE = 20
PLOIDY = 2
MODEL = "hudson"
NUM_REPLICATES = args.replicates

print(f"Workers (JAX CPU devices) : {jax.device_count()}")
print(f"Likelihood replicates     : {NUM_REPLICATES}")
print(f"Data windows              : {args.data_replicates}")
print(f"Sequence length           : {SEQUENCE_LENGTH:.2e} bp")
print(f"Sample size               : {SAMPLE_SIZE} diploids")
print(f"Model                     : {MODEL}")
print()

# ── Generate observed data at known true parameters ───────────────────────────
# We treat each MC replicate as an independent genomic window.

print("Generating observed data ...")

TRUE_NE = 10_000.0
TRUE_NE_C = 5_000.0
TRUE_NE_A = 15_000.0
TRUE_T0 = 500.0
TRUE_ALPHA = 0.002

sim_kwargs = dict(
    left_bins=left_bins,
    right_bins=right_bins,
    mutation_rate=MUTATION_RATE,
    recombination_rate=RECOMBINATION_RATE,
    sequence_length=SEQUENCE_LENGTH,
    sample_size=SAMPLE_SIZE,
    ploidy=PLOIDY,
    model=MODEL,
)

data_pi_constant, data_ld_constant = jax.block_until_ready(
    montecarlo.expected_constant(
        Ne=TRUE_NE,
        random_seed=0,
        num_replicates=args.data_replicates,
        **sim_kwargs,
    )
)
data_pi_exp, data_ld_exp = jax.block_until_ready(
    montecarlo.expected_piecewise_exponential(
        Ne_c=TRUE_NE_C,
        Ne_a=TRUE_NE_A,
        t0=TRUE_T0,
        alpha=TRUE_ALPHA,
        random_seed=0,
        num_replicates=args.data_replicates,
        **sim_kwargs,
    )
)
print("Done.\n")

# ── JIT-compiled likelihood wrappers ──────────────────────────────────────────
# Observed data, bins, and simulation hyperparameters are captured in the
# closure.  Only log_Ne / demographic params and random_seed are JAX-traced.
# num_replicates / ploidy / model are static (determine output shape of the
# inner MC call).

STATIC = dict(num_replicates=NUM_REPLICATES, ploidy=PLOIDY, model=MODEL)


@functools.partial(jax.jit, static_argnames=("num_replicates", "ploidy", "model"))
def loglik_constant(log_Ne, random_seed, *, num_replicates, ploidy, model):
    return surrogate_likelihoods.constant(
        data_pi_constant,
        data_ld_constant,
        log_Ne,
        left_bins,
        right_bins,
        MUTATION_RATE,
        RECOMBINATION_RATE,
        SEQUENCE_LENGTH,
        SAMPLE_SIZE,
        random_seed,
        num_replicates=num_replicates,
        ploidy=ploidy,
        model=model,
    )


@functools.partial(jax.jit, static_argnames=("num_replicates", "ploidy", "model"))
def loglik_piecewise_exponential(
    log_Ne_c, log_Ne_a, log_t0, alpha, random_seed, *, num_replicates, ploidy, model
):
    return surrogate_likelihoods.piecewise_exponential(
        data_pi_exp,
        data_ld_exp,
        log_Ne_c,
        log_Ne_a,
        log_t0,
        alpha,
        left_bins,
        right_bins,
        MUTATION_RATE,
        RECOMBINATION_RATE,
        SEQUENCE_LENGTH,
        SAMPLE_SIZE,
        random_seed,
        num_replicates=num_replicates,
        ploidy=ploidy,
        model=model,
    )


# ── Benchmark helper ──────────────────────────────────────────────────────────


def run(label, fn):
    print(f"=== {label} ===")
    # First call: JIT compile + run.
    t0 = time.perf_counter()
    val = jax.block_until_ready(fn())
    t_first = time.perf_counter() - t0

    # Second call: cached trace.
    t0 = time.perf_counter()
    val = jax.block_until_ready(fn())
    t_cached = time.perf_counter() - t0

    print(f"  log-likelihood           : {float(val):.4f}")
    print(f"  first call (compile+run) : {t_first:.2f}s")
    print(
        f"  second call (cached)     : {t_cached:.2f}s  "
        f"({t_cached / NUM_REPLICATES * 1000:.0f} ms/replicate)"
    )
    print()


# ── Constant Ne ───────────────────────────────────────────────────────────────

run(
    f"surrogate_likelihoods.constant  true Ne={TRUE_NE:.0f}",
    lambda: loglik_constant(jnp.log(jnp.float64(TRUE_NE)), jnp.int64(1), **STATIC),
)

# ── Piecewise exponential ─────────────────────────────────────────────────────

run(
    f"surrogate_likelihoods.piecewise_exponential  true Ne_c={TRUE_NE_C:.0f}  Ne_a={TRUE_NE_A:.0f}",
    lambda: loglik_piecewise_exponential(
        jnp.log(jnp.float64(TRUE_NE_C)),
        jnp.log(jnp.float64(TRUE_NE_A)),
        jnp.log(jnp.float64(TRUE_T0)),
        jnp.float64(TRUE_ALPHA),
        jnp.int64(1),
        **STATIC,
    ),
)
