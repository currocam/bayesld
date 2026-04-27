"""Regression tests: verify JAX-ported functions match pre-recorded PyTensor outputs."""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bayesld
from bayesld import deterministic, montecarlo

jax.config.update("jax_enable_x64", True)

LEFT_BINS, RIGHT_BINS = bayesld.linear_bins()
MUTATION_RATE = 1e-8
RECOMBINATION_RATE = 1e-8
SEQUENCE_LENGTH = RIGHT_BINS[-1] * 2 / RECOMBINATION_RATE
SAMPLE_SIZE = 10
PLOIDY = 2

DET_KWARGS = dict(
    left_bins=LEFT_BINS,
    right_bins=RIGHT_BINS,
    mutation_rate=MUTATION_RATE,
    sample_size=SAMPLE_SIZE,
    ploidy=PLOIDY,
)

MC_KWARGS = dict(
    left_bins=LEFT_BINS,
    right_bins=RIGHT_BINS,
    mutation_rate=MUTATION_RATE,
    recombination_rate=RECOMBINATION_RATE,
    sequence_length=SEQUENCE_LENGTH,
    sample_size=SAMPLE_SIZE,
    random_seed=42,
    num_replicates=2,
    ploidy=PLOIDY,
    model="hudson",
)


@pytest.fixture(scope="module")
def reference():
    ref_path = os.path.join(os.path.dirname(__file__), "reference_data.json")
    with open(ref_path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Deterministic regression tests
# ══════════════════════════════════════════════════════════════════════════════


class TestDeterministicRegression:
    """Verify JAX deterministic outputs match PyTensor golden data (rtol=1e-6)."""

    def test_constant(self, reference):
        pi, ld = deterministic.expected_constant(Ne=1000.0, **DET_KWARGS)
        ref = reference["deterministic"]["constant"]
        np.testing.assert_allclose(float(pi), ref["pi"], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(ld), ref["ld"], rtol=1e-6)

    def test_piecewise_exponential(self, reference):
        pi, ld = deterministic.expected_piecewise_exponential(
            Ne_c=500.0, Ne_a=2000.0, t0=50.0, alpha=0.01, **DET_KWARGS
        )
        ref = reference["deterministic"]["piecewise_exponential"]
        np.testing.assert_allclose(float(pi), ref["pi"], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(ld), ref["ld"], rtol=1e-6)

    def test_exponential_carrying_capacity(self, reference):
        pi, ld = deterministic.expected_exponential_carrying_capacity(
            Ne_c=500.0, Ne_a=2000.0, t0=20.0, t1=60.0, alpha=0.01, **DET_KWARGS
        )
        ref = reference["deterministic"]["exponential_carrying_capacity"]
        np.testing.assert_allclose(float(pi), ref["pi"], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(ld), ref["ld"], rtol=1e-6)

    def test_piecewise_constant(self, reference):
        pi, ld = deterministic.expected_piecewise_constant(
            Ne_values=[500.0, 1000.0, 2000.0],
            t_boundaries=[30.0, 60.0],
            **DET_KWARGS,
        )
        ref = reference["deterministic"]["piecewise_constant"]
        np.testing.assert_allclose(float(pi), ref["pi"], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(ld), ref["ld"], rtol=1e-6)

    def test_secondary_introduction(self, reference):
        pi, ld = deterministic.expected_secondary_introduction(
            Ne_1=3000.0,
            Ne_2=3000.0,
            Ne_a=6000.0,
            t0=20.0,
            t1=60.0,
            migration_rate=0.1,
            **DET_KWARGS,
        )
        ref = reference["deterministic"]["secondary_introduction"]
        np.testing.assert_allclose(float(pi), ref["pi"], rtol=1e-6)
        # Slightly looser tolerance: vmap vs vectorize gives ~2e-6 relative differences
        np.testing.assert_allclose(np.asarray(ld), ref["ld"], rtol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo regression tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMonteCarloRegression:
    """Verify montecarlo outputs match golden data exactly (same seed = same results)."""

    def test_constant(self, reference):
        pi_vec, ld_mat = montecarlo.expected_constant(Ne=1000.0, **MC_KWARGS)
        ref = reference["montecarlo"]["constant"]
        np.testing.assert_allclose(float(pi_vec.mean()), ref["pi_mean"])
        np.testing.assert_allclose(np.asarray(ld_mat.mean(axis=0)), ref["ld_mean"])

    def test_piecewise_exponential(self, reference):
        pi_vec, ld_mat = montecarlo.expected_piecewise_exponential(
            Ne_c=500.0, Ne_a=2000.0, t0=50.0, alpha=0.01, **MC_KWARGS
        )
        ref = reference["montecarlo"]["piecewise_exponential"]
        np.testing.assert_allclose(float(pi_vec.mean()), ref["pi_mean"])
        np.testing.assert_allclose(np.asarray(ld_mat.mean(axis=0)), ref["ld_mean"])

    def test_exponential_carrying_capacity(self, reference):
        pi_vec, ld_mat = montecarlo.expected_exponential_carrying_capacity(
            Ne_c=500.0, Ne_a=2000.0, t0=20.0, t1=60.0, alpha=0.01, **MC_KWARGS
        )
        ref = reference["montecarlo"]["exponential_carrying_capacity"]
        np.testing.assert_allclose(float(pi_vec.mean()), ref["pi_mean"])
        np.testing.assert_allclose(np.asarray(ld_mat.mean(axis=0)), ref["ld_mean"])

    def test_piecewise_constant(self, reference):
        pi_vec, ld_mat = montecarlo.expected_piecewise_constant(
            Ne_values=[500.0, 1000.0, 2000.0],
            t_boundaries=[30.0, 60.0],
            **MC_KWARGS,
        )
        ref = reference["montecarlo"]["piecewise_constant"]
        np.testing.assert_allclose(float(pi_vec.mean()), ref["pi_mean"])
        np.testing.assert_allclose(np.asarray(ld_mat.mean(axis=0)), ref["ld_mean"])

