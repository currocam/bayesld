"""
Tests for deterministic and montecarlo expected_* functions.

Includes:
- Smoke tests: all outputs are positive / finite for random inputs.
- Invariant tests: mathematical equivalences between demographic models.
- Gradient tests: JAX gradients are computable and finite for all deterministic functions.
- Comparison tests (slow): deterministic and montecarlo results are sufficiently close.
"""

import bayesld
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bayesld import deterministic, montecarlo

jax.config.update("jax_enable_x64", True)

LEFT_BINS, RIGHT_BINS = bayesld.linear_bins()
MUTATION_RATE = 1e-8
RECOMBINATION_RATE = 1e-8
SEQUENCE_LENGTH = RIGHT_BINS[-1] * 2 / RECOMBINATION_RATE
SAMPLE_SIZE = 10
PLOIDY = 2

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

DET_KWARGS = dict(
    left_bins=LEFT_BINS,
    right_bins=RIGHT_BINS,
    mutation_rate=MUTATION_RATE,
    sample_size=SAMPLE_SIZE,
    ploidy=PLOIDY,
)

# MC kwargs with more replicates for comparison tests
MC_KWARGS_COMPARISON = dict(
    left_bins=LEFT_BINS,
    right_bins=RIGHT_BINS,
    mutation_rate=MUTATION_RATE,
    recombination_rate=RECOMBINATION_RATE,
    sequence_length=SEQUENCE_LENGTH,
    sample_size=SAMPLE_SIZE,
    random_seed=42,
    num_replicates=200,
    ploidy=PLOIDY,
    model="hudson",
)


@pytest.fixture()
def rng():
    return np.random.default_rng(123)


def _assert_positive(pi_mean, ld_mean):
    """Check pi is positive (true invariant) and LD is finite (can be slightly negative with MC noise)."""
    assert np.all(np.asarray(pi_mean) > 0), f"pi_mean not positive: {pi_mean}"
    assert np.all(np.isfinite(np.asarray(ld_mean))), f"ld_mean not finite: {ld_mean}"


# ── Constant ─────────────────────────────────────────────────────────────────


def test_deterministic_constant(rng):
    Ne = rng.uniform(10, 2000)
    pi_mean, ld_mean = deterministic.expected_constant(Ne=Ne, **DET_KWARGS)
    _assert_positive(pi_mean, ld_mean)


def test_montecarlo_constant(rng):
    Ne = rng.uniform(10, 2000)
    pi_vec, ld_mat = montecarlo.expected_constant(Ne=Ne, **MC_KWARGS)
    _assert_positive(pi_vec.mean(), ld_mat.mean(axis=0))


# ── Piecewise Exponential ────────────────────────────────────────────────────


def test_deterministic_piecewise_exponential(rng):
    Ne_c = rng.uniform(10, 2000)
    Ne_a = rng.uniform(10, 2000)
    t0 = rng.uniform(1, 100)
    alpha = rng.uniform(0.001, 0.1)
    pi_mean, ld_mean = deterministic.expected_piecewise_exponential(
        Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, alpha=alpha, **DET_KWARGS
    )
    _assert_positive(pi_mean, ld_mean)


def test_montecarlo_piecewise_exponential(rng):
    Ne_c = rng.uniform(10, 2000)
    Ne_a = rng.uniform(10, 2000)
    t0 = rng.uniform(1, 100)
    alpha = rng.uniform(0.001, 0.1)
    pi_vec, ld_mat = montecarlo.expected_piecewise_exponential(
        Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, alpha=alpha, **MC_KWARGS
    )
    _assert_positive(pi_vec.mean(), ld_mat.mean(axis=0))


# ── Exponential Carrying Capacity ────────────────────────────────────────────


def test_deterministic_exponential_carrying_capacity(rng):
    Ne_c = rng.uniform(10, 2000)
    Ne_a = rng.uniform(10, 2000)
    t0 = rng.uniform(1, 50)
    t1 = t0 + rng.uniform(1, 50)
    alpha = rng.uniform(0.001, 0.1)
    pi_mean, ld_mean = deterministic.expected_exponential_carrying_capacity(
        Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, t1=t1, alpha=alpha, **DET_KWARGS
    )
    _assert_positive(pi_mean, ld_mean)


def test_montecarlo_exponential_carrying_capacity(rng):
    Ne_c = rng.uniform(10, 2000)
    Ne_a = rng.uniform(10, 2000)
    t0 = rng.uniform(1, 50)
    t1 = t0 + rng.uniform(1, 50)
    alpha = rng.uniform(0.001, 0.1)
    pi_vec, ld_mat = montecarlo.expected_exponential_carrying_capacity(
        Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, t1=t1, alpha=alpha, **MC_KWARGS
    )
    _assert_positive(pi_vec.mean(), ld_mat.mean(axis=0))


# ── Piecewise Constant ───────────────────────────────────────────────────────


def test_deterministic_piecewise_constant(rng):
    n_epochs = rng.integers(2, 5)
    Ne_values = rng.uniform(10, 2000, size=n_epochs)
    t_boundaries = np.sort(rng.uniform(1, 100, size=n_epochs - 1))
    pi_mean, ld_mean = deterministic.expected_piecewise_constant(
        Ne_values=Ne_values, t_boundaries=t_boundaries, **DET_KWARGS
    )
    _assert_positive(pi_mean, ld_mean)


def test_montecarlo_piecewise_constant(rng):
    n_epochs = rng.integers(2, 5)
    Ne_values = rng.uniform(10, 2000, size=n_epochs)
    t_boundaries = np.sort(rng.uniform(1, 100, size=n_epochs - 1))
    pi_vec, ld_mat = montecarlo.expected_piecewise_constant(
        Ne_values=Ne_values, t_boundaries=t_boundaries, **MC_KWARGS
    )
    _assert_positive(pi_vec.mean(), ld_mat.mean(axis=0))


# ── Secondary Introduction ───────────────────────────────────────────────────


def test_deterministic_secondary_introduction(rng):
    Ne_1 = rng.uniform(10, 2000)
    Ne_2 = rng.uniform(10, 2000)
    Ne_a = rng.uniform(10, 2000)
    t0 = rng.uniform(1, 50)
    t1 = t0 + rng.uniform(1, 50)
    migration_rate = rng.uniform(0.0001, 0.1)
    pi_mean, ld_mean = deterministic.expected_secondary_introduction(
        Ne_1=Ne_1,
        Ne_2=Ne_2,
        Ne_a=Ne_a,
        t0=t0,
        t1=t1,
        migration_rate=migration_rate,
        **DET_KWARGS,
    )
    _assert_positive(pi_mean, ld_mean)


def test_montecarlo_secondary_introduction(rng):
    Ne_1 = rng.uniform(10, 2000)
    Ne_2 = rng.uniform(10, 2000)
    Ne_a = rng.uniform(10, 2000)
    t0 = rng.uniform(1, 50)
    t1 = t0 + rng.uniform(1, 50)
    migration_rate = rng.uniform(0.0001, 0.1)
    pi_vec, ld_mat = montecarlo.expected_secondary_introduction(
        Ne_1=Ne_1,
        Ne_2=Ne_2,
        Ne_a=Ne_a,
        t0=t0,
        t1=t1,
        migration_rate=migration_rate,
        **MC_KWARGS,
    )
    _assert_positive(pi_vec.mean(), ld_mat.mean(axis=0))


# ══════════════════════════════════════════════════════════════════════════════
# Invariant tests
# ══════════════════════════════════════════════════════════════════════════════


class TestInvariants:
    """Mathematical equivalences between demographic models."""

    def test_piecewise_exponential_alpha_zero_equals_piecewise_constant(self):
        """When alpha=0, piecewise exponential Ne(t)=Ne_c for t<t0, Ne_a for t>=t0
        should equal a 2-epoch piecewise constant."""
        Ne_c, Ne_a, t0 = 500.0, 2000.0, 50.0
        pi_exp, ld_exp = deterministic.expected_piecewise_exponential(
            Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, alpha=0.0, **DET_KWARGS
        )
        pi_pc, ld_pc = deterministic.expected_piecewise_constant(
            Ne_values=np.array([Ne_c, Ne_a]),
            t_boundaries=np.array([t0]),
            **DET_KWARGS,
        )
        np.testing.assert_allclose(pi_exp, pi_pc, rtol=1e-4)
        np.testing.assert_allclose(ld_exp, ld_pc, rtol=1e-4)

    def test_piecewise_constant_identical_epochs_equals_constant(self):
        """When all Ne values are the same, piecewise constant should equal constant."""
        Ne = 1000.0
        pi_const, ld_const = deterministic.expected_constant(Ne=Ne, **DET_KWARGS)
        pi_pc, ld_pc = deterministic.expected_piecewise_constant(
            Ne_values=np.array([Ne, Ne, Ne]),
            t_boundaries=np.array([50.0, 100.0]),
            **DET_KWARGS,
        )
        np.testing.assert_allclose(pi_const, pi_pc, rtol=1e-4)
        np.testing.assert_allclose(ld_const, ld_pc, rtol=1e-4)

    def test_exponential_carrying_capacity_alpha_zero_equals_piecewise_constant(self):
        """When alpha=0, exp carrying capacity Ne(t)=Ne_c for t<t1, Ne_a for t>=t1
        should equal a 2-epoch piecewise constant."""
        Ne_c, Ne_a, t0, t1 = 500.0, 2000.0, 20.0, 60.0
        pi_ecc, ld_ecc = deterministic.expected_exponential_carrying_capacity(
            Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, t1=t1, alpha=0.0, **DET_KWARGS
        )
        # With alpha=0 the exponential piece collapses: Ne_c for t<t1, Ne_a for t>=t1
        pi_pc, ld_pc = deterministic.expected_piecewise_constant(
            Ne_values=np.array([Ne_c, Ne_a]),
            t_boundaries=np.array([t1]),
            **DET_KWARGS,
        )
        np.testing.assert_allclose(pi_ecc, pi_pc, rtol=1e-4)
        np.testing.assert_allclose(ld_ecc, ld_pc, rtol=1e-4)

    def test_piecewise_constant_two_identical_adjacent_pieces(self):
        """Two identical adjacent pieces [Ne_c, Ne_c, Ne_a] should equal [Ne_c, Ne_a]."""
        Ne_c, Ne_a = 500.0, 2000.0
        t0, t1 = 30.0, 60.0
        pi_3, ld_3 = deterministic.expected_piecewise_constant(
            Ne_values=np.array([Ne_c, Ne_c, Ne_a]),
            t_boundaries=np.array([t0, t1]),
            **DET_KWARGS,
        )
        pi_2, ld_2 = deterministic.expected_piecewise_constant(
            Ne_values=np.array([Ne_c, Ne_a]),
            t_boundaries=np.array([t1]),
            **DET_KWARGS,
        )
        np.testing.assert_allclose(pi_3, pi_2, rtol=1e-4)
        np.testing.assert_allclose(ld_3, ld_2, rtol=1e-4)

    def test_piecewise_exponential_alpha_zero_equals_piecewise_exponential_small_alpha(
        self,
    ):
        """Piecewise exponential with alpha=0 should be close to alpha very small."""
        Ne_c, Ne_a, t0 = 500.0, 2000.0, 50.0
        pi_zero, ld_zero = deterministic.expected_piecewise_exponential(
            Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, alpha=0.0, **DET_KWARGS
        )
        pi_small, ld_small = deterministic.expected_piecewise_exponential(
            Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, alpha=1e-8, **DET_KWARGS
        )
        np.testing.assert_allclose(pi_zero, pi_small, rtol=1e-3)
        np.testing.assert_allclose(ld_zero, ld_small, rtol=1e-3)

    def test_exponential_carrying_capacity_alpha_zero_continuous(self):
        """Exp carrying capacity with alpha=0 should be close to alpha very small."""
        Ne_c, Ne_a, t0, t1 = 500.0, 2000.0, 20.0, 60.0
        pi_zero, ld_zero = deterministic.expected_exponential_carrying_capacity(
            Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, t1=t1, alpha=0.0, **DET_KWARGS
        )
        pi_small, ld_small = deterministic.expected_exponential_carrying_capacity(
            Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, t1=t1, alpha=1e-8, **DET_KWARGS
        )
        np.testing.assert_allclose(pi_zero, pi_small, rtol=1e-3)
        np.testing.assert_allclose(ld_zero, ld_small, rtol=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# Gradient tests for deterministic functions
# ══════════════════════════════════════════════════════════════════════════════


class TestGradients:
    """Verify JAX gradients are computable and finite for all deterministic functions."""

    def test_gradient_constant(self):
        def f(Ne):
            pi, ld = deterministic.expected_constant(Ne=Ne, **DET_KWARGS)
            return pi + ld.sum()

        g = jax.grad(f)(1000.0)
        assert jnp.isfinite(g), f"gradient not finite: {g}"

    def test_gradient_piecewise_exponential(self):
        def f(Ne_c, Ne_a, t0, alpha):
            pi, ld = deterministic.expected_piecewise_exponential(
                Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, alpha=alpha, **DET_KWARGS
            )
            return pi + ld.sum()

        grads = jax.grad(f, argnums=(0, 1, 2, 3))(500.0, 2000.0, 50.0, 0.01)
        for i, g in enumerate(grads):
            assert jnp.isfinite(g), f"gradient {i} not finite: {g}"

    def test_gradient_exponential_carrying_capacity(self):
        def f(Ne_c, Ne_a, t0, t1, alpha):
            pi, ld = deterministic.expected_exponential_carrying_capacity(
                Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, t1=t1, alpha=alpha, **DET_KWARGS
            )
            return pi + ld.sum()

        grads = jax.grad(f, argnums=(0, 1, 2, 3, 4))(500.0, 2000.0, 20.0, 60.0, 0.01)
        for i, g in enumerate(grads):
            assert jnp.isfinite(g), f"gradient {i} not finite: {g}"

    def test_gradient_piecewise_constant(self):
        def f(Ne_values):
            pi, ld = deterministic.expected_piecewise_constant(
                Ne_values=Ne_values,
                t_boundaries=jnp.array([30.0, 60.0]),
                **DET_KWARGS,
            )
            return pi + ld.sum()

        g = jax.grad(f)(jnp.array([500.0, 1000.0, 2000.0]))
        assert jnp.all(jnp.isfinite(g)), f"gradient not finite: {g}"

    def test_gradient_secondary_introduction(self):
        def f(Ne_1, Ne_2, Ne_a, t0, t1, m):
            pi, ld = deterministic.expected_secondary_introduction(
                Ne_1=Ne_1,
                Ne_2=Ne_2,
                Ne_a=Ne_a,
                t0=t0,
                t1=t1,
                migration_rate=m,
                **DET_KWARGS,
            )
            return pi + ld.sum()

        grads = jax.grad(f, argnums=(0, 1, 2, 3, 4, 5))(
            3000.0, 3000.0, 6000.0, 20.0, 60.0, 0.1
        )
        for i, g in enumerate(grads):
            assert jnp.isfinite(g), f"gradient {i} not finite: {g}"


# ══════════════════════════════════════════════════════════════════════════════
# Comparison tests: deterministic vs Monte Carlo (slow)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestDeterministicVsMonteCarlo:
    """Check that deterministic and MC results are sufficiently close.

    These tests are slow because they require many MC replicates.
    Run with: pytest -m slow
    """

    def test_comparison_constant(self):
        Ne = 1000.0
        pi_det, ld_det = deterministic.expected_constant(Ne=Ne, **DET_KWARGS)
        pi_vec, ld_mat = montecarlo.expected_constant(Ne=Ne, **MC_KWARGS_COMPARISON)
        np.testing.assert_allclose(pi_det, pi_vec.mean(), rtol=0.15)
        np.testing.assert_allclose(ld_det, ld_mat.mean(axis=0), rtol=0.2)

    def test_comparison_piecewise_exponential(self):
        Ne_c, Ne_a, t0, alpha = 500.0, 2000.0, 50.0, 0.01
        pi_det, ld_det = deterministic.expected_piecewise_exponential(
            Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, alpha=alpha, **DET_KWARGS
        )
        pi_vec, ld_mat = montecarlo.expected_piecewise_exponential(
            Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, alpha=alpha, **MC_KWARGS_COMPARISON
        )
        np.testing.assert_allclose(pi_det, pi_vec.mean(), rtol=0.15)
        np.testing.assert_allclose(ld_det, ld_mat.mean(axis=0), rtol=0.2)

    def test_comparison_exponential_carrying_capacity(self):
        Ne_c, Ne_a, t0, t1, alpha = 500.0, 2000.0, 20.0, 60.0, 0.01
        pi_det, ld_det = deterministic.expected_exponential_carrying_capacity(
            Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, t1=t1, alpha=alpha, **DET_KWARGS
        )
        pi_vec, ld_mat = montecarlo.expected_exponential_carrying_capacity(
            Ne_c=Ne_c, Ne_a=Ne_a, t0=t0, t1=t1, alpha=alpha, **MC_KWARGS_COMPARISON
        )
        np.testing.assert_allclose(pi_det, pi_vec.mean(), rtol=0.15)
        np.testing.assert_allclose(ld_det, ld_mat.mean(axis=0), rtol=0.2)

    def test_comparison_piecewise_constant(self):
        Ne_values = np.array([500.0, 1000.0, 2000.0])
        t_boundaries = np.array([30.0, 60.0])
        pi_det, ld_det = deterministic.expected_piecewise_constant(
            Ne_values=Ne_values, t_boundaries=t_boundaries, **DET_KWARGS
        )
        pi_vec, ld_mat = montecarlo.expected_piecewise_constant(
            Ne_values=Ne_values, t_boundaries=t_boundaries, **MC_KWARGS_COMPARISON
        )
        np.testing.assert_allclose(pi_det, pi_vec.mean(), rtol=0.15)
        np.testing.assert_allclose(ld_det, ld_mat.mean(axis=0), rtol=0.2)

    def test_comparison_secondary_introduction(self):
        Ne_1, Ne_2, Ne_a = 3000.0, 3000.0, 6000.0
        # Approximation fails for small migration rates
        t0, t1, m = 20.0, 60.0, 0.1
        pi_det, ld_det = deterministic.expected_secondary_introduction(
            Ne_1=Ne_1,
            Ne_2=Ne_2,
            Ne_a=Ne_a,
            t0=t0,
            t1=t1,
            migration_rate=m,
            **DET_KWARGS,
        )
        pi_vec, ld_mat = montecarlo.expected_secondary_introduction(
            Ne_1=Ne_1,
            Ne_2=Ne_2,
            Ne_a=Ne_a,
            t0=t0,
            t1=t1,
            migration_rate=m,
            **MC_KWARGS_COMPARISON,
        )
        np.testing.assert_allclose(pi_det, pi_vec.mean(), rtol=0.15)
        np.testing.assert_allclose(ld_det, ld_mat.mean(axis=0), rtol=0.2)
