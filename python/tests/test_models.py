"""
Tests for bayesld.models — Stan-based demographic inference.

Strategy: use the deterministic module to compute noiseless expected statistics
for Ne=100, then simulate iid windows by adding Gaussian noise.
This exercises the full Stan compilation + optimization path without msprime.
"""

import jax
import numpy as np
import pytest

import bayesld
from bayesld import deterministic
from bayesld.models import ConstantDemography

jax.config.update("jax_enable_x64", True)

MUTATION_RATE = 1e-8
RECOMBINATION_RATE = 1e-8
NUM_SAMPLES = 50
NE_TRUE = 100.0
NUM_WINDOWS = 200
NOISE_FRACTION = 0.05

LEFT_BINS, RIGHT_BINS = bayesld.linear_bins()


@pytest.fixture(scope="module")
def window_data():
    """Synthetic iid windows for Ne=100 built from deterministic expected values."""
    pi_exp, ld_exp = deterministic.expected_constant(
        NE_TRUE, LEFT_BINS, RIGHT_BINS, MUTATION_RATE, sample_size=NUM_SAMPLES
    )
    pi_exp = float(pi_exp)
    ld_exp = np.array(ld_exp)
    rng = np.random.default_rng(42)
    diversity = rng.normal(pi_exp, pi_exp * NOISE_FRACTION, size=NUM_WINDOWS)
    ld = rng.normal(
        ld_exp, ld_exp[None, :] * NOISE_FRACTION, size=(NUM_WINDOWS, len(LEFT_BINS))
    )
    return diversity, ld


def _make_model(window_data, **kwargs):
    diversity, ld = window_data
    return ConstantDemography(
        diversity=diversity,
        ld=ld,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        num_samples=NUM_SAMPLES,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        **kwargs,
    )


def _synthetic_eval_points(n=20):
    """Pre-computed MC eval points with zero bias (for fast testing)."""
    ne_grid = np.exp(np.linspace(np.log(10), np.log(1000), n))
    pts = []
    for ne in ne_grid:
        _, ld = deterministic.expected_constant(
            ne, LEFT_BINS, RIGHT_BINS, MUTATION_RATE, sample_size=NUM_SAMPLES
        )
        ll = float(np.sum(np.array(ld)))
        pts.append({"ne": ne, "loglik_det": ll, "bc_loglik": ll, "epsilon": 0.01})
    return pts


# ---------------------------------------------------------------------------
# Approximate model tests
# ---------------------------------------------------------------------------

def test_optimize_recovers_ne(window_data):
    """MAP estimate should be within 5 % of the true Ne."""
    model = _make_model(window_data)
    result = model.optimize(show_console=False, seed=1)
    assert "Ne" in result
    assert "log_lik" in result
    assert len(result["log_lik"]) == NUM_WINDOWS
    assert abs(result["Ne"] - NE_TRUE) / NE_TRUE < 0.05


def test_sample_warns_without_eval_points(window_data):
    """sample() should warn when no eval points are present."""
    model = _make_model(window_data)
    with pytest.warns(UserWarning, match="approximate LD"):
        model.sample(show_console=False, seed=1, chains=1, iter_warmup=100, iter_sampling=50)


def test_custom_prior(window_data):
    """A tight prior centred on the true Ne should still recover it."""
    prior = f"log_Ne ~ normal({np.log(NE_TRUE):.4f}, 0.5);"
    model = _make_model(window_data, prior=prior)
    result = model.optimize(show_console=False, seed=1)
    assert abs(result["Ne"] - NE_TRUE) / NE_TRUE < 0.05


def test_extra_parameters(window_data):
    """Extra parameters can be declared and referenced in the prior."""
    prior = """
    mu_log_Ne ~ normal(11.5, 3.0);
    log_Ne ~ normal(mu_log_Ne, 0.5);
    """
    model = _make_model(
        window_data,
        prior=prior,
        extra_parameters="    real mu_log_Ne;",
    )
    result = model.optimize(show_console=False, seed=1)
    assert abs(result["Ne"] - NE_TRUE) / NE_TRUE < 0.05


# ---------------------------------------------------------------------------
# GP-surrogate model tests (via add_eval_points)
# ---------------------------------------------------------------------------

def test_surrogate_optimize(window_data):
    """Surrogate model returns GP-specific keys and recovers Ne within 10 %."""
    model = _make_model(window_data)
    model.add_eval_points(_synthetic_eval_points())
    result = model.optimize(show_console=False, seed=1)
    assert "gp_bias" in result
    assert "gp_rho" in result
    assert "gp_alpha" in result
    assert len(result["log_lik"]) == NUM_WINDOWS
    assert abs(result["Ne"] - NE_TRUE) / NE_TRUE < 0.10


def test_surrogate_no_warning_on_sample(window_data):
    """sample() should not warn when eval points are present."""
    model = _make_model(window_data)
    model.add_eval_points(_synthetic_eval_points())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.sample(show_console=False, seed=1, chains=1, iter_warmup=100, iter_sampling=50)


def test_add_eval_points_validates(window_data):
    """add_eval_points should reject dicts with missing keys."""
    model = _make_model(window_data)
    with pytest.raises(ValueError, match="keys"):
        model.add_eval_points([{"ne": 100.0}])


def test_eval_points_property(window_data):
    """eval_points returns a copy; mutating it does not affect the model."""
    model = _make_model(window_data)
    pts = _synthetic_eval_points(n=5)
    model.add_eval_points(pts)
    copy = model.eval_points
    copy.clear()
    assert len(model.eval_points) == 5


import warnings
