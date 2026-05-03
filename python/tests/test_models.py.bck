"""
Tests for bayesld.models — Stan-based demographic inference.

Strategy: use the deterministic module to compute noiseless expected statistics
for Ne=100, then simulate iid windows by adding Gaussian noise.
This exercises the full Stan compilation + optimization path without msprime.
"""

import warnings

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
    n_bins = len(LEFT_BINS)
    pts = []
    for ne in ne_grid:
        rel_bias = np.zeros(n_bins)
        eps_rel = np.full(n_bins, 0.01)
        pts.append({"rel_bias": rel_bias, "eps_rel": eps_rel})
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
    assert "gp_bias_ld" in result
    assert "gp_rho_r" in result
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


from bayesld.models import (
    PiecewiseConstantDemography,
    PiecewiseExponentialDemography,
    ExponentialCarryingCapacityDemography,
)

# ---------------------------------------------------------------------------
# PiecewiseConstantDemography fixtures and helpers
# ---------------------------------------------------------------------------

NE_VALUES_TRUE = np.array([5_000.0, 10_000.0, 50_000.0])
T_BOUNDARIES_TRUE = np.array([10.0, 1_000.0])


@pytest.fixture(scope="module")
def pc_window_data():
    """Synthetic iid windows built from a 3-epoch piecewise-constant demography."""
    pi_exp, ld_exp = deterministic.expected_piecewise_constant(
        NE_VALUES_TRUE, T_BOUNDARIES_TRUE,
        LEFT_BINS, RIGHT_BINS, MUTATION_RATE,
        sample_size=NUM_SAMPLES,
    )
    pi_exp = float(pi_exp)
    ld_exp = np.array(ld_exp)
    rng = np.random.default_rng(99)
    diversity = rng.normal(pi_exp, pi_exp * NOISE_FRACTION, size=NUM_WINDOWS)
    ld = rng.normal(
        ld_exp, ld_exp[None, :] * NOISE_FRACTION, size=(NUM_WINDOWS, len(LEFT_BINS))
    )
    return diversity, ld


def _make_pc_model(window_data, **kwargs):
    diversity, ld = window_data
    return PiecewiseConstantDemography(
        diversity=diversity,
        ld=ld,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        num_samples=NUM_SAMPLES,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        **kwargs,
    )


def _pc_synthetic_eval_points(ne_values_grid, t_bnd, fixed_t, n=20):
    """Zero-bias eval points for PiecewiseConstantDemography surrogate tests."""
    n_bins = len(LEFT_BINS)
    pts = []
    for ne_values in ne_values_grid:
        rel_bias = np.zeros(n_bins)
        eps_rel = np.full(n_bins, 0.01)
        pts.append({"rel_bias": rel_bias, "eps_rel": eps_rel})
    return pts


# ---------------------------------------------------------------------------
# PiecewiseConstantDemography: free-boundary tests (default, n_epochs=3)
# ---------------------------------------------------------------------------

def _pc_inits_free():
    return {
        "log_Ne_values": np.log(NE_VALUES_TRUE),
        "log_t_boundaries": np.log(T_BOUNDARIES_TRUE),
    }


def _pc_inits_fixed():
    return {"log_Ne_values": np.log(NE_VALUES_TRUE)}


def test_pc_optimize_free_boundaries(pc_window_data):
    """Approximate model with free boundaries compiles and returns expected keys."""
    model = _make_pc_model(pc_window_data)
    result = model.optimize(show_console=False, seed=1, inits=_pc_inits_free())
    assert "Ne_values" in result
    assert "t_boundaries" in result
    assert "E_pi" in result
    assert "approx_ld" in result
    assert "log_lik" in result
    assert result["Ne_values"].shape == (3,)
    assert result["t_boundaries"].shape == (2,)
    assert len(result["log_lik"]) == NUM_WINDOWS


def test_pc_sample_warns_without_eval_points(pc_window_data):
    """sample() warns when no eval points are present."""
    model = _make_pc_model(pc_window_data)
    with pytest.warns(UserWarning, match="approximate LD"):
        model.sample(
            show_console=False, seed=1, chains=1, iter_warmup=50, iter_sampling=30,
            inits=_pc_inits_free(),
        )


def test_pc_surrogate_free_boundaries(pc_window_data):
    """Surrogate model with free boundaries compiles and returns GP keys."""
    model = _make_pc_model(pc_window_data)
    rng = np.random.default_rng(7)
    # Vary both Ne and t_boundaries so no GP input dimension has zero variance.
    ne_grid = [NE_VALUES_TRUE * f for f in np.exp(np.linspace(-0.5, 0.5, 20))]
    t_grid = [T_BOUNDARIES_TRUE * f for f in np.exp(rng.uniform(-0.4, 0.4, size=20))]
    pts = [
        _pc_synthetic_eval_points([ne], t, fixed_t=False)[0]
        for ne, t in zip(ne_grid, t_grid)
    ]
    model.add_eval_points(pts)
    result = model.optimize(show_console=False, seed=1, inits=_pc_inits_free())
    assert "gp_bias_ld" in result
    assert "gp_rho_r" in result
    assert "gp_alpha" in result
    assert result["Ne_values"].shape == (3,)


# ---------------------------------------------------------------------------
# PiecewiseConstantDemography: fixed-boundary tests (n_epochs=3)
# ---------------------------------------------------------------------------

def test_pc_optimize_fixed_boundaries(pc_window_data):
    """Fixed-boundary model compiles, returns t_boundaries equal to fixed values."""
    model = _make_pc_model(pc_window_data, t_boundaries=T_BOUNDARIES_TRUE)
    result = model.optimize(show_console=False, seed=1, inits=_pc_inits_fixed())
    assert "Ne_values" in result
    assert "t_boundaries" in result
    assert result["Ne_values"].shape == (3,)
    np.testing.assert_array_equal(result["t_boundaries"], T_BOUNDARIES_TRUE)
    assert len(result["log_lik"]) == NUM_WINDOWS


def test_pc_optimize_two_epoch_boundary_shape(pc_window_data):
    """Two-epoch model returns a 1-D boundary array (shape (1,), not scalar)."""
    model = _make_pc_model(pc_window_data, n_epochs=2)
    result = model.optimize(
        show_console=False,
        seed=1,
        inits={
            "log_Ne_values": np.log(np.array([5_000.0, 20_000.0])),
            "log_t_boundaries": np.log(np.array([500.0])),
        },
    )
    assert result["Ne_values"].shape == (2,)
    assert result["t_boundaries"].shape == (1,)


def test_pc_custom_prior_and_extra_parameters(pc_window_data):
    """User-defined Stan prior and extra parameters are supported."""
    prior = """
    mu_log_ne ~ normal(11.5, 3.0);
    log_Ne_values ~ normal(mu_log_ne, 0.5);
    log_t_boundaries[1] ~ normal(log(10.0), 1.0);
    log_t_boundaries[2] ~ normal(log(200.0), 1.0);
    """
    model = _make_pc_model(
        pc_window_data,
        prior=prior,
        extra_parameters="    real mu_log_ne;",
    )
    result = model.optimize(show_console=False, seed=1, inits=_pc_inits_free())
    assert result["Ne_values"].shape == (3,)
    assert result["t_boundaries"].shape == (2,)


def test_pc_surrogate_fixed_boundaries(pc_window_data):
    """Surrogate model with fixed boundaries uses D=n_epochs GP input space."""
    model = _make_pc_model(pc_window_data, t_boundaries=T_BOUNDARIES_TRUE)
    ne_grid = [NE_VALUES_TRUE * f for f in np.exp(np.linspace(-0.5, 0.5, 20))]
    pts = _pc_synthetic_eval_points(ne_grid, T_BOUNDARIES_TRUE, fixed_t=True)
    model.add_eval_points(pts)
    result = model.optimize(show_console=False, seed=1, inits=_pc_inits_fixed())
    assert "gp_bias_ld" in result
    assert result["Ne_values"].shape == (3,)
    np.testing.assert_array_equal(result["t_boundaries"], T_BOUNDARIES_TRUE)


def test_pc_invalid_t_boundaries_shape(pc_window_data):
    """Providing t_boundaries with wrong shape raises ValueError."""
    with pytest.raises(ValueError, match="shape"):
        _make_pc_model(pc_window_data, t_boundaries=np.array([100.0]))


def test_pc_invalid_t_boundaries_not_sorted(pc_window_data):
    """Providing non-increasing t_boundaries raises ValueError."""
    with pytest.raises(ValueError, match="strictly increasing"):
        _make_pc_model(pc_window_data, t_boundaries=np.array([1000.0, 10.0]))


def test_pc_add_eval_points_validates(pc_window_data):
    """add_eval_points rejects dicts missing required keys."""
    model = _make_pc_model(pc_window_data)
    with pytest.raises(ValueError, match="keys"):
        model.add_eval_points([{"ne_values": NE_VALUES_TRUE}])


def test_pc_eval_points_property(pc_window_data):
    """eval_points returns a copy; mutating it does not affect the model."""
    model = _make_pc_model(pc_window_data)
    ne_grid = [NE_VALUES_TRUE * f for f in np.exp(np.linspace(-0.5, 0.5, 5))]
    pts = _pc_synthetic_eval_points(ne_grid, T_BOUNDARIES_TRUE, fixed_t=False)
    model.add_eval_points(pts)
    copy = model.eval_points
    copy.clear()
    assert len(model.eval_points) == 5


# ---------------------------------------------------------------------------
# PiecewiseExponentialDemography smoke tests
# ---------------------------------------------------------------------------

def _make_pe_model(window_data, **kwargs):
    diversity, ld = window_data
    return PiecewiseExponentialDemography(
        diversity=diversity,
        ld=ld,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        num_samples=NUM_SAMPLES,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        **kwargs,
    )


def test_pe_optimize(window_data):
    """PiecewiseExponentialDemography compiles and returns expected keys."""
    model = _make_pe_model(window_data)
    result = model.optimize(show_console=False, seed=1)
    assert "Ne_c" in result
    assert "Ne_a" in result
    assert "t0" in result
    assert "alpha" in result
    assert "log_fold_change" in result
    assert "E_pi" in result
    assert "approx_ld" in result
    assert "log_lik" in result
    assert len(result["log_lik"]) == NUM_WINDOWS


def test_pe_sample_warns_without_eval_points(window_data):
    """sample() should warn when no eval points are present."""
    model = _make_pe_model(window_data)
    with pytest.warns(UserWarning, match="approximate LD"):
        model.sample(
            show_console=False, seed=1, chains=1, iter_warmup=50, iter_sampling=30,
        )


# ---------------------------------------------------------------------------
# ExponentialCarryingCapacityDemography smoke tests
# ---------------------------------------------------------------------------

def _make_cc_model(window_data, **kwargs):
    diversity, ld = window_data
    return ExponentialCarryingCapacityDemography(
        diversity=diversity,
        ld=ld,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        num_samples=NUM_SAMPLES,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        **kwargs,
    )


def test_cc_optimize(window_data):
    """ExponentialCarryingCapacityDemography compiles and returns expected keys."""
    model = _make_cc_model(window_data)
    result = model.optimize(
        show_console=False, seed=1,
        require_converged=False,
        inits={
            "log_Ne_c": np.log(NE_TRUE),
            "log_Ne_a": np.log(NE_TRUE),
            "log_t_boundaries": np.log([10.0, 100.0]),
            "alpha": 0.0,
        },
    )
    assert "Ne_c" in result
    assert "Ne_a" in result
    assert "t0" in result
    assert "t1" in result
    assert "alpha" in result
    assert "E_pi" in result
    assert "approx_ld" in result
    assert "log_lik" in result
    assert len(result["log_lik"]) == NUM_WINDOWS


def test_cc_active_learning_requires_sequence_length(window_data):
    """surrogate_active_learning raises ValueError without sequence_length."""
    model = _make_cc_model(window_data)
    with pytest.raises(ValueError, match="sequence_length"):
        model.surrogate_active_learning()


def test_cc_sample_warns_without_eval_points(window_data):
    """sample() should warn when no eval points are present."""
    model = _make_cc_model(window_data)
    _inits = {
        "log_Ne_c": np.log(NE_TRUE),
        "log_Ne_a": np.log(NE_TRUE),
        "log_t_boundaries": np.log([10.0, 100.0]),
        "alpha": 0.0,
    }
    with pytest.warns(UserWarning, match="approximate LD"):
        model.sample(
            show_console=False, seed=1, chains=1, iter_warmup=50, iter_sampling=30,
            inits=_inits,
        )
