"""
Tests for bayesld.models.ConstantDemography.

Strategy: draw Ne from the prior log(Ne) ~ Normal(log(500), 1), compute
noiseless expected diversity and LD via the deterministic module, add a small
relative GP-like bias plus Gaussian noise, and verify the Stan model compiles,
runs, and returns expected types.
"""

import cmdstanpy
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
NUM_WINDOWS = 200
NOISE_REL_SD = 0.01

LEFT_BINS, RIGHT_BINS = bayesld.linear_bins()

LOG_NE_MU = np.log(500.0)
LOG_NE_SIGMA = 1.0


def _simulate(seed=42):
    rng = np.random.default_rng(seed)
    ne = float(np.exp(rng.normal(LOG_NE_MU, LOG_NE_SIGMA)))

    pi_exp, ld_exp = deterministic.expected_constant(
        ne, LEFT_BINS, RIGHT_BINS, MUTATION_RATE, sample_size=NUM_SAMPLES
    )
    pi_exp = float(pi_exp)
    ld_exp = np.asarray(ld_exp)

    # Simulate per-bin relative bias (GP-like)
    bias = rng.normal(0, 0.005, size=len(LEFT_BINS))
    ld_true = ld_exp * (1 + bias)

    diversity = rng.normal(pi_exp, abs(pi_exp) * NOISE_REL_SD, size=NUM_WINDOWS)
    ld = rng.normal(
        ld_true,
        np.abs(ld_true)[None, :] * NOISE_REL_SD,
        size=(NUM_WINDOWS, len(LEFT_BINS)),
    )
    assert np.all(np.isfinite(diversity)), "diversity contains non-finite values"
    assert np.all(np.isfinite(ld)), "ld contains non-finite values"
    assert np.all(np.std(ld, axis=0) > 0), "ld has zero-variance bins"
    assert np.std(diversity) > 0, "diversity has zero variance"
    return ne, diversity, ld


def _make_model(**kwargs):
    _, diversity, ld = _simulate()
    defaults = dict(
        diversity=diversity,
        ld=ld,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        num_samples=NUM_SAMPLES,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
    )
    defaults.update(kwargs)
    return ConstantDemography(**defaults)


# ── Non-slow tests ──────────────────────────────────────────────────────


def test_primitives():
    model = _make_model()
    code = model.get_stan_code()
    assert isinstance(code, str)
    assert "parameters" in code
    assert "n_synthetic" in code
    assert isinstance(model.model, cmdstanpy.CmdStanModel)

    data = model.stan_data()
    assert isinstance(data, dict)
    assert data["n_synthetic"] == 0
    assert data["eval_rel_bias"].shape == (0, len(LEFT_BINS))


def test_stan_data_keys():
    model = _make_model()
    data = model.stan_data()
    expected_keys = {
        "n_bins",
        "num_windows",
        "left_bins",
        "right_bins",
        "mutation_rate",
        "sample_size",
        "pi_array",
        "ld_mat",
        "n_synthetic",
        "eval_rel_bias",
        "eval_eps_rel",
        "hsgp_c",
        "hsgp_m",
        "gp_alpha_std",
    }
    assert expected_keys == set(data.keys())


def test_update_data():
    model = _make_model()
    old_model_obj = model.model
    _, new_div, new_ld = _simulate(seed=99)
    model.update_data(diversity=new_div, ld=new_ld)
    assert model.model is old_model_obj  # no recompilation
    data = model.stan_data()
    np.testing.assert_array_equal(data["pi_array"], new_div)


def test_update_prior_recompiles():
    model = _make_model()
    old_model_obj = model.model
    model.update_prior(prior="    log_Ne ~ normal(5, 2.0);")
    assert model.model is not old_model_obj  # recompiled


def test_update_prior_gp_alpha_std_no_recompile():
    model = _make_model()
    old_model_obj = model.model
    model.update_prior(gp_alpha_std=0.01)
    assert model.model is old_model_obj  # no recompilation
    assert model.stan_data()["gp_alpha_std"] == 0.01


def test_add_synthetic_points():
    model = _make_model()
    n_bins = len(LEFT_BINS)
    points = [
        {"rel_bias": np.zeros(n_bins), "eps_rel": np.ones(n_bins) * 0.01},
    ]
    model.add_synthetic_points(points)
    assert len(model.synthetic_points) == 1
    data = model.stan_data()
    assert data["n_synthetic"] == 1
    assert data["eval_rel_bias"].shape == (1, n_bins)


def test_add_synthetic_points_validation():
    model = _make_model()
    with pytest.raises(ValueError, match="keys"):
        model.add_synthetic_points([{"rel_bias": np.zeros(5)}])


def test_generate_stan_default():
    from bayesld.models.constant import _generate_stan

    code = _generate_stan()
    assert "log_Ne ~ normal(3, 1.0);" in code
    assert "real<offset=log_ne_offset> log_Ne;" in code
    assert "n_synthetic" in code
    assert "corrected_ld" in code


def test_generate_stan_custom():
    from bayesld.models.constant import _generate_stan

    code = _generate_stan(
        prior="    log_Ne ~ normal(10, 0.5);",
        parameters="    real log_Ne;",
    )
    assert "log_Ne ~ normal(10, 0.5);" in code
    assert "real log_Ne;" in code


# ── Slow inference tests ────────────────────────────────────────────────


@pytest.mark.slow
def test_sample_default_prior():
    model = _make_model()
    idata = model.sample(chains=2, iter_warmup=10, iter_sampling=10)
    assert "posterior" in idata.children
    assert "Ne" in idata["posterior"].ds


@pytest.mark.slow
def test_sample_true_prior():
    ne_true, diversity, ld = _simulate()
    log_ne_true = np.log(ne_true)
    model = ConstantDemography(
        diversity=diversity,
        ld=ld,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        num_samples=NUM_SAMPLES,
        left_bins=LEFT_BINS,
        right_bins=RIGHT_BINS,
        prior=f"    log_Ne ~ normal({log_ne_true:.4f}, 0.1);",
    )
    idata = model.sample(chains=2, iter_warmup=10, iter_sampling=10)
    assert "Ne" in idata["posterior"].ds
